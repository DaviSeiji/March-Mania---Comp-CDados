"""AutoGluon opcional: seleção interna agrupada e avaliação temporal externa."""

from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
import gc

import numpy as np
import pandas as pd

from .metrics import logistic_brier


def ambiente_autogluon():
    """Versões instaladas; presença do pacote não garante sucesso de um backend."""
    pacotes = ["autogluon.tabular", "numpy", "pandas", "scipy", "scikit-learn",
               "lightgbm", "catboost", "xgboost", "xgboost-cpu", "torch", "fastai", "ray"]
    resultado = {}
    for pacote in pacotes:
        try:
            resultado[pacote] = version(pacote)
        except PackageNotFoundError:
            resultado[pacote] = "não instalado"
    return resultado


def _backend():
    try:
        from autogluon.tabular import TabularPredictor
        from .metrics_autogluon import logistic_brier_scorer
    except ImportError as exc:
        raise ImportError("Instale requirements-autogluon.txt no ambiente do kernel.") from exc
    return TabularPredictor, logistic_brier_scorer


def _validar(tabela, features, *, historico=False):
    features = [features] if isinstance(features, str) else list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("Informe features únicas e não vazias.")
    if set(features) & {"Season", "Target", "Pred", "TeamA", "TeamB", "FoldInterno"}:
        raise ValueError("Alvo, temporada e identificadores não são features.")
    colunas = features + (["Season", "Target"] if historico else [])
    if tabela.empty or not set(colunas).issubset(tabela.columns):
        raise ValueError("Tabela vazia ou faltam colunas necessárias.")
    if not all(pd.api.types.is_numeric_dtype(tabela[c]) for c in colunas):
        raise ValueError("Valores devem ser numéricos.")
    if not np.isfinite(tabela[colunas].to_numpy(dtype=float)).all():
        raise ValueError("Valores ausentes ou infinitos.")
    if historico:
        if not tabela.Season.eq(np.floor(tabela.Season)).all() or tabela.Season.ge(2026).any():
            raise ValueError("Histórico deve conter temporadas inteiras anteriores a 2026.")
        if {"TeamA", "TeamB"}.issubset(tabela.columns) and not tabela.TeamA.lt(tabela.TeamB).all():
            raise ValueError("TeamA deve ser menor que TeamB.")
    return features


def auditar_folds(treino, *, n_folds=5, seed=42):
    """Audita o mesmo splitter da API 1.6.3, sem implementar outro agrupamento."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    splits, quantidade, repeticoes = ValidationStructure(group_on="Season").custom_splits(
        treino.drop(columns="Target"), treino.Target, num_folds=n_folds,
        num_repeats=1, random_state=seed, problem_type="regression",
    )
    if quantidade != n_folds or repeticoes != 1:
        raise ValueError("Quantidade inesperada de folds internos.")
    folds = np.full(len(treino), -1)
    for fold, (indices_treino, indices_val) in enumerate(splits):
        anos_treino = set(treino.iloc[indices_treino].Season)
        if not anos_treino.isdisjoint(treino.iloc[indices_val].Season):
            raise ValueError("Temporada compartilhada entre treino e validação interna.")
        if (folds[indices_val] != -1).any():
            raise ValueError("Validação interna repetida.")
        folds[indices_val] = fold
    resumo = pd.DataFrame({"Season": treino.Season.to_numpy(), "FoldInterno": folds})
    if (folds < 0).any() or not resumo.groupby("Season").FoldInterno.nunique().eq(1).all():
        raise ValueError("Fold ausente ou temporada dividida.")
    return resumo.drop_duplicates().sort_values("Season").reset_index(drop=True)


def resumir_leaderboard(predictor):
    """Ranking interno inferível; score_val negativo e LogisticBrier positivo."""
    tabela = predictor.leaderboard(extra_info=True, silent=True).copy()
    tabela = tabela.loc[tabela.can_infer & np.isfinite(tabela.score_val)].copy()
    if tabela.empty:
        raise RuntimeError("AutoGluon não produziu modelos utilizáveis. Consulte os logs.")
    def familia(linha):
        tipo = linha.get("child_model_type")
        if tipo is None or (isinstance(tipo, float) and np.isnan(tipo)):
            tipo = linha.get("model_type", "desconhecido")
        return getattr(tipo, "__name__", str(tipo))
    tabela["Familia"] = tabela.apply(familia, axis=1)
    tabela["LogisticBrier"] = -tabela.score_val
    tabela["_leader"] = tabela.model.eq(predictor.model_best)
    tabela = tabela.sort_values(["score_val", "_leader"], ascending=[False, False], kind="stable")
    colunas = ["model", "Familia", "score_val", "LogisticBrier", "fit_time", "pred_time_val", "stack_level"]
    return tabela[colunas].reset_index(drop=True)


def limpar_execucao(execucao):
    """Remove somente o diretório temporário pertencente a esta execução."""
    predictor = execucao.pop("predictor", None)
    try:
        if predictor is not None:
            predictor.unpersist()
    finally:
        del predictor
        gc.collect()
        temporario = execucao.pop("temporario", None)
        if temporario is not None:
            temporario.cleanup()


def treinar_automl(treino, *, features, ano_previsao, seed=42, preset="best_quality",
                   time_limit=300, n_folds=5):
    """Usa somente histórico; arquivos existem apenas até limpar_execucao()."""
    features = _validar(treino, features, historico=True)
    if not isinstance(ano_previsao, Integral) or not treino.Season.lt(ano_previsao).all():
        raise ValueError("Treino deve ser anterior ao ano de previsão.")
    if not isinstance(n_folds, Integral) or n_folds < 2 or treino.Season.nunique() < n_folds:
        raise ValueError("Temporadas insuficientes para os folds internos.")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit deve ser positivo e finito.")
    Predictor, scorer = _backend()
    entradas = treino[[*features, "Season", "Target"]].copy().reset_index(drop=True)
    folds = auditar_folds(entradas, n_folds=n_folds, seed=seed)
    execucao = {"temporario": TemporaryDirectory(prefix="march_autogluon_")}
    try:
        predictor = Predictor(
            label="Target", problem_type="regression", eval_metric=scorer,
            path=str(Path(execucao["temporario"].name) / "predictor"), verbosity=1,
            learner_kwargs={"ignored_columns": ["Season"], "random_state": seed},
        )
        execucao["predictor"] = predictor
        predictor.fit(
            train_data=entradas, presets=preset, time_limit=time_limit,
            validation_structure={"group_on": "Season"}, num_bag_folds=n_folds,
            num_bag_sets=1, num_stack_levels=1, dynamic_stacking=False,
            use_bag_holdout=False, feature_prune_kwargs=None,
            raise_on_model_failure=False,
        )
        usadas = set(predictor.features())
        if usadas != set(features):
            raise RuntimeError(f"Features inesperadas: {sorted(usadas)}")
        leaderboard = resumir_leaderboard(predictor)
        if leaderboard.iloc[0].model != predictor.model_best:
            raise RuntimeError("Leader não corresponde ao melhor score interno.")
        falhas = predictor.model_failures()
        colunas_falhas = [c for c in ["model", "model_type", "exc_type", "exc_str"] if c in falhas]
        execucao.update(
            leaderboard=leaderboard, model=predictor.model_best, folds=folds,
            falhas=falhas[colunas_falhas].astype(str).copy(),
        )
        return execucao
    except BaseException:
        limpar_execucao(execucao)
        raise


class AdaptadorAutoGluon:
    """Interface predict para submissão; devolve margens brutas de um modelo."""
    def __init__(self, predictor, features, model):
        self.predictor, self.features, self.model = predictor, list(features), model

    def predict(self, tabela):
        features = _validar(tabela, self.features)
        pred = np.asarray(self.predictor.predict(tabela[features].copy(), model=self.model), dtype=float)
        if pred.shape != (len(tabela),) or not np.isfinite(pred).all():
            raise ValueError("Previsões inválidas: tamanho, NaN ou infinito.")
        return pred


def avaliar_temporadas_autogluon(jogos, anos, *, features, **config):
    """Dez buscas independentes; média simples dos scores anuais combinados M/W."""
    anos = list(anos)
    if not anos or len(anos) != len(set(anos)) or any(
        not isinstance(a, Integral) or a >= 2026 for a in anos
    ):
        raise ValueError("Anos de validação únicos e anteriores a 2026 são necessários.")
    if set(jogos) != {"M", "W"}:
        raise ValueError("Informe as categorias M e W separadamente.")
    ids = {}
    for categoria, tabela in jogos.items():
        _validar(tabela, features, historico=True)
        ids[categoria] = set(tabela.TeamA) | set(tabela.TeamB)
    if not ids["M"].isdisjoint(ids["W"]):
        raise ValueError("Times compartilhados entre categorias.")
    previsoes, leaders, resultados, folds = [], [], [], []
    leaderboards, falhas = {}, {}
    for ano in anos:
        partes = []
        for categoria, tabela in jogos.items():
            treino = tabela.loc[tabela.Season < ano].copy()
            validacao = tabela.loc[tabela.Season == ano].copy()
            if treino.empty or validacao.empty:
                raise ValueError(f"Faltam jogos para {categoria}/{ano}.")
            execucao = treinar_automl(treino, features=features, ano_previsao=ano, **config)
            try:
                pred = AdaptadorAutoGluon(
                    execucao["predictor"], features, execucao["model"],
                ).predict(validacao)
                parte = validacao[["Season", "TeamA", "TeamB", "Target"]].assign(Pred=pred, Categoria=categoria)
                score = logistic_brier(parte.Target, parte.Pred)
                melhor = execucao["leaderboard"].iloc[0]
                registro = {"Season": ano, "Categoria": categoria, "Score": score,
                    "JogosTreino": len(treino), "JogosValidacao": len(validacao),
                    "UltimoAnoTreino": int(treino.Season.max())}
                resultados.append(registro)
                leaders.append({**registro, "model": execucao["model"], "Familia": melhor.Familia,
                    "score_val": float(melhor.score_val), "LogisticBrierInterno": float(melhor.LogisticBrier)})
                chave = f"{categoria}_{ano}"
                leaderboards[chave] = execucao["leaderboard"].copy()
                falhas[chave] = execucao["falhas"].copy()
                folds.append(execucao["folds"].assign(Categoria=categoria, AnoValidacao=ano))
                partes.append(parte)
                print(f"{categoria}/{ano}: {melhor.model} — Logistic Brier {score:.6f}")
            finally:
                limpar_execucao(execucao)
        combinado = pd.concat(partes, ignore_index=True)
        previsoes.append(combinado)
        resultados.append({"Season": ano, "Categoria": "Combinado",
            "Score": logistic_brier(combinado.Target, combinado.Pred),
            "JogosValidacao": len(combinado),
            "JogosTreino": sum(r["JogosTreino"] for r in leaders if r["Season"] == ano),
            "UltimoAnoTreino": max(r["UltimoAnoTreino"] for r in leaders if r["Season"] == ano)})
    resultados = pd.DataFrame(resultados)
    return {"resultados": resultados, "leaders": pd.DataFrame(leaders),
        "previsoes": pd.concat(previsoes, ignore_index=True), "folds": pd.concat(folds, ignore_index=True),
        "leaderboards": leaderboards, "falhas": falhas,
        "media": float(resultados.loc[resultados.Categoria.eq("Combinado"), "Score"].mean())}
