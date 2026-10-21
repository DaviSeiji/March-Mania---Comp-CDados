"""AutoML de regressão com CV interna por temporada e avaliação temporal externa.

H2O é opcional e importado apenas ao usar o backend. As execuções devem ser
sequenciais em uma sessão dedicada; a limpeza preserva objetos preexistentes.
"""

from numbers import Integral
from tempfile import mkdtemp
from uuid import uuid4

import numpy as np
import pandas as pd

from .metrics import logistic_brier


def _h2o():
    try:
        import h2o
    except ImportError as exc:
        raise ImportError("Instale requirements-h2o.txt no ambiente do kernel do notebook.") from exc
    return h2o


def inicializar_h2o(*, max_mem_size="4G", nthreads=2, port=54329):
    """Inicia/conecta uma vez; arquivos temporários ficam fora do repositório."""
    h2o = _h2o()
    if h2o.connection() is None or not h2o.cluster().is_running():
        h2o.init(ip="127.0.0.1", port=port, bind_to_localhost=True,
                 max_mem_size=max_mem_size, nthreads=nthreads,
                 ice_root=mkdtemp(prefix="march_mania_h2o_"))
    h2o.no_progress()
    from h2o.estimators.xgboost import H2OXGBoostEstimator
    disponivel = H2OXGBoostEstimator.available()
    return {
        "h2o_version": h2o.__version__, "cluster_version": h2o.cluster().version,
        "xgboost_disponivel": disponivel,
        "exclude_algos": ["DeepLearning"] + ([] if disponivel else ["XGBoost"]),
        "limitacao": None if disponivel else "XGBoost indisponível neste backend H2O; demais famílias habilitadas.",
    }


def criar_folds_temporadas(temporadas, *, n_folds=5, seed=42):
    """Embaralha anos únicos ordenados e distribui em round-robin, sem separar um ano.

    Esta CV agrupada não é expansiva: somente a avaliação externa é temporal.
    O mapeamento depende apenas dos anos de treino, nunca do alvo ou da validação.
    """
    temporadas = pd.Series(temporadas, copy=True)
    if not isinstance(n_folds, Integral) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("Informe pelo menos dois folds internos.")
    if temporadas.empty or not pd.api.types.is_numeric_dtype(temporadas):
        raise ValueError("Informe temporadas numéricas não vazias.")
    valores = temporadas.to_numpy(dtype=float)
    if not np.isfinite(valores).all() or not np.equal(valores, np.floor(valores)).all():
        raise ValueError("Temporadas devem ser inteiras e finitas.")
    anos = np.sort(temporadas.unique())
    if len(anos) < n_folds:
        raise ValueError("Há menos temporadas de treino que folds internos.")
    anos = np.random.default_rng(seed).permutation(anos)
    mapa = {ano: i % n_folds for i, ano in enumerate(anos)}
    return temporadas.map(mapa).astype(int).rename("FoldInterno")


def _validar_tabela(tabela, features, *, target=False):
    features = list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("Features devem ser não vazias e sem repetições.")
    if set(features) & {"Season", "Target", "Pred", "FoldInterno", "TeamA", "TeamB"}:
        raise ValueError("Features não podem conter identificadores, folds ou alvo.")
    colunas = features + (["Target"] if target else [])
    if tabela.empty or not set(colunas).issubset(tabela.columns):
        raise ValueError("Tabela vazia ou sem colunas necessárias.")
    if not all(pd.api.types.is_numeric_dtype(tabela[c]) for c in colunas):
        raise ValueError("Features e Target devem ser numéricos.")
    if not np.isfinite(tabela[colunas].to_numpy(dtype=float)).all():
        raise ValueError("Features e Target devem ser finitos.")
    return features


def _chaves_h2o():
    return set(_h2o().ls()["key"].tolist())


def limpar_execucao(execucao):
    """Libera modelos/frames desta busca; não reinicia o cluster nem apaga outros."""
    h2o = _h2o()
    chaves = set(execucao.get("chaves_h2o", [])) & _chaves_h2o()
    # Ensembles primeiro; o cascade também libera seus objetos dependentes.
    for chave in sorted(chaves, key=lambda k: (not k.startswith("StackedEnsemble"), k)):
        h2o.remove(chave)
    execucao["chaves_h2o"] = []
    execucao["automl"] = None
    execucao["leader"] = None


def treinar_automl(treino, *, features, ano_previsao, seed=42, max_models=20,
                   n_folds=5, exclude_algos=("DeepLearning",), nome="exp8"):
    """Treina somente dados anteriores ao ano previsto; leader escolhido por RMSE CV.

    Não aceita validation_frame/leaderboard_frame externos. Target é sempre real,
    mesmo quando suas margens são inteiras. IDs únicos evitam reutilizar buscas.
    """
    features = _validar_tabela(treino, features, target=True)
    if "Season" not in treino or not treino["Season"].lt(ano_previsao).all():
        raise ValueError("Treino deve conter apenas temporadas anteriores à previsão.")
    if not isinstance(max_models, Integral) or isinstance(max_models, bool) or max_models < 1:
        raise ValueError("max_models deve ser um inteiro positivo.")
    folds = criar_folds_temporadas(treino["Season"], n_folds=n_folds, seed=seed)
    mapa = pd.DataFrame({"Season": treino["Season"].to_numpy(), "FoldInterno": folds.to_numpy()})
    mapa = mapa.groupby(["Season", "FoldInterno"], as_index=False).size().rename(columns={"size": "Jogos"})
    h2o = _h2o()
    from h2o.automl import H2OAutoML
    from h2o.estimators.xgboost import H2OXGBoostEstimator
    excluidos = list(dict.fromkeys(["DeepLearning", *exclude_algos]))
    if "XGBoost" not in excluidos and not H2OXGBoostEstimator.available():
        excluidos.append("XGBoost")
    antes = _chaves_h2o()
    try:
        tabela = treino[[*features, "Target"]].astype(float).copy()
        tabela["FoldInterno"] = folds.to_numpy()
        frame = h2o.H2OFrame(tabela, column_types={**{c: "real" for c in [*features, "Target"]}, "FoldInterno": "int"})
        automl = H2OAutoML(
            seed=seed, max_models=int(max_models), max_runtime_secs=0,
            max_runtime_secs_per_model=0, nfolds=int(n_folds),
            sort_metric="RMSE", stopping_metric="RMSE", exclude_algos=excluidos,
            keep_cross_validation_models=False, keep_cross_validation_predictions=False,
            keep_cross_validation_fold_assignment=False,
            project_name=f"{nome}_{uuid4().hex[:12]}", verbosity="warn",
        )
        automl.train(x=features, y="Target", training_frame=frame, fold_column="FoldInterno")
        leader = automl.leader
        if leader is None:
            raise RuntimeError("AutoML não produziu um leader.")
        if leader._model_json["output"]["model_category"] != "Regression":
            raise RuntimeError("O leader deve ser um modelo de regressão.")
        leaderboard = automl.leaderboard.as_data_frame()
        eventos = automl.event_log.as_data_frame()
        if leaderboard.iloc[0]["model_id"] != leader.model_id:
            raise RuntimeError("Leader não corresponde ao primeiro modelo do leaderboard.")
        return {
            "automl": automl, "leader": leader, "leaderboard": leaderboard,
            "eventos": eventos, "folds": mapa, "exclude_algos": excluidos,
            "model_id": leader.model_id, "algoritmo": leader.algo,
            "rmse_cv": float(leaderboard.iloc[0]["rmse"]),
            "chaves_h2o": sorted(_chaves_h2o() - antes),
        }
    except BaseException:
        limpar_execucao({"chaves_h2o": _chaves_h2o() - antes})
        raise


def prever_leader(leader, dados, *, features):
    """Converte pandas para H2O, devolve margens numpy e remove frames temporários."""
    features = _validar_tabela(dados, features)
    h2o = _h2o()
    entrada = saida = None
    try:
        entrada = h2o.H2OFrame(dados[features].astype(float), column_types={c: "real" for c in features})
        saida = leader.predict(entrada)
        pred = saida.as_data_frame()["predict"].to_numpy(dtype=float)
        if pred.shape != (len(dados),) or not np.isfinite(pred).all():
            raise ValueError("Número de previsões inválido ou previsão não finita.")
        return pred
    finally:
        for frame in [saida, entrada]:
            if frame is not None:
                h2o.remove(frame)


class AdaptadorH2O:
    """Interface predict(pandas) para reutilizar prever_confrontos sem alterá-lo."""

    def __init__(self, leader, features):
        self.leader = leader
        self.features = list(features)

    def predict(self, dados):
        return prever_leader(self.leader, dados, features=self.features)


def avaliar_temporadas_h2o(dados, temporadas, *, features, seed=42, max_models=20,
                           n_folds=5, exclude_algos=("DeepLearning",)):
    """Avalia leaders internos, separando M/W; média simples dos scores combinados."""
    temporadas = list(temporadas)
    if set(dados) != {"M", "W"} or not temporadas or len(set(temporadas)) != len(temporadas):
        raise ValueError("Informe M/W separados e temporadas não vazias e únicas.")
    ids = {}
    for categoria, tabela in dados.items():
        _validar_tabela(tabela, features, target=True)
        if not {"TeamA", "TeamB", "Season"}.issubset(tabela.columns):
            raise ValueError("Faltam IDs/temporadas para verificar a separação M/W.")
        if tabela[["TeamA", "TeamB", "Season"]].isna().any().any() or not (tabela.TeamA < tabela.TeamB).all():
            raise ValueError("Chaves inválidas ou TeamA >= TeamB.")
        if tabela.Season.ge(2026).any():
            raise ValueError("A avaliação histórica não aceita jogos de 2026 em diante.")
        ids[categoria] = set(tabela.TeamA) | set(tabela.TeamB)
    if not ids["M"].isdisjoint(ids["W"]):
        raise ValueError("IDs compartilhados entre masculino e feminino.")
    linhas, previsoes, leaders, mapas, tabelas, eventos = [], [], [], [], {}, {}
    for ano in sorted(temporadas):
        partes = []
        for categoria, tabela in dados.items():
            treino = tabela.loc[tabela.Season < ano].copy()
            validacao = tabela.loc[tabela.Season == ano].copy()
            if treino.empty or validacao.empty:
                raise ValueError(f"Treino/validação vazios: {categoria}/{ano}.")
            execucao = treinar_automl(treino, features=features, ano_previsao=ano,
                seed=seed, max_models=max_models, n_folds=n_folds,
                exclude_algos=exclude_algos, nome=f"exp8_{categoria}_{ano}")
            try:
                validacao["Pred"] = prever_leader(execucao["leader"], validacao, features=features)
                validacao["Categoria"] = categoria
                score = logistic_brier(validacao.Target, validacao.Pred)
                linhas.append({"Season": ano, "Categoria": categoria, "Score": score,
                    "JogosTreino": len(treino), "JogosValidacao": len(validacao),
                    "UltimoAnoTreino": int(treino.Season.max())})
                leaders.append({"Season": ano, "Categoria": categoria,
                    "model_id": execucao["model_id"], "Algoritmo": execucao["algoritmo"],
                    "RMSE_CV": execucao["rmse_cv"]})
                mapas.append(execucao["folds"].assign(AnoValidacao=ano, Categoria=categoria))
                tabelas[f"{categoria}_{ano}"] = execucao["leaderboard"]
                eventos[f"{categoria}_{ano}"] = execucao["eventos"]
                partes.append(validacao)
                print(f"{categoria}/{ano}: {execucao['algoritmo']} — Logistic Brier {score:.6f}", flush=True)
            finally:
                limpar_execucao(execucao)
        combinado = pd.concat(partes, ignore_index=True)
        linhas.append({"Season": ano, "Categoria": "Combinado",
            "Score": logistic_brier(combinado.Target, combinado.Pred),
            "JogosTreino": sum(len(t.loc[t.Season < ano]) for t in dados.values()),
            "JogosValidacao": len(combinado),
            "UltimoAnoTreino": max(int(t.loc[t.Season < ano, "Season"].max()) for t in dados.values())})
        previsoes.append(combinado)
    resultados = pd.DataFrame(linhas)
    return {"media": float(resultados.query("Categoria == 'Combinado'")["Score"].mean()),
        "resultados": resultados, "previsoes": pd.concat(previsoes, ignore_index=True),
        "leaders": pd.DataFrame(leaders), "folds": pd.concat(mapas, ignore_index=True),
        "leaderboards": tabelas, "eventos": eventos}
