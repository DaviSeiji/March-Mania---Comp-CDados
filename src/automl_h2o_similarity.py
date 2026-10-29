"""Avaliação externa Top1/2/3; reutiliza integralmente o treino H2O do Exp. 8."""

from numbers import Integral

import numpy as np
import pandas as pd

from . import automl_h2o as backend
from .metrics import logistic_brier


def obter_top3_h2o(execucao, *, features):
    """Adaptadores e metadados das posições internas por RMSE, sem reordenar."""
    leaderboard = execucao["leaderboard"]
    if len(leaderboard) < 3:
        raise ValueError("O protocolo exige três modelos por categoria; não substituímos posições ausentes.")
    if not leaderboard.model_id.is_unique or not np.isfinite(leaderboard.rmse).all():
        raise ValueError("Leaderboard inválido.")
    if not leaderboard.rmse.is_monotonic_increasing:
        raise ValueError("Leaderboard deve estar ordenado por RMSE interna.")
    modelos, linhas = {}, []
    for posicao, linha in enumerate(leaderboard.head(3).itertuples(index=False), 1):
        modelo = backend._h2o().get_model(linha.model_id)
        modelos[posicao] = backend.AdaptadorH2O(modelo, features)
        linhas.append({"Posicao": posicao, "model_id": modelo.model_id,
            "Algoritmo": modelo.algo, "RMSE_CV": float(linha.rmse)})
    return modelos, pd.DataFrame(linhas)


def avaliar_top3_h2o(dados, temporadas, *, features, **config):
    """Treina uma vez por M/W/ano e avalia as três posições nos mesmos jogos."""
    temporadas = list(temporadas)
    if set(dados) != {"M", "W"} or not temporadas or len(set(temporadas)) != len(temporadas):
        raise ValueError("Informe M/W e temporadas únicas.")
    if any(not isinstance(a, Integral) or a >= 2026 for a in temporadas):
        raise ValueError("Validação deve usar temporadas históricas anteriores a 2026.")
    ids = {}
    for categoria, tabela in dados.items():
        backend._validar_tabela(tabela, features, target=True)
        if not {"Season", "TeamA", "TeamB"}.issubset(tabela.columns):
            raise ValueError("Faltam identificadores históricos.")
        if tabela[["Season", "TeamA", "TeamB"]].isna().any().any() or not tabela.TeamA.lt(tabela.TeamB).all():
            raise ValueError("Identificadores inválidos.")
        if not tabela.Season.eq(np.floor(tabela.Season)).all() or tabela.Season.ge(2026).any():
            raise ValueError("Histórico contém temporada inválida ou futura.")
        ids[categoria] = set(tabela.TeamA) | set(tabela.TeamB)
    if not ids["M"].isdisjoint(ids["W"]):
        raise ValueError("M/W compartilham times.")
    resultados, previsoes, modelos_usados, folds = [], [], [], []
    leaderboards = {}
    for ano in sorted(temporadas):
        partes = []
        for categoria, tabela in dados.items():
            treino, validacao = tabela.loc[tabela.Season < ano].copy(), tabela.loc[tabela.Season == ano].copy()
            if treino.empty or validacao.empty:
                raise ValueError(f"Treino ou validação vazios: {categoria}/{ano}.")
            execucao = backend.treinar_automl(treino, features=features, ano_previsao=ano, **config)
            modelos = {}
            try:
                modelos, resumo = obter_top3_h2o(execucao, features=features)
                modelos_usados.append(resumo.assign(Season=ano, Categoria=categoria,
                    JogosTreino=len(treino), JogosValidacao=len(validacao), UltimoAnoTreino=int(treino.Season.max())))
                folds.append(execucao["folds"].assign(Categoria=categoria, AnoValidacao=ano))
                leaderboards[f"{categoria}_{ano}"] = execucao["leaderboard"].copy()
                for posicao, modelo in modelos.items():
                    pred = np.asarray(modelo.predict(validacao), dtype=float)
                    if pred.shape != (len(validacao),) or not np.isfinite(pred).all():
                        raise ValueError("Previsões externas inválidas.")
                    parte = validacao[["Season", "TeamA", "TeamB", "Target"]].assign(
                        Categoria=categoria, Posicao=posicao, Pred=pred)
                    score = logistic_brier(parte.Target, parte.Pred)
                    resultados.append({"Season": ano, "Categoria": categoria, "Posicao": posicao, "Score": score})
                    partes.append(parte)
                    print(f"{categoria}/{ano} Top{posicao}: Logistic Brier {score:.6f}", flush=True)
            finally:
                modelos.clear()
                backend.limpar_execucao(execucao)
        anual = pd.concat(partes, ignore_index=True)
        for posicao, grupo in anual.groupby("Posicao"):
            resultados.append({"Season": ano, "Categoria": "Combinado", "Posicao": posicao,
                "Score": logistic_brier(grupo.Target, grupo.Pred)})
        previsoes.append(anual)
    resultados = pd.DataFrame(resultados)
    scores = resultados.loc[resultados.Categoria.eq("Combinado")].pivot(index="Season", columns="Posicao", values="Score")
    scores = scores.rename(columns={1: "Top1", 2: "Top2", 3: "Top3"}).reset_index()
    scores.columns.name = None
    return {"scores": scores, "resultados": resultados, "previsoes": pd.concat(previsoes, ignore_index=True),
        "modelos": pd.concat(modelos_usados, ignore_index=True), "folds": pd.concat(folds, ignore_index=True),
        "leaderboards": leaderboards}
