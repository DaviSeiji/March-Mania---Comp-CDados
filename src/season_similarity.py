"""Perfis pré-NCAA e seleção por similaridade, sem acesso a scores Kaggle."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_processing import preparar_seeds
from .features import (
    ESTATISTICAS_DETALHADAS, calcular_estatisticas_agregadas,
    calcular_estatisticas_detalhadas,
)


VARIAVEIS_PERFIL = (
    "WinRate", "AvgPointsFor", "AvgPointsAgainst", "OppWinRate",
    *ESTATISTICAS_DETALHADAS,
)
DIMENSOES_PERFIL = tuple(
    f"{categoria}_{variavel}_{estatistica}"
    for categoria in ("M", "W") for variavel in VARIAVEIS_PERFIL
    for estatistica in ("mean", "std")
)
TOPS = ("Top1", "Top2", "Top3")


def construir_perfis_temporadas(dados, limites, *, inicio=None, ano_maximo=2026):
    """Agrega participantes identificados pelas seeds, nunca confrontos do torneio.

    Lê apenas seeds e arquivos RegularSeason. Resultados NCAA presentes no dict
    não são acessados. Limites são datas exclusivas pré-torneio; para 2026, 134.
    Estatísticas da equipe usam todos os adversários regulares; só depois se
    restringe a população do perfil aos participantes. Desvio populacional (ddof=0).
    """
    inicio = {"M": 2003, "W": 2010} if inicio is None else dict(inicio)
    if set(dados) != {"M", "W"} or set(limites) != {"M", "W"} or set(inicio) != {"M", "W"}:
        raise ValueError("Informe dados, limites e início para M/W separadamente.")
    equipes, perfis_categoria, contagens, ids = [], [], [], {}
    por_categoria = {}
    for categoria in ("M", "W"):
        base = dados[categoria]
        sementes = preparar_seeds(base["seeds"])
        sementes = sementes.loc[sementes.Season.between(inicio[categoria], ano_maximo), ["Season", "TeamID"]]
        anos = set(sementes.Season)
        if not anos or not anos.issubset(limites[categoria]):
            raise ValueError("Faltam participantes ou limites pré-torneio para alguma temporada.")
        if 2026 in anos and limites[categoria][2026] != 134:
            raise ValueError("O perfil de 2026 exige DayNum < 134.")
        compact = base["temporada_regular"].loc[base["temporada_regular"].Season.isin(anos)].copy()
        detalhes = base["temporada_regular_detalhada"].loc[
            base["temporada_regular_detalhada"].Season.isin(anos)
        ].copy()
        agregadas = calcular_estatisticas_agregadas(compact, limites_daynum=limites[categoria])
        detalhadas = calcular_estatisticas_detalhadas(detalhes, limites_daynum=limites[categoria])
        participantes = sementes.merge(
            agregadas[["Season", "TeamID", *VARIAVEIS_PERFIL[:4]]],
            on=["Season", "TeamID"], how="left", validate="one_to_one",
        ).merge(detalhadas[["Season", "TeamID", *ESTATISTICAS_DETALHADAS]],
            on=["Season", "TeamID"], how="left", validate="one_to_one")
        if not np.isfinite(participantes[list(VARIAVEIS_PERFIL)].to_numpy(dtype=float)).all():
            raise ValueError("Participantes sem estatísticas completas e finitas.")
        ids[categoria] = set(participantes.TeamID)
        grupos = participantes.groupby("Season")[list(VARIAVEIS_PERFIL)]
        medias, desvios = grupos.mean(), grupos.std(ddof=0)
        perfil = pd.DataFrame(index=medias.index)
        for variavel in VARIAVEIS_PERFIL:
            perfil[variavel + "_mean"] = medias[variavel]
            perfil[variavel + "_std"] = desvios[variavel]
        contagem = participantes.groupby("Season").size().rename("Participantes")
        if contagem.lt(2).any():
            raise ValueError("Perfil exige ao menos dois participantes por categoria/temporada.")
        contagens.append(contagem.reset_index().assign(Categoria=categoria))
        equipes.append(participantes.assign(Categoria=categoria))
        perfis_categoria.append(perfil.reset_index().assign(Categoria=categoria))
        por_categoria[categoria] = perfil.add_prefix(categoria + "_")
    if not ids["M"].isdisjoint(ids["W"]):
        raise ValueError("IDs compartilhados entre M e W.")
    combinado = por_categoria["M"].join(por_categoria["W"], how="inner").reset_index()
    if combinado.empty:
        raise ValueError("Não há temporadas com perfis das duas categorias.")
    return {
        "perfis": combinado[["Season", *DIMENSOES_PERFIL]],
        "perfis_categoria": pd.concat(perfis_categoria, ignore_index=True),
        "equipes": pd.concat(equipes, ignore_index=True),
        "contagens": pd.concat(contagens, ignore_index=True),
    }


def calcular_distancias(referencia, candidatos, *, metrica="euclidiana"):
    """Distâncias entre vetores já padronizados; sem alvo ou score de modelos."""
    referencia, candidatos = np.asarray(referencia, dtype=float), np.asarray(candidatos, dtype=float)
    if referencia.ndim != 1 or candidatos.ndim != 2 or candidatos.shape[1] != len(referencia):
        raise ValueError("Dimensões incompatíveis dos perfis.")
    if not referencia.size or not np.isfinite(referencia).all() or not np.isfinite(candidatos).all():
        raise ValueError("Perfis devem ser não vazios e finitos.")
    delta = candidatos - referencia
    if metrica == "euclidiana":
        return np.sqrt(np.square(delta).sum(axis=1))
    if metrica == "manhattan":
        return np.abs(delta).sum(axis=1)
    raise ValueError("Métrica deve ser euclidiana ou manhattan.")


def ranquear_temporadas(perfis, ano_alvo, candidatos, *, metrica="euclidiana"):
    """Z-score ajustado em TODOS os perfis combinados anteriores ao alvo.

    As candidatas podem ser um subconjunto (2021–2025 para 2026). Nunca ajusta o
    scaler no alvo ou no futuro. Dimensões constantes usam escala 1 (StandardScaler).
    Empates de distância favorecem o ano mais recente, regra fixa sem usar scores.
    """
    if set(perfis.columns) != {"Season", *DIMENSOES_PERFIL} or not perfis.columns.is_unique:
        raise ValueError("Perfis devem conter somente Season e as 52 dimensões pré-torneio.")
    if perfis.empty or perfis.Season.duplicated().any() or not np.isfinite(perfis.to_numpy(dtype=float)).all():
        raise ValueError("Perfis vazios, duplicados ou não finitos.")
    if not perfis.Season.eq(np.floor(perfis.Season)).all() or not isinstance(ano_alvo, Integral):
        raise ValueError("Temporadas devem ser inteiras.")
    candidatos = list(candidatos)
    if not candidatos or len(set(candidatos)) != len(candidatos) or any(
        not isinstance(a, Integral) or a >= ano_alvo for a in candidatos
    ):
        raise ValueError("Candidatas devem ser únicas e anteriores ao ano alvo.")
    if not set([ano_alvo, *candidatos]).issubset(set(perfis.Season)):
        raise ValueError("Faltam perfis do alvo ou de temporadas candidatas.")
    historico = perfis.loc[perfis.Season < ano_alvo].sort_values("Season")
    if len(historico) < 2:
        raise ValueError("Padronização exige pelo menos duas temporadas históricas.")
    colunas = list(DIMENSOES_PERFIL)
    scaler = StandardScaler().fit(historico[colunas])
    alvo = scaler.transform(perfis.loc[perfis.Season.eq(ano_alvo), colunas])[0]
    tabela = perfis.set_index("Season").loc[candidatos, colunas]
    z = scaler.transform(tabela)
    distancias = calcular_distancias(alvo, z, metrica=metrica)
    ranking = pd.DataFrame({"Season": candidatos, "Distancia": distancias}).sort_values(
        ["Distancia", "Season"], ascending=[True, False], kind="stable",
    ).reset_index(drop=True)
    ranking["Ranking"] = np.arange(1, len(ranking) + 1)
    contribuicao = (z - alvo) ** 2 if metrica == "euclidiana" else np.abs(z - alvo)
    contribuicoes = pd.DataFrame(contribuicao, index=pd.Index(candidatos, name="Season"), columns=colunas)
    contribuicoes = contribuicoes.reset_index().melt(id_vars="Season", var_name="Dimensao", value_name="Contribuicao")
    totais = contribuicoes.groupby("Season").Contribuicao.transform("sum")
    contribuicoes["Fracao"] = np.divide(contribuicoes.Contribuicao, totais,
        out=np.zeros(len(totais)), where=totais.to_numpy() != 0)
    normalizacao = pd.DataFrame({"Dimensao": colunas, "MediaHistorica": scaler.mean_,
        "DesvioHistorico": np.sqrt(scaler.var_), "EscalaUsada": scaler.scale_})
    return {"ranking": ranking, "contribuicoes": contribuicoes, "normalizacao": normalizacao,
        "anos_padronizacao": historico.Season.astype(int).tolist(), "metrica": metrica, "ano_alvo": ano_alvo}


def _validar_scores(scores):
    if set(scores.columns) != {"Season", *TOPS} or not scores.columns.is_unique:
        raise ValueError("Use somente Season e scores externos Top1/Top2/Top3.")
    if scores.empty or scores.Season.duplicated().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Scores vazios, duplicados ou não finitos.")
    if not scores.Season.eq(np.floor(scores.Season)).all() or scores.Season.ge(2026).any():
        raise ValueError("Scores devem ser históricos, anteriores a 2026.")
    if not scores[list(TOPS)].ge(0).all().all() or not scores[list(TOPS)].le(1).all().all():
        raise ValueError("Logistic Brier deve estar entre 0 e 1.")


def selecionar_posicao(scores, ano_referencia):
    """Escolhe menor erro externo apenas no ano de referência; empate: menor Top."""
    _validar_scores(scores)
    linha = scores.loc[scores.Season.eq(ano_referencia), list(TOPS)]
    if len(linha) != 1:
        raise ValueError("Falta score do ano de referência.")
    return int(np.argmin(linha.iloc[0].to_numpy())) + 1


def ponderar_temporadas(ranking, *, n_referencias=3):
    """Seleciona os N anos mais próximos e normaliza pesos proporcionais a 1/d.

    Sem scores como entrada. Se faltarem anos, usa os disponíveis. Se houver
    distância zero, divide o peso entre esses empates e atribui zero aos demais.
    """
    if not isinstance(n_referencias, Integral) or isinstance(n_referencias, bool) or n_referencias < 1:
        raise ValueError("Número de referências deve ser inteiro positivo.")
    if not {"Season", "Distancia"}.issubset(ranking.columns) or ranking.empty:
        raise ValueError("Ranking vazio ou sem temporada/distância.")
    if ranking.Season.duplicated().any() or not np.isfinite(ranking[["Season", "Distancia"]].to_numpy(dtype=float)).all():
        raise ValueError("Ranking duplicado ou não finito.")
    if ranking.Distancia.lt(0).any() or not ranking.Season.eq(np.floor(ranking.Season)).all():
        raise ValueError("Distâncias devem ser não negativas e temporadas inteiras.")
    referencias = ranking[["Season", "Distancia"]].sort_values(
        ["Distancia", "Season"], ascending=[True, False], kind="stable",
    ).head(n_referencias).reset_index(drop=True)
    distancias = referencias.Distancia.to_numpy(dtype=float)
    zeros = distancias == 0
    pesos = zeros.astype(float) if zeros.any() else distancias.min() / distancias
    referencias["Peso"] = pesos / pesos.sum()
    return referencias


def selecionar_posicao_ponderada(scores, similaridade, *, n_referencias=3):
    """Minimiza a média externa ponderada dos Tops nos anos mais semelhantes."""
    _validar_scores(scores)
    referencias = ponderar_temporadas(similaridade["ranking"], n_referencias=n_referencias)
    if not referencias.Season.lt(similaridade["ano_alvo"]).all():
        raise ValueError("Referências devem ser anteriores ao ano alvo.")
    if not set(referencias.Season).issubset(set(scores.Season)):
        raise ValueError("Faltam scores de alguma referência; não substituímos anos ausentes.")
    detalhes = referencias.merge(scores, on="Season", how="left", validate="one_to_one", sort=False)
    for top in TOPS:
        detalhes[f"Contribuicao{top}"] = detalhes.Peso * detalhes[top]
    medias = np.array([detalhes[f"Contribuicao{top}"].sum() for top in TOPS])
    return {"posicao": int(np.argmin(medias)) + 1, "referencias": referencias, "detalhes": detalhes,
        "scores_ponderados": pd.DataFrame({"Posicao": [1, 2, 3], "ScorePonderado": medias})}


def backtest_similaridade(perfis, scores, *, anos=(2022, 2023, 2024, 2025), n_referencias=3):
    """Escolhe com passado; resultados do alvo só entram na avaliação/regret.

    Compara com Top1 fixo e com menor média simples dos anos de validação
    estritamente anteriores ao alvo. Usa até três referências, com pesos inversos
    às distâncias. O melhor Top do alvo é oráculo diagnóstico.
    """
    _validar_scores(scores)
    anos = list(anos)
    if not anos or len(set(anos)) != len(anos) or not set(anos).issubset(set(scores.Season)):
        raise ValueError("Anos de backtest devem ser únicos e possuir scores históricos.")
    linhas, auditorias = [], {}
    for ano in sorted(anos):
        passados = scores.loc[scores.Season < ano].copy()
        ranking = ranquear_temporadas(perfis, ano, passados.Season.astype(int).tolist())
        referencia = int(ranking["ranking"].iloc[0].Season)
        selecao = selecionar_posicao_ponderada(passados, ranking, n_referencias=n_referencias)
        posicao = selecao["posicao"]
        posicao_media = int(np.argmin(passados[list(TOPS)].mean().to_numpy())) + 1
        alvo = scores.set_index("Season").loc[ano]
        melhor = int(np.argmin(alvo[list(TOPS)].to_numpy())) + 1
        score = float(alvo[f"Top{posicao}"])
        linhas.append({"AnoAlvo": ano, "AnoReferencia": referencia,
            "AnosReferencia": selecao["referencias"].Season.astype(int).tolist(),
            "Pesos": selecao["referencias"].Peso.tolist(), "NReferencias": len(selecao["referencias"]),
            "Distancia": float(ranking["ranking"].iloc[0].Distancia), "TopEscolhido": posicao,
            "ScoreSimilaridade": score, "MelhorTopPossivel": melhor,
            "Regret": score - float(alvo[list(TOPS)].min()), "ScoreTop1": float(alvo.Top1),
            "TopMediaPassada": posicao_media, "ScoreMediaPassada": float(alvo[f"Top{posicao_media}"]),
            "UltimoAnoPadronizacao": max(ranking["anos_padronizacao"])})
        auditorias[ano] = {**ranking, "selecao": selecao}
    tabela = pd.DataFrame(linhas)
    medias = tabela[["ScoreSimilaridade", "ScoreTop1", "ScoreMediaPassada"]].mean().rename("MediaLogisticBrier")
    return {"resultados": tabela, "medias": medias, "auditorias": auditorias}
