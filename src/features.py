"""Features compartilhadas pelos jogos históricos e pela submissão."""

import pandas as pd
import numpy as np
from numbers import Integral


def adicionar_seeds(confrontos, seeds, *, permitir_ausentes=False):
    """Junta seeds por Season/TeamID e cria SeedDif e TemDuasSeeds.

    Ausências só são permitidas explicitamente, como nos pares do sample que
    envolvem times fora do torneio. Não altera os DataFrames recebidos.
    """
    seeds = seeds[["Season", "TeamID", "Seed"]].copy()
    if seeds[["Season", "TeamID"]].isna().any().any():
        raise ValueError("Seed sem temporada ou time.")
    if seeds.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Seed duplicada por temporada e time.")
    if not pd.api.types.is_numeric_dtype(seeds["Seed"]):
        raise ValueError("Prepare as seeds numéricas antes de criar features.")
    if not seeds["Seed"].isin(range(1, 17)).all():
        raise ValueError("Seeds devem ser inteiros entre 1 e 16.")
    if confrontos[["Season", "TeamA", "TeamB"]].isna().any().any():
        raise ValueError("Confronto sem temporada ou time.")
    resultado = confrontos.drop(
        columns=["SeedA", "SeedB", "SeedDif", "TemDuasSeeds"], errors="ignore"
    ).copy()
    for lado in ["A", "B"]:
        seeds_lado = seeds.rename(columns={"TeamID": f"Team{lado}", "Seed": f"Seed{lado}"})
        resultado = resultado.merge(
            seeds_lado, on=["Season", f"Team{lado}"], how="left",
            validate="many_to_one", sort=False,
        )
    resultado.index = confrontos.index
    resultado["TemDuasSeeds"] = resultado[["SeedA", "SeedB"]].notna().all(axis=1)
    if not permitir_ausentes and not resultado["TemDuasSeeds"].all():
        raise ValueError("Há jogos sem seed correspondente para um dos times.")
    resultado["SeedDif"] = resultado["SeedA"] - resultado["SeedB"]
    return resultado


def calcular_winrate(resultados_regulares, *, janela=None):
    """Vitórias / jogos por Season e TeamID, apenas da temporada regular.

    Recebe RegularSeasonCompactResults, nunca resultados do torneio NCAA.
    janela=None usa todos os jogos; um inteiro positivo usa os últimos N por
    DayNum de cada time/temporada. Se houver menos de N, usa os disponíveis.
    Jogos registra o denominador efetivamente usado. A taxa serve para prever
    o torneio após o encerramento dos jogos regulares fornecidos.
    """
    if janela is not None and (
        isinstance(janela, bool) or not isinstance(janela, Integral) or janela < 1
    ):
        raise ValueError("janela deve ser um inteiro positivo ou None (todos os jogos).")
    colunas = ["Season", "WTeamID", "LTeamID"]
    if janela is not None:
        colunas.append("DayNum")
    if resultados_regulares.empty or resultados_regulares[colunas].isna().any().any():
        raise ValueError("Informe jogos regulares não vazios, com temporada e times.")
    if resultados_regulares["WTeamID"].eq(resultados_regulares["LTeamID"]).any():
        raise ValueError("Um time não pode enfrentar a si mesmo.")
    extras = ["DayNum"] if janela is not None else []
    vitorias = resultados_regulares[["Season", "WTeamID", *extras]].rename(
        columns={"WTeamID": "TeamID"}
    ).assign(Vitoria=1)
    derrotas = resultados_regulares[["Season", "LTeamID", *extras]].rename(
        columns={"LTeamID": "TeamID"}
    ).assign(Vitoria=0)
    participacoes = pd.concat([vitorias, derrotas], ignore_index=True)
    if janela is not None:
        participacoes = participacoes.sort_values(
            ["Season", "TeamID", "DayNum"], kind="stable",
        ).groupby(["Season", "TeamID"], sort=False).tail(janela)
    resumo = participacoes.groupby(["Season", "TeamID"], as_index=False).agg(
        Vitorias=("Vitoria", "sum"), Jogos=("Vitoria", "size")
    )
    resumo["WinRate"] = resumo["Vitorias"] / resumo["Jogos"]
    return resumo


def adicionar_winrate(confrontos, winrates, *, permitir_ausentes=False):
    """Cria WinRateDiff; as taxas individuais são apenas intermediárias."""
    taxas = winrates[["Season", "TeamID", "WinRate"]].copy()
    if taxas[["Season", "TeamID"]].isna().any().any():
        raise ValueError("Win rate sem temporada ou time.")
    if taxas.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Win rate duplicado por temporada e time.")
    if not pd.api.types.is_numeric_dtype(taxas["WinRate"]) or not taxas["WinRate"].between(0, 1).all():
        raise ValueError("Win rates devem ser números entre 0 e 1.")
    if confrontos[["Season", "TeamA", "TeamB"]].isna().any().any():
        raise ValueError("Confronto sem temporada ou time.")
    resultado = confrontos.drop(
        columns=["WinRateA", "WinRateB", "WinRateDiff"], errors="ignore"
    ).copy()
    for lado in ["A", "B"]:
        taxas_lado = taxas.rename(columns={"TeamID": f"Team{lado}", "WinRate": f"WinRate{lado}"})
        resultado = resultado.merge(
            taxas_lado, on=["Season", f"Team{lado}"], how="left",
            validate="many_to_one", sort=False,
        )
    resultado.index = confrontos.index
    if not permitir_ausentes and resultado[["WinRateA", "WinRateB"]].isna().any().any():
        raise ValueError("Há confrontos sem win rate para um dos times.")
    resultado["WinRateDiff"] = resultado["WinRateA"] - resultado["WinRateB"]
    return resultado.drop(columns=["WinRateA", "WinRateB"])


def _participacoes_pre_torneio(resultados_regulares, limites_daynum):
    """Valida o recorte e transforma cada jogo regular em duas participações."""
    colunas = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
    faltantes = set(colunas) - set(resultados_regulares.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes nos jogos regulares: {sorted(faltantes)}")
    jogos = resultados_regulares[colunas].copy()
    if jogos.empty or not all(pd.api.types.is_numeric_dtype(jogos[c]) for c in colunas):
        raise ValueError("Informe jogos regulares não vazios com valores numéricos.")
    if not np.isfinite(jogos.to_numpy(dtype=float)).all():
        raise ValueError("Há valores ausentes ou infinitos nos jogos regulares.")
    if (jogos["WTeamID"] == jogos["LTeamID"]).any():
        raise ValueError("Um time não pode enfrentar a si mesmo.")
    if not (jogos["WScore"] > jogos["LScore"]).all() or (jogos[["WScore", "LScore"]] < 0).any().any():
        raise ValueError("Placares inválidos nos jogos regulares.")
    limites = jogos["Season"].map(dict(limites_daynum))
    if limites.isna().any() or not np.isfinite(limites.to_numpy(dtype=float)).all():
        raise ValueError("Falta um limite DayNum válido para alguma temporada.")
    if not (jogos["DayNum"] < limites).all():
        raise ValueError("Há jogos no início do torneio ou depois dele; use somente jogos pré-torneio.")
    chaves = jogos.assign(
        TeamA=jogos[["WTeamID", "LTeamID"]].min(axis=1),
        TeamB=jogos[["WTeamID", "LTeamID"]].max(axis=1),
    )
    if chaves.duplicated(["Season", "DayNum", "TeamA", "TeamB"]).any():
        raise ValueError("Há jogos duplicados na temporada regular.")
    vencedores = jogos.rename(columns={
        "WTeamID": "TeamID", "LTeamID": "OpponentID",
        "WScore": "PointsFor", "LScore": "PointsAgainst",
    }).assign(Vitoria=1)
    perdedores = jogos.rename(columns={
        "LTeamID": "TeamID", "WTeamID": "OpponentID",
        "LScore": "PointsFor", "WScore": "PointsAgainst",
    }).assign(Vitoria=0)
    return pd.concat([vencedores, perdedores], ignore_index=True)


def _opp_winrate(participacoes, winrates):
    adversarios = winrates[["Season", "TeamID", "WinRate"]].rename(
        columns={"TeamID": "OpponentID", "WinRate": "OpponentWinRate"}
    )
    tabela = participacoes.merge(
        adversarios, on=["Season", "OpponentID"], how="left", validate="many_to_one",
    )
    if tabela["OpponentWinRate"].isna().any():
        raise ValueError("Falta win rate para algum adversário na mesma temporada.")
    return tabela.groupby(["Season", "TeamID"], as_index=False).agg(
        OppWinRate=("OpponentWinRate", "mean"),
    )


def calcular_opp_winrate(resultados_regulares, *, limites_daynum):
    """Média dos win rates completos dos adversários, ponderada por partida.

    Inclui os jogos contra o próprio time no win rate do adversário. Reencontros
    contam novamente. Não é uma média por adversário distinto nem leave-one-out.
    limites_daynum mapeia Season para o primeiro dia proibido (limite exclusivo).
    """
    participacoes = _participacoes_pre_torneio(resultados_regulares, limites_daynum)
    taxas = participacoes.groupby(["Season", "TeamID"], as_index=False).agg(
        WinRate=("Vitoria", "mean"),
    )
    return _opp_winrate(participacoes, taxas)


def calcular_estatisticas_agregadas(resultados_regulares, *, limites_daynum):
    """WinRate, pontos feitos/sofridos e força dos adversários por Season/TeamID.

    Recebe exclusivamente RegularSeasonCompactResults pré-torneio. Rejeita datas
    fora do recorte em vez de descartá-las silenciosamente. Inclui Jogos para auditoria.
    """
    participacoes = _participacoes_pre_torneio(resultados_regulares, limites_daynum)
    resumo = participacoes.groupby(["Season", "TeamID"], as_index=False).agg(
        Vitorias=("Vitoria", "sum"), Jogos=("Vitoria", "size"),
        AvgPointsFor=("PointsFor", "mean"),
        AvgPointsAgainst=("PointsAgainst", "mean"),
    )
    resumo["WinRate"] = resumo["Vitorias"] / resumo["Jogos"]
    return resumo.merge(_opp_winrate(participacoes, resumo), on=["Season", "TeamID"], validate="one_to_one")


def adicionar_estatisticas(confrontos, estatisticas, *, permitir_ausentes=False):
    """Cria as quatro diferenças A - B; as estatísticas individuais são temporárias.

    Ausências são aceitas somente quando explicitadas e o confronto tem ID de
    sample e pelo menos uma seed ausente. Nunca libera ausências entre participantes.
    """
    nomes = ["WinRate", "AvgPointsFor", "AvgPointsAgainst", "OppWinRate"]
    stats = estatisticas[["Season", "TeamID", *nomes]].copy()
    if stats[["Season", "TeamID"]].isna().any().any() or stats.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Estatísticas com chave ausente ou duplicada por temporada/time.")
    if not all(pd.api.types.is_numeric_dtype(stats[n]) for n in nomes):
        raise ValueError("As estatísticas devem ser numéricas.")
    if not np.isfinite(stats[nomes].to_numpy(dtype=float)).all():
        raise ValueError("Estatísticas ausentes ou infinitas.")
    if not stats["WinRate"].between(0, 1).all() or not stats["OppWinRate"].between(0, 1).all():
        raise ValueError("Taxas devem estar entre 0 e 1.")
    if (stats[["AvgPointsFor", "AvgPointsAgainst"]] < 0).any().any():
        raise ValueError("Médias de pontos não podem ser negativas.")
    if confrontos[["Season", "TeamA", "TeamB"]].isna().any().any() or not (confrontos["TeamA"] < confrontos["TeamB"]).all():
        raise ValueError("Confrontos devem ter temporada e IDs válidos, com TeamA < TeamB.")
    resultado = confrontos.drop(
        columns=[f"{n}{s}" for n in nomes for s in ["A", "B", "Diff"]], errors="ignore",
    ).copy()
    for lado in ["A", "B"]:
        tabela = stats.rename(columns={"TeamID": f"Team{lado}", **{n: f"{n}{lado}" for n in nomes}})
        resultado = resultado.merge(tabela, on=["Season", f"Team{lado}"], how="left", validate="many_to_one", sort=False)
    resultado.index = confrontos.index
    individuais = [f"{n}{s}" for n in nomes for s in ["A", "B"]]
    ausentes = resultado[individuais].isna().any(axis=1)
    if permitir_ausentes:
        obrigatorias = {"ID", "TemDuasSeeds", "SeedA", "SeedB"}
        if not obrigatorias.issubset(resultado.columns) or resultado["ID"].isna().any():
            raise ValueError("Permissão de ausentes requer confrontos preparados do sample.")
        tem_seeds = resultado[["SeedA", "SeedB"]].notna().all(axis=1)
        if not resultado["TemDuasSeeds"].eq(tem_seeds).all():
            raise ValueError("Indicador TemDuasSeeds inconsistente.")
        ausentes = ausentes & tem_seeds
    if ausentes.any():
        raise ValueError("Há confrontos do torneio sem estatísticas disponíveis.")
    for nome in nomes:
        resultado[f"{nome}Diff"] = resultado[f"{nome}A"] - resultado[f"{nome}B"]
    return resultado.drop(columns=individuais)
