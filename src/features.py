"""Features compartilhadas pelos jogos históricos e pela submissão."""

import pandas as pd
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
