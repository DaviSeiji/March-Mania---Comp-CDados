"""Features compartilhadas pelos jogos históricos e pela submissão."""

import pandas as pd


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
