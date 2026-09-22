"""Leitura e organização dos dados, sem criar features de confronto."""

from pathlib import Path

import pandas as pd


def carregar_dados(pasta, categoria, *, stage=2, incluir_regular=False):
    """Carrega uma categoria ('M' ou 'W') e o sample do estágio escolhido."""
    categoria = categoria.upper()
    if categoria not in {"M", "W"}:
        raise ValueError("Categoria deve ser M ou W.")
    if stage not in {1, 2}:
        raise ValueError("Stage deve ser 1 ou 2.")
    pasta = Path(pasta)
    arquivos = {
        "seeds": f"{categoria}NCAATourneySeeds.csv",
        "resultados": f"{categoria}NCAATourneyCompactResults.csv",
        "times": f"{categoria}Teams.csv",
        "sample_submission": f"SampleSubmissionStage{stage}.csv",
    }
    if incluir_regular:
        arquivos["temporada_regular"] = f"{categoria}RegularSeasonCompactResults.csv"
    return {nome: pd.read_csv(pasta / arquivo) for nome, arquivo in arquivos.items()}


def preparar_seeds(seeds_raw):
    """Converte W01/W16a em 1/16, preservando SeedOriginal."""
    seeds = seeds_raw[["Season", "TeamID", "Seed"]].copy()
    seeds = seeds.rename(columns={"Seed": "SeedOriginal"})
    numero = seeds["SeedOriginal"].str.extract(r"^[WXYZ](\d{2})[ab]?$", expand=False)
    if numero.isna().any():
        raise ValueError("Há seeds ausentes ou em formato inesperado.")
    seeds["Seed"] = numero.astype(int)
    if not seeds["Seed"].between(1, 16).all():
        raise ValueError("As seeds devem estar entre 1 e 16.")
    if seeds[["Season", "TeamID"]].isna().any().any():
        raise ValueError("Seed sem temporada ou time.")
    if seeds.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Há mais de uma seed para o mesmo time na mesma temporada.")
    return seeds


def organizar_jogos(resultados):
    """Define A como menor ID e calcula a margem de A, sem juntar seeds."""
    colunas = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
    if resultados[colunas].isna().any().any():
        raise ValueError("Há valores ausentes nos resultados.")
    if (resultados["WTeamID"] == resultados["LTeamID"]).any():
        raise ValueError("Um time não pode enfrentar a si mesmo.")
    if not (resultados["WScore"] > resultados["LScore"]).all():
        raise ValueError("O placar do vencedor deve superar o do perdedor.")
    a_venceu = resultados["WTeamID"] < resultados["LTeamID"]
    jogos = resultados[["Season", "DayNum"]].copy()
    jogos["TeamA"] = resultados["WTeamID"].where(a_venceu, resultados["LTeamID"])
    jogos["TeamB"] = resultados["LTeamID"].where(a_venceu, resultados["WTeamID"])
    jogos["ScoreA"] = resultados["WScore"].where(a_venceu, resultados["LScore"])
    jogos["ScoreB"] = resultados["LScore"].where(a_venceu, resultados["WScore"])
    jogos["Target"] = jogos["ScoreA"] - jogos["ScoreB"]
    return jogos
