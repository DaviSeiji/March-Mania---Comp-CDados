"""Preparação e exportação de submissões de margens, como no baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from .features import adicionar_seeds


def preparar_confrontos(sample_submission, seeds_m, seeds_w, times_m, times_w, *, temporada=2026):

    confrontos = sample_submission[["ID"]].copy().reset_index(drop=True)
    if confrontos.empty or not confrontos["ID"].is_unique:
        raise ValueError("O sample deve conter IDs únicos e não pode ser vazio.")
    partes = confrontos["ID"].str.extract(r"^(\d{4})_(\d+)_(\d+)$")
    if partes.isna().any().any():
        raise ValueError("ID fora do formato Season_TeamA_TeamB.")
    confrontos[["Season", "TeamA", "TeamB"]] = partes.astype(int).to_numpy()
    if not confrontos["Season"].eq(temporada).all():
        raise ValueError("O sample contém outra temporada.")
    if not (confrontos["TeamA"] < confrontos["TeamB"]).all():
        raise ValueError("TeamA deve ser o menor ID.")

    times_m, times_w = set(times_m), set(times_w)
    if not times_m.isdisjoint(times_w):
        raise ValueError("Há IDs compartilhados entre as categorias.")
    masculino = confrontos["TeamA"].isin(times_m) & confrontos["TeamB"].isin(times_m)
    feminino = confrontos["TeamA"].isin(times_w) & confrontos["TeamB"].isin(times_w)
    if not (masculino | feminino).all():
        raise ValueError("Time desconhecido ou confronto entre categorias.")
    confrontos["Categoria"] = np.where(masculino, "Masculino", "Feminino")

    partes_seeds = []
    for seeds, times in [(seeds_m, times_m), (seeds_w, times_w)]:
        atuais = seeds.loc[seeds["Season"].eq(temporada), ["Season", "TeamID", "Seed"]]
        if atuais.empty:
            raise ValueError("Faltam seeds da temporada em uma das categorias.")
        if not atuais["TeamID"].isin(times).all():
            raise ValueError("Seed de time desconhecido ou de outra categoria.")
        partes_seeds.append(atuais)
    seeds = pd.concat(partes_seeds, ignore_index=True)
    confrontos = adicionar_seeds(confrontos, seeds, permitir_ausentes=True)
    for categoria, atuais in zip(["Masculino", "Feminino"], partes_seeds):
        quantidade = len(atuais)
        presentes = (confrontos["Categoria"].eq(categoria) & confrontos["TemDuasSeeds"]).sum()
        if presentes != quantidade * (quantidade - 1) // 2:
            raise ValueError(f"Faltam confrontos entre participantes: {categoria}.")
    return confrontos


def prever_confrontos(confrontos, modelo_m, modelo_w, *, features=("SeedDif",)):
    resultado = confrontos.copy()
    if not resultado["Categoria"].isin(["Masculino", "Feminino"]).all():
        raise ValueError("Categoria desconhecida.")
    resultado["Pred"] = 0.0
    for categoria, modelo in [("Masculino", modelo_m), ("Feminino", modelo_w)]:
        mask = resultado["Categoria"].eq(categoria) & resultado["TemDuasSeeds"]
        if not mask.any():
            continue
        entradas = resultado.loc[mask, list(features)]
        if not np.isfinite(entradas.to_numpy(dtype=float)).all():
            raise ValueError("Features ausentes ou não finitas nos confrontos participantes.")
        pred = np.asarray(modelo.predict(entradas), dtype=float)
        if pred.shape != (int(mask.sum()),) or not np.isfinite(pred).all():
            raise ValueError("O modelo retornou previsões inválidas.")
        resultado.loc[mask, "Pred"] = pred
    return resultado


def salvar_submission(confrontos, sample_submission, caminho):
    submission = confrontos[["ID", "Pred"]].copy().reset_index(drop=True)
    ids_esperados = sample_submission["ID"].reset_index(drop=True)
    if submission.empty or not submission["ID"].is_unique or submission["ID"].isna().any():
        raise ValueError("A submissão deve conter IDs únicos, não nulos e não ser vazia.")
    if not submission["ID"].equals(ids_esperados):
        raise ValueError("IDs ou ordem diferentes do sample submission.")
    if not np.isfinite(submission["Pred"].to_numpy(dtype=float)).all():
        raise ValueError("As previsões devem ser finitas.")
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(caminho, index=False)
    salva = pd.read_csv(caminho)
    if not salva["ID"].equals(ids_esperados) or not np.allclose(salva["Pred"], submission["Pred"]):
        raise ValueError("O CSV salvo não corresponde à submissão.")
    return caminho
