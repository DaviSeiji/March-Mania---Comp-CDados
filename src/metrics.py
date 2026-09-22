import numpy as np
from scipy.special import expit
from sklearn.base import clone

def logistic_brier(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Informe vetores não vazios e de mesmo tamanho.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("As margens devem ser valores finitos.")
    return float(np.mean((expit(y_pred / 7.0) - expit(y_true / 7.0)) ** 2))


def avaliar_modelo(dados, modelo, temporada, *, features=("SeedDif",)):
    if isinstance(features, str):
        features = [features]
    else:
        features = list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("Informe features não vazias e sem repetições.")
    if "Target" in features or "Pred" in features:
        raise ValueError("Target e Pred não podem ser usados como features.")
    faltantes = set(["Season", "Target", *features]) - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")
    if dados["Season"].isna().any():
        raise ValueError("Há jogos sem temporada.")

    treino = dados.loc[dados["Season"] < temporada].copy()
    validacao = dados.loc[dados["Season"] == temporada].copy()
    if treino.empty:
        raise ValueError(f"Não há jogos de treino anteriores a {temporada}.")
    if validacao.empty:
        raise ValueError(f"Não há jogos para validar em {temporada}.")
    for tabela in [treino, validacao]:
        valores = tabela[[*features, "Target"]].to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError("Features e Target devem ser numéricos e finitos.")

    modelo_treinado = clone(modelo)
    modelo_treinado.fit(treino[features], treino["Target"])
    pred = modelo_treinado.predict(validacao[features])
    score = logistic_brier(validacao["Target"], pred)
    score_zero = logistic_brier(validacao["Target"], np.zeros(len(validacao)))
    validacao["Pred"] = pred

    return {
        "modelo": modelo_treinado,
        "previsoes": validacao,
        "resumo": {
            "Temporada de validação": temporada,
            "Primeira temporada de treino": treino["Season"].min(),
            "Última temporada de treino": treino["Season"].max(),
            "Jogos de treino": len(treino),
            "Jogos de validação": len(validacao),
            "Logistic Brier": score,
            "Logistic Brier - margem zero": score_zero,
            "Ganho sobre margem zero": score_zero - score,
        },
    }
