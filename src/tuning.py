"""Busca de hiperparâmetros com validação temporal expansiva."""

import numpy as np
import pandas as pd

from .metrics import avaliar_modelo, logistic_brier


def avaliar_temporadas(dados, modelo, temporadas, *, features):
    """Avalia cada ano com treino anterior; dá o mesmo peso a cada ano.

    dados é um dicionário de categoria -> DataFrame. O estimador é clonado
    a cada categoria/ano. O score anual reúne os jogos das categorias.
    """
    temporadas = list(temporadas)
    if not dados or not temporadas or len(set(temporadas)) != len(temporadas):
        raise ValueError("Informe categorias e temporadas não vazias, sem anos repetidos.")
    linhas, previsoes = [], []
    for temporada in sorted(temporadas):
        partes_ano = []
        for categoria, tabela in dados.items():
            resultado = avaliar_modelo(tabela, modelo, temporada, features=features)
            resumo = resultado["resumo"]
            linhas.append({
                "Season": temporada, "Categoria": categoria,
                "Score": resumo["Logistic Brier"],
                "JogosTreino": resumo["Jogos de treino"],
                "JogosValidacao": resumo["Jogos de validação"],
                "UltimoAnoTreino": resumo["Última temporada de treino"],
            })
            partes_ano.append(resultado["previsoes"].assign(Categoria=categoria))
        ano = pd.concat(partes_ano, ignore_index=True)
        linhas.append({
            "Season": temporada, "Categoria": "Combinado",
            "Score": logistic_brier(ano["Target"], ano["Pred"]),
            "JogosTreino": sum(
                int(tabela["Season"].lt(temporada).sum()) for tabela in dados.values()
            ),
            "JogosValidacao": len(ano),
            "UltimoAnoTreino": max(
                tabela.loc[tabela["Season"] < temporada, "Season"].max()
                for tabela in dados.values()
            ),
        })
        previsoes.append(ano)
    resultados = pd.DataFrame(linhas)
    media = resultados.loc[resultados["Categoria"].eq("Combinado"), "Score"].mean()
    return {
        "media": float(media),
        "resultados": resultados,
        "previsoes": pd.concat(previsoes, ignore_index=True),
    }


def otimizar_modelo(dados, criar_modelo, espaco_busca, temporadas, *, features,
                   n_trials=30, parametros_fixos=None, seed=42, sampler=None,
                   nome="modelo"):
    """Seleciona uma configuração pela média anual do Logistic Brier.

    O espaço usa dicionários com tipo (int/float/categorical) e argumentos
    aceitos por trial.suggest_*. Parâmetros fixos não participam da busca.
    Todas as tentativas avaliam todos os anos, sem pruning.
    """
    import optuna

    if n_trials < 1:
        raise ValueError("n_trials deve ser positivo.")
    parametros_fixos = dict(parametros_fixos or {})
    if set(parametros_fixos) & set(espaco_busca):
        raise ValueError("Um parâmetro não pode ser fixo e otimizado ao mesmo tempo.")

    def objetivo(trial):
        params = dict(parametros_fixos)
        for parametro, especificacao in espaco_busca.items():
            spec = dict(especificacao)
            tipo = spec.pop("tipo")
            if tipo not in {"int", "float", "categorical"}:
                raise ValueError(f"Tipo de busca desconhecido: {tipo}")
            params[parametro] = getattr(trial, f"suggest_{tipo}")(parametro, **spec)
        avaliacao = avaliar_temporadas(
            dados, criar_modelo(**params), temporadas, features=features,
        )
        anuais = avaliacao["resultados"].query("Categoria == 'Combinado'")
        trial.set_user_attr("scores_anuais", {
            str(int(linha.Season)): float(linha.Score)
            for linha in anuais.itertuples()
        })
        return avaliacao["media"]

    estudo = optuna.create_study(
        study_name=nome, direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    def progresso(study, trial):
        if (trial.number + 1) % 5 == 0 or trial.number + 1 == n_trials:
            print(f"{nome}: {trial.number + 1}/{n_trials}; melhor média = {study.best_value:.6f}", flush=True)

    estudo.optimize(objetivo, n_trials=n_trials, n_jobs=1, callbacks=[progresso])
    melhores_params = {**parametros_fixos, **estudo.best_params}
    melhor_avaliacao = avaliar_temporadas(
        dados, criar_modelo(**melhores_params), temporadas, features=features,
    )
    if not np.isclose(melhor_avaliacao["media"], estudo.best_value):
        raise ValueError("A reavaliação não reproduziu a melhor tentativa.")
    return {
        "estudo": estudo, "parametros": melhores_params,
        **melhor_avaliacao,
    }
