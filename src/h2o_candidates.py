"""Configurações H2O fixas, avaliadas por validação temporal ponderada."""

from copy import deepcopy
from hashlib import sha256
import json
import warnings
from numbers import Integral
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from . import automl_h2o as backend
from .metrics import logistic_brier
from .season_similarity import ponderar_temporadas, ranquear_temporadas


def espacos_candidatos(n_features=14):
    """Espaços explícitos, sem informação de resultados ou anos de validação."""
    arvores = {"ntrees": [100, 200, 400, 600], "max_depth": [3, 5, 8, 12, 20],
        "min_rows": [2., 5., 10., 20.], "sample_rate": [.6, .8, 1.],
        "mtries": sorted(set([-1, min(3, n_features), min(5, n_features), min(8, n_features), n_features]))}
    return {
        "GLM": {"alpha": [0., .1, .25, .5, .75, 1.],
            "lambda_": [0., *np.logspace(-6, 1, 49).tolist()],
            "standardize": [False, True], "intercept": [False, True]},
        "GBM": {"ntrees": [100, 200, 400, 600], "max_depth": [2, 3, 4, 5, 7],
            "min_rows": [2., 5., 10., 20.], "learn_rate": [.01, .03, .05, .1],
            "sample_rate": [.6, .8, 1.], "col_sample_rate": [.6, .8, 1.]},
        "RF": deepcopy(arvores), "XRT": deepcopy(arvores),
    }


def gerar_candidatos_h2o(*, categoria, n_candidatos=200, seed=42, n_features=14):
    """Amostra sem reposição: 10% GLM, 45% GBM, 25% RF, 20% XRT.

    Identidade = família + hiperparâmetros + seed. Pesos ajustados são reaprendidos
    em cada ano; o candidato continua sendo a mesma configuração. Não há AutoML
    independente por temporada nem ensembles dependentes de rankings locais.
    """
    if categoria not in {"M", "W"} or not isinstance(n_candidatos, Integral) or isinstance(n_candidatos, bool) or n_candidatos < 1:
        raise ValueError("Informe categoria M/W e quantidade inteira positiva.")
    if not isinstance(n_features, Integral) or n_features < 1:
        raise ValueError("Número de features deve ser positivo.")
    espacos = espacos_candidatos(n_features)
    alvos = n_candidatos * np.array([.10, .45, .25, .20])
    cotas = np.floor(alvos).astype(int)
    for i in np.argsort(-(alvos - cotas), kind='stable')[:n_candidatos - cotas.sum()]:
        cotas[i] += 1
    candidatos = []
    for i, (familia, espaco) in enumerate(espacos.items()):
        quantidade = int(cotas[i])
        capacidade = int(np.prod([len(valores) for valores in espaco.values()]))
        if quantidade > capacidade:
            raise ValueError(f"Espaço insuficiente para {quantidade} configurações únicas de {familia}.")
        if quantidade == 0:
            continue
        for parametros in ParameterSampler(espaco, n_iter=quantidade, random_state=seed + i * 100 + (categoria == 'W')):
            params = {k: v.item() if isinstance(v, np.generic) else v for k, v in parametros.items()}
            params['seed'] = int(seed)
            if familia == 'GLM':
                params.update(family='gaussian', lambda_search=False, solver='IRLSM', max_iterations=100,
                              early_stopping=False)
            else:
                params.update(stopping_rounds=0)
                if familia == 'GBM':
                    params['distribution'] = 'gaussian'
                else:
                    params['histogram_type'] = 'Random' if familia == 'XRT' else 'UniformAdaptive'
            identidade = json.dumps({'Familia': familia, 'Parametros': params}, sort_keys=True)
            candidatos.append({'Candidato': f'{categoria}_{familia}_{sha256(identidade.encode()).hexdigest()[:12]}',
                'Categoria': categoria, 'Familia': familia, 'Parametros': params})
    assert len(candidatos) == n_candidatos
    assert len({c['Candidato'] for c in candidatos}) == n_candidatos
    return candidatos


def _estimador(familia):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    from h2o.estimators.random_forest import H2ORandomForestEstimator
    return {'GLM': H2OGeneralizedLinearEstimator, 'GBM': H2OGradientBoostingEstimator,
            'RF': H2ORandomForestEstimator, 'XRT': H2ORandomForestEstimator}[familia]


def treinar_candidato_h2o(treino, candidato, *, features, ano_previsao, n_folds=5, seed=42):
    """Treina exatamente os parâmetros do candidato, sem receber validação externa."""
    features = backend._validar_tabela(treino, features, target=True)
    if 'Season' not in treino or not treino.Season.lt(ano_previsao).all() or treino.Season.ge(2026).any():
        raise ValueError("Treino deve preceder o ano previsto e não pode incluir 2026.")
    folds = backend.criar_folds_temporadas(treino.Season, n_folds=n_folds, seed=seed)
    parametros = deepcopy(candidato['Parametros'])
    proibidos = {'training_frame', 'validation_frame', 'fold_column', 'nfolds', 'model_id', 'response_column',
                 'checkpoint', 'weights_column', 'offset_column'}
    if proibidos & set(parametros):
        raise ValueError("Configuração não pode carregar frames, modelos anteriores ou divisões de dados.")
    h2o = backend._h2o()
    antes = backend._chaves_h2o()
    try:
        tabela = treino[[*features, 'Target']].astype(float).copy()
        tabela['FoldInterno'] = folds.to_numpy()
        frame = h2o.H2OFrame(tabela, column_types={**{c: 'real' for c in [*features, 'Target']}, 'FoldInterno': 'int'})
        modelo = _estimador(candidato['Familia'])(**parametros,
            model_id=f"exp10_{candidato['Candidato']}_{ano_previsao}_{uuid4().hex[:8]}",
            keep_cross_validation_models=False, keep_cross_validation_predictions=False,
            keep_cross_validation_fold_assignment=False)
        with warnings.catch_warnings():
            # sample_rate=1 impede estimativa OOB; a validação temporal é externa.
            warnings.filterwarnings(
                'ignore',
                message=r'Sample rate is 100% and no validation dataset and no cross-validation\. '
                        r'There are no out-of-bag data to compute error estimates on the training data!',
                category=RuntimeWarning,
                module=r'h2o\.estimators\.estimator_base',
            )
            modelo.train(x=features, y='Target', training_frame=frame, fold_column='FoldInterno')
        if modelo._model_json['output']['model_category'] != 'Regression':
            raise RuntimeError("O candidato deve ser um regressor.")
        return {'leader': modelo, 'chaves_h2o': sorted(backend._chaves_h2o() - antes),
            'rmse_cv': float(modelo.rmse(xval=True))}
    except BaseException:
        backend.limpar_execucao({'chaves_h2o': backend._chaves_h2o() - antes})
        raise


def avaliar_candidatos_h2o(dados, candidatos, anos, *, features, n_folds=5, seed=42):
    """Cada configuração recebe um score em cada ano; falhas nunca viram médias parciais."""
    anos = sorted(anos)
    if set(dados) != {'M', 'W'} or set(candidatos) != {'M', 'W'}:
        raise ValueError("Dados e candidatos devem estar separados em M/W.")
    if not anos or len(set(anos)) != len(anos) or any(not isinstance(a, Integral) or a >= 2026 for a in anos):
        raise ValueError("Anos devem ser únicos e anteriores a 2026.")
    ids = {}
    for cat, tabela in dados.items():
        backend._validar_tabela(tabela, features, target=True)
        if not np.isfinite(tabela[['Season', 'TeamA', 'TeamB']].to_numpy(dtype=float)).all() or not tabela.TeamA.lt(tabela.TeamB).all():
            raise ValueError("Identificadores inválidos.")
        if tabela.Season.ge(2026).any() or not tabela.Season.eq(np.floor(tabela.Season)).all():
            raise ValueError("Histórico contém temporadas futuras ou inválidas.")
        ids[cat] = set(tabela.TeamA) | set(tabela.TeamB)
        banco = candidatos[cat]
        if not banco or any(c['Categoria'] != cat for c in banco) or len({c['Candidato'] for c in banco}) != len(banco):
            raise ValueError("Banco de candidatos vazio, duplicado ou de outra categoria.")
        for ano in anos:
            if not tabela.Season.lt(ano).any() or not tabela.Season.eq(ano).any():
                raise ValueError(f"Faltam jogos para {cat}/{ano}.")
    if not ids['M'].isdisjoint(ids['W']):
        raise ValueError("Times compartilhados entre M/W.")
    linhas, falhas = [], []
    previsoes = {}
    for cat, banco in candidatos.items():
        tabela = dados[cat]
        for i, candidato in enumerate(banco, 1):
            for ano in anos:
                treino = tabela.loc[tabela.Season < ano].copy()
                validacao = tabela.loc[tabela.Season == ano].copy()
                execucao = None
                try:
                    execucao = treinar_candidato_h2o(treino, candidato, features=features,
                        ano_previsao=ano, n_folds=n_folds, seed=seed)
                    pred = backend.prever_leader(execucao['leader'], validacao, features=features)
                    score = logistic_brier(validacao.Target, pred)
                    linhas.append({'Categoria': cat, 'Candidato': candidato['Candidato'],
                        'Familia': candidato['Familia'], 'Season': ano, 'Score': score,
                        'RMSE_CV': execucao['rmse_cv'], 'JogosTreino': len(treino),
                        'JogosValidacao': len(validacao), 'UltimoAnoTreino': int(treino.Season.max())})
                    previsoes[(cat, candidato['Candidato'], ano)] = validacao[
                        ['Season', 'TeamA', 'TeamB', 'Target']].assign(Pred=pred).reset_index(drop=True)
                except (ValueError, RuntimeError, EnvironmentError) as exc:
                    falhas.append({'Categoria': cat, 'Candidato': candidato['Candidato'], 'Season': ano,
                        'Erro': type(exc).__name__, 'Mensagem': str(exc)[:1500]})
                    print(f"Falha {cat}/{candidato['Candidato']}/{ano}: {type(exc).__name__}", flush=True)
                finally:
                    if execucao is not None:
                        backend.limpar_execucao(execucao)
            if i == 1 or i % 10 == 0 or i == len(banco):
                print(f"{cat}: {i}/{len(banco)} candidatos avaliados em {len(anos)} anos.", flush=True)
    return {'scores': pd.DataFrame(linhas, columns=['Categoria', 'Candidato', 'Familia', 'Season', 'Score',
        'RMSE_CV', 'JogosTreino', 'JogosValidacao', 'UltimoAnoTreino']),
        'falhas': pd.DataFrame(falhas), 'previsoes': previsoes}


def ranquear_candidatos(scores, referencias, *, ano_alvo=2026):
    """Score(c, categoria) = soma(Peso_ano * LogisticBrier(c, categoria, ano))."""
    refs = referencias[['Season', 'Peso']].copy()
    if refs.empty or refs.Season.duplicated().any() or not refs.Season.lt(ano_alvo).all():
        raise ValueError("Referências únicas devem ser anteriores ao alvo.")
    if not np.isfinite(refs.to_numpy(dtype=float)).all() or refs.Peso.lt(0).any() or not np.isclose(refs.Peso.sum(), 1.):
        raise ValueError("Pesos devem ser finitos, não negativos e somar 1.")
    usados = scores.loc[scores.Season.isin(refs.Season)].copy()
    if usados.duplicated(['Categoria', 'Candidato', 'Season']).any():
        raise ValueError("Scores repetidos por candidato/ano.")
    if not usados.Categoria.isin(['M', 'W']).all() or not np.isfinite(usados.Score).all() or not usados.Score.between(0, 1).all():
        raise ValueError("Scores ou categorias inválidos.")
    if not usados.groupby(['Categoria', 'Candidato']).Familia.nunique().eq(1).all():
        raise ValueError("A família de um candidato não pode mudar entre anos.")
    contagem = usados.groupby(['Categoria', 'Candidato']).Season.transform('nunique')
    completos = usados.loc[contagem.eq(len(refs))].merge(refs, on='Season', validate='many_to_one')
    completos['Contribuicao'] = completos.Score * completos.Peso
    ranking = completos.groupby(['Categoria', 'Candidato', 'Familia'], as_index=False).agg(
        ScorePonderado=('Contribuicao', 'sum'))
    anual = completos.pivot(index=['Categoria', 'Candidato'], columns='Season', values='Score')
    anual.columns = [f'LB_{int(ano)}' for ano in anual.columns]
    ranking = ranking.merge(anual.reset_index(), on=['Categoria', 'Candidato'], validate='one_to_one')
    ranking = ranking.sort_values(['Categoria', 'ScorePonderado', 'Candidato'], kind='stable').reset_index(drop=True)
    ranking['Posicao'] = ranking.groupby('Categoria').cumcount() + 1
    return ranking


def backtest_candidatos(perfis, scores, *, anos=(2022, 2023, 2024, 2025), n_referencias=3):
    """Seleciona configurações por categoria só com scores dos anos anteriores.

    Exige avaliação nos anos adicionais. Não escolhe substitutos olhando o alvo se
    o candidato previamente escolhido falhar nele: reporta NaN e a limitação.
    """
    linhas = []
    for ano in anos:
        passados = scores.loc[scores.Season < ano]
        candidatas = sorted(passados.Season.unique().astype(int).tolist())
        similaridade = ranquear_temporadas(perfis, ano, candidatas)
        refs = ponderar_temporadas(similaridade['ranking'], n_referencias=n_referencias)
        ponderado = ranquear_candidatos(passados, refs, ano_alvo=ano)
        iguais = pd.DataFrame({'Season': candidatas, 'Peso': 1 / len(candidatas)})
        uniforme = ranquear_candidatos(passados, iguais, ano_alvo=ano)
        for cat in ['M', 'W']:
            for estrategia, ranking in [('Similaridade', ponderado), ('MediaPassada', uniforme)]:
                elegiveis = ranking.loc[ranking.Categoria.eq(cat)]
                if elegiveis.empty:
                    raise ValueError(f"Nenhum candidato completo para {cat}/{ano}.")
                escolhido = elegiveis.iloc[0].Candidato
                alvo = scores.loc[scores.Season.eq(ano) & scores.Categoria.eq(cat) & scores.Candidato.eq(escolhido)]
                linhas.append({'AnoAlvo': ano, 'Categoria': cat, 'Estrategia': estrategia,
                    'Candidato': escolhido, 'Score': float(alvo.iloc[0].Score) if len(alvo) == 1 else np.nan,
                    'Referencias': refs.Season.astype(int).tolist() if estrategia == 'Similaridade' else candidatas,
                    'Pesos': refs.Peso.tolist() if estrategia == 'Similaridade' else iguais.Peso.tolist()})
    return pd.DataFrame(linhas)
