"""Descoberta H2O e reconstrução explícita de stacking por corte temporal."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from . import exp11 as comum
from . import automl_h2o as backend

REVISAO = 1
METALEARNER = {'non_negative': True, 'lambda': [0.], 'alpha': [0.],
              'family': 'gaussian', 'standardize': False, 'max_iterations': 1000}


def auditar_descoberta(resultado):
    """Separa origem nativa, dependências e candidatos, sem truncar o leaderboard.

    As reconstruções são ajustes novos dos mesmos candidatos; não acrescentam
    configurações ao catálogo. Metalearners e modelos dos folds não são candidatos.
    """
    snapshots = resultado['snapshots']
    ensembles = resultado['ensembles_originais']
    candidatos = resultado['candidatos']
    ids_lb = [r['model_id'] for r in resultado['leaderboard']]
    ids_ensemble = {r['ModeloOrigem'] for r in ensembles}
    ids_base = set(ids_lb) - ids_ensemble
    ids_snapshot = [s['model_id'] for s in snapshots]
    configuracoes_base = [c for c in candidatos if not c['Familia'].startswith('StackedEnsemble_')]
    receitas = [c for c in candidatos if c['Familia'].startswith('StackedEnsemble_')]
    dependencias_originais = {i for r in ensembles for i in r['BasesOriginais']}
    bases_auxiliares = sorted(dependencias_originais - set(ids_lb))
    dependencias_receitas = {i for r in receitas for i in r['Parametros']['Bases']}
    linhas = [{'Origem': 'AutoML: modelo-base',
        'Etapa': 'grid' if '_grid_' in s['model_id'] else 'inicial/outros',
        'Familia': s['algo'], 'ModeloOrigem': s['model_id']} for s in snapshots]
    linhas.extend({'Origem': 'AutoML: Stacked Ensemble', 'Etapa': 'stacking',
        'Familia': 'stackedensemble', 'ModeloOrigem': r['ModeloOrigem']} for r in ensembles)
    familias = (pd.DataFrame(linhas).groupby(['Origem', 'Etapa', 'Familia'], as_index=False).size()
                .rename(columns={'size': 'Quantidade'}).to_dict('records') if linhas else [])
    pedido = resultado['config']['max_models']
    resumo = {'max_models_solicitado': pedido, 'linhas_leaderboard': len(ids_lb),
        'modelos_base_automl': len(ids_base), 'ensembles_automl': len(ids_ensemble & set(ids_lb)),
        'excedente_base_sobre_max_models': max(0, len(ids_base) - pedido),
        'ids_leaderboard_repetidos': len(ids_lb) - len(set(ids_lb)),
        'ids_snapshots_repetidos': len(ids_snapshot) - len(set(ids_snapshot)),
        'snapshots_fora_leaderboard': sorted(set(ids_snapshot) - set(ids_lb)),
        'bases_auxiliares_fora_leaderboard': bases_auxiliares,
        'quantidade_bases_auxiliares_fora_leaderboard': len(bases_auxiliares),
        'metalearners_originais_fora_leaderboard': sorted({r['MetalearnerOrigem'] for r in ensembles
            if r.get('MetalearnerOrigem') and r['MetalearnerOrigem'] not in ids_lb}),
        'bases_unicas_elegiveis': len({c['Candidato'] for c in configuracoes_base}),
        'receitas_ensemble_avaliacao': len(receitas),
        'bases_reutilizadas_pelas_receitas': len(dependencias_receitas),
        'ajustes_cv_auxiliares_planejados_por_corte': len(dependencias_receitas) * resultado['config'].get('folds', 5),
        'metalearners_receitas_planejados_por_corte': len(receitas),
        'candidatos_unicos_avaliacao_externa': len({c['Candidato'] for c in candidatos}),
        'ids_candidatos_repetidos': len(candidatos) - len({c['Candidato'] for c in candidatos})}
    nota = ('O leaderboard nativo excedeu o max_models solicitado. Os modelos excedentes '
            'vieram do AutoML, não da reconstrução; nenhum candidato foi removido.'
            if resumo['excedente_base_sobre_max_models'] else
            'Contagem observada de bases dentro do max_models solicitado.')
    return {'resumo': resumo, 'familias_por_origem': familias, 'nota': nota}


def _adicionar_auditoria(resultado):
    # Somente metadados no nível raiz: não altera identidade de candidatos/cortes.
    resultado['auditoria_origens'] = auditar_descoberta(resultado)
    return resultado


def montar_receitas(bases, origens, *, limite_all=20, seed=42):
    """Composição do melhor BestOfFamily/AllModels por RMSE interno pré-2023.

    Receita equivalente: conserva dependências disponíveis; AllModels pode ser
    limitado às bases de menor RMSE interno. Metalearner GLM não negativo fixo.
    Não se afirma reprodução idêntica do metalearner do AutoML.
    """
    if limite_all is not None and (not isinstance(limite_all, int) or limite_all < 2):
        raise ValueError('Limite AllModels deve ser >= 2 ou None.')
    por_origem = {c['ModeloOrigem']: c for c in bases}
    saida, falhas = [], []
    for tipo in ['BestOfFamily', 'AllModels']:
        disponiveis = sorted([o for o in origens if tipo in o['ModeloOrigem']],
                             key=lambda o: (o['RMSEInterno'], o['ModeloOrigem']))
        if not disponiveis:
            falhas.append({'Familia': 'StackedEnsemble_' + tipo, 'Erro': 'AutoML não gerou esta família.'})
            continue
        origem = disponiveis[0]
        faltantes = sorted(set(origem['BasesOriginais']) - set(por_origem))
        if faltantes:
            falhas.append({'Familia': tipo, 'Erro': 'Dependências não reconstruíveis.', 'BasesAusentes': faltantes})
            continue
        membros = sorted({por_origem[n]['Candidato'] for n in origem['BasesOriginais']})
        mapa = {c['Candidato']: c for c in bases}
        membros.sort(key=lambda n: (mapa[n]['RMSEInterno'], n))
        antes = membros.copy()
        if tipo == 'AllModels' and limite_all is not None:
            membros = membros[:limite_all]
        if len(membros) < 2:
            falhas.append({'Familia': tipo, 'Erro': 'Menos de duas configurações-base distintas.'})
            continue
        receita = {'Bases': membros, 'Metalearner': deepcopy(METALEARNER), 'seed': seed,
                   'Tipo': tipo, 'Reconstrucao': 'receita_equivalente'}
        cat = bases[0]['Categoria']
        saida.append({'Candidato': f'{cat}_stack_{tipo}_{comum.identidade(receita)[:16]}',
            'Categoria': cat, 'Abordagem': 'h2o', 'Familia': 'StackedEnsemble_' + tipo,
            'Parametros': receita, 'RMSEInterno': origem['RMSEInterno'],
            'ModeloOrigem': origem['ModeloOrigem'], 'OrigemEnsemble': origem,
            'BasesAntesDoLimite': antes, 'BasesExcluidasPeloLimite': antes[len(membros):]})
    return saida, falhas


def descobrir_h2o(tabela, categoria, *, features, pasta, max_models=100,
                  n_folds=5, seed=42, limite_all=20, exclude_algos=('DeepLearning',)):
    treino = tabela.loc[tabela.Season < 2023].copy()
    config = {'revisao': REVISAO, 'dados': comum.assinatura_dados(treino, features),
        'features': list(features), 'categoria': categoria, 'max_models': max_models,
        'folds': n_folds, 'seed': seed, 'limite_all': limite_all,
        'exclude_algos': list(exclude_algos), 'versoes': comum.versoes()}
    path = Path(pasta) / f'descoberta_{categoria}_{comum.identidade(config)[:16]}.json'
    if path.exists():
        salvo = json.loads(path.read_text(encoding='utf-8'))
        # Snapshots permitem corrigir extração sem repetir o AutoML.
        bases, falhas = _extrair_todas_bases(salvo['snapshots'], categoria)
        bases, aliases = _unificar_bases(bases)
        conjuntos, erros = montar_receitas([*bases, *aliases], salvo['ensembles_originais'], limite_all=limite_all, seed=seed)
        salvo.update(candidatos=[*bases, *conjuntos], falhas=[*falhas, *erros])
        _adicionar_auditoria(salvo)
        comum.salvar_json(path, salvo)
        return salvo
    inicio = time.perf_counter()
    busca = backend.treinar_automl(treino, features=features, ano_previsao=2023,
        seed=seed, max_models=max_models, n_folds=n_folds, exclude_algos=exclude_algos,
        nome=f'exp12_{categoria}')
    salvo = False
    try:
        leaderboard = busca['leaderboard'].sort_values(['rmse', 'model_id'], kind='stable')
        snapshots, ensembles = [], []
        for posicao, row in enumerate(leaderboard.to_dict('records'), 1):
            modelo = backend._h2o().get_model(row['model_id'])
            if modelo.algo == 'stackedensemble':
                params = comum._parametros_simples(modelo.actual_params)
                ids = [x['name'] if isinstance(x, dict) else x for x in params['base_models']]
                meta = modelo.metalearner()
                ensembles.append({'ModeloOrigem': modelo.model_id, 'RMSEInterno': float(row['rmse']),
                    'BasesOriginais': ids, 'MetalearnerOrigem': meta.model_id,
                    'AlgoritmoMetalearnerOrigem': meta.algo,
                    'ParametrosMetalearnerOrigem': comum._parametros_simples(meta.actual_params)})
            else:
                snapshots.append(comum._snapshot_modelo(modelo, posicao, float(row['rmse'])))
        # Salva metadados ANTES de reconstruir configurações potencialmente incompatíveis.
        resultado = comum._normalizar_relatorio_descoberta({'config': config, 'snapshots': snapshots,
            'ensembles_originais': ensembles, 'leaderboard': leaderboard.to_dict('records'),
            'candidatos': [], 'falhas': [], 'segundos': time.perf_counter() - inicio})
        # O plano é o executado pelo H2O; nunca o injetamos como modeling_plan.
        eventos = busca.get('eventos')
        if isinstance(eventos, pd.DataFrame):
            resultado['eventos_automl'] = eventos.astype(object).where(eventos.notna(), None).to_dict('records')
        automl = busca.get('automl')
        if automl is not None:
            resultado['modeling_steps_executados'] = comum._parametros_simples(automl.modeling_steps)
        resultado['configuracao_automl_solicitada'] = {'max_models': max_models, 'seed': seed,
            'sort_metric': 'RMSE', 'stopping_metric': 'RMSE', 'n_folds': n_folds,
            'fold_column': 'FoldInterno', 'exclude_algos': list(exclude_algos),
            'modeling_plan': None, 'cotas_manuais': False}
        comum.salvar_json(path, resultado)
        salvo = True
        bases, falhas = _extrair_todas_bases(snapshots, categoria)
        bases, aliases = _unificar_bases(bases)
        conjuntos, erros = montar_receitas([*bases, *aliases], ensembles, limite_all=limite_all, seed=seed)
        resultado.update(candidatos=[*bases, *conjuntos], falhas=[*falhas, *erros])
        _adicionar_auditoria(resultado)
        comum.salvar_json(path, resultado)
        return resultado
    finally:
        if salvo:
            backend.limpar_execucao(busca)
        else:
            print('Snapshots não salvos: modelos preservados no cluster para recuperação.', flush=True)


def _unificar_bases(bases):
    unicos, aliases = {}, []
    for c in bases:
        if c['Candidato'] in unicos:
            aliases.append(c)
        else:
            unicos[c['Candidato']] = c
    return list(unicos.values()), aliases


def _extrair_todas_bases(snapshots, categoria):
    bases, falhas = [], []
    for snapshot in snapshots:
        candidatos, erros = comum._extrair_snapshots([snapshot], categoria)
        bases.extend(candidatos)
        falhas.extend(erros)
    return bases, falhas


class SessaoH2O:
    """Um frame por corte e cache de bases OOF compartilhado pelos ensembles.

    Todos os modelos usam a mesma ordem de linhas e mapa Season→fold. Apenas as
    bases necessárias ao stacking recebem CV; outras são reajustadas sem CV.
    A sessão remove somente seus objetos H2O, inclusive ao interromper.
    """

    def __init__(self, treino, candidatos, *, features, ano, n_folds=5, seed=42):
        backend._validar_tabela(treino, features, target=True)
        if not treino.Season.lt(ano).all() or treino.Season.ge(2026).any():
            raise ValueError('Todo treino, inclusive o metalearner, deve preceder o corte.')
        if len({c['Categoria'] for c in candidatos}) != 1:
            raise ValueError('Uma sessão deve conter uma única categoria.')
        self.treino = treino.copy()
        self.catalogo = {c['Candidato']: c for c in candidatos}
        self.features, self.ano, self.seed = list(features), ano, seed
        self.folds = backend.criar_folds_temporadas(treino.Season, n_folds=n_folds, seed=seed)
        self.necessarios = {i for c in candidatos if c['Familia'].startswith('StackedEnsemble_') for i in c['Parametros']['Bases']}
        if not self.necessarios <= self.catalogo.keys():
            raise ValueError('Faltam configurações-base necessárias ao ensemble.')
        self.modelos, self.custos = {}, {}
        self.frame = None
        self.audit = {'Ano': ano, 'UltimoAnoTreino': int(treino.Season.max()),
            'JogosTreino': len(treino), 'Features': self.features,
            'Folds': pd.DataFrame({'Season': treino.Season.to_numpy(), 'Fold': self.folds.to_numpy()})
                .drop_duplicates().sort_values('Season').to_dict('records'),
            'HashFolds': comum.identidade(self.folds.tolist()), 'BasesTreinadas': [], 'EnsemblesTreinados': []}

    def __enter__(self):
        self.antes = backend._chaves_h2o()
        try:
            tabela = self.treino[[*self.features, 'Target']].astype(float).copy()
            tabela['FoldInterno'] = self.folds.to_numpy()
            self.frame = backend._h2o().H2OFrame(tabela,
                column_types={**{c: 'real' for c in [*self.features, 'Target']}, 'FoldInterno': 'int'})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        backend.limpar_execucao({'chaves_h2o': backend._chaves_h2o() - self.antes})
        self.modelos.clear()
        self.frame = None

    def obter(self, candidato):
        ident = candidato['Candidato']
        if ident in self.modelos:
            return self.modelos[ident]
        inicio = time.perf_counter()
        if candidato['Familia'].startswith('StackedEnsemble_'):
            from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
            receita = candidato['Parametros']
            bases = [self.obter(self.catalogo[i]) for i in receita['Bases']]
            modelo = H2OStackedEnsembleEstimator(model_id='exp12_stack_' + uuid4().hex,
                base_models=[m.model_id for m in bases], metalearner_algorithm='glm',
                metalearner_params=deepcopy(receita['Metalearner']), metalearner_nfolds=0,
                keep_levelone_frame=False, seed=self.seed)
            modelo.train(x=self.features, y='Target', training_frame=self.frame)
            self.audit['EnsemblesTreinados'].append({'Candidato': ident, 'Modelo': modelo.model_id,
                'Bases': receita['Bases'], 'Metalearner': modelo.metalearner().model_id,
                'UltimoAnoTreinoMetalearner': int(self.treino.Season.max()),
                'Metodo': 'OOF das bases; GLM não negativo fixo, sem validação externa'})
        else:
            precisa_oof = ident in self.necessarios
            parametros = deepcopy(candidato['Parametros'])
            modelo = comum._classe_h2o(candidato['Familia'])(**parametros,
                model_id='exp12_base_' + uuid4().hex,
                keep_cross_validation_predictions=precisa_oof,
                keep_cross_validation_models=False, keep_cross_validation_fold_assignment=False)
            kwargs = {'fold_column': 'FoldInterno'} if precisa_oof else {}
            with warnings.catch_warnings():
                comum.configurar_avisos_h2o()
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                    message=r'Sample rate is 100% and no validation dataset and no cross-validation\.',
                    module=r'h2o\.estimators\.estimator_base')
                modelo.train(x=self.features, y='Target', training_frame=self.frame, **kwargs)
            self.audit['BasesTreinadas'].append({'Candidato': ident, 'Modelo': modelo.model_id,
                'OOF': precisa_oof, 'HashFolds': self.audit['HashFolds'] if precisa_oof else None})
        if modelo._model_json['output']['model_category'] != 'Regression':
            raise ValueError('Modelo deve ser regressor.')
        self.modelos[ident] = modelo
        self.custos[ident] = time.perf_counter() - inicio
        nodes = backend._h2o().api('GET /3/Cloud')['nodes']
        heap = sum(n['max_mem'] - n['free_mem'] for n in nodes) / 2**20
        self.audit.setdefault('HeapJVMObservadoMB', []).append({'Candidato': ident, 'HeapMB': heap})
        return modelo

    @contextmanager
    def adaptador(self, candidato):
        """Libera modelos individuais fora do conjunto compartilhado de stacking."""
        modelo = self.obter(candidato)
        try:
            yield backend.AdaptadorH2O(modelo, self.features)
        finally:
            ident = candidato['Candidato']
            if ident not in self.necessarios:
                # Dependências criadas por um ensemble devem continuar para o próximo.
                backend._h2o().remove(modelo.model_id)
                self.modelos.pop(ident, None)
