from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch, MagicMock
from types import ModuleType

import numpy as np
import pandas as pd

from src.efficiency_features import calcular_eficiencia, adicionar_eficiencia, ESTATISTICAS_EFICIENCIA
from src import exp11 as comum
from src import exp12 as e
from src.exp12_h2o import montar_receitas, SessaoH2O, descobrir_h2o


def partida(**alteracoes):
    jogo = dict(Season=2023, DayNum=10, WTeamID=1, LTeamID=2, WScore=80, LScore=70,
                WFGM=30, WFGA=60, WFGM3=8, WFTA=20, WOR=10, WDR=25, WTO=12,
                LFGM=25, LFGA=50, LFGM3=5, LFTA=10, LOR=8, LDR=20, LTO=15)
    return {**jogo, **alteracoes}


def dados():
    return {cat: pd.DataFrame([{'Season': ano, 'TeamA': base, 'TeamB': base + 1,
        'Target': float(i + 1), **{f: float(i) for f in e.FEATURES_EFICIENCIA}}
        for ano in range(2010, 2026) for i in range(3)]) for cat, base in [('M', 1000), ('W', 3000)]}


def bases(cat='M'):
    return [{'Candidato': f'{cat}_b{i}', 'Categoria': cat, 'Abordagem': 'h2o',
        'Familia': familia, 'Parametros': parametros, 'ModeloOrigem': f'{cat}_old{i}',
        'RMSEInterno': 1. + i} for i, (familia, parametros) in enumerate([
            ('glm', dict(family='gaussian', alpha=[0.], lambda_=[.01], seed=42)),
            ('gbm', dict(ntrees=3, max_depth=2, min_rows=2, seed=42)),
            ('drf', dict(ntrees=3, max_depth=2, min_rows=2, seed=42))])]


def ensembles(base):
    origens = [{'ModeloOrigem': f'StackedEnsemble_{tipo}_1',
        'BasesOriginais': [c['ModeloOrigem'] for c in base], 'RMSEInterno': 1.,
        'MetalearnerOrigem': 'metalearner'} for tipo in ['BestOfFamily', 'AllModels']]
    return montar_receitas(base, origens, limite_all=2)[0]


@contextmanager
def fake_modelo(treino, candidato, features, ano):
    assert treino.Season.lt(ano).all()
    class Modelo:
        def predict(self, X):
            return X[features[0]].to_numpy() + 1.
    yield Modelo()


class FakeSessao:
    def __init__(self, treino, candidatos, *, features, ano, **kwargs):
        self.treino, self.features, self.ano = treino, features, ano
        self.audit = {'Ano': ano}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def adaptador(self, candidato):
        return fake_modelo(self.treino, candidato, self.features, self.ano)


class EficienciaTest(unittest.TestCase):
    def test_formulas_vencedor_perdedor_e_ritmo(self):
        original = pd.DataFrame([partida()])
        antes = original.copy(deep=True)
        stats = calcular_eficiencia(original, limites_daynum={2023: 134}).set_index('TeamID')
        a, b = stats.loc[1], stats.loc[2]
        p, q = 60 - 10 + 12 + .44*20, 50 - 8 + 15 + .44*10
        self.assertAlmostEqual(a.OffensiveRating, 100*80/p)
        self.assertAlmostEqual(a.DefensiveRating, 100*70/q)
        self.assertAlmostEqual(a.NetRating, 100*80/p - 100*70/q)
        self.assertAlmostEqual(a.PossessionsPerGame, (p+q)/2)
        self.assertAlmostEqual(a.EffectiveFGPercentage, (30+.5*8)/60)
        self.assertAlmostEqual(a.TurnoverRate, 12/(60+.44*20+12))
        self.assertAlmostEqual(a.OffensiveReboundRate, 10/(10+20))
        self.assertAlmostEqual(a.FreeThrowRate, 20/60)
        self.assertAlmostEqual(b.OffensiveRating, a.DefensiveRating)
        self.assertAlmostEqual(b.DefensiveRating, a.OffensiveRating)
        self.assertAlmostEqual(b.OffensiveReboundRate, 8/(8+25))
        pd.testing.assert_frame_equal(original, antes)

    def test_totais_sazonais_nao_media_de_percentuais(self):
        raw = pd.DataFrame([partida(), partida(DayNum=20, WTeamID=2, LTeamID=1,
            LFGA=100, LFGM=40, LFGM3=10, LScore=75, WScore=90), partida(Season=2024)])
        stats = calcular_eficiencia(raw, limites_daynum={2023: 134, 2024: 134}).set_index(['Season', 'TeamID'])
        self.assertAlmostEqual(stats.loc[(2023, 1), 'EffectiveFGPercentage'], (30+40+.5*(8+10))/(60+100))
        self.assertEqual(stats.loc[(2023, 1), 'JogosEficiencia'], 2)
        self.assertEqual(stats.loc[(2024, 1), 'JogosEficiencia'], 1)
        self.assertAlmostEqual(stats.loc[(2024, 1), 'FreeThrowRate'], 20/60)

    def test_sem_leakage_duplicados_zero_inf_e_autoconfronto(self):
        for raw in [pd.DataFrame([partida(DayNum=134)]), pd.DataFrame([partida(), partida()]),
                    pd.DataFrame([partida(WFGA=np.inf)]), pd.DataFrame([partida(WTeamID=2)]),
                    pd.DataFrame([partida(WOR=0, LDR=0)]),
                    pd.DataFrame([partida(WFGA=0, WFGM=0, WFGM3=0)])]:
            with self.assertRaises(ValueError):
                calcular_eficiencia(raw, limites_daynum={2023: 134})

    def test_diferencas_ausencias_seeds_e_chaves(self):
        stats = calcular_eficiencia(pd.DataFrame([partida()]), limites_daynum={2023: 134})
        pares = pd.DataFrame({'Season': [2023], 'TeamA': [1], 'TeamB': [2]}, index=[7])
        antes = pares.copy()
        resultado = adicionar_eficiencia(pares, stats)
        for nome in ESTATISTICAS_EFICIENCIA:
            self.assertAlmostEqual(resultado.loc[7, nome+'Diff'], stats.iloc[0][nome]-stats.iloc[1][nome])
        pd.testing.assert_frame_equal(pares, antes)
        with self.assertRaises(ValueError):
            adicionar_eficiencia(pares, pd.concat([stats, stats]))
        sample = pares.assign(TeamB=3, ID='2023_1_3', SeedA=1., SeedB=np.nan, TemDuasSeeds=False)
        self.assertTrue(adicionar_eficiencia(sample, stats, permitir_ausentes=True).NetRatingDiff.isna().all())
        with self.assertRaises(ValueError):
            adicionar_eficiencia(sample.assign(SeedB=2., TemDuasSeeds=True), stats, permitir_ausentes=True)
        with self.assertRaises(ValueError):
            adicionar_eficiencia(pares.assign(TeamA=2, TeamB=1), stats)


class Exp12Test(unittest.TestCase):
    def test_automl_nativo_sem_cotas_recebe_max100_seed_rmse_e_folds(self):
        from src.automl_h2o import treinar_automl
        aml = MagicMock()
        aml.leader.model_id = 'glm'
        aml.leader._model_json = {'output': {'model_category': 'Regression'}}
        aml.leaderboard.as_data_frame.return_value = pd.DataFrame({'model_id': ['glm'], 'rmse': [1.]})
        aml.event_log.as_data_frame.return_value = pd.DataFrame()
        modulo = ModuleType('h2o.automl')
        modulo.H2OAutoML = MagicMock(return_value=aml)
        xgb = ModuleType('h2o.estimators.xgboost')
        xgb.H2OXGBoostEstimator = MagicMock()
        xgb.H2OXGBoostEstimator.available.return_value = False
        tabela = dados()['M'].query('Season < 2023')
        with patch.dict(comum.sys.modules, {'h2o.automl': modulo, 'h2o.estimators.xgboost': xgb}), \
             patch('src.automl_h2o._h2o', return_value=MagicMock()), \
             patch('src.automl_h2o._chaves_h2o', return_value=set()):
            treinar_automl(tabela, features=e.FEATURES_BASE, ano_previsao=2023, max_models=100, seed=42)
        kwargs = modulo.H2OAutoML.call_args.kwargs
        self.assertEqual(kwargs['max_models'], 100)
        self.assertEqual(kwargs['seed'], 42)
        self.assertEqual(kwargs['sort_metric'], 'RMSE')
        self.assertEqual(kwargs['nfolds'], 5)
        self.assertEqual(kwargs['exclude_algos'], ['DeepLearning', 'XGBoost'])
        for nome in ['modeling_plan', 'include_algos', 'exploitation_ratio', 'algo_parameters']:
            self.assertNotIn(nome, kwargs)
        self.assertEqual(aml.train.call_args.kwargs['fold_column'], 'FoldInterno')

    def test_auditoria_108_bases_nativas_separadas_de_auxiliares_e_receitas(self):
        from src.exp12_h2o import auditar_descoberta
        snapshots = [{'model_id': f'GBM_grid_1_model_{i}', 'algo': 'gbm'} for i in range(100)]
        snapshots += [{'model_id': f'GBM_{i}', 'algo': 'gbm'} for i in range(1, 6)]
        snapshots += [{'model_id': ident, 'algo': algo} for ident, algo in [('GLM_1','glm'), ('DRF_1','drf'), ('XRT_1','drf')]]
        ids = [s['model_id'] for s in snapshots]
        origens = [{'ModeloOrigem': f'StackedEnsemble_{tipo}', 'BasesOriginais': ids,
                    'MetalearnerOrigem': f'meta_{tipo}'} for tipo in ['BestOfFamily', 'AllModels']]
        candidatos = [{'Candidato': ident, 'Familia': s['algo']} for ident, s in zip(ids, snapshots)]
        candidatos += [{'Candidato': f'stack_{i}', 'Familia': 'StackedEnsemble_' + str(i),
                         'Parametros': {'Bases': ids[:20]}} for i in range(2)]
        resultado = {'config': {'max_models': 100}, 'snapshots': snapshots,
            'ensembles_originais': origens, 'candidatos': candidatos,
            'leaderboard': [{'model_id': ident} for ident in ids + [r['ModeloOrigem'] for r in origens]]}
        antes = deepcopy(resultado)
        auditoria = auditar_descoberta(resultado)
        resumo = auditoria['resumo']
        self.assertEqual(resumo['modelos_base_automl'], 108)
        self.assertEqual(resumo['ensembles_automl'], 2)
        self.assertEqual(resumo['excedente_base_sobre_max_models'], 8)
        self.assertEqual(resumo['quantidade_bases_auxiliares_fora_leaderboard'], 0)
        self.assertEqual(resumo['candidatos_unicos_avaliacao_externa'], 110)
        self.assertEqual(resumo['ids_leaderboard_repetidos'], 0)
        self.assertEqual(len(resumo['metalearners_originais_fora_leaderboard']), 2)
        self.assertEqual(resultado, antes)
        extras = deepcopy(resultado)
        extras['ensembles_originais'][0]['BasesOriginais'].append('base_extra')
        self.assertEqual(auditar_descoberta(extras)['resumo']['bases_auxiliares_fora_leaderboard'], ['base_extra'])

    def test_tabpfn_corrige_instancia_sem_mudar_identidade_n_estimators_ou_avisos_importantes(self):
        for candidato in e.candidatos_tabpfn('M', verificar_api=False):
            parametros = candidato['Parametros']
            self.assertNotIn('auto_scale_n_estimators', parametros)
            legado = {**parametros, 'auto_scale_n_estimators': False}
            self.assertEqual(candidato['Candidato'], 'M_tabpfn12_' + comum.identidade(legado)[:16])
            candidato_legado = deepcopy(candidato)
            candidato_legado['Parametros'] = legado
            with patch.object(comum, 'modelo_ajustado', side_effect=fake_modelo) as fit:
                with e._modelo_individual(dados()['M'].query('Season < 2023'), candidato_legado, e.FEATURES_BASE, 2023):
                    efetivos = fit.call_args.args[1]['Parametros']
                    self.assertNotIn('auto_scale_n_estimators', efetivos)
                    self.assertEqual(efetivos['n_estimators'], parametros['n_estimators'])
            self.assertIn('auto_scale_n_estimators', candidato_legado['Parametros'])
        for n in ['auto', 0, -1, True]:
            with self.assertRaises(ValueError):
                e._parametros_tabpfn({'n_estimators': n, 'auto_scale_n_estimators': False})
        msg = 'auto_scale_n_estimators is deprecated and will be removed in v9. Only auto is affected.'
        candidato = e.candidatos_tabpfn('M', max_candidatos=1, verificar_api=False)[0]
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter('always')
            with patch.object(comum, 'modelo_ajustado', side_effect=fake_modelo):
                with e._modelo_individual(dados()['M'].query('Season < 2023'), candidato, e.FEATURES_BASE, 2023):
                    warnings.warn_explicit(msg, FutureWarning, 'regressor.py', 1030, module='tabpfn.regressor')
                    warnings.warn_explicit('CUDA memory/feature limit warning', RuntimeWarning, 'regressor.py', 1030, module='tabpfn.regressor')
                    warnings.warn_explicit('Numeric warning', FutureWarning, 'regressor.py', 1030, module='tabpfn.regressor')
                    warnings.warn_explicit(msg, FutureWarning, 'other.py', 1, module='other')
            warnings.warn_explicit(msg, FutureWarning, 'regressor.py', 1030, module='tabpfn.regressor')
        self.assertEqual(len(capturados), 4)

    def test_checkpoints_legados_validos_reutilizados_sem_retreino_ou_regravacao(self):
        candidato = e.candidatos_tabpfn('M', max_candidatos=1, verificar_api=False)[0]
        legado = deepcopy(candidato)
        legado['Parametros']['auto_scale_n_estimators'] = False
        base = dados()['M']
        with TemporaryDirectory() as tmp, patch.object(comum, 'modelo_ajustado', side_effect=fake_modelo) as fit:
            original = e._avaliar(base, legado, 2023, features=e.FEATURES_BASE, pasta=tmp)
            _, path = e._checkpoint(base, legado, 2023, e.FEATURES_BASE, tmp)
            conteudo = path.read_bytes()
            reusado = e._avaliar(base, candidato, 2023, features=e.FEATURES_BASE, pasta=tmp)
            self.assertEqual(reusado, original)
            fit.assert_called_once()
            self.assertEqual(path.read_bytes(), conteudo)
            self.assertEqual(len(list(Path(tmp).glob('corte_*.json'))), 1)
            alterado = deepcopy(candidato)
            alterado['Parametros']['n_estimators'] = 8
            e._avaliar(base, alterado, 2023, features=e.FEATURES_BASE, pasta=tmp)
            self.assertEqual(fit.call_count, 2)
            # Mesma compatibilidade para previsões finais, sem mover/sobrescrever arquivos.
            config_final = {'revisao': 1, 'candidato': candidato, 'features': e.FEATURES_BASE,
                            'dados': 'hash-treino', 'sample': 'hash-sample', 'versoes': comum.versoes()}
            config_final_antiga = deepcopy(config_final)
            config_final_antiga['candidato']['Parametros']['auto_scale_n_estimators'] = False
            path_final = Path(tmp) / f'final_{candidato["Candidato"]}_{comum.identidade(config_final)[:16]}.json'
            path_antigo = path_final.with_name(path_final.stem.rsplit('_', 1)[0] + '_' + comum.identidade(config_final_antiga)[:16] + '.json')
            registro = {'config': config_final_antiga, 'ID': ['2026_1_2'], 'Pred': [2.5], 'segundos': 1.}
            comum.salvar_json(path_antigo, registro)
            self.assertEqual(e._ler_checkpoint_compativel(path_final, config_final), registro)
            self.assertFalse(path_final.exists())
            alterado = deepcopy(config_final)
            alterado['candidato']['Parametros']['softmax_temperature'] = .8
            outro_path = path_final.with_name('final_' + candidato['Candidato'] + '_' + comum.identidade(alterado)[:16] + '.json')
            self.assertIsNone(e._ler_checkpoint_compativel(outro_path, alterado))

    def test_catalogo_100_estrutural_sem_amostragem(self):
        banco = e.candidatos_tabpfn('M', verificar_api=False)
        self.assertEqual(len(banco), 100)
        self.assertEqual(len({c['Candidato'] for c in banco}), 100)
        self.assertEqual(len({c['Estrutura'] for c in banco}), 100)
        self.assertEqual({c['Parametros']['n_estimators'] for c in banco}, {4,8,16,32,64})
        self.assertTrue(all(c['Parametros']['random_state'] == 42 for c in banco))
        self.assertTrue(all(not c['Parametros']['ignore_pretraining_limits'] for c in banco))
        self.assertTrue(all('SUBSAMPLE_SAMPLES' not in c['Parametros']['inference_config'] for c in banco))
        self.assertEqual(banco, e.candidatos_tabpfn('M', verificar_api=False))

    def test_receitas_origem_limite_e_dependencias(self):
        base = bases()
        antes = deepcopy(base)
        stacks = ensembles(base)
        self.assertEqual(len(stacks), 2)
        allmodels = next(c for c in stacks if c['Familia'].endswith('AllModels'))
        self.assertEqual(allmodels['Parametros']['Bases'], ['M_b0', 'M_b1'])
        self.assertEqual(allmodels['BasesExcluidasPeloLimite'], ['M_b2'])
        self.assertEqual(allmodels['Parametros']['Reconstrucao'], 'receita_equivalente')
        self.assertEqual(base, antes)
        self.assertEqual(montar_receitas(base, [dict(ModeloOrigem='StackedEnsemble_AllModels',
            BasesOriginais=['desconhecido'], RMSEInterno=1.)])[0], [])

    def test_uma_busca_pre2023_e_retomada_snapshots(self):
        from types import SimpleNamespace
        from src import exp12_h2o as h
        modelos = {str(i): SimpleNamespace(algo='gbm', model_id=str(i)) for i in range(2)}
        busca = {'leaderboard': pd.DataFrame({'model_id':['0','1'], 'rmse':[1.,2.], 'rmsle':[np.nan,np.nan]})}
        def extrair(snapshots, cat):
            return bases(cat)[:2], []
        with TemporaryDirectory() as tmp, patch.object(h.backend, 'treinar_automl', return_value=busca) as fit, \
             patch.object(h.backend, '_h2o', return_value=SimpleNamespace(get_model=modelos.get)), \
             patch.object(h.backend, 'limpar_execucao'), \
             patch.object(h.comum, '_snapshot_modelo', side_effect=lambda m, *args: {'model_id':m.model_id, 'algo':m.algo}), \
             patch.object(h, '_extrair_todas_bases', side_effect=extrair):
            result = descobrir_h2o(dados()['M'], 'M', features=e.FEATURES_BASE, pasta=tmp)
            self.assertTrue(fit.call_args.args[0].Season.lt(2023).all())
            self.assertEqual(fit.call_args.kwargs['max_models'], 100)
            self.assertEqual(fit.call_args.kwargs['n_folds'], 5)
            descobrir_h2o(dados()['M'], 'M', features=e.FEATURES_BASE, pasta=tmp)
            fit.assert_called_once()
            self.assertTrue(result['falhas'])  # Ausência dos ensembles explicitamente registrada.

    def test_avaliacao_retomada_rankings_controle_e_quatro_csvs(self):
        base = dados()
        banco = [c for cat in ['M','W'] for c in [*bases(cat), *e.candidatos_tabpfn(cat, max_candidatos=2, verificar_api=False)]]
        with TemporaryDirectory() as tmp, patch.object(e, 'SessaoH2O', FakeSessao), \
             patch.object(comum, 'modelo_ajustado', side_effect=fake_modelo) as fit:
            r = e.avaliar_banco(base, banco, features=e.FEATURES_EFICIENCIA, pasta=tmp)
            self.assertEqual(len(r['avaliacoes']), 30)
            self.assertTrue(all(a['status']=='ok' for a in r['avaliacoes']))
            self.assertEqual(fit.call_count, 12)
            auditoria = e.auditar_checkpoints(base, banco, features=e.FEATURES_EFICIENCIA, pasta=tmp)
            self.assertEqual(auditoria.query("Abordagem == 'h2o'").CandidatosComTresCortesValidos.tolist(), [3,3])
            self.assertEqual(auditoria.query("Abordagem == 'tabpfn'").CandidatosComTresCortesValidos.tolist(), [2,2])
            self.assertEqual(fit.call_count, 12)
            novamente = e.avaliar_banco(base, banco, features=e.FEATURES_EFICIENCIA, pasta=tmp)
            self.assertEqual(fit.call_count, 12)
            refs = pd.DataFrame({'Season':[2023,2024,2025], 'Peso':[.2,.3,.5]})
            scores, ranking = comum.rankings(r['avaliacoes'], refs)
            self.assertTrue(scores.RMSE_pond.eq(0).all())
            self.assertTrue(e.validar_controle(base, deepcopy(base)))
            diferente = deepcopy(base)
            diferente['M'].loc[0,'Target'] += 1
            with self.assertRaises(AssertionError):
                e.validar_controle(base, diferente)
            tabela = e.comparar_rankings({'eficiencia':{'ranking':ranking,'candidatos':banco},
                                         'baseline':{'ranking':ranking,'candidatos':banco}})
            self.assertEqual(set(tabela.Conjunto), {'eficiencia','baseline'})
            pares = pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
                Categoria=nome,TemDuasSeeds=i!=2, **{f:float(i) for f in e.FEATURES_EFICIENCIA})
                for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
            sample = pares[['ID']].assign(Pred=0.)
            final = e.gerar_submissions(base,banco,ranking,pares,sample,features=e.FEATURES_EFICIENCIA,
                pasta=Path(tmp)/'final',destino=Path(tmp)/'csv')
            self.assertEqual(len(final['submissions']),4)
            for row in final['submissions']:
                csv = pd.read_csv(Path(tmp)/'csv'/row['Arquivo'])
                self.assertEqual(csv.ID.tolist(),sample.ID.tolist())
                self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
                self.assertTrue(np.isfinite(csv.Pred).all())
            n_fit = fit.call_count
            e.gerar_submissions(base,banco,ranking,pares,sample,features=e.FEATURES_EFICIENCIA,
                pasta=Path(tmp)/'final',destino=Path(tmp)/'csv')
            self.assertEqual(fit.call_count,n_fit)
            with self.assertRaises(ValueError):
                e.gerar_submissions(base,banco,ranking,pares,sample,features=e.FEATURES_BASE,
                    pasta=Path(tmp)/'final',destino=Path(tmp)/'csv')
            e.comum.salvar_json(Path(tmp)/'status.json',novamente)

    def test_falha_exclui_ranking_e_orcamento_registra_pendentes(self):
        banco = e.candidatos_tabpfn('M', max_candidatos=3, verificar_api=False)
        with TemporaryDirectory() as tmp, patch.object(comum,'modelo_ajustado',side_effect=RuntimeError('falha')):
            r = e.avaliar_banco(dados(),banco,features=e.FEATURES_EFICIENCIA,pasta=tmp,
                                orcamento_tabpfn_por_categoria=1e-9)
            self.assertEqual(len(r['avaliacoes']),3)
            self.assertEqual(len(r['pendentes']),2)
            with self.assertRaises(ValueError):
                comum.rankings(r['avaliacoes'],pd.DataFrame({'Season':[2023,2024,2025],'Peso':[.2,.3,.5]}))

    def test_folds_por_temporada_e_rejeicao_de_futuro(self):
        from src.automl_h2o import criar_folds_temporadas
        tabela = dados()['M'].query('Season < 2023')
        folds = criar_folds_temporadas(tabela.Season, n_folds=5, seed=42)
        self.assertTrue(pd.DataFrame({'Season':tabela.Season,'Fold':folds}).groupby('Season').Fold.nunique().eq(1).all())
        with self.assertRaises(ValueError):
            SessaoH2O(dados()['M'], bases(), features=e.FEATURES_EFICIENCIA, ano=2023)


@unittest.skipUnless(os.environ.get('MARCH_MANIA_H2O_INTEGRATION')=='1', 'Integração H2O pequena habilitada explicitamente.')
class StackingRealTest(unittest.TestCase):
    def test_stacking_real_oof_reuso_e_metalearner_sem_futuro(self):
        import h2o
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = sock.getsockname()[1]
        with TemporaryDirectory(prefix='exp12_h2o_', ignore_cleanup_errors=True) as tmp:
            h2o.init(ip='127.0.0.1',port=port,bind_to_localhost=True,nthreads=1,max_mem_size='1G',ice_root=tmp)
            h2o.no_progress()
            try:
                x = np.random.default_rng(42).normal(size=80)
                treino = pd.DataFrame({'Season':np.repeat(np.arange(2010,2018),10),
                    'X':x,'Target':2*x + np.sin(x)})
                base = bases()
                stacks = ensembles(base)
                with SessaoH2O(treino,[*base,*stacks],features=['X'],ano=2018,n_folds=2) as sessao:
                    modelo = sessao.obter(stacks[0])
                    self.assertEqual(len(sessao.audit['BasesTreinadas']),3)
                    for c in base:
                        m = sessao.modelos[c['Candidato']]
                        self.assertEqual(m.actual_params['fold_column']['column_name'],'FoldInterno')
                        self.assertEqual(m.cross_validation_holdout_predictions().nrows,len(treino))
                    meta = modelo.metalearner()
                    self.assertEqual(meta._model_json['output']['training_metrics']._metric_json['nobs'],len(treino))
                    with sessao.adaptador(stacks[1]) as adaptador:
                        self.assertTrue(np.isfinite(adaptador.predict(treino[['X']].head(4))).all())
                    self.assertEqual(len(sessao.audit['BasesTreinadas']),3)
                    hashes = {r['HashFolds'] for r in sessao.audit['BasesTreinadas']}
                    self.assertEqual(len(hashes),1)
                    self.assertTrue(all(r['HeapMB'] >= 0 for r in sessao.audit['HeapJVMObservadoMB']))
                    self.assertTrue(all(r['UltimoAnoTreinoMetalearner']==2017 for r in sessao.audit['EnsemblesTreinados']))
                    # O snapshot é reconstruível mesmo quando vem de get_model.
                    obtido = h2o.get_model(sessao.modelos[base[1]['Candidato']].model_id)
                    comum.extrair_configuracao(obtido,'M',1,1.)
                self.assertEqual(len(h2o.ls()),0)
                # Também verifica a extração real do grafo de um AutoML minúsculo.
                with TemporaryDirectory() as checkpoints:
                    descoberta = descobrir_h2o(treino,'M',features=['X'],pasta=checkpoints,
                        max_models=3,n_folds=2,limite_all=2,exclude_algos=['DeepLearning','XGBoost'])
                    receitas = [c for c in descoberta['candidatos'] if c['Familia'].startswith('StackedEnsemble_')]
                    self.assertTrue(receitas, descoberta['falhas'])
                    self.assertTrue(descoberta['ensembles_originais'])
                    self.assertEqual(len(h2o.ls()),0)
                    with SessaoH2O(treino,descoberta['candidatos'],features=['X'],ano=2023,n_folds=2) as sessao:
                        with sessao.adaptador(receitas[0]) as adaptador:
                            self.assertTrue(np.isfinite(adaptador.predict(treino[['X']].head(4))).all())
            finally:
                h2o.cluster().shutdown(prompt=False)
                servidor = h2o.connection().local_server
                if servidor is not None and servidor._process is not None:
                    servidor._process.wait(timeout=20)


if __name__=='__main__':
    unittest.main()
