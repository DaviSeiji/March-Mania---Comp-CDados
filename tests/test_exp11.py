from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
import os
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import exp11 as e


def dados():
    return {cat: pd.DataFrame([{'Season': ano, 'TeamA': base, 'TeamB': base + 1,
        'X': float(i), 'Target': float(i + 1)} for ano in range(2015, 2026) for i in range(3)])
        for cat, base in [('M', 1000), ('W', 3000)]}


def candidatos():
    return [dict(Candidato=f'{cat}_{abordagem}_{i}', Categoria=cat, Abordagem=abordagem,
                 Familia='fake', Parametros={'device': 'cpu', 'deslocamento': i})
            for abordagem in ['h2o', 'tabpfn'] for cat in ['M', 'W'] for i in range(3)]


def modelo_mock(ident, algo='gbm'):
    return SimpleNamespace(algo=algo, model_id=ident, actual_params={},
        _model_json={'output': {'model_summary': SimpleNamespace(
            as_data_frame=lambda: pd.DataFrame({'number_of_trees': [3]}))}})


@contextmanager
def falso_modelo(treino, candidato, features, ano):
    assert treino.Season.lt(ano).all()
    assert treino.TeamA.nunique() == 1
    class Modelo:
        def predict(self, X):
            return X.X.to_numpy() + candidato['Parametros']['deslocamento']
    yield Modelo()


class Exp11Test(unittest.TestCase):
    def test_diagnostico_separa_chave_licenca_e_conexao(self):
        for valido, aceita, mensagem in [(False, True, 'rejeitou'), (None, True, 'consultar'),
                                        (True, False, 'ainda não foi aceita'),
                                        (True, None, 'indisponível'), (True, True, None)]:
            def nome_licenca(repo):
                self.assertEqual(repo, 'tabpfn_3_5')
                return 'tabpfn-3-5-license-v1.0'
            auth = SimpleNamespace(verify_token=lambda *args: valido,
                check_license_accepted=lambda *args: aceita, _get_license_name=nome_licenca)
            settings = SimpleNamespace(settings=SimpleNamespace(tabpfn=SimpleNamespace(auth_api_url='https://example.invalid')))
            with self.subTest(valido=valido, aceita=aceita), \
                 patch.dict(e.sys.modules, {'tabpfn.browser_auth': auth, 'tabpfn.settings': settings}), \
                 patch.dict(os.environ, {'TABPFN_TOKEN': 'credencial-ficticia'}):
                if mensagem:
                    with self.assertRaisesRegex(RuntimeError, mensagem) as erro:
                        e.validar_acesso_tabpfn()
                    self.assertNotIn('credencial-ficticia', str(erro.exception))
                else:
                    self.assertEqual(e.validar_acesso_tabpfn(), {
                        'chave_valida': True, 'licenca': 'tabpfn-3-5-license-v1.0', 'licenca_aceita': True})

    def test_token_local_sem_saida_ou_retorno(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / '.env.tabpfn'
            path.write_text('TABPFN_TOKEN="chave-ficticia-teste"\n', encoding='utf-8')
            output = StringIO()
            with redirect_stdout(output):
                self.assertIsNone(e.carregar_token_tabpfn(path))
            self.assertEqual(output.getvalue(), '')
            self.assertEqual(os.environ['TABPFN_TOKEN'], 'chave-ficticia-teste')
            for conteudo in ['TABPFN_TOKEN=COLE_SUA_CHAVE_AQUI', 'TABPFN_TOKEN=',
                              'ERRADO=chave-ficticia-teste', 'TABPFN_TOKEN=chave com espaco']:
                path.write_text(conteudo, encoding='utf-8')
                with self.assertRaises(ValueError) as erro:
                    e.carregar_token_tabpfn(path)
                self.assertNotIn('chave-ficticia-teste', str(erro.exception))

    def test_filtro_oculta_so_avisos_esperados_de_parada(self):
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter('always')
            e.configurar_avisos_h2o()
            for msg in ['Stopping metric is ignored for _stopping_rounds=0.',
                        'Stopping tolerance is ignored for _stopping_rounds=0.',
                        'Falha numérica inesperada']:
                warnings.warn_explicit(msg, RuntimeWarning, 'estimator_base.py', 232,
                    module='h2o.estimators.estimator_base')
            warnings.warn_explicit('Stopping metric is ignored for _stopping_rounds=0.',
                RuntimeWarning, 'outro.py', 1, module='outro')
        self.assertEqual(len(capturados), 2)
        self.assertEqual(str(capturados[0].message), 'Falha numérica inesperada')

    def test_piloto_pode_repetir_falha_sem_repetir_sucesso(self):
        with TemporaryDirectory() as tmp:
            with patch.object(e, 'modelo_ajustado', side_effect=RuntimeError('licença pendente')) as fit:
                erro = e.avaliar_corte(dados()['M'], candidatos()[0], 2023, features=['X'], pasta=tmp)
                self.assertEqual(erro['status'], 'falha')
            with patch.object(e, 'modelo_ajustado', side_effect=falso_modelo) as fit:
                sucesso = e.avaliar_corte(dados()['M'], candidatos()[0], 2023, features=['X'], pasta=tmp, repetir_falhas=True)
                self.assertEqual(sucesso['status'], 'ok')
                e.avaliar_corte(dados()['M'], candidatos()[0], 2023, features=['X'], pasta=tmp, repetir_falhas=True)
                fit.assert_called_once()

    def test_descoberta_uma_busca_so_pre2023_sem_ensembles_e_retomavel(self):
        modelos = {str(i): modelo_mock(str(i), 'stackedensemble' if i == 0 else 'gbm') for i in range(5)}
        busca = {'leaderboard': pd.DataFrame({'model_id': list(modelos), 'rmse': [1., 2., 3., 4., 5.],
                                             'rmsle': [np.nan] * 5})}
        def extrair(modelo, categoria, posicao, rmse):
            return {'Candidato': modelo.model_id, 'Categoria': categoria, 'PosicaoLeaderboard': posicao}
        with TemporaryDirectory() as tmp, patch.object(e.h2ob, 'treinar_automl', return_value=busca) as fit, \
             patch.object(e.h2ob, '_h2o', return_value=SimpleNamespace(get_model=modelos.get)), \
             patch.object(e.h2ob, 'limpar_execucao') as limpar, patch.object(e, 'extrair_configuracao', side_effect=extrair):
            resultado = e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp, top_n=3)
            self.assertTrue(fit.call_args.args[0].Season.lt(2023).all())
            self.assertEqual(fit.call_args.kwargs['n_folds'], 5)
            self.assertEqual(fit.call_args.kwargs['max_models'], 100)
            self.assertEqual(resultado['ensembles_excluidos'], ['0'])
            self.assertTrue(all(row['rmsle'] is None for row in resultado['leaderboard']))
            self.assertEqual([c['Candidato'] for c in resultado['candidatos']], ['1', '2', '3'])
            e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp, top_n=3)
            self.assertEqual(fit.call_count, 1)
            limpar.assert_called_once()

    def test_nan_auxiliar_nao_altera_rmse_nem_parametros(self):
        original = {'leaderboard': [{'rmse': 2.5, 'rmsle': np.nan, 'outro': np.inf}],
                    'candidatos': [{'Parametros': {'alpha': .5}}]}
        normalizado = e._normalizar_relatorio_descoberta(original)
        self.assertEqual(normalizado['leaderboard'], [{'rmse': 2.5, 'rmsle': None, 'outro': None}])
        self.assertEqual(normalizado['candidatos'], original['candidatos'])
        self.assertTrue(np.isnan(original['leaderboard'][0]['rmsle']))
        with self.assertRaises(ValueError):
            e._normalizar_relatorio_descoberta({'leaderboard': [{'rmse': np.nan}]})
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                e.salvar_json(Path(tmp) / 'invalido.json', {'RMSE': np.nan})

    def test_recupera_busca_concluida_do_traceback_sem_retreinar(self):
        modelos = {str(i): modelo_mock(str(i)) for i in range(3)}
        busca = {'leaderboard': pd.DataFrame({'model_id': list(modelos), 'rmse': [1., 2., 3.], 'rmsle': [np.nan]*3})}
        with TemporaryDirectory() as tmp, patch.object(e.h2ob, 'treinar_automl', return_value=busca) as fit, \
             patch.object(e.h2ob, '_h2o', return_value=SimpleNamespace(get_model=modelos.get)), \
             patch.object(e.h2ob, 'limpar_execucao'), \
             patch.object(e, 'extrair_configuracao', side_effect=lambda m, *args: {'Candidato': m.model_id}):
            try:
                with patch.object(e, 'salvar_json', side_effect=ValueError('erro de gravação')):
                    e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            except ValueError as exc:
                with patch.object(e.sys, 'last_traceback', exc.__traceback__, create=True):
                    path = e.recuperar_descoberta_interrompida()
            self.assertTrue(path.exists())
            resultado = e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            self.assertEqual(len(resultado['candidatos']), 3)
            fit.assert_called_once()

    def test_wrappers_h2o_sao_convertidos_sem_deepcopy(self):
        class Resposta(dict):
            def __new__(cls, keyvals):
                return dict.__new__(cls)
        original = Resposta({'seed': 42, 'training_frame': Resposta({'name': 'frame'}),
                             'alpha': [np.float64(.5)]})
        with self.assertRaises(TypeError):
            deepcopy(original)
        simples = e._parametros_simples(original)
        self.assertIs(type(simples), dict)
        self.assertIs(type(simples['training_frame']), dict)
        self.assertEqual(simples, original)
        self.assertEqual(deepcopy(simples), simples)

    def test_checkpoint_antigo_irrecuperavel_e_arquivado_antes_de_refazer(self):
        modelos = {str(i): modelo_mock(str(i)) for i in range(3)}
        busca = {'leaderboard': pd.DataFrame({'model_id': list(modelos), 'rmse': [1., 2., 3.]})}
        with TemporaryDirectory() as tmp, patch.object(e.h2ob, 'treinar_automl', return_value=busca) as fit, \
             patch.object(e.h2ob, '_h2o', return_value=SimpleNamespace(get_model=modelos.get)), \
             patch.object(e.h2ob, 'limpar_execucao'), \
             patch.object(e, 'extrair_configuracao', side_effect=lambda m, *args: {'Candidato': m.model_id}):
            e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            path = next(Path(tmp).glob('descoberta_*.json'))
            antigo = json.loads(path.read_text(encoding='utf-8'))
            antigo.pop('snapshots')
            antigo.update(candidatos=[], falhas=[{'Modelo': '0', 'Erro': "H2OResponse.__new__() missing 1 required positional argument: 'keyvals'"}])
            e.salvar_json(path, antigo)
            recuperado = e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(len(recuperado['candidatos']), 3)
            self.assertTrue(path.with_suffix('.falha_extracao.json').exists())

    def test_falhas_de_extracao_reprocessadas_sem_nova_busca(self):
        modelos = {str(i): modelo_mock(str(i)) for i in range(3)}
        busca = {'leaderboard': pd.DataFrame({'model_id': list(modelos), 'rmse': [1., 2., 3.]})}
        with TemporaryDirectory() as tmp, patch.object(e.h2ob, 'treinar_automl', return_value=busca) as fit, \
             patch.object(e.h2ob, '_h2o', return_value=SimpleNamespace(get_model=modelos.get)), \
             patch.object(e.h2ob, 'limpar_execucao'):
            with patch.object(e, 'extrair_configuracao', side_effect=ValueError('bug de extração')):
                salvo = e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            self.assertEqual(len(salvo['snapshots']), 3)
            self.assertFalse(salvo['candidatos'])
            with patch.object(e, 'extrair_configuracao', side_effect=lambda m, *args: {'Candidato': m.model_id}):
                recuperado = e.descobrir_h2o(dados()['M'], 'M', features=['X'], pasta=tmp)
            self.assertEqual(len(recuperado['candidatos']), 3)
            self.assertFalse(recuperado['falhas'])
            fit.assert_called_once()

    def test_extracao_arvores_reais_e_parametro_nao_portavel(self):
        class Estimador:
            def __init__(self, ntrees=50, seed=42, stopping_rounds=0, max_runtime_secs=0):
                self._parms = dict(ntrees=ntrees, seed=seed, stopping_rounds=stopping_rounds, max_runtime_secs=max_runtime_secs)
        modelo = SimpleNamespace(algo='drf', model_id='a', actual_params={
            'ntrees': 50, 'seed': 42, 'stopping_rounds': 3, 'offset_column': None, 'training_frame': 'old'},
            _model_json={'output': {'model_summary': SimpleNamespace(as_data_frame=lambda: pd.DataFrame({'number_of_trees': [12]}))}})
        with patch.object(e, '_classe_h2o', return_value=Estimador):
            c = e.extrair_configuracao(modelo, 'M', 1, 2.)
            self.assertEqual(c['Parametros']['ntrees'], 12)
            self.assertEqual(c['Parametros']['stopping_rounds'], 0)
            self.assertNotIn('training_frame', c['Parametros'])
            self.assertEqual(c['ParametrosNulosOmitidos'], ['offset_column'])
            modelo.actual_params['offset_column'] = 'external'
            with self.assertRaises(ValueError):
                e.extrair_configuracao(modelo, 'M', 1, 2.)

    def test_fluxo_retomada_metricas_rankings_e_doze_csvs(self):
        with TemporaryDirectory() as tmp, patch.object(e, 'modelo_ajustado', side_effect=falso_modelo) as fit:
            base, banco = dados(), candidatos()
            antes = deepcopy(base)
            aval = e.avaliar_banco(base, banco, features=['X'], pasta=tmp)
            self.assertEqual(fit.call_count, 36)
            self.assertTrue(all(r['status'] == 'ok' for r in aval['avaliacoes']))
            again = e.avaliar_banco(base, banco, features=['X'], pasta=tmp)
            self.assertEqual(fit.call_count, 36)
            self.assertEqual(aval, again)
            refs = pd.DataFrame({'Season': [2023, 2024, 2025], 'Peso': [.2, .3, .5]})
            scores, ranks = e.rankings(aval['avaliacoes'], refs)
            self.assertEqual(len(scores), 12)
            self.assertTrue(ranks.query('Posicao == 1').Candidato.str.endswith('_1').all())
            self.assertTrue(scores.loc[scores.Candidato.str.endswith('_1'), 'RMSE_pond'].eq(0).all())
            rows = []
            for cat, base_id, nome in [('M', 1000, 'Masculino'), ('W', 3000, 'Feminino')]:
                for i in range(3):
                    rows.append({'ID': f'2026_{base_id}_{base_id + i + 1}', 'Season': 2026,
                        'TeamA': base_id, 'TeamB': base_id + i + 1, 'Categoria': nome,
                        'TemDuasSeeds': i != 2, 'X': float(i) if i != 2 else np.nan})
            pares = pd.DataFrame(rows)
            sample = pares[['ID']].assign(Pred=0.)
            final = e.gerar_submissions(base, banco, ranks, pares, sample, features=['X'],
                                       pasta=Path(tmp) / 'final', destino=Path(tmp) / 'csv')
            self.assertEqual(len(final['submissions']), 12)
            self.assertFalse(final['falhas'])
            self.assertEqual(fit.call_count, 48)
            self.assertTrue(any(r['IdenticaA'] for r in final['submissions']))
            for r in final['submissions']:
                csv = pd.read_csv(Path(tmp) / 'csv' / r['Arquivo'])
                self.assertEqual(csv.ID.tolist(), sample.ID.tolist())
                self.assertTrue(csv.loc[~pares.TemDuasSeeds, 'Pred'].eq(0).all())
                self.assertTrue(np.isfinite(csv.Pred).all())
            e.gerar_submissions(base, banco, ranks, pares, sample, features=['X'],
                               pasta=Path(tmp) / 'final', destino=Path(tmp) / 'csv')
            self.assertEqual(fit.call_count, 48)
            for cat in base:
                pd.testing.assert_frame_equal(base[cat], antes[cat])

    def test_pesos_formula_falha_empate_e_duplicados(self):
        refs = pd.DataFrame({'Season': [2023, 2024, 2025], 'Peso': [.2, .3, .5]})
        rows = [{'Categoria': 'M', 'Abordagem': 'h2o', 'Candidato': ident,
                 'Season': ano, 'status': 'ok', 'RMSE': value * 10, 'LB': value}
                for ident in ['a', 'b'] for ano, value in zip(e.ANOS, [.1, .2, .3])]
        scores, ranks = e.rankings(rows, refs)
        self.assertAlmostEqual(scores.iloc[0].LB_pond, .23)
        self.assertAlmostEqual(scores.iloc[0].RMSE_pond, 2.3)
        self.assertTrue(ranks.query('Posicao == 1').Candidato.eq('a').all())
        rows[0]['status'] = 'falha'
        self.assertEqual(e.rankings(rows, refs)[0].Candidato.tolist(), ['b'])
        self.assertEqual(e.rankings(rows[1:], refs)[0].Candidato.tolist(), ['b'])
        with self.assertRaises(ValueError):
            e.rankings(rows + [rows[0]], refs)
        with self.assertRaises(ValueError):
            e.rankings(rows, refs.assign(Peso=.1))

    def test_retomada_invalida_com_dados_ou_parametros_alterados(self):
        with TemporaryDirectory() as tmp, patch.object(e, 'modelo_ajustado', side_effect=falso_modelo) as fit:
            base, c = dados()['M'], candidatos()[0]
            e.avaliar_corte(base, c, 2023, features=['X'], pasta=tmp)
            e.avaliar_corte(base.assign(Target=base.Target + 1), c, 2023, features=['X'], pasta=tmp)
            c['Parametros']['deslocamento'] = 4
            e.avaliar_corte(base, c, 2023, features=['X'], pasta=tmp)
            self.assertEqual(fit.call_count, 3)

    def test_falhas_nao_substituidas_e_orcamento_pendente(self):
        with TemporaryDirectory() as tmp, patch.object(e, 'modelo_ajustado', side_effect=RuntimeError('falha real')):
            result = e.avaliar_banco(dados(), candidatos()[:2], features=['X'], pasta=tmp,
                                   orcamento_segundos=1e-9)
            self.assertEqual(len(result['avaliacoes']), 3)
            self.assertEqual(len(result['pendentes']), 1)
            self.assertTrue(all(r['status'] == 'falha' and 'falha real' in r['erro'] for r in result['avaliacoes']))

    def test_separacao_temporal_e_categorias(self):
        base = dados()
        base['M'].loc[0, 'Season'] = 2026
        with self.assertRaises(ValueError):
            e.validar_historico(base, ['X'])
        base = dados()
        base['W'] = base['M'].copy()
        with self.assertRaises(ValueError):
            e.validar_historico(base, ['X'])
        with self.assertRaises(ValueError):
            with e.modelo_ajustado(dados()['M'], candidatos()[0], ['X'], 2023):
                pass

    def test_tabpfn_oito_configs_sem_ignorar_limites(self):
        banco = e.candidatos_tabpfn('M')
        self.assertEqual(len(banco), 8)
        self.assertEqual({c['Familia'] for c in banco}, {'V3_5', 'V3_5_FAST'})
        self.assertTrue(all(c['Parametros']['device'] == 'cuda:0' for c in banco))
        self.assertTrue(all(c['Parametros']['ignore_pretraining_limits'] is False for c in banco))


if __name__ == '__main__':
    unittest.main()
