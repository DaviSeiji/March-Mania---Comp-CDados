import unittest
from unittest.mock import patch, MagicMock
from types import ModuleType

import numpy as np
import pandas as pd

from src.automl_h2o import criar_folds_temporadas, avaliar_temporadas_h2o, treinar_automl, AdaptadorH2O
from src.metrics import logistic_brier
from src.submission import prever_confrontos


class ExclusoesAutoMLTest(unittest.TestCase):
    def test_dl_opcional_sem_alterar_padrao_exp8(self):
        backend = MagicMock()
        aml = MagicMock()
        aml.leader.model_id = 'modelo'
        aml.leader.algo = 'glm'
        aml.leader._model_json = {'output': {'model_category': 'Regression'}}
        aml.leaderboard.as_data_frame.return_value = pd.DataFrame({'model_id': ['modelo'], 'rmse': [1.]})
        aml.event_log.as_data_frame.return_value = pd.DataFrame()
        automl_module = ModuleType('h2o.automl')
        automl_module.H2OAutoML = MagicMock(return_value=aml)
        xgb_module = ModuleType('h2o.estimators.xgboost')
        xgb_module.H2OXGBoostEstimator = MagicMock()
        xgb_module.H2OXGBoostEstimator.available.return_value = False
        tabela = pd.DataFrame({'Season': range(2010, 2016), 'X': range(6), 'Target': range(6)})
        with patch.dict('sys.modules', {'h2o.automl': automl_module, 'h2o.estimators.xgboost': xgb_module}), \
             patch('src.automl_h2o._h2o', return_value=backend), \
             patch('src.automl_h2o._chaves_h2o', return_value=set()):
            treinar_automl(tabela, features=['X'], ano_previsao=2021)
            self.assertEqual(automl_module.H2OAutoML.call_args.kwargs['exclude_algos'], ['DeepLearning', 'XGBoost'])
            treinar_automl(tabela, features=['X'], ano_previsao=2021, exclude_algos=[])
            self.assertEqual(automl_module.H2OAutoML.call_args.kwargs['exclude_algos'], ['XGBoost'])
            self.assertEqual(aml.train.call_args.kwargs['fold_column'], 'FoldInterno')
            treinar_automl(tabela, features=['X'], ano_previsao=2021, max_models=10,
                          include_algos=['DeepLearning'], exclude_algos=None, max_runtime_secs_per_model=120)
            config = automl_module.H2OAutoML.call_args.kwargs
            self.assertEqual(config['include_algos'], ['DeepLearning'])
            self.assertNotIn('exclude_algos', config)
            self.assertEqual(config['max_runtime_secs_per_model'], 120)
            self.assertEqual(config['max_models'], 10)


class FoldsTemporadasTest(unittest.TestCase):
    def test_determinismo_agrupamento_e_ordem(self):
        anos = pd.Series(np.repeat(np.arange(2003, 2021), 3), index=np.arange(54) + 100)
        antes = anos.copy()
        folds = criar_folds_temporadas(anos)
        pd.testing.assert_series_equal(folds, criar_folds_temporadas(anos))
        pd.testing.assert_series_equal(folds, criar_folds_temporadas(anos.sample(frac=1, random_state=7)).reindex(anos.index))
        self.assertEqual(set(folds), set(range(5)))
        self.assertTrue(pd.DataFrame({'Season': anos, 'Fold': folds}).groupby('Season').Fold.nunique().eq(1).all())
        pd.testing.assert_series_equal(anos, antes)

    def test_anos_invalidos(self):
        for anos in [[], [2010, 2011], [2010, np.nan, 2012, 2013, 2014], [2010.5, 2011, 2012, 2013, 2014]]:
            with self.assertRaises(ValueError):
                criar_folds_temporadas(anos)

    def test_rejeita_futuro_e_alvo_como_feature(self):
        tabela = pd.DataFrame({'Season': [2026], 'X': [1.], 'Target': [1.]})
        with self.assertRaises(ValueError):
            treinar_automl(tabela, features=['X'], ano_previsao=2026)
        with self.assertRaises(ValueError):
            treinar_automl(tabela, features=['Target'], ano_previsao=2027)
        with self.assertRaises(ValueError):
            treinar_automl(tabela.assign(Target=np.inf), features=['X'], ano_previsao=2027)


class AvaliacaoTemporalTest(unittest.TestCase):
    def setUp(self):
        self.dados = {}
        for cat, base, quantidade in [('M', 1000, 2), ('W', 3000, 3)]:
            self.dados[cat] = pd.DataFrame([
                {'Season': ano, 'TeamA': base, 'TeamB': base + 1, 'X': float(i), 'Target': float(i + 2)}
                for ano in [*range(2010, 2020), 2021, 2022] for i in range(quantidade)
            ])

    def falso_treino(self, tabela, **kw):
        self.assertTrue(tabela.Season.lt(kw['ano_previsao']).all())
        self.assertNotIn('validation_frame', kw)
        self.assertNotIn('leaderboard_frame', kw)
        self.assertEqual(tabela.TeamA.nunique(), 1)
        return {'leader': 'interno', 'model_id': 'leader_rmse', 'algoritmo': 'glm', 'rmse_cv': 1.,
                'folds': pd.DataFrame({'Season': [int(tabela.Season.min())], 'FoldInterno': [0]}),
                'leaderboard': pd.DataFrame({'model_id': ['leader_rmse'], 'rmse': [1.]}),
                'eventos': pd.DataFrame(), 'chaves_h2o': []}

    def test_score_combinado_temporal_e_limpeza(self):
        with patch('src.automl_h2o.treinar_automl', side_effect=self.falso_treino) as treino, \
             patch('src.automl_h2o.prever_leader', side_effect=lambda leader, df, **kw: df.X.to_numpy()) as prever, \
             patch('src.automl_h2o.limpar_execucao') as limpar:
            resultado = avaliar_temporadas_h2o(self.dados, [2021, 2022], features=['X'])
        self.assertEqual(treino.call_count, 4)
        self.assertEqual(limpar.call_count, 4)
        self.assertTrue(all(c.args[0] == 'interno' for c in prever.call_args_list))
        self.assertTrue(all(c.args[1].Season.nunique() == 1 for c in prever.call_args_list))
        self.assertEqual(len(resultado['previsoes']), 10)
        medias = []
        for ano, grupo in resultado['previsoes'].groupby('Season'):
            esperado = logistic_brier(grupo.Target, grupo.Pred)
            medias.append(esperado)
            linha = resultado['resultados'].query("Season == @ano and Categoria == 'Combinado'").iloc[0]
            self.assertAlmostEqual(linha.Score, esperado)
        self.assertAlmostEqual(resultado['media'], np.mean(medias))

    def test_limpa_quando_previsao_falha(self):
        with patch('src.automl_h2o.treinar_automl', side_effect=self.falso_treino), \
             patch('src.automl_h2o.prever_leader', side_effect=ValueError('previsao invalida')), \
             patch('src.automl_h2o.limpar_execucao') as limpar:
            with self.assertRaises(ValueError):
                avaliar_temporadas_h2o(self.dados, [2021], features=['X'])
        limpar.assert_called_once()

    def test_categorias_sobrepostas(self):
        self.dados['W'] = self.dados['M'].copy()
        with self.assertRaises(ValueError):
            avaliar_temporadas_h2o(self.dados, [2021], features=['X'])

    def test_adaptador_preserva_fallback_e_margem(self):
        pares = pd.DataFrame({'ID': ['a', 'b', 'c'], 'Categoria': ['Masculino', 'Feminino', 'Masculino'],
                             'TemDuasSeeds': [True, True, False], 'X': [1., 2., np.nan]})
        with patch('src.automl_h2o.prever_leader', side_effect=lambda leader, df, **kw: np.full(len(df), leader)):
            resultado = prever_confrontos(pares, AdaptadorH2O(8.2, ['X']), AdaptadorH2O(-4.7, ['X']), features=['X'])
        np.testing.assert_array_equal(resultado.Pred, [8.2, -4.7, 0.])
        self.assertTrue(resultado.ID.equals(pares.ID))


if __name__ == '__main__':
    unittest.main()
