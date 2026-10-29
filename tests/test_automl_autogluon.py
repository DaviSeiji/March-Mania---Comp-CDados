"""Testes rápidos; backends pesados são substituídos por mocks."""

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.automl_autogluon import (
    AdaptadorAutoGluon, auditar_folds, avaliar_temporadas_autogluon,
    limpar_execucao, resumir_leaderboard, treinar_automl,
)
from src.metrics import logistic_brier
from src.submission import preparar_confrontos, prever_confrontos, salvar_submission


def historico(base=1000, quantidade=2):
    return pd.DataFrame([
        {"Season": ano, "TeamA": base, "TeamB": base + 1,
         "X": float(i), "Target": float(i + 2)}
        for ano in [*range(2010, 2020), 2021, 2022] for i in range(quantidade)
    ])


def ranking():
    return pd.DataFrame({"model": ["melhor", "outro"], "Familia": ["RFModel", "LGBModel"],
        "score_val": [-.05, -.08], "LogisticBrier": [.05, .08],
        "fit_time": [1., 2.], "pred_time_val": [.1, .2], "stack_level": [1, 1]})


class AutoGluonTest(unittest.TestCase):
    def test_validacoes_antes_de_importar_backend(self):
        tabela = historico()
        for features in [["Season"], ["Target"], ["X", "X"], []]:
            with self.assertRaises(ValueError):
                treinar_automl(tabela, features=features, ano_previsao=2026)
        for invalida in [tabela.assign(Season=2026), tabela.assign(X=np.nan),
                         tabela.assign(Target=np.inf), tabela.assign(Season=2010.5),
                         tabela.assign(TeamB=tabela.TeamA)]:
            with self.assertRaises(ValueError):
                treinar_automl(invalida, features=["X"], ano_previsao=2026)
        with self.assertRaises(ValueError):
            treinar_automl(tabela, features=["X"], ano_previsao=2022)

    def test_config_dados_features_e_limpeza(self):
        tabela = historico()
        antes = tabela.copy(deep=True)
        Predictor = MagicMock()
        predictor = Predictor.return_value
        predictor.features.return_value = ["X"]
        predictor.model_best = "melhor"
        predictor.model_failures.return_value = pd.DataFrame()
        with patch('src.automl_autogluon._backend', return_value=(Predictor, 'scorer')), \
             patch('src.automl_autogluon.auditar_folds', return_value=pd.DataFrame()), \
             patch('src.automl_autogluon.resumir_leaderboard', return_value=ranking()):
            execucao = treinar_automl(tabela, features=["X"], ano_previsao=2026)
        diretorio = Path(execucao['temporario'].name)
        self.assertTrue(diretorio.is_dir())
        kwargs = Predictor.call_args.kwargs
        self.assertEqual(kwargs['learner_kwargs'], {'ignored_columns': ['Season'], 'random_state': 42})
        self.assertEqual(kwargs['eval_metric'], 'scorer')
        self.assertNotIn('groups', kwargs)
        fit = predictor.fit.call_args.kwargs
        self.assertEqual(list(fit['train_data']), ['X', 'Season', 'Target'])
        self.assertEqual(fit['validation_structure'], {'group_on': 'Season'})
        self.assertEqual((fit['num_bag_folds'], fit['num_bag_sets'], fit['time_limit']), (5, 1, 300))
        self.assertNotIn('tuning_data', fit)
        self.assertNotIn('excluded_model_types', fit)
        self.assertIsNone(fit['feature_prune_kwargs'])
        limpar_execucao(execucao)
        limpar_execucao(execucao)
        self.assertFalse(diretorio.exists())
        predictor.unpersist.assert_called_once()
        pd.testing.assert_frame_equal(tabela, antes)

    def test_limpa_erro_fit(self):
        Predictor = MagicMock()
        Predictor.return_value.fit.side_effect = RuntimeError('falhou')
        with patch('src.automl_autogluon._backend', return_value=(Predictor, 'scorer')), \
             patch('src.automl_autogluon.auditar_folds', return_value=pd.DataFrame()):
            with self.assertRaises(RuntimeError):
                treinar_automl(historico(), features=['X'], ano_previsao=2026)
        caminho = Path(Predictor.call_args.kwargs['path']).parent
        self.assertFalse(caminho.exists())

    def test_ranking_sinal_tipo_e_inferencia(self):
        predictor = MagicMock(model_best='melhor')
        tabela = ranking().drop(columns=['Familia', 'LogisticBrier'])
        tabela['can_infer'] = [True, False]
        tabela['model_type'] = ['BaggedEnsembleModel'] * 2
        tabela['child_model_type'] = ['RFModel', 'LGBModel']
        predictor.leaderboard.return_value = tabela
        resultado = resumir_leaderboard(predictor)
        self.assertEqual(resultado.model.tolist(), ['melhor'])
        self.assertEqual(resultado.Familia.tolist(), ['RFModel'])
        self.assertEqual(resultado.LogisticBrier.tolist(), [.05])
        self.assertNotIn('data', predictor.leaderboard.call_args.kwargs)

    def test_predict_valida_formato_e_valores(self):
        predictor = MagicMock()
        adaptador = AdaptadorAutoGluon(predictor, ['X'], 'modelo')
        for invalida in [[np.nan], [np.inf], [1., 2.], [[1.]]]:
            predictor.predict.return_value = invalida
            with self.assertRaises(ValueError):
                adaptador.predict(pd.DataFrame({'X': [1.]}))
        predictor.predict.return_value = [8.2]
        np.testing.assert_array_equal(adaptador.predict(pd.DataFrame({'X': [1.], 'Season': [2026]})), [8.2])
        self.assertEqual(list(predictor.predict.call_args.args[0]), ['X'])

    def falso_treino(self, tabela, **config):
        self.assertTrue(tabela.Season.lt(config['ano_previsao']).all())
        self.assertTrue(tabela.Season.lt(2026).all())
        self.assertEqual(tabela.TeamA.nunique(), 1)
        self.assertNotIn('tuning_data', config)
        predictor = MagicMock()
        predictor.predict.side_effect = lambda df, **kw: df.X.to_numpy()
        return {'predictor': predictor, 'model': 'melhor', 'leaderboard': ranking(),
            'falhas': pd.DataFrame(), 'folds': pd.DataFrame({'Season': [2010], 'FoldInterno': [0]})}

    def test_temporal_combinado_separacao_e_limpeza(self):
        dados = {'M': historico(), 'W': historico(3000, 3)}
        with patch('src.automl_autogluon.treinar_automl', side_effect=self.falso_treino) as treino, \
             patch('src.automl_autogluon.limpar_execucao') as limpar:
            resultado = avaliar_temporadas_autogluon(dados, [2021, 2022], features=['X'])
        self.assertEqual(treino.call_count, 4)
        self.assertEqual(limpar.call_count, 4)
        self.assertEqual(len(resultado['previsoes']), 10)
        medias = []
        for ano, grupo in resultado['previsoes'].groupby('Season'):
            self.assertEqual(set(grupo.Categoria), {'M', 'W'})
            score = logistic_brier(grupo.Target, grupo.Pred)
            medias.append(score)
            linha = resultado['resultados'].query("Season == @ano and Categoria == 'Combinado'").iloc[0]
            self.assertEqual(linha.Score, score)
        self.assertEqual(resultado['media'], np.mean(medias))

    def test_rejeita_categorias_sobrepostas_e_futuro(self):
        with self.assertRaises(ValueError):
            avaliar_temporadas_autogluon({'M': historico(), 'W': historico()}, [2021], features=['X'])
        with self.assertRaises(ValueError):
            avaliar_temporadas_autogluon({'M': historico(), 'W': historico(3000)}, [2026], features=['X'])

    def test_limpa_erro_previsao(self):
        with patch('src.automl_autogluon.treinar_automl', side_effect=self.falso_treino), \
             patch.object(AdaptadorAutoGluon, 'predict', side_effect=ValueError('falhou')), \
             patch('src.automl_autogluon.limpar_execucao') as limpar:
            with self.assertRaises(ValueError):
                avaliar_temporadas_autogluon({'M': historico(), 'W': historico(3000)}, [2021], features=['X'])
        limpar.assert_called_once()

    def test_submission_real_ids_ordem_fallback_margens(self):
        sample = pd.DataFrame({'ID': ['2026_3000_3001', '2026_1000_1002', '2026_1000_1001'], 'Pred': [0., 0., 0.]})
        sementes = lambda base: pd.DataFrame({'Season': [2026, 2026], 'TeamID': [base, base + 1], 'Seed': [1, 2]})
        pares = preparar_confrontos(sample, sementes(1000), sementes(3000), [1000, 1001, 1002], [3000, 3001])
        pares['X'] = [1., np.nan, 2.]
        m, w = MagicMock(), MagicMock()
        m.predict.return_value, w.predict.return_value = [8.2], [-4.7]
        pred = prever_confrontos(pares, AdaptadorAutoGluon(m, ['X'], 'm'), AdaptadorAutoGluon(w, ['X'], 'w'), features=['X'])
        with TemporaryDirectory() as tmp:
            arquivo = salvar_submission(pred, sample, Path(tmp) / 'submission.csv')
            salvo = pd.read_csv(arquivo)
            self.assertTrue(salvo.ID.equals(sample.ID))
            np.testing.assert_array_equal(salvo.Pred, [-4.7, 0., 8.2])


class APIOpcionalTest(unittest.TestCase):
    def test_scorer_oficial_sinal_pickle(self):
        try:
            from src.metrics_autogluon import logistic_brier_scorer
        except ModuleNotFoundError as exc:
            if exc.name.startswith('autogluon'):
                self.skipTest('Dependência opcional AutoGluon não instalada.')
            raise
        scorer = pickle.loads(pickle.dumps(logistic_brier_scorer))
        real, pred = np.array([-10., 0., 20.]), np.array([-7., 2., 11.])
        self.assertEqual(scorer.optimum, 0)
        self.assertFalse(scorer.greater_is_better_internal)
        self.assertEqual(-scorer(real, pred), logistic_brier(real, pred))
        self.assertEqual(scorer(real, real), 0)

    def test_splitter_real_nao_divide_temporadas(self):
        try:
            folds = auditar_folds(historico(), n_folds=5, seed=42)
        except ModuleNotFoundError as exc:
            if exc.name.startswith('autogluon'):
                self.skipTest('Dependência opcional AutoGluon não instalada.')
            raise
        self.assertEqual(set(folds.FoldInterno), set(range(5)))
        self.assertTrue(folds.Season.is_unique)
        pd.testing.assert_frame_equal(folds, auditar_folds(historico(), n_folds=5, seed=42))


if __name__ == '__main__':
    unittest.main()
