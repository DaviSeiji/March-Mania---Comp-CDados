from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.h2o_candidates import (
    gerar_candidatos_h2o, treinar_candidato_h2o, avaliar_candidatos_h2o,
    ranquear_candidatos, backtest_candidatos,
)
from src.season_similarity import DIMENSOES_PERFIL


def bancos(n=4):
    return {cat: gerar_candidatos_h2o(categoria=cat, n_candidatos=n, n_features=1) for cat in ['M', 'W']}


def dados():
    return {cat: pd.DataFrame([
        {'Season': ano, 'TeamA': base, 'TeamB': base + 1, 'X': float(i), 'Target': float(i + 2)}
        for ano in range(2010, 2026) for i in range(3)
    ]) for cat, base in [('M', 1000), ('W', 3000)]}


class CandidatosTest(unittest.TestCase):
    def test_200_unicos_reproduziveis_e_divisao_de_familias(self):
        for cat in ['M', 'W']:
            banco = gerar_candidatos_h2o(categoria=cat)
            self.assertEqual(banco, gerar_candidatos_h2o(categoria=cat))
            self.assertEqual(len(banco), 200)
            self.assertEqual(len({c['Candidato'] for c in banco}), 200)
            self.assertEqual(len({json.dumps([c['Familia'], c['Parametros']], sort_keys=True) for c in banco}), 200)
            self.assertEqual(pd.Series([c['Familia'] for c in banco]).value_counts().to_dict(),
                             {'GBM': 90, 'RF': 50, 'XRT': 40, 'GLM': 20})
            for c in banco:
                self.assertEqual(c['Categoria'], cat)
                self.assertNotIn('training_frame', c['Parametros'])
                self.assertNotIn('validation_frame', c['Parametros'])
        self.assertNotEqual(gerar_candidatos_h2o(categoria='M', seed=42), gerar_candidatos_h2o(categoria='M', seed=43))

    def test_mesma_config_tres_anos_mw_separados_e_limpeza(self):
        catalogos, chamadas = bancos(), []
        antes = deepcopy(catalogos)
        def fit(treino, candidato, **kw):
            self.assertTrue(treino.Season.lt(kw['ano_previsao']).all())
            self.assertEqual(treino.TeamA.nunique(), 1)
            self.assertNotIn('validation_frame', kw)
            chamadas.append((candidato['Candidato'], kw['ano_previsao'], deepcopy(candidato['Parametros'])))
            return {'leader': candidato['Candidato'], 'rmse_cv': 1.}
        with patch('src.h2o_candidates.treinar_candidato_h2o', side_effect=fit), \
             patch('src.h2o_candidates.backend.prever_leader', side_effect=lambda modelo, df, **kw: df.X.to_numpy()), \
             patch('src.h2o_candidates.backend.limpar_execucao') as limpar:
            resultado = avaliar_candidatos_h2o(dados(), catalogos, [2023, 2024, 2025], features=['X'])
        self.assertEqual(len(chamadas), 24)
        self.assertEqual(limpar.call_count, 24)
        self.assertEqual(catalogos, antes)
        for c in [c for banco in catalogos.values() for c in banco]:
            selecao = [p for ident, ano, p in chamadas if ident == c['Candidato']]
            self.assertEqual(selecao, [c['Parametros']] * 3)
        self.assertTrue(resultado['falhas'].empty)
        self.assertEqual(len(resultado['previsoes']), 24)

    def test_ranking_por_categoria_formula_sem_media_parcial(self):
        scores = pd.DataFrame([
            {'Categoria': cat, 'Candidato': f'{cat}{c}', 'Familia': 'GBM', 'Season': ano,
             'Score': valor}
            for cat in ['M', 'W'] for c, valores in [(1, [.1, .4, .4]), (2, [.2, .1, .1])]
            for ano, valor in zip([2023, 2024, 2025], valores)
        ])
        refs = pd.DataFrame({'Season': [2025, 2024, 2023], 'Peso': [.5, .3, .2]})
        ranking = ranquear_candidatos(scores, refs)
        self.assertEqual(ranking.query('Posicao == 1').Candidato.tolist(), ['M2', 'W2'])
        self.assertAlmostEqual(ranking.query("Candidato == 'M2'").iloc[0].ScorePonderado, .12)
        self.assertAlmostEqual(ranking.query("Candidato == 'M1'").iloc[0].ScorePonderado, .34)
        incompleto = scores.loc[~(scores.Candidato.eq('M2') & scores.Season.eq(2023))]
        self.assertNotIn('M2', ranquear_candidatos(incompleto, refs).Candidato.tolist())
        with self.assertRaises(ValueError):
            ranquear_candidatos(scores, refs, ano_alvo=2025)
        with self.assertRaises(ValueError):
            ranquear_candidatos(pd.concat([scores, scores]), refs)

    def test_falhas_registradas_e_limpeza(self):
        with patch('src.h2o_candidates.treinar_candidato_h2o', return_value={'leader': 'a', 'rmse_cv': 1.}), \
             patch('src.h2o_candidates.backend.prever_leader', side_effect=ValueError('previsao invalida')), \
             patch('src.h2o_candidates.backend.limpar_execucao') as limpar:
            resultado = avaliar_candidatos_h2o(dados(), bancos(1), [2023, 2024, 2025], features=['X'])
        self.assertEqual(len(resultado['falhas']), 6)
        self.assertTrue(resultado['scores'].empty)
        self.assertEqual(limpar.call_count, 6)

    def test_fit_nao_recebe_externo_e_preserva_parametros(self):
        candidato = bancos(1)['M'][0]
        antes = deepcopy(candidato)
        modelo, h2o, classe = MagicMock(), MagicMock(), MagicMock()
        classe.return_value = modelo
        modelo._model_json = {'output': {'model_category': 'Regression'}}
        modelo.rmse.return_value = 1.2
        historico = dados()['M'].query('Season < 2023')
        with patch('src.h2o_candidates.backend._h2o', return_value=h2o), \
             patch('src.h2o_candidates.backend._chaves_h2o', side_effect=[set(), {'novo'}]), \
             patch('src.h2o_candidates._estimador', return_value=classe):
            resultado = treinar_candidato_h2o(historico, candidato, features=['X'], ano_previsao=2023)
        for k, v in candidato['Parametros'].items():
            self.assertEqual(classe.call_args.kwargs[k], v)
        self.assertEqual(modelo.train.call_args.kwargs['x'], ['X'])
        self.assertEqual(modelo.train.call_args.kwargs['fold_column'], 'FoldInterno')
        self.assertNotIn('validation_frame', modelo.train.call_args.kwargs)
        self.assertEqual(list(h2o.H2OFrame.call_args.args[0]), ['X', 'Target', 'FoldInterno'])
        self.assertEqual(resultado['chaves_h2o'], ['novo'])
        self.assertEqual(candidato, antes)

    def test_nao_aceita_2026_ou_categorias_sobrepostas(self):
        with self.assertRaises(ValueError):
            avaliar_candidatos_h2o(dados(), bancos(), [2026], features=['X'])
        iguais = dados()
        iguais['W'] = iguais['M']
        with self.assertRaises(ValueError):
            avaliar_candidatos_h2o(iguais, bancos(), [2023], features=['X'])
        with self.assertRaises(ValueError):
            treinar_candidato_h2o(dados()['M'].assign(Season=2026), bancos(1)['M'][0], features=['X'], ano_previsao=2026)

    def test_backtest_seleciona_sem_olhar_scores_futuros(self):
        anos = [2010, 2011, 2021, 2022, 2023, 2024, 2025, 2026]
        perfis = pd.DataFrame({'Season': anos, **{d: np.array(anos, dtype=float) for d in DIMENSOES_PERFIL}})
        scores = pd.DataFrame([
            {'Categoria': cat, 'Candidato': f'{cat}{c}', 'Familia': 'GLM', 'Season': ano, 'Score': .1 * c}
            for cat in ['M', 'W'] for c in [1, 2, 3] for ano in range(2021, 2026)
        ])
        antes = backtest_candidatos(perfis, scores)
        scores.loc[scores.Season.eq(2025) & scores.Candidato.str.endswith('1'), 'Score'] = .9
        depois = backtest_candidatos(perfis, scores)
        self.assertEqual(antes.Candidato.tolist(), depois.Candidato.tolist())
        pd.testing.assert_frame_equal(antes.query('AnoAlvo < 2025'), depois.query('AnoAlvo < 2025'))
        self.assertTrue(all(max(r.Referencias) < r.AnoAlvo for r in depois.itertuples()))


if __name__ == '__main__':
    unittest.main()
