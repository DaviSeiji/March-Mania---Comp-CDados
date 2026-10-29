import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.season_similarity import (
    DIMENSOES_PERFIL, construir_perfis_temporadas, calcular_distancias,
    ranquear_temporadas, selecionar_posicao, backtest_similaridade,
    ponderar_temporadas, selecionar_posicao_ponderada,
)
from src.automl_h2o_similarity import avaliar_top3_h2o, obter_top3_h2o
from src.metrics import logistic_brier
from src.submission import preparar_confrontos, prever_confrontos, salvar_submission


def dados_regulares():
    dados, limites = {}, {}
    for categoria, base, pontos in [('M', 1000, 80), ('W', 3000, 60)]:
        jogos = pd.DataFrame([
            {'Season': ano, 'DayNum': 20 + i, 'WTeamID': base + i,
             'LTeamID': base + i + 1, 'WScore': pontos + i, 'LScore': pontos - 10}
            for ano in [2021, 2022, 2026] for i in [0, 1]
        ])
        detalhes = jogos.copy()
        for lado in ['W', 'L']:
            for nome, valor in {'FGM': 20, 'FGA': 40, 'FGM3': 4, 'FGA3': 10,
                                'FTM': 8, 'FTA': 10, 'Ast': 12, 'OR': 8, 'DR': 20,
                                'TO': 10, 'Stl': 4, 'Blk': 3}.items():
                detalhes[lado + nome] = valor
        sementes = pd.DataFrame([
            {'Season': ano, 'TeamID': base + i, 'Seed': f'W0{i + 1}'}
            for ano in [2021, 2022, 2026] for i in [0, 1]
        ])
        dados[categoria] = {'seeds': sementes, 'temporada_regular': jogos,
            'temporada_regular_detalhada': detalhes,
            'resultados': pd.DataFrame({'Season': [2026], 'Target': [1000.]})}
        limites[categoria] = {2021: 134, 2022: 134, 2026: 134}
    return dados, limites


def perfis_sinteticos():
    anos = [2010, 2011, 2012, 2021, 2022, 2023, 2024, 2025, 2026]
    return pd.DataFrame({'Season': anos, **{nome: np.array(anos, dtype=float) - 2010 for nome in DIMENSOES_PERFIL}})


def scores_sinteticos():
    return pd.DataFrame({'Season': [2021, 2022, 2023, 2024, 2025],
        'Top1': [.2, .1, .3, .2, .2], 'Top2': [.1, .3, .2, .1, .3], 'Top3': [.3, .2, .1, .3, .1]})


class PerfisTest(unittest.TestCase):
    def test_participantes_mw_separados_desvio_e_imutabilidade(self):
        dados, limites = dados_regulares()
        antes = copy.deepcopy(dados)
        resultado = construir_perfis_temporadas(dados, limites)
        self.assertEqual(len(resultado['perfis']), 3)
        self.assertEqual(len(resultado['perfis'].columns), 53)
        self.assertTrue(resultado['perfis'].Season.is_unique)
        self.assertTrue(resultado['contagens'].Participantes.eq(2).all())
        self.assertNotIn(1002, set(resultado['equipes'].TeamID))
        self.assertNotIn(3002, set(resultado['equipes'].TeamID))
        linha = resultado['perfis'].set_index('Season').loc[2021]
        np.testing.assert_allclose(linha.M_AvgPointsFor_mean, (80 + (70 + 81)/2)/2)
        self.assertGreater(linha.M_AvgPointsFor_mean, linha.W_AvgPointsFor_mean)
        np.testing.assert_allclose(linha.M_AvgPointsFor_std, np.std([80, (70 + 81)/2], ddof=0))
        for categoria in dados:
            for chave in dados[categoria]:
                pd.testing.assert_frame_equal(dados[categoria][chave], antes[categoria][chave])

    def test_resultados_torneio_e_kaggle_nao_sao_lidos(self):
        dados, limites = dados_regulares()
        esperado = construir_perfis_temporadas(dados, limites)['perfis']
        for base in dados.values():
            base['resultados'] = object()
            base['score_kaggle'] = 'não acessar'
            base['seeds']['Seed'] = ['X05', 'X06'] * 3
        atual = construir_perfis_temporadas(dados, limites)['perfis']
        pd.testing.assert_frame_equal(esperado, atual)

    def test_rejeita_jogos_no_torneio_inclusive_2026(self):
        for chave in ['temporada_regular', 'temporada_regular_detalhada']:
            dados, limites = dados_regulares()
            dados['M'][chave].loc[dados['M'][chave].Season.eq(2026), 'DayNum'] = 134
            with self.assertRaises(ValueError):
                construir_perfis_temporadas(dados, limites)
        dados, limites = dados_regulares()
        limites['M'][2026] = 140
        with self.assertRaises(ValueError):
            construir_perfis_temporadas(dados, limites)

    def test_duplicados_ausentes_e_categorias_sobrepostas(self):
        for modo in ['duplicado', 'ausente', 'categoria']:
            dados, limites = dados_regulares()
            if modo == 'duplicado':
                dados['M']['seeds'] = pd.concat([dados['M']['seeds']] * 2)
            elif modo == 'ausente':
                dados['M']['seeds'].loc[0, 'TeamID'] = 1999
            else:
                dados['W'] = copy.deepcopy(dados['M'])
            with self.assertRaises(ValueError):
                construir_perfis_temporadas(dados, limites)


class SimilaridadeTest(unittest.TestCase):
    def test_distancia_zero_e_valores_conhecidos(self):
        for metrica, esperado in [('euclidiana', 5.), ('manhattan', 7.)]:
            np.testing.assert_array_equal(calcular_distancias([0, 0], [[0, 0], [3, 4]], metrica=metrica), [0, esperado])

    def test_scaler_passado_ranking_contribuicoes(self):
        perfis = perfis_sinteticos()
        antes = perfis.copy(deep=True)
        resultado = ranquear_temporadas(perfis, 2026, list(range(2021, 2026)))
        self.assertEqual(resultado['ranking'].Season.tolist(), [2025, 2024, 2023, 2022, 2021])
        self.assertEqual(resultado['anos_padronizacao'], perfis.loc[perfis.Season.lt(2026), 'Season'].tolist())
        esperado = perfis.loc[perfis.Season.lt(2026), list(DIMENSOES_PERFIL)].mean().to_numpy()
        np.testing.assert_allclose(resultado['normalizacao'].MediaHistorica, esperado)
        contribuicoes = resultado['contribuicoes'].groupby('Season').Contribuicao.sum()
        for linha in resultado['ranking'].itertuples():
            self.assertAlmostEqual(contribuicoes.loc[linha.Season], linha.Distancia ** 2)
        pd.testing.assert_frame_equal(perfis, antes)

    def test_futuro_nao_altera_ranking_ou_normalizacao(self):
        perfis = perfis_sinteticos()
        antes = ranquear_temporadas(perfis, 2023, [2021, 2022])
        perfis.loc[perfis.Season.gt(2023), list(DIMENSOES_PERFIL)] = 1e6
        depois = ranquear_temporadas(perfis, 2023, [2021, 2022])
        for chave in ['ranking', 'normalizacao', 'contribuicoes']:
            pd.testing.assert_frame_equal(antes[chave], depois[chave])

    def test_nao_aceita_target_kaggle_duplicados_futuro(self):
        perfis = perfis_sinteticos()
        for invalido in [perfis.assign(Target=0.), perfis.assign(Kaggle=.1), pd.concat([perfis, perfis]),
                         perfis.assign(M_WinRate_mean=np.nan)]:
            with self.assertRaises(ValueError):
                ranquear_temporadas(invalido, 2026, [2025])
        with self.assertRaises(ValueError):
            ranquear_temporadas(perfis, 2025, [2026])
        with self.assertRaises(ValueError):
            selecionar_posicao(scores_sinteticos().assign(Kaggle=.1), 2021)

    def test_empate_distancia_recente_e_score_menor_top(self):
        perfis = perfis_sinteticos()
        perfis.loc[perfis.Season.isin([2024, 2025]), list(DIMENSOES_PERFIL)] = 1
        resultado = ranquear_temporadas(perfis, 2026, [2024, 2025])
        self.assertEqual(resultado['ranking'].iloc[0].Season, 2025)
        scores = scores_sinteticos().assign(Top1=.1, Top2=.1, Top3=.1)
        self.assertEqual(selecionar_posicao(scores, 2021), 1)

    def test_top_selecionado_e_backtest_sem_futuro(self):
        scores = scores_sinteticos()
        self.assertEqual(selecionar_posicao(scores, 2021), 2)
        self.assertEqual(selecionar_posicao(scores, 2023), 3)
        resultado = backtest_similaridade(perfis_sinteticos(), scores, n_referencias=1)
        tabela = resultado['resultados']
        self.assertEqual(tabela.AnoReferencia.tolist(), [2021, 2022, 2023, 2024])
        self.assertEqual(tabela.TopEscolhido.tolist(), [2, 1, 3, 2])
        self.assertTrue(tabela.UltimoAnoPadronizacao.lt(tabela.AnoAlvo).all())
        self.assertTrue(tabela.Regret.ge(0).all())
        scores.loc[scores.Season.ge(2024), ['Top1', 'Top2', 'Top3']] = [.99, .98, 0.]
        outro = backtest_similaridade(perfis_sinteticos(), scores, n_referencias=1)['resultados']
        pd.testing.assert_frame_equal(tabela.iloc[:2], outro.iloc[:2])
        self.assertEqual(tabela.iloc[2].TopEscolhido, outro.iloc[2].TopEscolhido)
        self.assertEqual(tabela.iloc[2].TopMediaPassada, outro.iloc[2].TopMediaPassada)

    def test_pesos_inversos_tres_mais_proximas_e_imutabilidade(self):
        ranking = pd.DataFrame({'Season': [2021, 2022, 2023, 2024], 'Distancia': [4., 3., 2., 1.]})
        antes = ranking.copy(deep=True)
        referencias = ponderar_temporadas(ranking)
        self.assertEqual(referencias.Season.tolist(), [2024, 2023, 2022])
        np.testing.assert_allclose(referencias.Peso, np.array([1., 1/2, 1/3]) / (1 + 1/2 + 1/3))
        self.assertAlmostEqual(referencias.Peso.sum(), 1)
        self.assertTrue(referencias.Peso.is_monotonic_decreasing)
        pd.testing.assert_frame_equal(ranking, antes)

    def test_pesos_zero_empates_e_historico_curto(self):
        ranking = pd.DataFrame({'Season': [2021, 2022, 2023], 'Distancia': [0., 0., 2.]})
        resultado = ponderar_temporadas(ranking)
        self.assertEqual(resultado.Season.tolist(), [2022, 2021, 2023])
        np.testing.assert_allclose(resultado.Peso, [.5, .5, 0.])
        self.assertEqual(len(ponderar_temporadas(ranking.iloc[:1])), 1)
        self.assertEqual(ponderar_temporadas(ranking.iloc[:1]).Peso.iloc[0], 1.)
        for n in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                ponderar_temporadas(ranking, n_referencias=n)
        with self.assertRaises(ValueError):
            ponderar_temporadas(ranking.assign(Distancia=-1))

    def test_selecao_ponderada_difere_da_referencia_unica(self):
        scores = pd.DataFrame({'Season': [2021, 2022, 2023, 2024],
            'Top1': [.4, .4, .1, .9], 'Top2': [0., 0., .11, 0.], 'Top3': [.5, .5, .5, .9]})
        similaridade = {'ano_alvo': 2026, 'ranking': pd.DataFrame({
            'Season': [2021, 2022, 2023, 2024], 'Distancia': [3., 2., 1., 4.]})}
        resultado = selecionar_posicao_ponderada(scores, similaridade)
        self.assertEqual(selecionar_posicao(scores, 2023), 1)
        self.assertEqual(resultado['posicao'], 2)
        esperado = np.array([[.1, .11, .5], [.4, 0., .5], [.4, 0., .5]])
        np.testing.assert_allclose(resultado['scores_ponderados'].ScorePonderado, np.array([6, 3, 2]) @ esperado / 11)
        scores.loc[scores.Season.eq(2024), ['Top1', 'Top2', 'Top3']] = [0., 1., 0.]
        outro = selecionar_posicao_ponderada(scores, similaridade)
        pd.testing.assert_frame_equal(resultado['scores_ponderados'], outro['scores_ponderados'])
        with self.assertRaises(ValueError):
            selecionar_posicao_ponderada(scores.query('Season != 2023'), similaridade)
        with self.assertRaises(ValueError):
            selecionar_posicao_ponderada(scores, {**similaridade, 'ano_alvo': 2023})

    def test_backtest_ponderado_so_usa_passado_e_ate_tres_anos(self):
        scores = scores_sinteticos()
        resultado = backtest_similaridade(perfis_sinteticos(), scores)
        tabela = resultado['resultados']
        self.assertEqual(tabela.NReferencias.tolist(), [1, 2, 3, 3])
        for ano, auditoria in resultado['auditorias'].items():
            referencias = auditoria['selecao']['referencias']
            self.assertTrue(referencias.Season.lt(ano).all())
            self.assertAlmostEqual(referencias.Peso.sum(), 1)
        scores.loc[scores.Season.ge(2024), ['Top1', 'Top2', 'Top3']] = [.99, 0., .98]
        futuro = perfis_sinteticos()
        futuro.loc[futuro.Season.gt(2024), list(DIMENSOES_PERFIL)] = 1e9
        depois = backtest_similaridade(futuro, scores)['resultados']
        pd.testing.assert_frame_equal(tabela.iloc[:2], depois.iloc[:2])
        self.assertEqual(tabela.iloc[2].TopEscolhido, depois.iloc[2].TopEscolhido)
        np.testing.assert_allclose(tabela.iloc[2].Pesos, depois.iloc[2].Pesos)


class Top3H2OTest(unittest.TestCase):
    def setUp(self):
        self.dados = {
            cat: pd.DataFrame([
                {'Season': ano, 'TeamA': base, 'TeamB': base + 1, 'X': float(i), 'Target': float(i + 2)}
                for ano in [2015, 2016, 2017, 2018, 2019, 2021, 2022] for i in range(n)
            ]) for cat, base, n in [('M', 1000, 2), ('W', 3000, 3)]
        }
        self.backend = MagicMock()
        self.backend.get_model.side_effect = lambda ident: MagicMock(model_id=ident, algo='glm')

    def treino(self, tabela, **kw):
        self.assertTrue(tabela.Season.lt(kw['ano_previsao']).all())
        self.assertEqual(tabela.TeamA.nunique(), 1)
        self.assertNotIn('leaderboard_frame', kw)
        self.assertEqual(kw['max_models'], 100)
        return {'leaderboard': pd.DataFrame({'model_id': ['modelo1', 'modelo2', 'modelo3'], 'rmse': [1., 2., 3.]}),
            'folds': pd.DataFrame({'Season': [2015], 'FoldInterno': [0]})}

    def prever(self, modelo, tabela, **kw):
        self.assertEqual(kw['features'], ['X'])
        return tabela.X.to_numpy() + int(modelo.model_id[-1])

    def test_tres_posicoes_mesmas_partidas_combinacao_e_limpeza(self):
        with patch('src.automl_h2o_similarity.backend.treinar_automl', side_effect=self.treino) as treino, \
             patch('src.automl_h2o_similarity.backend._h2o', return_value=self.backend), \
             patch('src.automl_h2o_similarity.backend.prever_leader', side_effect=self.prever), \
             patch('src.automl_h2o_similarity.backend.limpar_execucao') as limpar:
            resultado = avaliar_top3_h2o(self.dados, [2021, 2022], features=['X'], max_models=100)
        self.assertEqual(treino.call_count, 4)
        self.assertEqual(limpar.call_count, 4)
        self.assertEqual(len(resultado['modelos']), 12)
        self.assertEqual(len(resultado['previsoes']), 30)
        for (ano, posicao), grupo in resultado['previsoes'].groupby(['Season', 'Posicao']):
            self.assertEqual(set(grupo.Categoria), {'M', 'W'})
            esperado = logistic_brier(grupo.Target, grupo.Pred)
            self.assertEqual(resultado['scores'].set_index('Season').loc[ano, f'Top{posicao}'], esperado)
        self.assertTrue(resultado['scores'].Top2.eq(0).all())

    def test_falta_top3_e_limpa_erro_previsao(self):
        with self.assertRaises(ValueError):
            obter_top3_h2o({'leaderboard': pd.DataFrame({'model_id': ['a'], 'rmse': [1.]})}, features=['X'])
        with patch('src.automl_h2o_similarity.backend.treinar_automl', side_effect=self.treino), \
             patch('src.automl_h2o_similarity.backend._h2o', return_value=self.backend), \
             patch('src.automl_h2o_similarity.backend.prever_leader', side_effect=RuntimeError('erro')), \
             patch('src.automl_h2o_similarity.backend.limpar_execucao') as limpar:
            with self.assertRaises(RuntimeError):
                avaliar_top3_h2o(self.dados, [2021], features=['X'], max_models=100)
        limpar.assert_called_once()

    def test_rejeita_categorias_misturadas_ou_2026(self):
        for dados in [{'M': self.dados['M'], 'W': self.dados['M']},
                      {'M': self.dados['M'].assign(Season=2026), 'W': self.dados['W']}]:
            with self.assertRaises(ValueError):
                avaliar_top3_h2o(dados, [2021], features=['X'])

    def test_submission_posicao_fixa_margens_ids_e_fallback(self):
        execucao = {'leaderboard': pd.DataFrame({'model_id': ['modelo1', 'modelo2', 'modelo3'], 'rmse': [1., 2., 3.]})}
        sample = pd.DataFrame({'ID': ['2026_3000_3001', '2026_1000_1002', '2026_1000_1001'], 'Pred': [0., 0., 0.]})
        seeds = lambda base: pd.DataFrame({'Season': [2026, 2026], 'TeamID': [base, base + 1], 'Seed': [1, 2]})
        confrontos = preparar_confrontos(sample, seeds(1000), seeds(3000), [1000, 1001, 1002], [3000, 3001])
        confrontos['X'] = [-6.7, np.nan, 6.2]
        with patch('src.automl_h2o_similarity.backend._h2o', return_value=self.backend), \
             patch('src.automl_h2o_similarity.backend.prever_leader', side_effect=self.prever):
            modelos, resumo = obter_top3_h2o(execucao, features=['X'])
            pred = prever_confrontos(confrontos, modelos[2], modelos[2], features=['X'])
        with TemporaryDirectory() as pasta:
            caminho = salvar_submission(pred, sample, Path(pasta) / 'submission.csv')
            salvo = pd.read_csv(caminho)
            self.assertTrue(salvo.ID.equals(sample.ID))
            np.testing.assert_allclose(salvo.Pred, [-4.7, 0., 8.2])
        self.assertEqual(resumo.Posicao.tolist(), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
