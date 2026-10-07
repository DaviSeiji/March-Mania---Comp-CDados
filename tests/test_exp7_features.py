import unittest

import numpy as np
import pandas as pd

from src.features import (
    ESTATISTICAS_DETALHADAS, calcular_estatisticas_detalhadas,
    adicionar_estatisticas_detalhadas,
)


class FeaturesDetalhadasTest(unittest.TestCase):
    def setUp(self):
        self.jogos = pd.DataFrame({
            'Season': [2024, 2024], 'DayNum': [10, 20],
            'WTeamID': [1, 2], 'LTeamID': [2, 1], 'WScore': [80, 90], 'LScore': [70, 60],
        })
        for lado in ['W', 'L']:
            for nome, valores in {
                'FGM': [10, 30], 'FGA': [20, 100], 'FGM3': [2, 6], 'FGA3': [5, 20],
                'FTM': [5, 10], 'FTA': [10, 30], 'Ast': [4, 8], 'OR': [6, 10],
                'DR': [10, 20], 'TO': [3, 7], 'Stl': [2, 4], 'Blk': [1, 5],
            }.items():
                self.jogos[lado + nome] = valores

    def calcular(self, tabela=None):
        return calcular_estatisticas_detalhadas(
            self.jogos if tabela is None else tabela, limites_daynum={2024: 134, 2025: 134},
        )

    def test_percentuais_totais_e_medias_vitoria_derrota(self):
        antes = self.jogos.copy(deep=True)
        linha = self.calcular().set_index('TeamID').loc[1]
        for nome, esperado in zip(ESTATISTICAS_DETALHADAS, [40/120, 8/25, 15/40, 6, 8, 15, 5, 3, 3]):
            self.assertAlmostEqual(linha[nome], esperado)
        self.assertEqual(linha.JogosDetalhados, 2)
        pd.testing.assert_frame_equal(self.jogos, antes)

    def test_temporadas_independentes(self):
        extra = self.jogos.iloc[:1].assign(Season=2025)
        stats = self.calcular(pd.concat([self.jogos, extra]))
        pd.testing.assert_frame_equal(stats.query('Season == 2024').reset_index(drop=True), self.calcular())
        self.assertEqual(stats.query('Season == 2025 and TeamID == 1').iloc[0].FGPercentage, 0.5)

    def test_corte_e_dados_invalidos(self):
        for invalido in [self.jogos.assign(DayNum=134), pd.concat([self.jogos, self.jogos]),
                         self.jogos.assign(WAst=np.nan), self.jogos.assign(LBlk=np.inf),
                         self.jogos.assign(WTO=-1), self.jogos.assign(WFGM=101),
                         self.jogos.assign(WFGA3=0, WFGM3=0, LFGA3=0, LFGM3=0)]:
            with self.assertRaises(ValueError):
                self.calcular(invalido)

    def test_diferencas_e_chaves(self):
        stats = self.calcular(self.jogos.assign(WAst=[7, 20]))
        par = pd.DataFrame({'Season': [2024], 'TeamA': [1], 'TeamB': [2]}, index=[8])
        antes, antes_stats = par.copy(deep=True), stats.copy(deep=True)
        resultado = adicionar_estatisticas_detalhadas(par, stats)
        lookup = stats.set_index('TeamID')
        for nome in ESTATISTICAS_DETALHADAS:
            self.assertAlmostEqual(resultado.loc[8, nome + 'Diff'], lookup.loc[1, nome] - lookup.loc[2, nome])
        pd.testing.assert_frame_equal(par, antes)
        pd.testing.assert_frame_equal(stats, antes_stats)
        for invalido in [pd.concat([stats, stats]), stats.assign(TeamID=np.nan), stats.assign(FTPercentage=2)]:
            with self.assertRaises(ValueError):
                adicionar_estatisticas_detalhadas(par, invalido)

    def test_fallback_apenas_fora_torneio(self):
        stats = self.calcular()
        par = pd.DataFrame({'Season': [2024], 'TeamA': [1], 'TeamB': [99]})
        with self.assertRaises(ValueError):
            adicionar_estatisticas_detalhadas(par, stats)
        with self.assertRaises(ValueError):
            adicionar_estatisticas_detalhadas(par, stats, permitir_ausentes=True)
        sample = par.assign(ID='2024_1_99', SeedA=1, SeedB=np.nan, TemDuasSeeds=False)
        self.assertTrue(adicionar_estatisticas_detalhadas(sample, stats, permitir_ausentes=True).FGPercentageDiff.isna().all())
        with self.assertRaises(ValueError):
            adicionar_estatisticas_detalhadas(sample.assign(SeedB=2, TemDuasSeeds=True), stats, permitir_ausentes=True)


if __name__ == '__main__':
    unittest.main()
