"""Cálculos conhecidos e proteção temporal das features do Experimento 4."""

import unittest

import numpy as np
import pandas as pd

from src.features import (
    adicionar_estatisticas, calcular_estatisticas_agregadas,
    calcular_opp_winrate, calcular_winrate,
)


class FeaturesAgregadasTest(unittest.TestCase):
    def setUp(self):
        self.jogos = pd.DataFrame({
            "Season": [2024] * 4, "DayNum": [10, 20, 30, 40],
            "WTeamID": [1, 2, 1, 2], "LTeamID": [2, 1, 3, 3],
            "WScore": [80, 90, 75, 85], "LScore": [70, 60, 65, 55],
        })
        self.limites = {2024: 134}

    def calcular(self, jogos=None):
        return calcular_estatisticas_agregadas(
            self.jogos if jogos is None else jogos, limites_daynum=self.limites,
        )

    def test_pontos_vitorias_e_derrotas(self):
        stats = self.calcular().set_index("TeamID")
        self.assertAlmostEqual(stats.loc[1, "AvgPointsFor"], 215 / 3)
        self.assertEqual(stats.loc[1, "AvgPointsAgainst"], 75)
        self.assertAlmostEqual(stats.loc[1, "WinRate"], 2 / 3)
        self.assertEqual(stats.loc[3, "WinRate"], 0)
        self.assertEqual(stats.loc[3, "Jogos"], 2)

    def test_adversarios_contam_por_partida(self):
        stats = self.calcular().set_index("TeamID")
        self.assertAlmostEqual(stats.loc[1, "OppWinRate"], 4 / 9)
        self.assertAlmostEqual(stats.loc[3, "OppWinRate"], 2 / 3)
        separado = calcular_opp_winrate(self.jogos, limites_daynum=self.limites)
        pd.testing.assert_frame_equal(
            separado, self.calcular()[["Season", "TeamID", "OppWinRate"]],
        )

    def test_separacao_temporadas(self):
        outro = pd.DataFrame({"Season": [2025], "DayNum": [10], "WTeamID": [3], "LTeamID": [1], "WScore": [100], "LScore": [10]})
        stats = calcular_estatisticas_agregadas(pd.concat([self.jogos, outro]), limites_daynum={2024: 134, 2025: 134})
        pd.testing.assert_frame_equal(stats.query("Season == 2024").reset_index(drop=True), self.calcular())
        self.assertEqual(stats.query("Season == 2025 and TeamID == 1").iloc[0].AvgPointsFor, 10)

    def test_diferencas_e_nao_mutacao(self):
        original = self.jogos.copy(deep=True)
        stats = self.calcular()
        copia = stats.copy(deep=True)
        pares = pd.DataFrame({"Season": [2024, 2024], "TeamA": [1, 1], "TeamB": [2, 3]}, index=[8, 2])
        original_pares = pares.copy(deep=True)
        resultado = adicionar_estatisticas(pares, stats)
        for nome in ["WinRate", "AvgPointsFor", "AvgPointsAgainst", "OppWinRate"]:
            taxas = stats.set_index("TeamID")[nome]
            np.testing.assert_allclose(resultado[f"{nome}Diff"], pares.TeamA.map(taxas) - pares.TeamB.map(taxas))
            self.assertNotIn(f"{nome}A", resultado)
        pd.testing.assert_frame_equal(self.jogos, original)
        pd.testing.assert_frame_equal(stats, copia)
        pd.testing.assert_frame_equal(pares, original_pares)
        self.assertTrue(resultado.index.equals(pares.index))

    def test_winrate_compatibilidade(self):
        antigo = calcular_winrate(self.jogos)
        atual = self.calcular()[antigo.columns]
        pd.testing.assert_frame_equal(antigo, atual)

    def test_limite_exclusivo_e_jogos_torneio(self):
        for dia in [134, 140]:
            contaminado = self.jogos.copy()
            contaminado.loc[0, "DayNum"] = dia
            with self.assertRaises(ValueError):
                self.calcular(contaminado)
        with self.assertRaises(ValueError):
            calcular_estatisticas_agregadas(self.jogos, limites_daynum={})

    def test_dados_invalidos(self):
        with self.assertRaises(ValueError):
            self.calcular(pd.concat([self.jogos, self.jogos.iloc[:1]]))
        for valor in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                self.calcular(self.jogos.assign(WScore=valor))

    def test_chaves_ausentes_duplicadas_e_inf(self):
        stats = self.calcular()
        pares = pd.DataFrame({"Season": [2024], "TeamA": [1], "TeamB": [2]})
        for invalido in [pd.concat([stats, stats.iloc[:1]]), stats.assign(TeamID=np.nan), stats.assign(OppWinRate=np.inf)]:
            with self.assertRaises(ValueError):
                adicionar_estatisticas(pares, invalido)
        with self.assertRaises(ValueError):
            adicionar_estatisticas(pares.assign(TeamA=3), stats)

    def test_ausentes_somente_no_sample_sem_duas_seeds(self):
        stats = self.calcular()
        par = pd.DataFrame({"Season": [2024], "TeamA": [1], "TeamB": [99]})
        with self.assertRaises(ValueError):
            adicionar_estatisticas(par, stats)
        with self.assertRaises(ValueError):
            adicionar_estatisticas(par, stats, permitir_ausentes=True)
        sample = par.assign(ID="2024_1_99", SeedA=1, SeedB=np.nan, TemDuasSeeds=False)
        resultado = adicionar_estatisticas(sample, stats, permitir_ausentes=True)
        self.assertTrue(resultado.WinRateDiff.isna().all())
        with self.assertRaises(ValueError):
            adicionar_estatisticas(sample.assign(SeedB=2, TemDuasSeeds=True), stats, permitir_ausentes=True)


if __name__ == "__main__":
    unittest.main()
