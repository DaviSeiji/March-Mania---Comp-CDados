import unittest

import numpy as np
import pandas as pd

from src.features import calcular_elo, adicionar_elo


class EloTest(unittest.TestCase):
    def setUp(self):
        self.jogos = pd.DataFrame({"Season": [2025], "DayNum": [10], "WTeamID": [1], "LTeamID": [2], "WScore": [80], "LScore": [70]})

    def ratings(self, jogos):
        return calcular_elo(jogos, limites_daynum={2025: 134, 2026: 134}).set_index(["Season", "TeamID"])

    def test_vitoria_derrota_e_conservacao(self):
        ratings = self.ratings(self.jogos)
        self.assertEqual(ratings.loc[(2025, 1), "Elo"], 1510)
        self.assertEqual(ratings.loc[(2025, 2), "Elo"], 1490)
        self.assertEqual(ratings.Elo.sum(), 3000)

    def test_zebra(self):
        favorito = pd.concat([self.jogos, self.jogos.assign(DayNum=20)])
        zebra = pd.concat([self.jogos, self.jogos.assign(DayNum=20, WTeamID=2, LTeamID=1)])
        ganho_esperado = self.ratings(favorito).loc[(2025, 1), "Elo"] - 1510
        ganho_zebra = self.ratings(zebra).loc[(2025, 2), "Elo"] - 1490
        self.assertGreater(ganho_zebra, ganho_esperado)
        self.assertGreater(ganho_zebra, 10)

    def test_ordem_reset_e_nao_mutacao(self):
        jogos = pd.concat([self.jogos.assign(Season=2026), self.jogos.assign(DayNum=20, WTeamID=2, LTeamID=1), self.jogos])
        original = jogos.copy(deep=True)
        ratings = self.ratings(jogos)
        pd.testing.assert_frame_equal(ratings, self.ratings(jogos.sort_values(["Season", "DayNum"])))
        pd.testing.assert_frame_equal(jogos, original)
        self.assertEqual(ratings.loc[(2026, 1), "Elo"], 1510)

    def test_sem_margem_ou_mando(self):
        pd.testing.assert_frame_equal(self.ratings(self.jogos), self.ratings(self.jogos.assign(WScore=150, LScore=10, WLoc="H")))

    def test_corte_torneio(self):
        for dia in [134, 150]:
            with self.assertRaises(ValueError):
                self.ratings(self.jogos.assign(DayNum=dia))
        with self.assertRaises(ValueError):
            calcular_elo(self.jogos, limites_daynum={})

    def test_diff_chaves_finitude(self):
        elos = self.ratings(self.jogos).reset_index()
        par = pd.DataFrame({"Season": [2025], "TeamA": [1], "TeamB": [2]}, index=[8])
        antes = par.copy(deep=True)
        saida = adicionar_elo(par, elos)
        self.assertEqual(saida.loc[8, "EloDiff"], 20)
        self.assertNotIn("EloA", saida)
        pd.testing.assert_frame_equal(par, antes)
        for invalido in [pd.concat([elos, elos]), elos.assign(Elo=np.inf)]:
            with self.assertRaises(ValueError):
                adicionar_elo(par, invalido)

    def test_fallback_somente_fora_torneio(self):
        elos = self.ratings(self.jogos).reset_index()
        par = pd.DataFrame({"Season": [2025], "TeamA": [1], "TeamB": [99]})
        with self.assertRaises(ValueError):
            adicionar_elo(par, elos, permitir_ausentes=True)
        sample = par.assign(ID="2025_1_99", SeedA=1, SeedB=np.nan, TemDuasSeeds=False)
        self.assertTrue(adicionar_elo(sample, elos, permitir_ausentes=True).EloDiff.isna().all())
        with self.assertRaises(ValueError):
            adicionar_elo(sample.assign(SeedB=2, TemDuasSeeds=True), elos, permitir_ausentes=True)


if __name__ == "__main__":
    unittest.main()
