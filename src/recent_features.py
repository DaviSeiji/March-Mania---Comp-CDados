"""Forma recente pré-NCAA, com janelas independentes por equipe e temporada."""

import numpy as np
import pandas as pd

from .features import _participacoes_pre_torneio

ESTATISTICAS_RECENTES = (
    'RecentWinRate10', 'RecentAvgMargin10',
    'RecentAvgPointsFor10', 'RecentAvgPointsAgainst10',
)


def calcular_forma_recente(regulares, *, limites_daynum):
    """Últimos dez jogos de cada Season/TeamID; usa os disponíveis se forem menos.

    A entrada deve ser RegularSeasonCompactResults anterior ao limite exclusivo.
    Empate em DayNum é resolvido pelo OpponentID crescente, independentemente
    da ordem do CSV. Cada equipe escolhe sua própria janela de participações W/L.
    """
    participacoes = _participacoes_pre_torneio(regulares, limites_daynum)
    recentes = participacoes.sort_values(['Season', 'TeamID', 'DayNum', 'OpponentID'], kind='stable')
    recentes = recentes.groupby(['Season', 'TeamID'], sort=False).tail(10).copy()
    recentes['Margin'] = recentes.PointsFor - recentes.PointsAgainst
    resumo = recentes.groupby(['Season', 'TeamID'], as_index=False).agg(
        RecentGames10=('Vitoria', 'size'), RecentFirstDayNum10=('DayNum', 'min'),
        RecentLastDayNum10=('DayNum', 'max'), RecentWinRate10=('Vitoria', 'mean'),
        RecentAvgMargin10=('Margin', 'mean'), RecentAvgPointsFor10=('PointsFor', 'mean'),
        RecentAvgPointsAgainst10=('PointsAgainst', 'mean'))
    if not resumo.RecentGames10.between(1, 10).all() or not np.isfinite(resumo.to_numpy()).all():
        raise ValueError('Janela recente inválida.')
    np.testing.assert_allclose(resumo.RecentAvgMargin10,
                               resumo.RecentAvgPointsFor10 - resumo.RecentAvgPointsAgainst10)
    return resumo


def adicionar_forma_recente(confrontos, estatisticas, *, permitir_ausentes=False):
    """Junta estatísticas por Season/TeamID; retorna quatro diferenças A − B."""
    nomes = list(ESTATISTICAS_RECENTES)
    stats = estatisticas[['Season', 'TeamID', *nomes]].copy()
    if stats[['Season', 'TeamID']].isna().any().any() or stats.duplicated(['Season', 'TeamID']).any():
        raise ValueError('Forma recente com chaves ausentes ou duplicadas.')
    if not all(pd.api.types.is_numeric_dtype(stats[n]) for n in nomes) or not np.isfinite(stats[nomes].to_numpy(dtype=float)).all():
        raise ValueError('Forma recente deve ser numérica e finita.')
    if not stats.RecentWinRate10.between(0, 1).all() or (stats[nomes[2:]] < 0).any().any():
        raise ValueError('Taxas ou pontos recentes inválidos.')
    np.testing.assert_allclose(stats.RecentAvgMargin10, stats.RecentAvgPointsFor10 - stats.RecentAvgPointsAgainst10)
    if confrontos[['Season', 'TeamA', 'TeamB']].isna().any().any() or not confrontos.TeamA.lt(confrontos.TeamB).all():
        raise ValueError('Confrontos exigem chaves válidas e TeamA < TeamB.')
    resultado = confrontos.drop(columns=[n + s for n in nomes for s in ['A', 'B', 'Diff']], errors='ignore').copy()
    for lado in ['A', 'B']:
        resultado = resultado.merge(stats.rename(columns={'TeamID': 'Team' + lado, **{n: n + lado for n in nomes}}),
            on=['Season', 'Team' + lado], how='left', validate='many_to_one', sort=False)
    resultado.index = confrontos.index
    individuais = [n + s for n in nomes for s in ['A', 'B']]
    ausentes = resultado[individuais].isna().any(axis=1)
    if permitir_ausentes:
        if not {'ID', 'TemDuasSeeds', 'SeedA', 'SeedB'} <= set(resultado) or resultado.ID.isna().any():
            raise ValueError('Ausências explícitas exigem confrontos preparados do sample.')
        sementes = resultado[['SeedA', 'SeedB']].notna().all(axis=1)
        if not resultado.TemDuasSeeds.eq(sementes).all():
            raise ValueError('TemDuasSeeds inconsistente.')
        ausentes &= sementes
    if ausentes.any():
        raise ValueError('Participantes sem forma recente pré-torneio.')
    for nome in nomes:
        resultado[nome + 'Diff'] = resultado[nome + 'A'] - resultado[nome + 'B']
    return resultado.drop(columns=individuais)
