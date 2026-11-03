"""Estatísticas sazonais de eficiência calculadas exclusivamente antes do NCAA."""

import numpy as np
import pandas as pd

from .features import _participacoes_pre_torneio

ESTATISTICAS_EFICIENCIA = (
    'OffensiveRating', 'DefensiveRating', 'NetRating', 'PossessionsPerGame',
    'EffectiveFGPercentage', 'TurnoverRate', 'OffensiveReboundRate', 'FreeThrowRate',
)


def calcular_eficiencia(regulares, *, limites_daynum):
    """Totais por Season/TeamID; nunca faz média simples dos percentuais.

    Posses próprias = FGA - OR + TO + .44*FTA, calculadas dos dois lados.
    Ritmo = média por partida de (Poss próprias + Poss adversárias)/2.
    ORtg/DRtg usam, respectivamente, posses próprias/adversárias acumuladas.
    .44 é uma aproximação fixa das posses consumidas por lances livres.
    Não há ajuste de prorrogação ou força de calendário nestas oito estatísticas.
    """
    _participacoes_pre_torneio(regulares, limites_daynum)
    siglas = ['FGM', 'FGA', 'FGM3', 'FTA', 'OR', 'DR', 'TO']
    cols = [lado + s for lado in ['W', 'L'] for s in siglas]
    if not set(cols) <= set(regulares):
        raise ValueError('Faltam estatísticas detalhadas para eficiência.')
    if not all(pd.api.types.is_numeric_dtype(regulares[c]) for c in cols):
        raise ValueError('Estatísticas de eficiência devem ser numéricas.')
    valores = regulares[cols].to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError('Estatísticas ausentes, negativas ou infinitas.')
    partes = []
    for lado, outro in [('W', 'L'), ('L', 'W')]:
        tabela = regulares[['Season', f'{lado}TeamID', f'{lado}Score', f'{outro}Score',
                            *[lado + s for s in siglas]]].rename(columns={
            f'{lado}TeamID': 'TeamID', f'{lado}Score': 'PointsFor', f'{outro}Score': 'PointsAgainst',
            **{lado + s: s for s in siglas}}).copy()
        if (tabela.FGM > tabela.FGA).any() or (tabela.FGM3 > tabela.FGM).any():
            raise ValueError('Acertos inconsistentes com tentativas.')
        tabela['OppDR'] = regulares[f'{outro}DR'].to_numpy()
        tabela['Poss'] = tabela.FGA - tabela.OR + tabela.TO + .44 * tabela.FTA
        tabela['OppPoss'] = (regulares[f'{outro}FGA'] - regulares[f'{outro}OR']
                            + regulares[f'{outro}TO'] + .44 * regulares[f'{outro}FTA']).to_numpy()
        if (tabela[['Poss', 'OppPoss']] <= 0).any().any():
            raise ValueError('Posses estimadas devem ser positivas em cada jogo.')
        partes.append(tabela)
    grupos = pd.concat(partes, ignore_index=True).groupby(['Season', 'TeamID'])
    totais = grupos[['PointsFor', 'PointsAgainst', *siglas, 'OppDR', 'Poss', 'OppPoss']].sum()
    jogos = grupos.size()
    denominadores = pd.DataFrame({'FGA': totais.FGA, 'Poss': totais.Poss, 'OppPoss': totais.OppPoss,
        'TOVDen': totais.FGA + .44 * totais.FTA + totais.TO, 'OREBDen': totais.OR + totais.OppDR})
    if (denominadores <= 0).any().any():
        raise ValueError('Denominador sazonal zero/inválido; não imputamos taxas.')
    resumo = pd.DataFrame(index=totais.index)
    resumo['JogosEficiencia'] = jogos
    resumo['OffensiveRating'] = 100 * totais.PointsFor / totais.Poss
    resumo['DefensiveRating'] = 100 * totais.PointsAgainst / totais.OppPoss
    resumo['NetRating'] = resumo.OffensiveRating - resumo.DefensiveRating
    resumo['PossessionsPerGame'] = (totais.Poss + totais.OppPoss) / (2 * jogos)
    resumo['EffectiveFGPercentage'] = (totais.FGM + .5 * totais.FGM3) / totais.FGA
    resumo['TurnoverRate'] = totais.TO / denominadores.TOVDen
    resumo['OffensiveReboundRate'] = totais.OR / denominadores.OREBDen
    resumo['FreeThrowRate'] = totais.FTA / totais.FGA
    if not np.isfinite(resumo.to_numpy()).all():
        raise ValueError('Estatísticas de eficiência não finitas.')
    return resumo.reset_index()


def adicionar_eficiencia(confrontos, estatisticas, *, permitir_ausentes=False):
    """Diferenças A − B; ausentes somente em pares do sample sem duas seeds."""
    nomes = list(ESTATISTICAS_EFICIENCIA)
    stats = estatisticas[['Season', 'TeamID', *nomes]].copy()
    if stats[['Season', 'TeamID']].isna().any().any() or stats.duplicated(['Season', 'TeamID']).any():
        raise ValueError('Chaves de eficiência ausentes ou duplicadas.')
    if not all(pd.api.types.is_numeric_dtype(stats[n]) for n in nomes) or not np.isfinite(stats[nomes].to_numpy(dtype=float)).all():
        raise ValueError('Estatísticas de eficiência devem ser numéricas e finitas.')
    if (stats[[n for n in nomes if n != 'NetRating']] < 0).any().any():
        raise ValueError('Eficiência negativa fora de NetRating.')
    if (stats[['TurnoverRate', 'OffensiveReboundRate']] > 1).any().any():
        raise ValueError('Taxas fora de 0–1.')
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
            raise ValueError('Permissão de ausentes exige confrontos preparados do sample.')
        seeds = resultado[['SeedA', 'SeedB']].notna().all(axis=1)
        if not resultado.TemDuasSeeds.eq(seeds).all():
            raise ValueError('TemDuasSeeds inconsistente.')
        ausentes &= seeds
    if ausentes.any():
        raise ValueError('Participantes sem estatísticas de eficiência.')
    for nome in nomes:
        resultado[nome + 'Diff'] = resultado[nome + 'A'] - resultado[nome + 'B']
    return resultado.drop(columns=individuais)
