from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import exp11 as comum, exp12 as anterior, exp13 as e
from src.exp12_h2o import montar_receitas
from src.recent_features import calcular_forma_recente, adicionar_forma_recente, ESTATISTICAS_RECENTES


REFS = pd.DataFrame({'Season': [2023, 2024, 2025], 'Distancia': [3., 2., 1.], 'Peso': [.2, .3, .5]})


def partida(**kwargs):
    return dict(dict(Season=2023, DayNum=10, WTeamID=1, LTeamID=2, WScore=80, LScore=70), **kwargs)


def dados():
    return {cat: pd.DataFrame([dict(Season=ano, TeamA=base, TeamB=base+1, Target=float(i+1),
        **{f: float(i) for f in e.FEATURES}) for ano in range(2010, 2026) for i in range(3)])
        for cat, base in [('M', 1000), ('W', 3000)]}


def catalogo(n_tab=3):
    banco = []
    for cat in ['M', 'W']:
        bases = [dict(Candidato=f'{cat}_b{i}', Categoria=cat, Abordagem='h2o', Familia=familia,
            Parametros={'seed': 42, 'ntrees': 3} if familia != 'glm' else {'seed': 42, 'family': 'gaussian'},
            ModeloOrigem=f'{cat}_old{i}', RMSEInterno=float(i+1))
            for i, familia in enumerate(['glm', 'gbm', 'drf'])]
        origens = [dict(ModeloOrigem=f'StackedEnsemble_{tipo}_1', BasesOriginais=[c['ModeloOrigem'] for c in bases],
            RMSEInterno=1., MetalearnerOrigem='metalearner') for tipo in ['BestOfFamily', 'AllModels']]
        stacks = montar_receitas(bases, origens, limite_all=2)[0]
        tab = [dict(Candidato=f'{cat}_t{i}', Categoria=cat, Abordagem='tabpfn', Familia='TabPFN',
            Parametros=dict(version='V3_5_FAST' if i % 2 else 'V3_5', n_estimators=4,
                random_state=42, device='cuda:0', softmax_temperature=1.+i/1000)) for i in range(n_tab)]
        banco.extend([*bases, *stacks, *tab])
    return banco


@contextmanager
def fake_modelo(treino, candidato, features, ano):
    assert treino.Season.lt(ano).all()
    class Modelo:
        def predict(self, X):
            return X[features[0]].to_numpy() + 3.
    yield Modelo()


class FakeSessao:
    registros = []
    def __init__(self, treino, candidatos, *, features, ano, **kwargs):
        assert treino.Season.lt(ano).all()
        assert kwargs == {'n_folds': 5, 'seed': 42}
        self.treino, self.features, self.ano = treino, features, ano
        self.banco = {c['Candidato']: deepcopy(c) for c in candidatos}
        self.audit = {'Ano': ano, 'UltimoAnoTreino': int(treino.Season.max())}
        self.registros.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def adaptador(self, candidato):
        assert self.banco[candidato['Candidato']] == candidato
        if candidato['Familia'].startswith('StackedEnsemble_'):
            assert set(candidato['Parametros']['Bases']) <= self.banco.keys()
        return fake_modelo(self.treino, candidato, self.features, self.ano)


def fonte_exp12(pasta):
    pasta = Path(pasta)
    banco = catalogo(100)
    comum.salvar_json(pasta/'registro.json', dict(features=e.FEATURES_CONTROLE, anos=[2023,2024,2025],
        seed=42, n_folds=5, pesos=REFS.to_dict('records')))
    comum.salvar_json(pasta/'similaridade.json', dict(referencias=REFS.to_dict('records'), anos_padronizacao=[2010,2025]))
    for cat in ['M', 'W']:
        comum.salvar_json(pasta/'eficiencia'/'descoberta'/f'descoberta_{cat}_teste.json',
            dict(config=dict(features=e.FEATURES_CONTROLE, seed=42, folds=5, max_models=100),
                 candidatos=[c for c in banco if c['Categoria']==cat and c['Abordagem']=='h2o']))
    comum.salvar_json(pasta/'eficiencia'/'avaliacao'/'ultima_avaliacao.json', dict(avaliacoes=[
        dict(config=dict(features=e.FEATURES_CONTROLE, candidato=c)) for c in banco if c['Abordagem']=='tabpfn']))
    return banco


class RecentesTest(unittest.TestCase):
    def test_ultimos10_w_l_menos_de10_separacao_anos_sem_mutacao(self):
        raw = pd.DataFrame([partida(DayNum=d, WTeamID=1 if d%2 else 2, LTeamID=2 if d%2 else 1,
            WScore=80+d, LScore=60+d) for d in range(1,13)] + [partida(Season=2024)])
        antes = raw.copy(deep=True)
        stats = calcular_forma_recente(raw, limites_daynum={2023:134, 2024:134}).set_index(['Season','TeamID'])
        a, b = stats.loc[(2023,1)], stats.loc[(2023,2)]
        self.assertEqual((a.RecentGames10, a.RecentFirstDayNum10, a.RecentLastDayNum10), (10,3,12))
        self.assertEqual(a.RecentWinRate10, .5)
        self.assertEqual(a.RecentAvgMargin10, 0.)
        self.assertEqual(a.RecentAvgPointsFor10, np.mean([80+d if d%2 else 60+d for d in range(3,13)]))
        self.assertEqual(a.RecentAvgPointsFor10, b.RecentAvgPointsAgainst10)
        self.assertEqual(stats.loc[(2024,1),'RecentGames10'], 1)
        self.assertEqual(stats.loc[(2024,2),'RecentWinRate10'], 0)
        pd.testing.assert_frame_equal(raw, antes)

    def test_desempate_deterministico_e_rejeicao_leakage(self):
        raw = pd.DataFrame([partida(DayNum=10, LTeamID=i, WScore=80+i) for i in range(2,14)])
        a = calcular_forma_recente(raw, limites_daynum={2023:134})
        b = calcular_forma_recente(raw.sample(frac=1,random_state=42), limites_daynum={2023:134})
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.set_index('TeamID').loc[1,'RecentAvgPointsFor10'], np.mean(np.arange(4,14)+80))
        for invalido in [pd.DataFrame([partida(DayNum=134)]), pd.DataFrame([partida(DayNum=150)]),
            pd.DataFrame([partida(),partida()]), pd.DataFrame([partida(WScore=np.inf)])]:
            with self.assertRaises(ValueError): calcular_forma_recente(invalido, limites_daynum={2023:134})

    def test_diferencas_copias_duplicados_e_fallback(self):
        stats = calcular_forma_recente(pd.DataFrame([partida()]), limites_daynum={2023:134})
        pares = pd.DataFrame([dict(Season=2023,TeamA=1,TeamB=2)], index=[7])
        antes = pares.copy()
        resultado = adicionar_forma_recente(pares,stats)
        lookup = stats.set_index('TeamID')
        for n in ESTATISTICAS_RECENTES:
            self.assertEqual(resultado.loc[7,n+'Diff'], lookup.loc[1,n]-lookup.loc[2,n])
        pd.testing.assert_frame_equal(antes,pares)
        with self.assertRaises(ValueError): adicionar_forma_recente(pares,pd.concat([stats,stats]))
        with self.assertRaises(ValueError): adicionar_forma_recente(pares.assign(TeamA=2,TeamB=1),stats)
        sample = pares.assign(TeamB=3,ID='2023_1_3',SeedA=1.,SeedB=np.nan,TemDuasSeeds=False)
        self.assertTrue(adicionar_forma_recente(sample,stats,permitir_ausentes=True).RecentWinRate10Diff.isna().all())
        with self.assertRaises(ValueError):
            adicionar_forma_recente(sample.assign(SeedB=2.,TemDuasSeeds=True),stats,permitir_ausentes=True)


class Exp13Test(unittest.TestCase):
    def test_preparo_rejeita_sobreposicao_com_fonte_ncaa(self):
        raw = pd.DataFrame([partida()])
        fonte = dict(jogos={'M': pd.DataFrame({'Season':[2023]})}, limites={'M':{2023:134}},
            dados={'M':{'temporada_regular':raw,'resultados':raw.copy()}})
        with patch.object(anterior,'preparar_dados',return_value=fonte):
            with self.assertRaisesRegex(ValueError,'contém jogos NCAA'):
                e.preparar_dados('raw_data')

    def test_le_catalogo_fixo_com_receitas_sem_busca_ou_geracao_tabpfn(self):
        with TemporaryDirectory() as tmp:
            banco = fonte_exp12(tmp)
            antes = {p:p.read_bytes() for p in Path(tmp).rglob('*.json')}
            with patch.object(anterior,'candidatos_tabpfn',side_effect=AssertionError('nova geração proibida')), \
                 patch('src.exp12_h2o.descobrir_h2o',side_effect=AssertionError('nova busca proibida')):
                manifesto = e.carregar_catalogo(tmp)
            self.assertEqual({c['Candidato']:c for c in manifesto['candidatos']}, {c['Candidato']:c for c in banco})
            self.assertEqual(len(manifesto['candidatos']),210)
            pd.testing.assert_frame_equal(manifesto['referencias'],REFS)
            self.assertEqual(antes,{p:p.read_bytes() for p in Path(tmp).rglob('*.json')})
            path=Path(tmp)/'eficiencia'/'avaliacao'/'ultima_avaliacao.json'
            salvo=json.loads(path.read_text(encoding='utf8')); salvo['avaliacoes'].pop()
            comum.salvar_json(path,salvo)
            with self.assertRaisesRegex(FileNotFoundError,'99/100'): e.carregar_catalogo(tmp)

    def test_configuracoes_fixas_novos_estimadores_tres_cortes_receitas_e_retomada(self):
        banco, base = catalogo(), dados()
        antes = deepcopy(banco)
        FakeSessao.registros=[]
        with TemporaryDirectory() as tmp, patch.object(anterior,'SessaoH2O',FakeSessao), \
             patch.object(comum,'modelo_ajustado',side_effect=fake_modelo) as fit:
            resultado=e.avaliar_novo(base,banco,pasta=tmp)
            self.assertEqual(len(resultado['avaliacoes']),48)
            self.assertTrue(all(r['status']=='ok' for r in resultado['avaliacoes']))
            self.assertEqual(len(FakeSessao.registros),6)
            self.assertEqual(fit.call_count,18)
            for c in banco:
                linhas=[r for r in resultado['avaliacoes'] if r['Candidato']==c['Candidato']]
                self.assertEqual({r['Season'] for r in linhas},{2023,2024,2025})
                self.assertTrue(all(r['UltimoAnoTreino']==r['Season']-1 for r in linhas))
                self.assertTrue(all(r['config']['candidato']==c for r in linhas))
            for cat in ['M','W']:
                sessoes=[s for s in FakeSessao.registros if next(iter(s.banco.values()))['Categoria']==cat]
                self.assertEqual(len({id(s) for s in sessoes}),3)
                self.assertEqual(sessoes[0].banco,sessoes[1].banco)
                self.assertEqual(sessoes[0].banco,sessoes[2].banco)
            for chamada in fit.call_args_list:
                treino,c,features,ano=chamada.args
                self.assertTrue(treino.Season.lt(ano).all())
                self.assertEqual(c['Parametros']['n_estimators'],4)
            e.avaliar_novo(base,banco,pasta=tmp)
            self.assertEqual(fit.call_count,18)
            self.assertEqual(len(FakeSessao.registros),6)
            self.assertEqual(antes,banco)

    def test_controle_reusa_apenas_equivalentes_origem_intacta(self):
        base=dados(); c=catalogo()[0]
        with TemporaryDirectory() as tmp, patch.object(comum,'modelo_ajustado',side_effect=fake_modelo) as fit:
            origem=Path(tmp)/'exp12'; destino=Path(tmp)/'exp13'
            for ano in comum.ANOS:
                anterior._avaliar(base['M'],c,ano,features=e.FEATURES_CONTROLE,pasta=origem/'eficiencia'/'avaliacao')
            antes={p:p.read_bytes() for p in origem.rglob('*.json')}
            r=e.reutilizar_controle(base,[c],pasta_exp12=origem,destino=destino)
            self.assertTrue(r.Status.eq('reutilizado_ok').all())
            self.assertEqual(len(e.ler_controle(base,[c],pasta=destino)['avaliacoes']),3)
            self.assertEqual(fit.call_count,3)
            diferente=deepcopy(base); diferente['M'].loc[0,e.FEATURES_CONTROLE[0]]+=1
            r=e.reutilizar_controle(diferente,[c],pasta_exp12=origem,destino=destino)
            self.assertTrue(r.Status.eq('ausente_ou_nao_equivalente').all())
            with patch.object(comum,'versoes',return_value={'torch':'outra-versao'}):
                self.assertEqual(len(e.ler_controle(base,[c],pasta=destino)['pendentes']),3)
            with self.assertRaises(ValueError):
                e.reutilizar_controle(base,[c],pasta_exp12=origem,destino=origem/'controle')
            self.assertEqual(antes,{p:p.read_bytes() for p in origem.rglob('*.json')})

    def test_rmse_ponderado_unico_criterio_pareamento_e_cobertura(self):
        banco=catalogo()
        registros=[dict(Categoria=c['Categoria'],Abordagem=c['Abordagem'],Candidato=c['Candidato'],Season=ano,
            status='ok',RMSE=float(i+ano-2022),LB=1/(i+2)) for i,c in enumerate(banco) for ano in comum.ANOS]
        scores,ranking=e.ranquear(registros,REFS)
        self.assertEqual(set(ranking.Metrica),{'rmse'})
        self.assertEqual(ranking.query("Categoria=='M' and Abordagem=='h2o'").iloc[0].Candidato,banco[0]['Candidato'])
        self.assertAlmostEqual(scores.iloc[0].RMSE_pond,2.3)
        self.assertAlmostEqual(scores.iloc[0].RMSE_media,2.)
        controle=scores.iloc[:-1].copy(); controle['RMSE_pond']+=2; controle['LB_pond']+=.01
        pares,cobertura=e.comparar_pareado(controle,scores,banco)
        self.assertEqual(len(pares),len(scores)-1)
        np.testing.assert_allclose(pares.Delta_RMSE_pond,-2.)
        self.assertEqual(cobertura.ParesCompletos.sum(),len(pares))
        self.assertEqual(cobertura.Catalogo.sum(),len(banco))
        registros[0]['status']='falha'
        incompletos,_=e.ranquear(registros,REFS)
        self.assertNotIn(banco[0]['Candidato'],incompletos.Candidato.tolist())

    def test_seis_csvs_margens_brutas_ids_fallback_duplicados_e_cache(self):
        banco,base=catalogo(),dados()
        registros=[dict(Categoria=c['Categoria'],Abordagem=c['Abordagem'],Candidato=c['Candidato'],Season=ano,
            status='ok',RMSE=float(i+1),LB=.1) for i,c in enumerate(banco) for ano in comum.ANOS]
        _,ranking=e.ranquear(registros,REFS)
        # Coloca as duas receitas entre os vencedores para verificar dependências no treino final.
        for cat in ['M','W']:
            stacks=[c['Candidato'] for c in banco if c['Categoria']==cat and c['Familia'].startswith('StackedEnsemble_')]
            mask=ranking.Categoria.eq(cat)&ranking.Abordagem.eq('h2o')
            ordem=[*stacks,f'{cat}_b0',f'{cat}_b1',f'{cat}_b2']
            ranking.loc[mask,'Posicao']=ranking.loc[mask,'Candidato'].map({c:i+1 for i,c in enumerate(ordem)})
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i!=2,**{f:float(i-7) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        sample=pares[['ID']].assign(Pred=0.)
        with TemporaryDirectory() as tmp, patch.object(anterior,'SessaoH2O',FakeSessao), \
             patch.object(comum,'modelo_ajustado',side_effect=fake_modelo) as fit:
            args=dict(pasta=Path(tmp)/'final',destino=Path(tmp)/'csv')
            final=e.gerar_submissions(base,banco,ranking,pares,sample,**args)
            self.assertEqual(len(final['submissions']),6)
            self.assertEqual(sum(r['IdenticaA'] is not None for r in final['submissions']),5)
            for r in final['submissions']:
                csv=pd.read_csv(args['destino']/r['Arquivo'])
                self.assertEqual(csv.ID.tolist(),sample.ID.tolist())
                self.assertEqual(csv.columns.tolist(),['ID','Pred'])
                self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
                self.assertTrue(csv.loc[pares.TemDuasSeeds,'Pred'].lt(-1).all())
                self.assertEqual(r['ModeloM'].split('_')[0],'M')
                self.assertEqual(r['ModeloW'].split('_')[0],'W')
            n=fit.call_count
            e.gerar_submissions(base,banco,ranking,pares,sample,**args)
            self.assertEqual(fit.call_count,n)
            with self.assertRaises(ValueError):
                e.gerar_submissions(base,banco,ranking.assign(Metrica='lb'),pares,sample,**args)


if __name__=='__main__':
    unittest.main()
