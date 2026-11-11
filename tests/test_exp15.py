from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import exp11 as comum, exp12, exp13, exp14, exp15 as e

REFS = pd.DataFrame({'Season':[2023,2024,2025],'Distancia':[3.,2.,1.],'Peso':[.2,.3,.5]})


def dados():
    return {cat:pd.DataFrame([dict(Season=a,TeamA=b,TeamB=b+1,Target=2.,ScoreA=72.,ScoreB=70.,
        **{f:float(i) for f in e.FEATURES}) for a in range(2010,2026) for i in range(2)])
        for cat,b in [('M',1000),('W',3000)]}


def catalogo():
    return [dict(Candidato=f'{cat}_t{i}',Categoria=cat,Abordagem='tabpfn',Familia='V3_5_FAST',
        Parametros=dict(version='V3_5_FAST',n_estimators=4,random_state=42,device='cuda:0',
            softmax_temperature=1.+i/100)) for cat in ['M','W'] for i in range(3)]


@contextmanager
def modelo(treino,c,features,ano):
    assert treino.Season.lt(ano).all()
    class Modelo:
        def predict(self,X):
            assert X.columns.tolist()==features
            return np.full(len(X),2.+len(features)+c['Parametros']['softmax_temperature']-1)
    yield Modelo()


@contextmanager
def recursos(cuda=False):
    yield {'segundos':.001,'rss_pico_mb':1.}


def fonte(pasta):
    banco=catalogo()
    rs=[dict(Categoria=c['Categoria'],Abordagem='tabpfn',Candidato=c['Candidato'],Season=a,status='ok',
        RMSE=26.+i,LB=.1+i/1000) for i,c in enumerate(banco) for a in e.ANOS]
    scores,ranking=exp13.ranquear(rs,REFS)
    comum.salvar_json(Path(pasta)/'catalogo.json',dict(candidatos=banco,referencias=REFS.to_dict('records'),
        assinaturas={c['Candidato']:exp13.assinatura_candidato(c) for c in banco}))
    comum.salvar_json(Path(pasta)/'recente26_rankings.json',dict(scores=scores.to_dict('records'),ranking=ranking.to_dict('records')))
    comum.salvar_json(Path(pasta)/'registro.json',dict(features=e.FEATURES,anos=e.ANOS,seed=42,n_folds=5,
        pesos=REFS.to_dict('records'),versoes=comum.versoes()))


class Exp15Test(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.dados=dados(); self.banco=catalogo()
        self.fit=patch.object(comum,'modelo_ajustado',side_effect=modelo).start()
        self.addCleanup(patch.stopall)
        patch.object(comum,'medir_recursos',side_effect=recursos).start()
        self.stdout=redirect_stdout(io.StringIO()); self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__,None,None,None)

    def avaliador(self,**kw):
        return e.Avaliador(self.dados,self.banco,REFS,pasta=self.root/'exp15',pasta_exp13=self.root/'exp13',**kw)

    def test_particao_explicita_corresponde_a_implementacao(self):
        self.assertTrue(e.validar_classes())
        self.assertEqual(set(e.CLASSES),set('ABCD'))
        self.assertEqual([len(c) for c in e.CLASSES.values()],[5,9,8,4])
        self.assertEqual([f for c in e.CLASSES.values() for f in c],e.FEATURES)
        for grupos in [dict(A=e.CLASSES['A']),{**deepcopy(e.CLASSES),'B':e.CLASSES['A']},
                       {**deepcopy(e.CLASSES),'D':e.CLASSES['D'][:-1]}]:
            with self.assertRaises(ValueError):e.validar_classes(grupos)

    def test_15_combinacoes_quantidades_ordem_assinaturas_e_sem_seed(self):
        planos=e.gerar_combinacoes()
        self.assertEqual([p['Combinacao'] for p in planos],['A','B','C','D','A+B','A+C','A+D',
            'B+C','B+D','C+D','A+B+C','A+B+D','A+C+D','B+C+D','A+B+C+D'])
        self.assertEqual([p['NFeatures'] for p in planos],[5,9,8,4,14,13,9,17,13,12,22,18,17,21,26])
        self.assertEqual(len({p['Assinatura'] for p in planos}),15)
        for p in planos:
            nomes={f for c in p['Classes'] for f in e.CLASSES[c]}
            self.assertEqual(p['Features'],[f for f in e.FEATURES if f in nomes])
        self.assertNotIn('SeedDif',planos[1]['Features'])
        self.assertEqual(planos,e.gerar_combinacoes())
        planos[0]['Features'].clear()
        self.assertEqual(len(e.CLASSES['A']),5)

    def test_origem_top3_fixa_sem_busca_e_sem_remover_features_individuais(self):
        fonte(self.root/'exp13')
        with patch.object(exp12,'candidatos_tabpfn',side_effect=AssertionError('nova busca')), \
             patch.object(exp14,'avaliar_individuais',side_effect=AssertionError('seleção individual proibida')):
            m=e.carregar_origem(self.root/'exp13')
            self.assertEqual(m['candidatos'],self.banco)
            a=self.avaliador()
            a.avaliar('M_t0','B')
            with self.assertRaises(ValueError):a.avaliar('M_t0',e.FEATURES[1:])

    def test_plano_completo270_cortes_mesmos_modelos_m_w_sem_leakage_e_retomada(self):
        a=self.avaliador(); antes=deepcopy(self.banco)
        r=e.avaliar_combinacoes(a)
        self.assertEqual(len(r),90)
        self.assertTrue(r.status.eq('ok').all())
        self.assertEqual(self.fit.call_count,270)
        self.assertEqual(r.groupby('Categoria').size().to_dict(),{'M':45,'W':45})
        for chamada in self.fit.call_args_list:
            treino,c,features,ano=chamada.args
            self.assertTrue(treino.Season.lt(ano).all())
            self.assertEqual(c,next(x for x in antes if x['Candidato']==c['Candidato']))
            self.assertTrue(treino.TeamA.lt(2000).all() if c['Categoria']=='M' else treino.TeamA.gt(2000).all())
            self.assertEqual(features,a._plano(features)['Features'])
        self.assertEqual(len(list((a.pasta/'cortes').glob('*.json'))),270)
        novo=self.avaliador()
        e.avaliar_combinacoes(novo)
        self.assertEqual(self.fit.call_count,270)
        self.assertEqual(antes,self.banco)
        ranking=novo.ranking()
        resumo=e.resumir_combinacoes(ranking)
        self.assertEqual(len(resumo),30)
        self.assertTrue(resumo.ConfiguracoesValidas.eq(3).all())
        self.assertEqual(resumo.groupby('Categoria').head(1).Combinacao.tolist(),['D','D'])
        adicoes=e.comparar_adicoes(ranking)
        self.assertEqual(len(adicoes),168)
        self.assertTrue(adicoes.MenorFoiMelhor.all())
        self.assertTrue((adicoes.NFeaturesDepois>adicoes.NFeaturesAntes).all())
        self.assertTrue(((adicoes.Antes=='A+B')&(adicoes.Depois=='A+B+C')).any())
        self.assertEqual(ranking.loc[ranking.Posicao.le(3)].Combinacao.tolist(),['D']*6)

    def test_piloto_reuso_26_equivalencia_estrita_e_origem_intacta(self):
        for ano in e.ANOS:
            exp12._avaliar(self.dados['M'],self.banco[0],ano,features=e.FEATURES,
                pasta=self.root/'exp13'/'recente'/'avaliacao')
        antes={p:p.read_bytes() for p in (self.root/'exp13').rglob('*.json')}
        self.fit.reset_mock()
        a=self.avaliador()
        r=e.avaliar_combinacoes(a,combinacoes=['A+B+C+D','D'],n_configuracoes=1,etapa='piloto')
        self.assertEqual(len(r),4); self.assertEqual(self.fit.call_count,9)
        self.assertEqual(r.loc[r.Configuracao.eq('M_t0')&r.Combinacao.eq('A+B+C+D'),'CortesReutilizadosExp13'].iloc[0],3)
        self.assertEqual(antes,{p:p.read_bytes() for p in (self.root/'exp13').rglob('*.json')})
        dados_novos=deepcopy(self.dados); dados_novos['M'].loc[0,e.FEATURES[0]]+=1
        b=e.Avaliador(dados_novos,self.banco,REFS,pasta=self.root/'outro',pasta_exp13=self.root/'exp13')
        self.assertTrue(b.auditar_controle().Status.eq('ausente_ou_incompativel').all())
        with patch.object(comum,'versoes',return_value={'tabpfn':'outra'}):
            self.assertNotEqual(self.avaliador().path_estado,a.path_estado)
        self.assertEqual(self.fit.call_count,9)

    def test_limite_global_persiste_entre_piloto_e_completo_e_aumentar_retoma(self):
        a=self.avaliador(max_avaliacoes=2)
        r=e.avaliar_combinacoes(a,combinacoes=['D','A'],n_configuracoes=1,etapa='piloto')
        self.assertEqual(r.status.tolist(),['ok','ok','pendente_orcamento','pendente_orcamento'])
        n=self.fit.call_count
        e.avaliar_combinacoes(self.avaliador(max_avaliacoes=2))
        self.assertEqual(self.fit.call_count,n)
        b=self.avaliador(max_avaliacoes=3)
        self.assertEqual(a.path_estado,b.path_estado)
        b.avaliar('M_t0','A')
        self.assertEqual(self.fit.call_count,n+3)
        self.assertEqual(len(b.estado['avaliacoes']),3)

    def test_retomada_interrupcao_no_segundo_corte_nao_repete_primeiro(self):
        @contextmanager
        def interromper(treino,c,features,ano):
            if ano==2024:raise KeyboardInterrupt()
            with modelo(treino,c,features,ano) as m:yield m
        self.fit.side_effect=interromper
        a=self.avaliador(max_avaliacoes=1)
        with self.assertRaises(KeyboardInterrupt):a.avaliar('M_t0','B')
        self.fit.side_effect=modelo; self.fit.reset_mock()
        r=self.avaliador(max_avaliacoes=1).avaliar('M_t0','B')
        self.assertEqual(r['status'],'ok'); self.assertEqual(self.fit.call_count,2)
        self.assertEqual(r['Classes'],['B'])

    def test_falha_nao_concorre_e_lb_nao_muda_ranking(self):
        a=self.avaliador()
        self.fit.side_effect=RuntimeError('falha simulada')
        a.avaliar('M_t0','A')
        self.assertTrue(a.ranking().empty)
        self.fit.side_effect=modelo
        a.avaliar('M_t0','D'); a.avaliar('M_t1','D')
        ranking=a.ranking()
        antes=ranking.Candidato.tolist()
        for r in a.estado['avaliacoes'].values():
            if r['status']=='ok':r['LB_pond']=.001 if r['Configuracao']=='M_t1' else .9
        self.assertEqual(a.ranking().Candidato.tolist(),antes)
        self.assertEqual(ranking.iloc[0].RMSE_pond,4.)
        self.assertEqual(ranking.iloc[0].RMSE_2023,4.)

    def test_tres_csvs_m_w_classes_diferentes_margens_formato_e_retomada(self):
        a=self.avaliador()
        for cat,grupos in [('M',['D','A','A+B']),('W',['B','C','A+B+D'])]:
            for grupo in grupos:a.avaliar(f'{cat}_t0',grupo)
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        sample=pares[['ID']].assign(Pred=0.)
        saidas=e.gerar_submissions(a,a.ranking(),pares,sample,destino=self.root/'csv')
        self.assertEqual(len(saidas['submissions']),3)
        self.assertEqual(saidas['submissions'][0]['CombinacaoM'],'D')
        self.assertEqual(saidas['submissions'][0]['CombinacaoW'],'C')
        for r in saidas['submissions']:
            self.assertTrue(r['Arquivo'].startswith('submission_exp15_'))
            csv=pd.read_csv(self.root/'csv'/r['Arquivo'])
            self.assertEqual(csv.ID.tolist(),sample.ID.tolist())
            self.assertEqual(csv.columns.tolist(),['ID','Pred'])
            self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
            self.assertTrue(csv.loc[pares.TemDuasSeeds,'Pred'].gt(1).all())
        n=self.fit.call_count
        e.gerar_submissions(a,a.ranking(),pares,sample,destino=self.root/'csv')
        self.assertEqual(self.fit.call_count,n)

    def test_rejeita_piloto_invalido_destino_anterior_e_futuro(self):
        a=self.avaliador()
        for opts in [dict(combinacoes=[]),dict(combinacoes=['A','A']),dict(combinacoes=['Z']),dict(n_configuracoes=0)]:
            with self.assertRaises(ValueError):e.avaliar_combinacoes(a,**opts)
        for limite in [-1,True,.5]:
            with self.assertRaises(ValueError):self.avaliador(max_avaliacoes=limite)
        for antigo in ['exp12','exp13','exp14']:
            with self.assertRaises(ValueError):
                e.Avaliador(self.dados,self.banco,REFS,pasta=self.root/antigo,pasta_exp13=self.root/'exp13')
        futuro=deepcopy(self.dados); futuro['M'].loc[0,'Season']=2026
        with self.assertRaises(ValueError):
            e.Avaliador(futuro,self.banco,REFS,pasta=self.root/'novo',pasta_exp13=self.root/'exp13')

    def test_busca20_estratificada_sem_vencedores_e_prefixo_estavel(self):
        banco=e.gerar_catalogo_busca(20)
        self.assertEqual(len(banco),40)
        self.assertEqual(banco,e.gerar_catalogo_busca(20))
        self.assertNotEqual(banco,e.gerar_catalogo_busca(20,seed=43))
        maior=e.gerar_catalogo_busca(30)
        for cat in ['M','W']:
            lista=[c for c in banco if c['Categoria']==cat]
            self.assertEqual(len({c['Candidato'] for c in lista}),20)
            contagem={}
            for c in lista:
                p=c['Parametros']; key=(p['version'],p['n_estimators'])
                contagem[key]=contagem.get(key,0)+1
                self.assertEqual(p['random_state'],42)
            self.assertEqual(len(contagem),10)
            self.assertEqual(set(contagem.values()),{2})
            self.assertEqual(lista,[c for c in maior if c['Categoria']==cat][:20])
        self.assertEqual([c['Parametros'] for c in banco[:20]],[c['Parametros'] for c in banco[20:]])
        fonte(self.root/'exp13')
        (self.root/'exp13'/'catalogo.json').unlink()
        (self.root/'exp13'/'recente26_rankings.json').unlink()
        with patch.object(e,'carregar_origem',side_effect=AssertionError('ranking anterior proibido')):
            m=e.carregar_busca(self.root/'exp13')
        self.assertEqual(m['candidatos'],banco)
        pd.testing.assert_frame_equal(m['referencias'],REFS)
        self.fit.assert_not_called()

    def test_plano20_por_combinacao600_candidatos1800_cortes_sem_treinar(self):
        banco=e.gerar_catalogo_busca()
        a=e.Avaliador(self.dados,banco,REFS,pasta=self.root/'novo',pasta_exp13=self.root/'exp13')
        def planejar(ident,grupo,**kw):
            return dict(Configuracao=ident,Combinacao=grupo,Categoria=a.candidatos[ident]['Categoria'],
                Anos=list(e.ANOS),Parametros=deepcopy(a.candidatos[ident]['Parametros']))
        with patch.object(a,'avaliar',side_effect=planejar):r=e.avaliar_combinacoes(a)
        self.assertEqual(len(r),600)
        self.assertEqual(sum(len(anos) for anos in r.Anos),1800)
        self.assertEqual(set(r.groupby(['Categoria','Combinacao']).size()),{20})
        self.assertEqual(r.groupby('Categoria').size().tolist(),[300,300])
        self.fit.assert_not_called()

    def test_vencedor_independente_por_features_mesmos_parametros_nos_tres_cortes(self):
        a=self.avaliador()
        @contextmanager
        def modelo_por_dados(treino,c,features,ano):
            assert treino.Season.lt(ano).all()
            numero=int(c['Candidato'][-1])
            melhor=0 if features==e.CLASSES['A'] else 1
            class Modelo:
                def predict(self,X):return np.full(len(X),2.+abs(numero-melhor)+1.)
            yield Modelo()
        self.fit.side_effect=modelo_por_dados
        e.avaliar_combinacoes(a,combinacoes=['A','D'],n_configuracoes=2)
        melhores=e.resumir_combinacoes(a.ranking())
        for cat in ['M','W']:
            grupo=melhores.loc[melhores.Categoria.eq(cat)].set_index('Combinacao')
            self.assertEqual(grupo.loc['A','Configuracao'],cat+'_t0')
            self.assertEqual(grupo.loc['D','Configuracao'],cat+'_t1')
        for chamada in self.fit.call_args_list:
            treino,c,features,ano=chamada.args
            self.assertEqual(c,next(x for x in self.banco if x['Candidato']==c['Candidato']))
            self.assertEqual(int(treino.Season.max()),ano-1)

    def test_top5_de_cada_categoria_cinco_submissions_e_cache(self):
        a=self.avaliador()
        e.avaliar_combinacoes(a,combinacoes=['D','A'],n_configuracoes=3)
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        sample=pares[['ID']].assign(Pred=0.)
        saidas=e.gerar_submissions(a,a.ranking(),pares,sample,destino=self.root/'csv5',top_k=5)
        self.assertEqual(len(saidas['submissions']),5)
        self.assertEqual([r['Posicao'] for r in saidas['submissions']],[1,2,3,4,5])
        for saida in saidas['submissions']:
            csv=pd.read_csv(self.root/'csv5'/saida['Arquivo'])
            self.assertEqual(csv.ID.tolist(),sample.ID.tolist())
            self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
            self.assertTrue(csv.loc[pares.TemDuasSeeds,'Pred'].gt(1).all())
        n=self.fit.call_count
        e.gerar_submissions(a,a.ranking(),pares,sample,destino=self.root/'csv5',top_k=5)
        self.assertEqual(self.fit.call_count,n)

    def test_notebook_busca20_top5_em_ordem_sem_modificar_resultados_anteriores(self):
        from contextlib import ExitStack
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fonte_path=self.root/'checkpoints'/'exp13'
        fonte(fonte_path)
        fonte15=self.root/'checkpoints'/'exp15'/'ranking.json'
        comum.salvar_json(fonte15,[{'anterior':'preservado'}])
        (self.root/'raw_data').mkdir()
        (self.root/'submissions').mkdir()
        anterior=self.root/'submissions'/'submission_exp15_tabpfn_rmse_top1_2026.csv'
        anterior.write_text('ID,Pred\n2026_1000_1001,7\n',encoding='utf8')
        antes={p:p.read_bytes() for p in [*fonte_path.glob('*.json'),fonte15,anterior]}
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        preparo=dict(jogos=self.dados,confrontos=pares,sample=pares[['ID']].assign(Pred=0.),
            auditoria=pd.DataFrame({'Categoria':['M','W']}))
        path=Path(__file__).resolve().parents[1]/'notebooks'/'features_incrementais'/'exp15_busca_independente.ipynb'
        nb=json.loads(path.read_text(encoding='utf8'))
        ambiente={}
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path,'cwd',return_value=self.root))
            stack.enter_context(patch.object(e,'preparar_dados',return_value=preparo))
            stack.enter_context(patch.object(e,'carregar_origem',side_effect=AssertionError('Não usar vencedores anteriores')))
            stack.enter_context(patch.object(comum,'detectar_tabpfn',return_value={'device':'cuda:0'}))
            stack.enter_context(patch.object(comum,'carregar_token_tabpfn'))
            stack.enter_context(patch.object(comum,'validar_acesso_tabpfn',return_value={'fixture':True}))
            stack.enter_context(patch('IPython.display.display'))
            stack.enter_context(patch.object(plt,'show',side_effect=lambda:plt.close('all')))
            for i,cell in enumerate(nb['cells']):
                if cell['cell_type']!='code':continue
                source=''.join(cell['source']).replace('MAX_AVALIACOES=None','MAX_AVALIACOES=10')
                source=source.replace('RODAR_COMPLETO=False','RODAR_COMPLETO=True').replace(
                    'GERAR_SUBMISSIONS=False','GERAR_SUBMISSIONS=True').replace('RODAR_PILOTO=False','RODAR_PILOTO=True')
                source=source.replace("COMBINACOES_PILOTO=['A+B+C+D','A+B','D']",
                    "COMBINACOES_PILOTO=['A+B+C+D','A+B','D','A','B']")
                exec(compile(source,f'exp15_busca_celula_{i}','exec'),ambiente)
        self.assertEqual(ambiente['N_TRIALS'],20)
        self.assertEqual(ambiente['TOP_K'],5)
        self.assertEqual(len(ambiente['completo']),600)
        self.assertEqual(ambiente['completo'].status.eq('pendente_orcamento').sum(),590)
        self.assertEqual(ambiente['ranking'].groupby('Categoria').size().tolist(),[5,5])
        self.assertEqual(len(list((self.root/'submissions'/'exp15_busca_independente').glob('*.csv'))),5)
        for p,conteudo in antes.items():self.assertEqual(p.read_bytes(),conteudo)


if __name__=='__main__':unittest.main()
