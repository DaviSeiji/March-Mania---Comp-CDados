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

from src import exp11 as comum, exp12 as anterior, exp13, exp14 as e


REFS=pd.DataFrame({'Season':[2023,2024,2025],'Distancia':[3.,2.,1.],'Peso':[.2,.3,.5]})


def dados():
    return {cat:pd.DataFrame([dict(Season=a,TeamA=b,TeamB=b+1,Target=2.,ScoreA=72.,ScoreB=70.,
        **{f:float(i+1) for f in e.FEATURES}) for a in range(2010,2026) for i in range(2)])
        for cat,b in [('M',1000),('W',3000)]}


def catalogo(n=3):
    return [dict(Candidato=f'{cat}_t{i}',Categoria=cat,Abordagem='tabpfn',Familia='V3_5_FAST',
        Parametros=dict(version='V3_5_FAST',n_estimators=4,random_state=42,device='cuda:0',
            softmax_temperature=1.+i/100)) for cat in ['M','W'] for i in range(n)]


@contextmanager
def recursos(cuda=False):
    yield {'segundos':.001,'rss_pico_mb':1.,'cuda_pico_mb':1.}


@contextmanager
def modelo(treino,candidato,features,ano):
    assert treino.Season.lt(ano).all()
    margem=10. - 2.*(e.FEATURES[0] not in features) - (e.FEATURES[1] not in features)
    margem+=2.*(e.FEATURES[2] not in features)
    class Modelo:
        def predict(self,X):
            assert X.columns.tolist()==features
            return np.full(len(X),2.+margem)
    yield Modelo()


def fonte(pasta):
    pasta=Path(pasta); banco=catalogo()
    registros=[dict(Categoria=c['Categoria'],Abordagem='tabpfn',Candidato=c['Candidato'],Season=a,
        status='ok',RMSE=10.+i,LB=.1+i/1000) for i,c in enumerate(banco) for a in e.ANOS]
    scores,ranking=exp13.ranquear(registros,REFS)
    comum.salvar_json(pasta/'catalogo.json',dict(candidatos=banco,referencias=REFS.to_dict('records'),
        assinaturas={c['Candidato']:exp13.assinatura_candidato(c) for c in banco}))
    comum.salvar_json(pasta/'recente26_rankings.json',dict(scores=scores.to_dict('records'),ranking=ranking.to_dict('records')))
    comum.salvar_json(pasta/'registro.json',dict(features=e.FEATURES,anos=e.ANOS,seed=42,n_folds=5,
        pesos=REFS.to_dict('records'),versoes=comum.versoes()))
    return banco


class Exp14Test(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=dados()
        self.banco=catalogo(1)
        self.fit=patch.object(comum,'modelo_ajustado',side_effect=modelo).start()
        self.addCleanup(patch.stopall)
        patch.object(comum,'medir_recursos',side_effect=recursos).start()
        self.silencio=redirect_stdout(io.StringIO()); self.silencio.__enter__()
        self.addCleanup(self.silencio.__exit__,None,None,None)

    def avaliador(self,**kwargs):
        return e.Avaliador(self.base,self.banco,REFS,pasta=self.root/'exp14',pasta_exp13=self.root/'exp13',**kwargs)

    def test_le_top3_parametros_pesos_sem_busca_e_sem_modificar_origem(self):
        banco=fonte(self.root/'exp13')
        antes={p:p.read_bytes() for p in (self.root/'exp13').glob('*.json')}
        with patch.object(anterior,'candidatos_tabpfn',side_effect=AssertionError('nova busca proibida')):
            m=e.carregar_origem(self.root/'exp13')
        self.assertEqual(m['candidatos'],banco)
        self.assertEqual(m['top13'].NFeatures.tolist(),[26]*6)
        self.assertEqual(m['top13'].Features.tolist(),[e.FEATURES]*6)
        pd.testing.assert_frame_equal(m['referencias'],REFS)
        self.assertEqual(antes,{p:p.read_bytes() for p in (self.root/'exp13').glob('*.json')})
        self.fit.assert_not_called()
        (self.root/'exp13'/'catalogo.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'catalogo.json'):e.carregar_origem(self.root/'exp13')

    def test_rejeita_ranking_de_origem_inconsistente(self):
        fonte(self.root/'exp13')
        path=self.root/'exp13'/'recente26_rankings.json'
        r=json.loads(path.read_text(encoding='utf8')); r['ranking'][0]['Posicao']=3
        comum.salvar_json(path,r)
        with self.assertRaises(AssertionError):e.carregar_origem(self.root/'exp13')

    def test_identidade_subconjunto_ano_parametros_fixos_e_nao_mutacao(self):
        a=self.avaliador(); antes=deepcopy(self.base); original=deepcopy(self.banco)
        for features in [e.FEATURES,e.FEATURES[1:]]:
            r=a.avaliar('M_t0',features,etapa='teste')
            self.assertEqual(r['status'],'ok')
            self.assertEqual([t['Season'] for t in r['Cortes']],e.ANOS)
            self.assertEqual(r['NFeatures'],len(features))
        self.assertEqual(self.fit.call_count,6)
        for chamada in self.fit.call_args_list:
            treino,c,features,ano=chamada.args
            self.assertTrue(treino.Season.lt(ano).all())
            self.assertEqual(c,original[0])
            self.assertEqual(int(treino.Season.max()),ano-1)
        for cat in ['M','W']:pd.testing.assert_frame_equal(self.base[cat],antes[cat])
        self.assertEqual(self.banco,original)
        self.assertEqual(len(list((a.pasta/'cortes').glob('*.json'))),6)
        self.assertEqual(e.identidade_avaliacao(self.banco[0],e.FEATURES[1:]),
            e.identidade_avaliacao(self.banco[0],list(reversed(e.FEATURES[1:]))))
        a.avaliar('M_t0',e.FEATURES[1:],etapa='outra_etapa')
        self.assertEqual(self.fit.call_count,6)
        self.assertEqual(a.obter('M_t0',e.FEATURES[1:])['Etapas'],['teste','outra_etapa'])

    def test_reuso_compatibilidade_completa_e_controle_nao_gasta_orcamento(self):
        for ano in e.ANOS:
            anterior._avaliar(self.base['M'],self.banco[0],ano,features=e.FEATURES,
                pasta=self.root/'exp13'/'recente'/'avaliacao')
        antes={p:p.read_bytes() for p in (self.root/'exp13').rglob('*.json')}
        self.fit.reset_mock()
        a=self.avaliador(max_avaliacoes_por_configuracao=0)
        self.assertEqual(len(a.reutilizar_controles()),1)
        self.assertEqual(a.obter('M_t0',e.FEATURES)['CortesReutilizadosExp13'],3)
        self.fit.assert_not_called()
        self.assertTrue(a.contagem().AvaliacoesUsadas.eq(0).all())
        self.assertEqual(a.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')['status'],'pendente_orcamento')
        self.assertEqual(antes,{p:p.read_bytes() for p in (self.root/'exp13').rglob('*.json')})
        alterados=deepcopy(self.base); alterados['M'].loc[0,e.FEATURES[0]]+=1
        b=e.Avaliador(alterados,self.banco,REFS,pasta=self.root/'outro',pasta_exp13=self.root/'exp13')
        self.assertTrue(b.auditar_controle().Status.eq('ausente_ou_incompativel').all())
        self.assertEqual(b.reutilizar_controles(),[])
        with patch.object(comum,'versoes',return_value={'tabpfn':'versao_diferente'}):
            c=self.avaliador()
            self.assertTrue(c.auditar_controle().Status.eq('ausente_ou_incompativel').all())
        self.fit.assert_not_called()

    def test_piloto_reaproveitado_individuais_todos_partem_de26_e_limite_por_configuracao(self):
        a=self.avaliador(max_avaliacoes_por_configuracao=2)
        piloto=e.avaliar_individuais(a,features=e.FEATURES[:2],configuracoes=['M_t0'],piloto=True)
        self.assertEqual(self.fit.call_count,9)
        self.assertTrue(piloto.NFeatures.eq(25).all())
        completo=e.avaliar_individuais(a,configuracoes=['M_t0'])
        self.assertEqual(self.fit.call_count,9)
        self.assertEqual(completo.status.eq('pendente_orcamento').sum(),24)
        self.assertEqual(a.contagem().set_index('Configuracao').loc['M_t0','AvaliacoesUsadas'],2)
        a.avaliar('W_t0',e.FEATURES[1:],etapa='piloto')
        self.assertEqual(a.contagem().set_index('Configuracao').loc['W_t0','AvaliacoesUsadas'],1)

    def test_retomada_de_interrupcao_reutiliza_primeiro_corte_e_reserva_orcamento(self):
        original=self.fit.side_effect
        chamadas=[]
        @contextmanager
        def interromper(treino,c,features,ano):
            chamadas.append(ano)
            if ano==2024:raise KeyboardInterrupt()
            with original(treino,c,features,ano) as m:yield m
        self.fit.side_effect=interromper
        a=self.avaliador(max_avaliacoes_por_configuracao=1)
        with self.assertRaises(KeyboardInterrupt):a.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')
        self.assertEqual(len(list((a.pasta/'cortes').glob('*.json'))),1)
        self.fit.side_effect=original; self.fit.reset_mock()
        b=self.avaliador(max_avaliacoes_por_configuracao=1)
        r=b.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')
        self.assertEqual(r['status'],'ok'); self.assertEqual(self.fit.call_count,2)
        self.assertEqual(b.contagem().set_index('Configuracao').loc['M_t0','AvaliacoesUsadas'],1)

    def test_progressiva_exige_individuais_e_desativacao_nao_testa_remocoes(self):
        a=self.avaliador()
        r=e.selecao_progressiva(a,'M_t0')
        self.assertEqual(r['Motivo'],'analise_individual_incompleta')
        n=self.fit.call_count
        self.assertEqual(e.selecao_progressiva(a,'M_t0',ativar=False)['Motivo'],'desativada')
        self.assertEqual(self.fit.call_count,n)

    def test_backward_remove_melhor_uma_a_uma_limites_minimo_e_melhora(self):
        a=self.avaliador()
        individuais=e.avaliar_individuais(a,configuracoes=['M_t0'])
        self.assertEqual(individuais.iloc[0].FeatureRemovida,e.FEATURES[0])
        n=self.fit.call_count
        r=e.selecao_progressiva(a,'M_t0',max_removidas=2)
        self.assertEqual(r['RemovidasNaProgressiva'],e.FEATURES[:2])
        self.assertEqual(r['NFeatures'],24); self.assertEqual(r['Motivo'],'limite_remocoes')
        self.assertEqual(self.fit.call_count-n,25*3)
        n=self.fit.call_count
        r=e.selecao_progressiva(a,'M_t0',min_features=25)
        self.assertEqual(r['NFeatures'],25); self.assertEqual(r['Motivo'],'minimo_features')
        self.assertEqual(self.fit.call_count,n)
        r=e.selecao_progressiva(a,'M_t0',melhora_minima=3.)
        self.assertEqual(r['NFeatures'],26); self.assertEqual(r['Motivo'],'sem_melhora_minima')

    def test_orcamento_nao_aceita_rodada_parcial_retomada_com_orcamento_maior(self):
        a=self.avaliador(max_avaliacoes_por_configuracao=27)
        e.avaliar_individuais(a,configuracoes=['M_t0'])
        r=e.selecao_progressiva(a,'M_t0',max_removidas=2)
        self.assertEqual(r['Motivo'],'orcamento_rodada_incompleta'); self.assertEqual(r['NFeatures'],25)
        self.assertFalse(r['Rodadas'][-1]['Aceita'])
        b=self.avaliador(max_avaliacoes_por_configuracao=100)
        r=e.selecao_progressiva(b,'M_t0',max_removidas=2)
        self.assertEqual(r['NFeatures'],24)
        self.assertEqual(b.contagem().set_index('Configuracao').loc['M_t0','AvaliacoesUsadas'],51)

    def test_falhas_registradas_sem_ranking_e_tentar_novamente_so_se_pedido(self):
        self.fit.side_effect=RuntimeError('erro controlado')
        a=self.avaliador()
        r=a.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')
        self.assertEqual(r['status'],'falha'); self.assertTrue(a.ranking().empty)
        self.assertTrue(all('erro controlado' in c['erro'] for c in r['Cortes']))
        a.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')
        self.assertEqual(self.fit.call_count,3)
        self.fit.side_effect=modelo
        b=self.avaliador(repetir_falhas=True)
        self.assertEqual(b.avaliar('M_t0',e.FEATURES[1:],etapa='piloto')['status'],'ok')

    def test_rankings_rmse_e_top_features_por_categoria_comparacao(self):
        a=self.avaliador()
        for c in self.banco:
            a.avaliar(c['Candidato'],e.FEATURES,etapa='controle')
            a.avaliar(c['Candidato'],e.FEATURES[1:],etapa='individual')
            a.avaliar(c['Candidato'],e.FEATURES[2:],etapa='progressiva')
        ranking=a.ranking()
        self.assertEqual(ranking.groupby('Categoria').head(1).NFeatures.tolist(),[24,24])
        self.assertTrue(ranking.Metrica.eq('rmse').all())
        self.assertTrue(ranking.groupby('Categoria').head(1).Delta_RMSE_pond.eq(-3.).all())
        # Invertemos o diagnóstico LB para provar que não participa da ordenação.
        antes=ranking.Candidato.tolist()
        for i,r in enumerate(a.estado['avaliacoes'].values()):r['LB_pond']=i/100
        self.assertEqual(a.ranking().Candidato.tolist(),antes)
        comparacao=e.comparar_categorias(ranking)
        self.assertEqual(len(comparacao),3)
        self.assertEqual(comparacao.iloc[0].MantidasAmbas,e.FEATURES[2:])

    def test_tres_submissions_conjuntos_diferentes_m_w_margens_ids_e_retomada(self):
        a=self.avaliador()
        for cat,subsets in [('M',[e.FEATURES,e.FEATURES[1:],e.FEATURES[2:]]),
                            ('W',[e.FEATURES,e.FEATURES[1:],[f for f in e.FEATURES if f!=e.FEATURES[1]]])]:
            for f in subsets:a.avaliar(f'{cat}_t0',f,etapa='teste')
        ranking=a.ranking()
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        sample=pares[['ID']].assign(Pred=0.)
        r=e.gerar_submissions(a,ranking,pares,sample,destino=self.root/'csv')
        self.assertEqual(len(r['submissions']),3)
        self.assertEqual((r['submissions'][0]['NFeaturesM'],r['submissions'][0]['NFeaturesW']),(24,25))
        self.assertEqual(r['submissions'][0]['FeaturesM'],e.FEATURES[2:])
        for saida in r['submissions']:
            csv=pd.read_csv(self.root/'csv'/saida['Arquivo'])
            self.assertEqual(csv.ID.tolist(),sample.ID.tolist())
            self.assertEqual(csv.columns.tolist(),['ID','Pred'])
            self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
            self.assertTrue(csv.loc[pares.TemDuasSeeds,'Pred'].gt(1).all())
        n=self.fit.call_count
        e.gerar_submissions(a,ranking,pares,sample,destino=self.root/'csv')
        self.assertEqual(self.fit.call_count,n)
        with self.assertRaises(ValueError):e.gerar_submissions(a,ranking.assign(Metrica='lb'),pares,sample,destino=self.root/'csv')

    def test_validacoes_features_orcamento_e_destino(self):
        for f in [[],['inexistente'],[e.FEATURES[0]]*2]:
            with self.assertRaises(ValueError):e.validar_features(f)
        for limite in [-1,True,.5]:
            with self.assertRaises(ValueError):self.avaliador(max_avaliacoes_por_configuracao=limite)
        with self.assertRaises(ValueError):
            e.Avaliador(self.base,self.banco,REFS,pasta=self.root/'exp13'/'subpasta',pasta_exp13=self.root/'exp13')
        futura=deepcopy(self.base); futura['M'].loc[0,'Season']=2026
        with self.assertRaises(ValueError):
            e.Avaliador(futura,self.banco,REFS,pasta=self.root/'outro',pasta_exp13=self.root/'exp13')

    def test_notebook_em_ordem_piloto_sem_fit_real_nem_treino_final(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from contextlib import ExitStack
        fonte_path=self.root/'checkpoints'/'exp13'
        fonte(fonte_path)
        (self.root/'raw_data').mkdir()
        antes={p:p.read_bytes() for p in fonte_path.rglob('*.json')}
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        preparo=dict(jogos=self.base,confrontos=pares,sample=pares[['ID']].assign(Pred=0.),
            auditoria=pd.DataFrame({'Categoria':['M','W']}))
        path=Path(__file__).resolve().parents[1]/'notebooks'/'features_incrementais'/'exp14_selecao_individual_features.ipynb'
        notebook=json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(notebook['nbformat'],4)
        ambiente={}
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path,'cwd',return_value=self.root))
            stack.enter_context(patch.object(e,'preparar_dados',return_value=preparo))
            stack.enter_context(patch.object(comum,'detectar_tabpfn',return_value={'device':'cuda:0'}))
            stack.enter_context(patch.object(comum,'carregar_token_tabpfn'))
            stack.enter_context(patch.object(comum,'validar_acesso_tabpfn',return_value={'fixture':True}))
            mostrar=stack.enter_context(patch('IPython.display.display'))
            stack.enter_context(patch.object(plt,'show',side_effect=lambda:plt.close('all')))
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code':
                    source=''.join(cell['source'])
                    exec(compile(source,f'exp14_celula_{i}','exec'),ambiente)
            self.assertTrue(mostrar.called)
        self.assertEqual(ambiente['ranking'].groupby('Categoria').size().tolist(),[3,3])
        self.assertFalse(ambiente['GERAR_SUBMISSIONS'])
        self.assertFalse(any(c.args[3]==2026 for c in self.fit.call_args_list))
        self.assertFalse(list(self.root.rglob('*.csv')))
        self.assertEqual(antes,{p:p.read_bytes() for p in fonte_path.rglob('*.json')})


if __name__=='__main__':unittest.main()
