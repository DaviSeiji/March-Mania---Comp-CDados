from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import exp11 as comum, exp14 as base, exp14_busca as e
from test_exp14 import dados, catalogo, fonte, recursos, REFS


@contextmanager
def modelo(treino,c,features,ano):
    assert treino.Season.lt(ano).all()
    idx=1 if c['Candidato'].endswith('_t1') else 0
    falta0=e.FEATURES[0] not in features
    falta1=e.FEATURES[1] not in features
    erro=(5. if idx==0 else 6.) if falta0 and falta1 else (
        (9. if idx==0 else 7.) if falta0 else (8. if idx==0 else 9.) if falta1 else 10.+idx)
    class Modelo:
        def predict(self,X):
            assert X.columns.tolist()==features
            return np.full(len(X),2.+erro)
    yield Modelo()


class Exp14BuscaTest(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.dados=dados(); self.catalogo=catalogo(2)
        self.fit=patch.object(comum,'modelo_ajustado',side_effect=modelo).start()
        self.addCleanup(patch.stopall)
        patch.object(comum,'medir_recursos',side_effect=recursos).start()
        self.silencio=redirect_stdout(io.StringIO());self.silencio.__enter__()
        self.addCleanup(self.silencio.__exit__,None,None,None)

    def avaliador(self,**kw):
        return e.Avaliador(self.dados,self.catalogo,REFS,pasta=self.root/'exp14'/'busca_independente',
            pasta_exp13=self.root/'exp13',**kw)

    def test_cinco_trials_sem_scores_anteriores(self):
        fonte(self.root/'exp13')
        (self.root/'exp13'/'catalogo.json').unlink()
        (self.root/'exp13'/'recente26_rankings.json').unlink()
        m=e.carregar_busca(self.root/'exp13')
        self.assertEqual(len(m['candidatos']),10)
        self.assertEqual([c['Categoria'] for c in m['candidatos']].count('M'),5)
        self.assertEqual([c['Categoria'] for c in m['candidatos']].count('W'),5)
        self.assertEqual(m['candidatos'],e.carregar_busca(self.root/'exp13')['candidatos'])
        pd.testing.assert_frame_equal(m['referencias'],REFS)
        self.fit.assert_not_called()

    def test_analyse_individual_planeja780_cortes_mais30_controles(self):
        a=self.avaliador()
        a.candidatos={c['Candidato']:c for c in catalogo(5)}
        chamadas=[]
        def planejar(ident,features,**kw):
            chamadas.append((ident,features))
            return dict(Categoria=a.candidatos[ident]['Categoria'],Configuracao=ident,Candidato=ident+str(len(features)),
                Features=features,NFeatures=len(features),status='ok',RMSE_pond=1.,LB_pond=.1)
        with patch.object(a,'avaliar',side_effect=planejar):r=e.avaliar_individuais(a)
        self.assertEqual(len(r),260)
        self.assertEqual(len([1 for _,f in chamadas if len(f)==25])*3,780)
        self.assertEqual(len([1 for _,f in chamadas if len(f)==26])*3,30)
        self.assertEqual(set(r.groupby(['Categoria','FeatureRemovida']).size()),{5})
        self.fit.assert_not_called()

    def test_pool_escolhe_modelo_diferente_por_subconjunto_temporal_fixo(self):
        a=self.avaliador()
        cheio=e.avaliar_subconjunto(a,'M',e.FEATURES)
        reduzido=e.avaliar_subconjunto(a,'M',e.FEATURES[1:])
        self.assertEqual(cheio['Melhor']['Configuracao'],'M_t0')
        self.assertEqual(reduzido['Melhor']['Configuracao'],'M_t1')
        self.assertEqual(self.fit.call_count,12)
        for chamada in self.fit.call_args_list:
            treino,c,f,ano=chamada.args
            self.assertTrue(treino.Season.lt(ano).all())
            self.assertEqual(c,next(x for x in self.catalogo if x['Candidato']==c['Candidato']))
        self.assertEqual(e.avaliar_subconjunto(a,'W',e.FEATURES,executar=False)['status'],'pendente')

    def test_progressiva_muda_vencedor_e_reusa_rodada_individual(self):
        a=self.avaliador()
        ids=['M_t0','M_t1']
        individuais=e.avaliar_individuais(a,configuracoes=ids)
        resumo=e.resumir_individuais(a,individuais)
        f=resumo.loc[resumo.FeatureRemovida.eq(e.FEATURES[0])].iloc[0]
        self.assertEqual(f.Configuracao,'M_t1'); self.assertTrue(f.MudouConfiguracao)
        n=self.fit.call_count
        primeiro=e.selecao_progressiva(a,'M',max_removidas=1)
        self.assertEqual(self.fit.call_count,n)
        self.assertEqual(primeiro['Configuracao'],'M_t1')
        self.assertEqual(primeiro['Removidas'],e.FEATURES[:1])
        completo=e.selecao_progressiva(a,'M',max_removidas=2)
        self.assertEqual(completo['Configuracao'],'M_t0')
        self.assertEqual(completo['Removidas'],e.FEATURES[:2])
        self.assertEqual(completo['NFeatures'],24)
        self.assertTrue(all(r['MudouConfiguracao'] for r in completo['Rodadas']))
        n=self.fit.call_count
        e.selecao_progressiva(self.avaliador(),'M',max_removidas=2)
        self.assertEqual(self.fit.call_count,n)

    def test_orcamento_nao_aceita_pool_ou_rodada_parcial_e_exige_individuais(self):
        a=self.avaliador(max_avaliacoes_por_configuracao=26)
        self.assertEqual(e.selecao_progressiva(a,'M')['Motivo'],'analise_individual_incompleta')
        e.avaliar_individuais(a,configuracoes=['M_t0','M_t1'])
        r=e.selecao_progressiva(a,'M',max_removidas=2)
        self.assertEqual(r['Motivo'],'orcamento_rodada_incompleta')
        self.assertEqual(r['NFeatures'],25)
        self.assertFalse(r['Rodadas'][-1]['Aceita'])
        n=self.fit.call_count
        incompleto=e.avaliar_subconjunto(a,'M',e.FEATURES[2:])
        self.assertEqual(incompleto['status'],'pendente')
        self.assertEqual(self.fit.call_count,n)

    def test_falhas_nao_concorrem_e_limites_sao_verificados(self):
        a=self.avaliador()
        self.fit.side_effect=RuntimeError('falha simulada')
        pool=e.avaliar_subconjunto(a,'M',e.FEATURES)
        self.assertEqual(pool['status'],'falha')
        self.assertIsNone(pool['Melhor'])
        self.assertEqual(e.selecao_progressiva(a,'M')['Motivo'],'controle_invalido')
        for kw in [dict(max_removidas=-1),dict(min_features=0),dict(melhora_minima=-1)]:
            with self.assertRaises(ValueError):e.selecao_progressiva(a,'M',**kw)

    def test_notebook_completo_simulado_preserva_anteriores(self):
        from contextlib import ExitStack
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fonte_path=self.root/'checkpoints'/'exp13'; fonte(fonte_path)
        (self.root/'raw_data').mkdir()
        anterior=self.root/'checkpoints'/'exp14'/'ranking.json'; comum.salvar_json(anterior,{'preservado':True})
        antes={p:p.read_bytes() for p in [*fonte_path.glob('*.json'),anterior]}
        pares=pd.DataFrame([dict(ID=f'2026_{b}_{b+i+1}',Season=2026,TeamA=b,TeamB=b+i+1,
            Categoria=nome,TemDuasSeeds=i<2,**{f:float(i) for f in e.FEATURES})
            for b,nome in [(1000,'Masculino'),(3000,'Feminino')] for i in range(3)])
        preparo=dict(jogos=self.dados,confrontos=pares,sample=pares[['ID']].assign(Pred=0.),
            auditoria=pd.DataFrame({'Categoria':['M','W']}))
        path=Path(__file__).resolve().parents[1]/'notebooks'/'features_incrementais'/'exp14_busca_independente.ipynb'
        nb=json.loads(path.read_text(encoding='utf8')); ambiente={}
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path,'cwd',return_value=self.root))
            stack.enter_context(patch.object(e,'preparar_dados',return_value=preparo))
            stack.enter_context(patch.object(comum,'detectar_tabpfn',return_value={'device':'cuda:0'}))
            stack.enter_context(patch.object(comum,'carregar_token_tabpfn'))
            stack.enter_context(patch.object(comum,'validar_acesso_tabpfn',return_value={'fixture':True}))
            stack.enter_context(patch('IPython.display.display'))
            stack.enter_context(patch.object(plt,'show',side_effect=lambda:plt.close('all')))
            for i,c in enumerate(nb['cells']):
                if c['cell_type']!='code':continue
                source=''.join(c['source']).replace('N_TRIALS=5','N_TRIALS=1').replace('MAX_FEATURES_REMOVIDAS=10','MAX_FEATURES_REMOVIDAS=1')
                source=source.replace('RODAR_ANALISE_COMPLETA=False','RODAR_ANALISE_COMPLETA=True').replace(
                    'RODAR_PROGRESSIVA=False','RODAR_PROGRESSIVA=True').replace('GERAR_SUBMISSIONS=False','GERAR_SUBMISSIONS=True')
                exec(compile(source,f'exp14_busca_celula_{i}','exec'),ambiente)
        self.assertEqual(len(ambiente['individuais']),52)
        self.assertEqual(len(ambiente['progressivas']),2)
        saidas=list((self.root/'submissions'/'exp14_busca_independente').glob('*.csv'))
        self.assertEqual(len(saidas),5)
        self.assertEqual(len(ambiente['comuns']),5)
        self.assertEqual(set(ambiente['ranking'].loc[ambiente['ranking'].Posicao.le(5),'Categoria']),{'M','W'})
        for path in saidas:
            csv=pd.read_csv(path)
            self.assertEqual(csv.ID.tolist(),pares.ID.tolist())
            self.assertTrue(np.isfinite(csv.Pred).all())
            self.assertTrue(csv.loc[~pares.TemDuasSeeds,'Pred'].eq(0).all())
        for p,b in antes.items():self.assertEqual(p.read_bytes(),b)


if __name__=='__main__':unittest.main()
