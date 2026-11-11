"""Exp. 14: ablação individual e backward elimination com TabPFN fixo."""

from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import exp11 as comum, exp12 as avaliacao, exp13 as origem
from .submission import salvar_submission

FEATURES = list(origem.FEATURES)
ANOS = list(comum.ANOS)
REVISAO = 1
preparar_dados = origem.preparar_dados


def _ler(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validar_features(features):
    nomes = list(features)
    if not nomes or len(set(nomes)) != len(nomes) or not set(nomes) <= set(FEATURES):
        raise ValueError('Informe um subconjunto não vazio, sem duplicatas, das 26 features.')
    return [f for f in FEATURES if f in nomes]


def identidade_avaliacao(candidato, features):
    return candidato['Candidato'] + '__f' + comum.identidade(validar_features(features))[:16]


def carregar_origem(pasta_exp13):
    """Lê o Top 3 TabPFN M/W e seus parâmetros; nunca descobre configurações."""
    pasta = Path(pasta_exp13).resolve()
    paths = [pasta/'catalogo.json', pasta/'recente26_rankings.json', pasta/'registro.json']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Falta o artefato do Exp. 13: {path}')
    catalogo, historico, registro = map(_ler, paths)
    if (registro['features'] != FEATURES or registro['anos'] != ANOS
            or registro['seed'] != 42 or registro['n_folds'] != 5):
        raise ValueError('O Exp. 13 deve usar 26 features, 2023–2025, seed 42 e cinco folds.')
    referencias = pd.DataFrame(registro['pesos'])
    pesos_catalogo = pd.DataFrame(catalogo['referencias'])
    pd.testing.assert_frame_equal(referencias.sort_values('Season').reset_index(drop=True),
        pesos_catalogo[referencias.columns].sort_values('Season').reset_index(drop=True))
    anuais = []
    for r in historico['scores']:
        if r['Abordagem'] == 'tabpfn':
            anuais.extend({**{k: r[k] for k in ['Categoria', 'Abordagem', 'Candidato']},
                'Season': ano, 'status': 'ok', 'RMSE': r[f'RMSE_{ano}'], 'LB': r[f'LB_{ano}']}
                for ano in ANOS)
    _, ranking = origem.ranquear(anuais, referencias)
    top = ranking.loc[ranking.Posicao.le(3)].copy()
    esperado = {(cat, p) for cat in ['M', 'W'] for p in [1, 2, 3]}
    if len(top) != 6 or set(map(tuple, top[['Categoria','Posicao']].to_numpy())) != esperado:
        raise ValueError('Faltam três configurações TabPFN válidas por categoria no Exp. 13.')
    publicado = pd.DataFrame(historico['ranking'])
    publicado = publicado.loc[publicado.Abordagem.eq('tabpfn') & publicado.Posicao.le(3)]
    chaves = ['Categoria', 'Candidato', 'Posicao']
    pd.testing.assert_frame_equal(top[chaves].sort_values(chaves).reset_index(drop=True),
        publicado[chaves].sort_values(chaves).reset_index(drop=True))
    mapa = {c['Candidato']: c for c in catalogo['candidatos']}
    candidatos = []
    for ident in top.Candidato:
        if ident not in mapa:
            raise ValueError(f'Faltam os parâmetros de {ident} em {paths[0]}.')
        c = deepcopy(mapa[ident])
        if c['Abordagem'] != 'tabpfn' or c['Categoria'] != top.set_index('Candidato').loc[ident,'Categoria']:
            raise ValueError('Identidade inconsistente no catálogo do Exp. 13.')
        p = avaliacao._parametros_tabpfn(c['Parametros'])
        if p.get('version') not in {'V3_5', 'V3_5_FAST'} or not p.get('device', '').startswith('cuda'):
            raise ValueError('Esperava os candidatos V3.5/3.5-Fast em CUDA do Exp. 13.')
        assinatura = origem.assinatura_candidato(c)
        if catalogo['assinaturas'].get(ident) != assinatura:
            raise ValueError(f'Parâmetros de origem alterados: {ident}.')
        candidatos.append(c)
    top['Configuracao'] = top.Candidato
    top['NFeatures'] = len(FEATURES)
    top['NRemovidas'] = 0
    top['Features'] = [list(FEATURES) for _ in range(len(top))]
    top['Removidas'] = [[] for _ in range(len(top))]
    return {'candidatos': candidatos, 'referencias': referencias, 'top13': top,
        'fontes': [str(p) for p in paths], 'versoes_exp13': registro['versoes']}


def _destino_seguro(destino, fonte):
    destino, fonte = Path(destino).resolve(), Path(fonte).resolve()
    if destino == fonte or fonte in destino.parents:
        raise ValueError('Checkpoints do Exp. 14 devem ficar fora do Exp. 13.')
    return destino


def _compativel(path, config):
    salvo = avaliacao._ler_checkpoint_compativel(path, config)
    if salvo is None:
        return None
    efetiva, esperada = deepcopy(salvo['config']), deepcopy(config)
    for cfg in [efetiva, esperada]:
        cfg['candidato']['Parametros'] = avaliacao._parametros_tabpfn(cfg['candidato']['Parametros'])
    if efetiva != esperada:
        raise ValueError(f'Checkpoint com configuração inconsistente: {path}')
    return salvo


class Avaliador:
    """Avaliações únicas configuração × features × ano, com orçamento retomável.

    O limite conta subconjuntos distintos por configuração (cada um = três cortes),
    incluindo falhas e parciais, mas excluindo o controle de 26 features. O orçamento
    não é reiniciado ao trocar de etapa ou reabrir o notebook.
    """

    def __init__(self, dados, candidatos, referencias, *, pasta, pasta_exp13,
                 max_avaliacoes_por_configuracao=100, repetir_falhas=False, verbose=False):
        comum.validar_historico(dados, FEATURES)
        if (max_avaliacoes_por_configuracao is not None and
                (isinstance(max_avaliacoes_por_configuracao, bool) or
                 not isinstance(max_avaliacoes_por_configuracao, Integral) or max_avaliacoes_por_configuracao < 0)):
            raise ValueError('Limite de avaliações deve ser inteiro não negativo ou None.')
        if len({c['Candidato'] for c in candidatos}) != len(candidatos) or not candidatos:
            raise ValueError('Catálogo vazio ou duplicado.')
        if any(c['Abordagem'] != 'tabpfn' or c['Categoria'] not in ['M','W'] for c in candidatos):
            raise ValueError('O Exp. 14 aceita somente TabPFN separado por M/W.')
        for cat in ['M','W']:
            if not set(ANOS) <= set(dados[cat].Season) or dados[cat].query('Season < 2023').empty:
                raise ValueError('Faltam dados para os três cortes históricos.')
        # Reutiliza a validação de pesos sem avaliar nenhum estimador.
        comum.rankings([dict(Categoria='M', Abordagem='tabpfn', Candidato='validar',
            Season=a, status='ok', RMSE=1., LB=.1) for a in ANOS], referencias)
        self.dados = dados
        self.candidatos = {c['Candidato']: deepcopy(c) for c in candidatos}
        self.referencias = referencias.copy()
        self.pasta_exp13 = Path(pasta_exp13).resolve()
        self.pasta = _destino_seguro(pasta, self.pasta_exp13)
        self.max_avaliacoes = max_avaliacoes_por_configuracao
        self.repetir_falhas = repetir_falhas
        self.verbose = verbose
        self.contexto = {'exp14_revisao': REVISAO, 'features': FEATURES, 'anos': ANOS, 'seed':42,
            'n_folds':5, 'candidatos': list(self.candidatos.values()), 'pesos':referencias.to_dict('records'),
            'dados': {cat:comum.assinatura_dados(t, FEATURES) for cat,t in dados.items()},
            'versoes':comum.versoes()}
        self.path_estado = self.pasta / ('estado_' + comum.identidade(self.contexto)[:16] + '.json')
        estado = _ler(self.path_estado) if self.path_estado.exists() else {'contexto':self.contexto,'avaliacoes':{}}
        if estado['contexto'] != self.contexto:
            raise ValueError('Estado do Exp. 14 incompatível.')
        self.estado = estado

    def _salvar(self):
        comum.salvar_json(self.path_estado, self.estado)

    def contagem(self):
        return pd.DataFrame([{'Categoria':c['Categoria'], 'Configuracao':ident,
            'AvaliacoesUsadas':sum(r['Configuracao']==ident and r['NFeatures']<26
                                  for r in self.estado['avaliacoes'].values()),
            'Limite':self.max_avaliacoes} for ident,c in self.candidatos.items()])

    def auditar_controle(self):
        linhas = []
        for c in self.candidatos.values():
            for ano in ANOS:
                config,path = avaliacao._checkpoint(self.dados[c['Categoria']],c,ano,FEATURES,
                    self.pasta_exp13/'recente'/'avaliacao')
                salvo = _compativel(path,config)
                linhas.append({'Categoria':c['Categoria'],'Configuracao':c['Candidato'],'Season':ano,
                    'Status':'compativel_'+salvo['status'] if salvo else 'ausente_ou_incompativel'})
        return pd.DataFrame(linhas)

    def reutilizar_controles(self):
        """Importa controles completos equivalentes, sem executar nenhum fit."""
        importados=[]
        for ident,c in self.candidatos.items():
            completos=True
            for ano in ANOS:
                config,path=avaliacao._checkpoint(self.dados[c['Categoria']],c,ano,FEATURES,self.pasta/'cortes')
                _,fonte=avaliacao._checkpoint(self.dados[c['Categoria']],c,ano,FEATURES,
                    self.pasta_exp13/'recente'/'avaliacao')
                registros=[_compativel(p,config) for p in [path,fonte]]
                completos &= any(r is not None and r['status']=='ok' for r in registros)
            if completos:
                importados.append(self.avaliar(ident,FEATURES,etapa='controle'))
        return importados

    def avaliar(self, ident, features, *, etapa):
        c = self.candidatos[ident]
        features = validar_features(features)
        chave = identidade_avaliacao(c,features)
        banco = self.estado['avaliacoes']
        anterior = banco.get(chave)
        metadados = {'Categoria':c['Categoria'],'Abordagem':'tabpfn','Candidato':chave,
            'Configuracao':ident,'Features':features,'NFeatures':len(features),
            'Removidas':[f for f in FEATURES if f not in features], 'NRemovidas':26-len(features)}
        if anterior is None and features != FEATURES:
            usados = sum(r['Configuracao']==ident and r['NFeatures']<26 for r in banco.values())
            if self.max_avaliacoes is not None and usados >= self.max_avaliacoes:
                return {**metadados,'status':'pendente_orcamento'}
        etapas = list(dict.fromkeys([*(anterior or {}).get('Etapas',[]),etapa]))
        banco[chave] = {**metadados,'status':'incompleto','Etapas':etapas}
        self._salvar()  # Reserva antes do primeiro corte; interrupções não perdem o orçamento.
        registros, reusados = [], 0
        for ano in ANOS:
            tabela = self.dados[c['Categoria']]
            config,path = avaliacao._checkpoint(tabela,c,ano,features,self.pasta/'cortes')
            local = _compativel(path,config)
            if features == FEATURES and (local is None or local['status']!='ok'):
                _,fonte = avaliacao._checkpoint(tabela,c,ano,features,self.pasta_exp13/'recente'/'avaliacao')
                salvo = _compativel(fonte,config)
                if salvo is not None:
                    nome = f'corte_{ident}_{ano}_{comum.identidade(salvo["config"])[:16]}.json'
                    comum.salvar_json(self.pasta/'cortes'/nome,salvo)
                    reusados += 1
            r = avaliacao._avaliar(tabela,c,ano,features=features,pasta=self.pasta/'cortes',
                n_folds=5,seed=42,repetir_falhas=self.repetir_falhas)
            registros.append(r)
        resumo = {**metadados,'Etapas':etapas,'status':'ok' if all(r['status']=='ok' for r in registros) else 'falha',
            'CortesReutilizadosExp13':max(reusados,(anterior or {}).get('CortesReutilizadosExp13',0)),
            'Segundos':sum(r.get('recursos',{}).get('segundos',0.) for r in registros),
            'Cortes':[{k:r.get(k) for k in ['Season','status','RMSE','LB','UltimoAnoTreino',
                'JogosTreino','JogosValidacao','SegundosTreino','SegundosInferencia','recursos','erro']} for r in registros]}
        if resumo['status']=='ok':
            score,_ = origem.ranquear(registros,self.referencias)
            resumo.update({k:v for k,v in score.iloc[0].to_dict().items()
                if k not in ['Categoria','Abordagem','Candidato']})
        banco[chave] = resumo
        self._salvar()
        if self.verbose:
            print(f'{c["Categoria"]}/{ident}: {len(features)} features — {resumo["status"]}',flush=True)
        return deepcopy(resumo)

    def obter(self, ident, features):
        chave = identidade_avaliacao(self.candidatos[ident],features)
        return deepcopy(self.estado['avaliacoes'].get(chave))

    def ranking(self):
        linhas = [deepcopy(r) for r in self.estado['avaliacoes'].values() if r['status']=='ok']
        if not linhas:
            return pd.DataFrame()
        tabela = pd.DataFrame(linhas).sort_values(['Categoria','RMSE_pond','Candidato'],kind='stable')
        tabela['Posicao'] = tabela.groupby('Categoria').cumcount()+1
        tabela['Metrica'] = 'rmse'
        controles = {i:self.obter(i,FEATURES) for i in self.candidatos}
        for metrica in ['RMSE_pond','LB_pond']:
            tabela['Delta_'+metrica] = tabela.apply(lambda r: r[metrica]-controles[r.Configuracao][metrica]
                if controles[r.Configuracao] and controles[r.Configuracao]['status']=='ok' else np.nan, axis=1)
        return tabela.reset_index(drop=True)


def avaliar_individuais(avaliador, *, features=None, configuracoes=None, piloto=False):
    """Cada remoção parte das 26 features, mantendo a configuração de referência."""
    testar = FEATURES if features is None else validar_features(features)
    ids = list(avaliador.candidatos) if configuracoes is None else list(configuracoes)
    linhas = []
    for ident in ids:
        controle = avaliador.avaliar(ident,FEATURES,etapa='controle')
        for f in testar:
            r = avaliador.avaliar(ident,[n for n in FEATURES if n!=f],etapa='piloto' if piloto else 'individual')
            valido = controle['status']=='ok' and r['status']=='ok'
            delta = r['RMSE_pond']-controle['RMSE_pond'] if valido else None
            linhas.append({'Categoria':controle['Categoria'],'Configuracao':ident,'FeatureRemovida':f,
                'Avaliacao':r['Candidato'],'NFeatures':r['NFeatures'],'Features':r['Features'],
                'status':r['status'] if controle['status']=='ok' else 'controle_invalido',
                'RMSE_controle':controle.get('RMSE_pond'),'RMSE_sem':r.get('RMSE_pond'),
                'LB_controle':controle.get('LB_pond'),'LB_sem':r.get('LB_pond'),
                'Delta_RMSE':delta,'Delta_LB':r['LB_pond']-controle['LB_pond'] if valido else None,
                'Impacto':('melhorou' if delta<0 else 'piorou' if delta>0 else 'igual') if valido else 'não avaliado'})
    tabela = pd.DataFrame(linhas).sort_values(['Categoria','Configuracao','Delta_RMSE','FeatureRemovida'],kind='stable')
    tabela['PosicaoImpacto'] = pd.Series(pd.NA,index=tabela.index,dtype='Int64')
    validos = tabela.status.eq('ok')
    tabela.loc[validos,'PosicaoImpacto'] = tabela.loc[validos].groupby(['Categoria','Configuracao']).cumcount()+1
    return tabela.reset_index(drop=True)


def selecao_progressiva(avaliador, ident, *, ativar=True, max_removidas=10,
                        min_features=5, melhora_minima=.001):
    """Backward elimination por configuração; só decide após uma rodada inteira.

    Rodada parcial por orçamento não aceita remoção; pode ser retomada aumentando
    o limite. Falhas são registradas, e apenas candidatos válidos podem ser aceitos.
    """
    for n,nome in [(max_removidas,'max_removidas'),(min_features,'min_features')]:
        if isinstance(n,bool) or not isinstance(n,Integral) or n < (1 if nome=='min_features' else 0):
            raise ValueError(f'{nome} inválido.')
    if min_features>26 or not np.isfinite(melhora_minima) or melhora_minima<0:
        raise ValueError('Mínimo de features ou melhora mínima inválidos.')
    atual = avaliador.avaliar(ident,FEATURES,etapa='controle')
    rodadas, removidas = [], []
    motivo = 'desativada' if not ativar else 'controle_invalido' if atual['status']!='ok' else None
    if motivo is None:
        faltam = [f for f in FEATURES if not avaliador.obter(ident,[n for n in FEATURES if n!=f])
                  or avaliador.obter(ident,[n for n in FEATURES if n!=f])['status'] not in ['ok','falha']]
        if faltam:
            motivo = 'analise_individual_incompleta'
    while motivo is None:
        if len(removidas)>=max_removidas:
            motivo='limite_remocoes'; break
        if atual['NFeatures']<=min_features:
            motivo='minimo_features'; break
        testes = []
        for f in atual['Features']:
            r = avaliador.avaliar(ident,[n for n in atual['Features'] if n!=f],etapa='progressiva')
            testes.append({'FeatureRemovida':f,'Avaliacao':r['Candidato'],'status':r['status'],
                'Features':r['Features'],'NFeatures':r['NFeatures'],'RMSE_pond':r.get('RMSE_pond'),
                'LB_pond':r.get('LB_pond'),'Delta_RMSE':r['RMSE_pond']-atual['RMSE_pond'] if r['status']=='ok' else None})
        rodada = {'Rodada':len(removidas)+1,'Referencia':atual['Candidato'],'FeaturesAntes':atual['Features'],
            'NFeaturesAntes':atual['NFeatures'],'RMSE_antes':atual['RMSE_pond'],'LB_antes':atual['LB_pond'],
            'Testes':testes,'Aceita':False,'Removida':None}
        rodadas.append(rodada)
        if any(t['status']=='pendente_orcamento' for t in testes):
            motivo='orcamento_rodada_incompleta'; break
        validos = sorted([t for t in testes if t['status']=='ok'],key=lambda t:(t['RMSE_pond'],t['FeatureRemovida']))
        if not validos:
            motivo='nenhuma_remocao_valida'; break
        melhor = validos[0]
        if atual['RMSE_pond']-melhor['RMSE_pond']<=melhora_minima:
            motivo='sem_melhora_minima'; break
        rodada.update(Aceita=True,Removida=melhor['FeatureRemovida'],MelhoraRMSE=-melhor['Delta_RMSE'])
        removidas.append(melhor['FeatureRemovida'])
        atual = avaliador.obter(ident,melhor['Features'])
    resultado = {'Categoria':atual['Categoria'],'Configuracao':ident,'Rodadas':rodadas,'Motivo':motivo,
        'RemovidasNaProgressiva':removidas,'NRemovidas':atual['NRemovidas'],'NFeatures':atual['NFeatures'],
        'Features':atual['Features'],'Removidas':atual['Removidas'],'AvaliacaoFinal':atual['Candidato'],
        'RMSE_pond':atual.get('RMSE_pond'),'LB_pond':atual.get('LB_pond')}
    config = {'ativar':ativar,'max_removidas':max_removidas,'min_features':min_features,
        'melhora_minima':melhora_minima,'contexto':comum.identidade(avaliador.contexto)}
    comum.salvar_json(avaliador.pasta/f'progressiva_{ident}_{comum.identidade(config)[:16]}.json',
        {'config':config,**resultado})
    return resultado


def comparar_categorias(ranking):
    """Features comuns e exclusivas nas mesmas posições Top 1/2/3 M/W."""
    linhas=[]
    for pos in [1,2,3]:
        grupo = ranking.loc[ranking.Posicao.eq(pos)]
        if set(grupo.Categoria)!={'M','W'} or len(grupo)!=2:
            continue
        m,w = (set(grupo.set_index('Categoria').loc[c,'Features']) for c in ['M','W'])
        ordenar=lambda nomes:[f for f in FEATURES if f in nomes]
        linhas.append({'Posicao':pos,'NFeaturesM':len(m),'NFeaturesW':len(w),
            'MantidasAmbas':ordenar(m&w),'RemovidasSomenteM':ordenar(w-m),
            'RemovidasSomenteW':ordenar(m-w),'RemovidasAmbas':ordenar(set(FEATURES)-m-w)})
    return pd.DataFrame(linhas)


def gerar_submissions(avaliador, ranking, confrontos, sample, *, destino):
    """Opcional: Top 1/2/3 com configuração e subconjunto próprios por M/W."""
    if ranking.empty or not ranking.Metrica.eq('rmse').all():
        raise ValueError('Exp. 14 seleciona exclusivamente por RMSE ponderado.')
    top=ranking.loc[ranking.Posicao.le(3)].copy()
    esperado={(c,p) for c in ['M','W'] for p in [1,2,3]}
    if len(top)!=6 or set(map(tuple,top[['Categoria','Posicao']].to_numpy()))!=esperado:
        raise ValueError('Faltam três combinações completas por categoria.')
    previsoes,tempos,falhas={},{},[]
    for r in top.to_dict('records'):
        c=avaliador.candidatos[r['Configuracao']]
        features=validar_features(r['Features'])
        valido=avaliador.obter(c['Candidato'],features)
        if (valido is None or valido['status']!='ok' or r['Candidato']!=valido['Candidato']
                or r['Categoria']!=c['Categoria'] or r['NFeatures']!=len(features)):
            raise ValueError('Combinação sem avaliação histórica válida.')
        nome={'M':'Masculino','W':'Feminino'}[c['Categoria']]
        mask=confrontos.Categoria.eq(nome)&confrontos.TemDuasSeeds
        pares=confrontos.loc[mask]
        treino=avaliador.dados[c['Categoria']]
        assert treino.Season.lt(2026).all()
        config={'exp14_revisao':REVISAO,'candidato':c,'features':features,'ano':2026,
            'treino':comum.assinatura_dados(treino,features),'sample':comum.assinatura_dados(pares,features,target=False),
            'seed':42,'n_folds':5,'versoes':comum.versoes()}
        path=avaliador.pasta/'final'/f'final_{r["Candidato"]}_{comum.identidade(config)[:16]}.json'
        try:
            if path.exists():
                salvo=_ler(path)
                if salvo['config']!=config: raise ValueError('Configuração final inconsistente.')
            else:
                inicio=time.perf_counter()
                with avaliacao._modelo_individual(treino,c,features,2026) as modelo:
                    pred=np.asarray(modelo.predict(pares[features]),dtype=float)
                if pred.shape!=(len(pares),) or not np.isfinite(pred).all():
                    raise ValueError('Previsões finais inválidas.')
                salvo={'config':config,'ID':pares.ID.tolist(),'Pred':pred.tolist(),'segundos':time.perf_counter()-inicio}
                comum.salvar_json(path,salvo)
            if (salvo['ID']!=pares.ID.tolist() or len(salvo['Pred'])!=len(pares)
                    or not np.isfinite(salvo['Pred']).all()):
                raise ValueError('IDs ou previsões finais inválidas.')
            previsoes[r['Candidato']]=(mask,salvo['Pred'])
            tempos[r['Candidato']]=salvo['segundos']
        except Exception as exc:
            falhas.append({'Avaliacao':r['Candidato'],'Erro':f'{type(exc).__name__}: {exc}'})
    if falhas:
        comum.salvar_json(avaliador.pasta/'final'/'falhas.json',falhas)
        raise RuntimeError('Falhas no treino final; previsões concluídas preservadas no Exp. 14.')
    saidas,hashes=[],{}
    for pos in [1,2,3]:
        grupo=top.loc[top.Posicao.eq(pos)].set_index('Categoria')
        csv=confrontos[['ID']].copy().assign(Pred=0.)
        for ident in grupo.Candidato:
            mask,pred=previsoes[ident]
            csv.loc[mask,'Pred']=pred
        assert csv.loc[~confrontos.TemDuasSeeds,'Pred'].eq(0).all()
        nome=f'submission_exp14_tabpfn_rmse_top{pos}_2026.csv'
        salvar_submission(csv,sample,Path(destino)/nome)
        digest=comum.identidade(csv.Pred.tolist())
        saidas.append({'Arquivo':nome,'Posicao':pos,'Linhas':len(csv),'IdenticaA':hashes.get(digest),
            **{f'{col}{cat}':grupo.loc[cat,col] for cat in ['M','W']
               for col in ['Candidato','Configuracao','NFeatures','Features','Removidas']}})
        hashes.setdefault(digest,nome)
    resultado={'submissions':saidas,'segundos_finais':tempos}
    comum.salvar_json(avaliador.pasta/'final'/'submissions.json',resultado)
    return resultado
