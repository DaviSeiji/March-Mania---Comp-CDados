"""Exp. 14: cinco trials por subconjunto e eliminação conjunta de features/modelo."""

from copy import deepcopy
from numbers import Integral

import pandas as pd
import numpy as np

from . import exp11 as comum, exp14 as base, exp15

FEATURES = list(base.FEATURES)
ANOS = list(base.ANOS)
Avaliador = base.Avaliador
preparar_dados = base.preparar_dados
avaliar_individuais = base.avaliar_individuais


def comparar_categorias(ranking, *, top_k=5):
    """Compara quantidades e listas de features do Top 5 M/W."""
    return base.comparar_categorias(ranking,top_k=top_k)


def gerar_submissions(avaliador, ranking, confrontos, sample, *, destino, top_k=5):
    """Gera cinco CSVs, pareando as mesmas posições de M/W."""
    return base.gerar_submissions(avaliador,ranking,confrontos,sample,destino=destino,top_k=top_k)


def carregar_busca(pasta_exp13, *, n_trials=5, seed=42):
    """Reutiliza o sorteio sem scores do Exp. 15, sem consultar seus vencedores."""
    return exp15.carregar_busca(pasta_exp13,n_trials=n_trials,seed=seed)


def avaliar_subconjunto(avaliador, categoria, features, *, executar=True, etapa='progressiva_conjunta'):
    """Avalia os cinco trials do pool; não escolhe um pool parcialmente executado."""
    nomes = base.validar_features(features)
    if categoria not in ['M','W']:
        raise ValueError('Use M ou W.')
    configs = [i for i,c in avaliador.candidatos.items() if c['Categoria']==categoria]
    if not configs:
        raise ValueError(f'Nenhum trial para {categoria}.')
    trials=[]
    for ident in configs:
        r = (avaliador.avaliar(ident,nomes,etapa=etapa) if executar else avaliador.obter(ident,nomes))
        trials.append(r or {'Configuracao':ident,'status':'ausente'})
    pendente = any(r['status'] not in ['ok','falha'] for r in trials)
    validos = sorted([r for r in trials if r['status']=='ok'],key=lambda r:(r['RMSE_pond'],r['Candidato']))
    return {'Categoria':categoria,'Features':nomes,'NFeatures':len(nomes),'NTrials':len(configs),
        'TrialsValidos':len(validos),'status':'pendente' if pendente else 'ok' if validos else 'falha',
        'Melhor':deepcopy(validos[0]) if validos else None,
        'Trials':[{k:r.get(k) for k in ['Configuracao','Candidato','status','RMSE_pond','LB_pond']} for r in trials]}


def resumir_individuais(avaliador, tabela):
    """Melhor configuração por remoção vs. melhor controle; relatório fixo também permanece."""
    linhas=[]
    for (cat,feature),_ in tabela.groupby(['Categoria','FeatureRemovida'],sort=False):
        antes=avaliar_subconjunto(avaliador,cat,FEATURES,executar=False)
        depois=avaliar_subconjunto(avaliador,cat,[f for f in FEATURES if f!=feature],executar=False)
        valido=antes['status']=='ok' and depois['status']=='ok'
        a,b=antes['Melhor'],depois['Melhor']
        linhas.append({'Categoria':cat,'FeatureRemovida':feature,'NFeatures':25,
            'status':depois['status'] if antes['status']=='ok' else 'controle_incompleto',
            'TrialsValidos':depois['TrialsValidos'],'TrialsPrevistos':depois['NTrials'],
            'Configuracao':b['Configuracao'] if b else None,'ConfiguracaoControle':a['Configuracao'] if a else None,
            'RMSE_controle':a['RMSE_pond'] if a else None,'RMSE_sem':b['RMSE_pond'] if b else None,
            'Delta_RMSE':b['RMSE_pond']-a['RMSE_pond'] if valido else None,
            'LB_controle':a['LB_pond'] if a else None,'LB_sem':b['LB_pond'] if b else None,
            'MudouConfiguracao':a['Configuracao']!=b['Configuracao'] if valido else None})
    return pd.DataFrame(linhas).sort_values(['Categoria','Delta_RMSE','FeatureRemovida'],kind='stable').reset_index(drop=True)


def selecao_progressiva(avaliador, categoria, *, ativar=True, max_removidas=10,
                        min_features=5, melhora_minima=.001):
    """Uma trajetória por categoria: cada remoção concorre com os cinco trials.

    Hiperparâmetros ficam fixos nos três anos de cada trial. O vencedor pode mudar
    entre subconjuntos. Rodadas ou pools incompletos por orçamento não são aceitos.
    """
    for n,nome,piso in [(max_removidas,'max_removidas',0),(min_features,'min_features',1)]:
        if isinstance(n,bool) or not isinstance(n,Integral) or n<piso:
            raise ValueError(f'{nome} inválido.')
    if min_features>26 or not np.isfinite(melhora_minima) or melhora_minima<0:
        raise ValueError('Piso de features ou melhora mínima inválidos.')
    pool=avaliar_subconjunto(avaliador,categoria,FEATURES,etapa='controle')
    atual=pool['Melhor']
    nomes=list(FEATURES)
    rodadas,removidas=[],[]
    motivo='desativada' if not ativar else 'controle_invalido' if pool['status']!='ok' else None
    if motivo is None:
        pendentes=[f for f in FEATURES if avaliar_subconjunto(avaliador,categoria,
            [n for n in FEATURES if n!=f],executar=False)['status']=='pendente']
        if pendentes:
            motivo='analise_individual_incompleta'
    while motivo is None:
        if len(removidas)>=max_removidas:
            motivo='limite_remocoes'; break
        if len(nomes)<=min_features:
            motivo='minimo_features'; break
        testes=[]
        for f in nomes:
            candidato=avaliar_subconjunto(avaliador,categoria,[n for n in nomes if n!=f])
            melhor=candidato['Melhor']
            testes.append({'FeatureRemovida':f,'Features':candidato['Features'],'NFeatures':candidato['NFeatures'],
                'status':candidato['status'],'TrialsValidos':candidato['TrialsValidos'],'NTrials':candidato['NTrials'],
                'Trials':candidato['Trials'],'Avaliacao':melhor['Candidato'] if melhor else None,
                'Configuracao':melhor['Configuracao'] if melhor else None,
                'RMSE_pond':melhor['RMSE_pond'] if melhor else None,'LB_pond':melhor['LB_pond'] if melhor else None,
                'Delta_RMSE':melhor['RMSE_pond']-atual['RMSE_pond'] if melhor else None})
        rodada={'Rodada':len(removidas)+1,'Referencia':atual['Candidato'],'ConfiguracaoAntes':atual['Configuracao'],
            'FeaturesAntes':list(nomes),'NFeaturesAntes':len(nomes),'RMSE_antes':atual['RMSE_pond'],
            'LB_antes':atual['LB_pond'],'Testes':testes,'Aceita':False,'Removida':None}
        rodadas.append(rodada)
        if any(t['status']=='pendente' for t in testes):
            motivo='orcamento_rodada_incompleta'; break
        validos=sorted([t for t in testes if t['status']=='ok'],key=lambda t:(t['RMSE_pond'],t['FeatureRemovida'],t['Avaliacao']))
        if not validos:
            motivo='nenhuma_remocao_valida'; break
        vencedor=validos[0]
        if atual['RMSE_pond']-vencedor['RMSE_pond']<=melhora_minima:
            motivo='sem_melhora_minima'; break
        rodada.update(Aceita=True,Removida=vencedor['FeatureRemovida'],
            ConfiguracaoDepois=vencedor['Configuracao'],MudouConfiguracao=atual['Configuracao']!=vencedor['Configuracao'],
            MelhoraRMSE=-vencedor['Delta_RMSE'])
        removidas.append(vencedor['FeatureRemovida'])
        nomes=vencedor['Features']
        atual=avaliador.obter(vencedor['Configuracao'],nomes)
    resultado={'Categoria':categoria,'Configuracao':atual['Configuracao'] if atual else None,
        'Rodadas':rodadas,'Motivo':motivo,'Features':nomes,'NFeatures':len(nomes),
        'Removidas':removidas,'NRemovidas':len(removidas),
        'AvaliacaoFinal':atual['Candidato'] if atual else None,
        'RMSE_pond':atual['RMSE_pond'] if atual else None,'LB_pond':atual['LB_pond'] if atual else None}
    config={'metodo':'backward conjunto por categoria','contexto':comum.identidade(avaliador.contexto),
        'ativar':ativar,'max_removidas':max_removidas,'min_features':min_features,'melhora_minima':melhora_minima}
    comum.salvar_json(avaliador.pasta/f'progressiva_conjunta_{categoria}_{comum.identidade(config)[:16]}.json',
        {'config':config,**resultado})
    return resultado
