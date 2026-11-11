"""Exp. 15: combinações fechadas de classes, com o avaliador temporal existente."""

from copy import deepcopy
from itertools import combinations
from numbers import Integral
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import exp11 as comum, exp13, exp14 as base
from .efficiency_features import ESTATISTICAS_EFICIENCIA
from .features import ESTATISTICAS_DETALHADAS
from .recent_features import ESTATISTICAS_RECENTES
from .submission import salvar_submission

FEATURES = list(exp13.FEATURES)
ANOS = list(comum.ANOS)
REVISAO = 2
preparar_dados = exp13.preparar_dados
carregar_origem = base.carregar_origem

CLASSES = {
    'A': ['SeedDif', 'WinRateDiff', 'AvgPointsForDiff', 'AvgPointsAgainstDiff', 'OppWinRateDiff'],
    'B': ['FGPercentageDiff', 'ThreePointPercentageDiff', 'FTPercentageDiff', 'AssistsPerGameDiff',
          'OffensiveReboundsPerGameDiff', 'DefensiveReboundsPerGameDiff', 'TurnoversPerGameDiff',
          'StealsPerGameDiff', 'BlocksPerGameDiff'],
    'C': ['OffensiveRatingDiff', 'DefensiveRatingDiff', 'NetRatingDiff', 'PossessionsPerGameDiff',
          'EffectiveFGPercentageDiff', 'TurnoverRateDiff', 'OffensiveReboundRateDiff', 'FreeThrowRateDiff'],
    'D': ['RecentWinRate10Diff', 'RecentAvgMargin10Diff', 'RecentAvgPointsFor10Diff', 'RecentAvgPointsAgainst10Diff'],
}


def validar_classes(classes=None):
    classes = CLASSES if classes is None else classes
    implementadas = {'A': base.avaliacao.FEATURES_BASE[:5],
        'B': [n+'Diff' for n in ESTATISTICAS_DETALHADAS],
        'C': [n+'Diff' for n in ESTATISTICAS_EFICIENCIA],
        'D': [n+'Diff' for n in ESTATISTICAS_RECENTES]}
    todas = [f for grupo in classes.values() for f in grupo]
    if (set(classes) != set('ABCD') or len(todas) != 26 or len(set(todas)) != 26
            or set(todas) != set(FEATURES) or classes != implementadas):
        raise ValueError('As classes devem corresponder à partição implementada A:5/B:9/C:8/D:4.')
    return True


def gerar_combinacoes():
    """As 15 uniões não vazias, sem remover features internas às classes."""
    validar_classes()
    planos = []
    for n in range(1, 5):
        for grupos in combinations('ABCD', n):
            nomes = {f for g in grupos for f in CLASSES[g]}
            features = [f for f in FEATURES if f in nomes]
            ident = '+'.join(grupos)
            planos.append({'Combinacao': ident, 'Classes': list(grupos), 'NClasses': n,
                'Features': features, 'NFeatures': len(features),
                'Assinatura': comum.identidade({'Classes': list(grupos), 'Features': features})})
    return planos


def gerar_catalogo_busca(n_trials=20, *, seed=42):
    """Amostra configurações sem consultar scores ou vencedores anteriores.

    Ordem aleatória estratificada por versão/n_estimators. Com 20 trials, cada
    um dos dez estratos recebe duas configurações distintas. O prefixo permanece
    estável se o orçamento aumentar. Mesmos parâmetros para M/W e para as 15
    combinações; cada par dados/configuração é ajustado e selecionado separadamente.
    """
    if isinstance(n_trials,bool) or not isinstance(n_trials,Integral) or not 1<=n_trials<=100:
        raise ValueError('Use de 1 a 100 trials distintos por combinação/categoria.')
    if isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0:
        raise ValueError('Seed deve ser inteiro não negativo.')
    referencia=base.avaliacao.candidatos_tabpfn('M',seed=int(seed),verificar_api=False)
    rng=np.random.default_rng(seed)
    estratos=[]
    for n in [4,8,16,32,64]:
        for versao in ['V3_5_FAST','V3_5']:
            indices=[i for i,c in enumerate(referencia)
                if c['Parametros']['n_estimators']==n and c['Parametros']['version']==versao]
            estratos.append(rng.permutation(indices).tolist())
    ordem=rng.permutation(len(estratos)).tolist()
    indices=[estratos[j][rodada] for rodada in range(10) for j in ordem][:n_trials]
    return [banco[i] for cat in ['M','W']
        for banco in [base.avaliacao.candidatos_tabpfn(cat,seed=int(seed),verificar_api=False)] for i in indices]


def carregar_busca(pasta_exp13, *, n_trials=20, seed=42):
    """Lê somente o protocolo/pesos do Exp. 13; gera o catálogo sem seu ranking."""
    path=Path(pasta_exp13).resolve()/'registro.json'
    if not path.is_file():
        raise FileNotFoundError(f'Falta o protocolo/pesos de origem: {path}')
    registro=base._ler(path)
    if (registro['features']!=FEATURES or registro['anos']!=ANOS
            or registro['seed']!=42 or registro['n_folds']!=5):
        raise ValueError('Protocolo de origem deve usar 26 features, 2023–2025, seed 42 e folds=5.')
    referencias=pd.DataFrame(registro['pesos'])
    comum.rankings([dict(Categoria='M',Abordagem='tabpfn',Candidato='pesos',Season=a,
        status='ok',RMSE=1.,LB=.1) for a in ANOS],referencias)
    return {'candidatos':gerar_catalogo_busca(n_trials,seed=seed),'referencias':referencias,
        'fontes':[str(path)],'versoes_exp13':registro['versoes'],
        'busca':{'metodo':'amostragem aleatoria estratificada sem scores anteriores',
            'n_trials_por_combinacao_categoria':int(n_trials),'seed_amostragem':int(seed),
            'versoes':['V3_5_FAST','V3_5'],'n_estimators':[4,8,16,32,64],
            'perfis':['padrao','alvo_identidade','alvo_safepower','colunas_rotate','colunas_sem_shift'],
            'temperaturas':[1.,.7],'espaco_total':100}}


class Avaliador(base.Avaliador):
    """Reutiliza os fits/checkpoints do Exp. 14, aceitando só as 15 uniões.

    O estado e os resultados ficam exclusivamente no Exp. 15. O limite global
    conta candidatos distintos configuração × combinação, inclusive controles,
    falhas, parciais e resultados reutilizados; cada candidato tem três cortes.
    """

    def __init__(self, dados, candidatos, referencias, *, pasta, pasta_exp13,
                 max_avaliacoes=None, repetir_falhas=False, verbose=False):
        if max_avaliacoes is not None and (isinstance(max_avaliacoes, bool)
                or not isinstance(max_avaliacoes, Integral) or max_avaliacoes < 0):
            raise ValueError('max_avaliacoes deve ser inteiro não negativo ou None.')
        destino, fonte = Path(pasta).resolve(), Path(pasta_exp13).resolve()
        for anterior in [fonte, fonte.parent/'exp12', fonte.parent/'exp14']:
            if destino == anterior or anterior in destino.parents:
                raise ValueError('O Exp. 15 não pode escrever dentro dos experimentos anteriores.')
        self.planos = {p['Combinacao']: p for p in gerar_combinacoes()}
        super().__init__(dados, candidatos, referencias, pasta=destino, pasta_exp13=fonte,
            max_avaliacoes_por_configuracao=None, repetir_falhas=repetir_falhas, verbose=verbose)
        self.limite = max_avaliacoes
        self.contexto = {k:v for k,v in self.contexto.items() if k != 'exp14_revisao'} | {
            'exp15_revisao': REVISAO, 'backend_exp14_revisao': base.REVISAO,
            'classes': deepcopy(CLASSES), 'combinacoes': list(self.planos.values())}
        self.path_estado = self.pasta / ('estado_exp15_' + comum.identidade(self.contexto)[:16] + '.json')
        self.estado = (base._ler(self.path_estado) if self.path_estado.exists()
            else {'contexto': self.contexto, 'avaliacoes': {}})
        if self.estado['contexto'] != self.contexto:
            raise ValueError('Estado do Exp. 15 incompatível.')

    def _plano(self, combinacao):
        if isinstance(combinacao, str):
            if combinacao not in self.planos:
                raise ValueError(f'Combinação desconhecida: {combinacao}.')
            return self.planos[combinacao]
        features = base.validar_features(combinacao)
        for p in self.planos.values():
            if p['Features'] == features:
                return p
        raise ValueError('Somente classes inteiras são permitidas no Exp. 15.')

    def avaliar(self, ident, combinacao, *, etapa='combinacoes'):
        plano = self._plano(combinacao)
        candidato = self.candidatos[ident]
        chave = base.identidade_avaliacao(candidato, plano['Features'])
        if (chave not in self.estado['avaliacoes'] and self.limite is not None
                and len(self.estado['avaliacoes']) >= self.limite):
            return {**deepcopy(plano), 'Categoria': candidato['Categoria'], 'Abordagem': 'tabpfn',
                'Configuracao': ident, 'Candidato': chave, 'status': 'pendente_orcamento'}
        r = super().avaliar(ident, plano['Features'], etapa=etapa)
        r.update(deepcopy(plano))
        self.estado['avaliacoes'][chave] = r
        self._salvar()
        return deepcopy(r)

    def ranking(self):
        tabela = super().ranking()
        if tabela.empty:
            return tabela
        for i, r in tabela.iterrows():
            plano = self._plano(r.Features)
            for col in ['Combinacao', 'NClasses', 'Assinatura']:
                tabela.loc[i, col] = plano[col]
        tabela['NClasses'] = tabela.NClasses.astype(int)
        tabela['Classes'] = [self._plano(f)['Classes'] for f in tabela.Features]
        trials={}
        for cat in ['M','W']:
            ids=[i for i,c in self.candidatos.items() if c['Categoria']==cat]
            trials.update({ident:n for n,ident in enumerate(ids,1)})
        tabela['Trial']=tabela.Configuracao.map(trials)
        return tabela

    def contagem(self):
        return pd.DataFrame([{'Categoria': c['Categoria'], 'Configuracao': ident,
            'CandidatosRegistrados': sum(r['Configuracao']==ident for r in self.estado['avaliacoes'].values()),
            'CandidatosValidos': sum(r['Configuracao']==ident and r['status']=='ok'
                                    for r in self.estado['avaliacoes'].values()),
            'CombinacoesPrevistas': 15} for ident, c in self.candidatos.items()])


def avaliar_combinacoes(avaliador, *, combinacoes=None, n_configuracoes=None, etapa='completo'):
    """Avalia cada configuração fixa nos três anos, separadamente por dados/M/W."""
    if n_configuracoes is not None and (isinstance(n_configuracoes, bool)
            or not isinstance(n_configuracoes, Integral) or not 1<=n_configuracoes<=100):
        raise ValueError('Use de uma a 100 configurações por categoria, ou None para todas.')
    ids = list(avaliador.planos) if combinacoes is None else list(combinacoes)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= avaliador.planos.keys():
        raise ValueError('Lista de combinações vazia, duplicada ou inválida.')
    selecionados = []
    for cat in ['M', 'W']:
        # Ordem da amostragem, sem classificação prévia por desempenho.
        banco = [c for c in avaliador.candidatos.values() if c['Categoria']==cat]
        if not banco or (n_configuracoes is not None and len(banco)<n_configuracoes):
            raise ValueError(f'Faltam configurações de origem para {cat}.')
        selecionados.extend(banco[:n_configuracoes])
    resultados = [avaliador.avaliar(c['Candidato'], ident, etapa=etapa)
        for ident in ids for c in selecionados]
    return pd.DataFrame(resultados)


def resumir_combinacoes(ranking):
    """Melhor configuração por combinação; mantém cobertura e ranking por M/W."""
    if ranking.empty:
        return ranking.copy()
    melhor = ranking.sort_values(['Categoria','RMSE_pond','Candidato'], kind='stable').groupby(
        ['Categoria','Combinacao'], sort=False).head(1).copy()
    contagens = ranking.groupby(['Categoria','Combinacao']).size()
    melhor['ConfiguracoesValidas'] = [int(contagens.loc[(r.Categoria,r.Combinacao)]) for r in melhor.itertuples()]
    melhor['PosicaoCombinacao'] = melhor.groupby('Categoria').cumcount()+1
    return melhor.reset_index(drop=True)


def comparar_adicoes(ranking):
    """Compara X com X+uma classe, sempre com a mesma configuração/categoria."""
    linhas = []
    if ranking.empty:
        return pd.DataFrame()
    for (cat, config), grupo in ranking.groupby(['Categoria','Configuracao']):
        mapa = {r['Combinacao']:r for r in grupo.to_dict('records')}
        for antes in mapa.values():
            for classe in 'ABCD':
                if classe in antes['Classes']:
                    continue
                ident = '+'.join(c for c in 'ABCD' if c in [*antes['Classes'],classe])
                if ident not in mapa:
                    continue
                depois = mapa[ident]
                linhas.append({'Categoria':cat,'Configuracao':config,'Antes':antes['Combinacao'],
                    'Depois':ident,'ClasseAdicionada':classe,'NFeaturesAntes':antes['NFeatures'],
                    'NFeaturesDepois':depois['NFeatures'],'RMSE_antes':antes['RMSE_pond'],
                    'RMSE_depois':depois['RMSE_pond'],'Delta_RMSE':depois['RMSE_pond']-antes['RMSE_pond'],
                    'LB_antes':antes['LB_pond'],'LB_depois':depois['LB_pond'],
                    'Delta_LB':depois['LB_pond']-antes['LB_pond'],
                    'MenorFoiMelhor':antes['RMSE_pond']<depois['RMSE_pond']})
    return pd.DataFrame(linhas)


def gerar_submissions(avaliador, ranking, confrontos, sample, *, destino, top_k=3):
    """Top K M/W por RMSE; top_k=3 mantém a API anterior; a busca nova usa 5."""
    if isinstance(top_k,bool) or not isinstance(top_k,Integral) or top_k<1:
        raise ValueError('top_k deve ser inteiro positivo.')
    if ranking.empty or not ranking.Metrica.eq('rmse').all():
        raise ValueError('Seleção do Exp. 15 exige ranking por RMSE ponderado.')
    top = ranking.loc[ranking.Posicao.le(top_k)]
    if len(top)!=2*top_k or set(map(tuple,top[['Categoria','Posicao']].to_numpy())) != {
            (c,p) for c in ['M','W'] for p in range(1,top_k+1)}:
        raise ValueError(f'São necessários {top_k} candidatos válidos por categoria.')
    previsoes, tempos = {}, {}
    for r in top.to_dict('records'):
        c = avaliador.candidatos[r['Configuracao']]
        plano = avaliador._plano(r['Combinacao'])
        features = plano['Features']
        valido = avaliador.obter(c['Candidato'],features)
        if (valido is None or valido['status']!='ok' or valido['Candidato']!=r['Candidato']
                or c['Categoria']!=r['Categoria'] or r['Features']!=features or r['NFeatures']!=len(features)):
            raise ValueError('Candidato sem avaliação histórica válida para essas classes.')
        mask = confrontos.Categoria.eq({'M':'Masculino','W':'Feminino'}[c['Categoria']]) & confrontos.TemDuasSeeds
        pares, treino = confrontos.loc[mask], avaliador.dados[c['Categoria']]
        assert treino.Season.lt(2026).all()
        config = {'exp15_revisao':REVISAO,'candidato':c,'combinacao':plano,'ano':2026,'seed':42,'n_folds':5,
            'treino':comum.assinatura_dados(treino,features),'sample':comum.assinatura_dados(pares,features,target=False),
            'versoes':comum.versoes()}
        path = avaliador.pasta/'final'/f'final_{r["Candidato"]}_{comum.identidade(config)[:16]}.json'
        try:
            if path.exists():
                salvo = base._ler(path)
                if salvo['config']!=config:
                    raise ValueError('Checkpoint final com configuração inconsistente.')
            else:
                inicio = time.perf_counter()
                with base.avaliacao._modelo_individual(treino,c,features,2026) as modelo:
                    pred = np.asarray(modelo.predict(pares[features]),dtype=float)
                if pred.shape!=(len(pares),) or not np.isfinite(pred).all():
                    raise ValueError('Previsões finais inválidas.')
                salvo = {'config':config,'ID':pares.ID.tolist(),'Pred':pred.tolist(),'segundos':time.perf_counter()-inicio}
                comum.salvar_json(path,salvo)
            if (salvo['ID']!=pares.ID.tolist() or len(salvo['Pred'])!=len(pares)
                    or not np.isfinite(salvo['Pred']).all()):
                raise ValueError('IDs ou previsões finais inconsistentes.')
            previsoes[r['Candidato']] = (mask,salvo['Pred'])
            tempos[r['Candidato']] = salvo['segundos']
        except Exception as exc:
            comum.salvar_json(avaliador.pasta/'final'/'falha.json',
                {'Candidato':r['Candidato'],'Erro':f'{type(exc).__name__}: {exc}'})
            raise
    saidas, hashes = [], {}
    for pos in range(1,top_k+1):
        grupo = top.loc[top.Posicao.eq(pos)].set_index('Categoria')
        saida = confrontos[['ID']].copy().assign(Pred=0.)
        for ident in grupo.Candidato:
            mask,pred = previsoes[ident]
            saida.loc[mask,'Pred'] = pred
        assert saida.loc[~confrontos.TemDuasSeeds,'Pred'].eq(0).all()
        nome = f'submission_exp15_tabpfn_rmse_top{pos}_2026.csv'
        salvar_submission(saida,sample,Path(destino)/nome)
        digest = comum.identidade(saida.Pred.tolist())
        saidas.append({'Arquivo':nome,'Posicao':pos,'Linhas':len(saida),'IdenticaA':hashes.get(digest),
            **{f'{col}{cat}':grupo.loc[cat,col] for cat in ['M','W']
               for col in ['Candidato','Configuracao','Combinacao','NFeatures','Features']}})
        hashes.setdefault(digest,nome)
    resultado = {'submissions':saidas,'segundos_finais':tempos}
    comum.salvar_json(avaliador.pasta/'final'/'submissions.json',resultado)
    return resultado
