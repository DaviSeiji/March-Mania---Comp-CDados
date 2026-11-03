"""Exp. 13: mesmo catálogo do Exp. 12, forma recente e Top 3 por RMSE."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import exp11 as comum
from . import exp12 as anterior
from .recent_features import ESTATISTICAS_RECENTES, calcular_forma_recente, adicionar_forma_recente
from .submission import salvar_submission

FEATURES_CONTROLE = list(anterior.FEATURES_EFICIENCIA)
FEATURES = [*FEATURES_CONTROLE, *[n + 'Diff' for n in ESTATISTICAS_RECENTES]]
REVISAO = 1


def preparar_dados(pasta):
    base = anterior.preparar_dados(pasta)
    jogos, recentes = {}, {}
    for cat, tabela in base['jogos'].items():
        regulares = base['dados'][cat]['temporada_regular']
        regulares = regulares.loc[regulares.Season.isin(base['limites'][cat]) &
                                  regulares.Season.ge(tabela.Season.min())].copy()
        ncaa = base['dados'][cat]['resultados']
        if not regulares.merge(ncaa[['Season', 'DayNum', 'WTeamID', 'LTeamID']],
                              on=['Season', 'DayNum', 'WTeamID', 'LTeamID']).empty:
            raise ValueError('Fonte da forma recente contém jogos NCAA.')
        recentes[cat] = calcular_forma_recente(regulares, limites_daynum=base['limites'][cat])
        jogos[cat] = adicionar_forma_recente(tabela, recentes[cat])
        lookup = recentes[cat].set_index(['Season', 'TeamID'])
        for nome in ESTATISTICAS_RECENTES:
            a = lookup[nome].reindex(pd.MultiIndex.from_frame(tabela[['Season', 'TeamA']])).to_numpy()
            b = lookup[nome].reindex(pd.MultiIndex.from_frame(tabela[['Season', 'TeamB']])).to_numpy()
            np.testing.assert_allclose(jogos[cat][nome + 'Diff'], a - b)
        pd.testing.assert_frame_equal(jogos[cat][tabela.columns], tabela)
    comum.validar_historico(jogos, FEATURES)
    atuais = pd.concat([t.loc[t.Season.eq(2026)] for t in recentes.values()], ignore_index=True)
    confrontos = adicionar_forma_recente(base['confrontos'], atuais, permitir_ausentes=True)
    assert np.isfinite(confrontos.loc[confrontos.TemDuasSeeds, FEATURES].to_numpy()).all()
    assert confrontos.ID.tolist() == base['sample'].ID.tolist()
    return {**base, 'controle': base['jogos'], 'jogos': jogos, 'recentes': recentes, 'confrontos': confrontos}


def _ler(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assinatura_candidato(candidato):
    params = (anterior._parametros_tabpfn(candidato['Parametros']) if candidato['Abordagem'] == 'tabpfn'
              else deepcopy(candidato['Parametros']))
    return comum.identidade({k: candidato[k] for k in ['Candidato', 'Categoria', 'Abordagem', 'Familia']} | {'Parametros': params})


def carregar_catalogo(pasta_exp12):
    """Lê exclusivamente artefatos do Exp. 12; não gera modelos ou configurações.

    Exige uma descoberta inequívoca de 22 features por categoria, metadados do
    protocolo e as 100 configurações TabPFN salvas por categoria. Ausências e
    ambiguidades causam erro com os caminhos necessários.
    """
    pasta = Path(pasta_exp12).resolve()
    obrigatorios = [pasta / 'registro.json', pasta / 'similaridade.json']
    faltantes = [str(p) for p in obrigatorios if not p.is_file()]
    if faltantes:
        raise FileNotFoundError('Faltam checkpoints do Exp. 12: ' + '; '.join(faltantes))
    registro, similaridade = map(_ler, obrigatorios)
    if registro['features'] != FEATURES_CONTROLE or registro['anos'] != list(comum.ANOS):
        raise ValueError('Exp. 12 deve usar as 22 features e validação 2023–2025.')
    if registro['seed'] != 42 or registro['n_folds'] != 5:
        raise ValueError('O protocolo de origem não corresponde a seed=42/folds=5.')
    referencias = pd.DataFrame(similaridade['referencias'])
    if (set(referencias.Season) != set(comum.ANOS) or referencias.Season.duplicated().any()
            or not np.isfinite(referencias[['Distancia', 'Peso']]).all().all()
            or referencias.Peso.lt(0).any() or not np.isclose(referencias.Peso.sum(), 1)):
        raise ValueError('Pesos históricos de origem inválidos.')
    if max(similaridade['anos_padronizacao']) >= 2026:
        raise ValueError('Padronização de origem contém o ano alvo.')
    if not np.allclose(referencias.set_index('Season').Peso.sort_index(),
                       pd.DataFrame(registro['pesos']).set_index('Season').Peso.sort_index()):
        raise ValueError('Pesos divergentes entre registro e similaridade do Exp. 12.')
    candidatos, fontes = [], [str(p) for p in obrigatorios]
    for cat in ['M', 'W']:
        paths = [p for p in (pasta / 'eficiencia' / 'descoberta').glob(f'descoberta_{cat}_*.json')
                 if _ler(p).get('config', {}).get('features') == FEATURES_CONTROLE]
        if len(paths) != 1:
            raise FileNotFoundError(f'Esperava uma descoberta de 22 features para {cat}: '
                f'{pasta / "eficiencia" / "descoberta" / ("descoberta_" + cat + "_*.json")}; encontradas {len(paths)}. Se houver várias, selecione um diretório de origem inequívoco.')
        descoberta = _ler(paths[0])
        config = descoberta['config']
        if config['seed'] != 42 or config['folds'] != 5 or config['max_models'] != 100:
            raise ValueError('Descoberta H2O de origem não usa o protocolo esperado.')
        banco = descoberta['candidatos']
        if not banco or any(c['Categoria'] != cat or c['Abordagem'] != 'h2o' for c in banco):
            raise ValueError('Catálogo H2O de origem inconsistente.')
        tipos = {c['Familia'] for c in banco if c['Familia'].startswith('StackedEnsemble_')}
        if tipos != {'StackedEnsemble_BestOfFamily', 'StackedEnsemble_AllModels'}:
            raise ValueError(f'Faltam receitas de stacking no checkpoint {paths[0]}.')
        ids = {c['Candidato'] for c in banco}
        if any(not set(c['Parametros']['Bases']) <= ids for c in banco if c['Familia'].startswith('StackedEnsemble_')):
            raise ValueError('Receita sem suas configurações-base fixas.')
        candidatos.extend(deepcopy(banco))
        fontes.append(str(paths[0]))
    avaliacao = pasta / 'eficiencia' / 'avaliacao'
    tabpfn = {}
    def registrar(r):
        candidato = r.get('config', {}).get('candidato')
        if not candidato or candidato['Abordagem'] != 'tabpfn' or r['config']['features'] != FEATURES_CONTROLE:
            return
        ident = candidato['Candidato']
        if ident in tabpfn and assinatura_candidato(tabpfn[ident]) != assinatura_candidato(candidato):
            raise ValueError(f'Configurações conflitantes no Exp. 12 para {ident}.')
        tabpfn.setdefault(ident, deepcopy(candidato))
    ultimo = avaliacao / 'ultima_avaliacao.json'
    if ultimo.exists():
        for r in _ler(ultimo)['avaliacoes']:
            registrar(r)
        fontes.append(str(ultimo))
    if any(sum(c['Categoria'] == cat for c in tabpfn.values()) < 100 for cat in ['M', 'W']):
        for path in sorted(avaliacao.glob('corte_*.json')):
            registrar(_ler(path))
        fontes.append(str(avaliacao / 'corte_*.json'))
    for cat in ['M', 'W']:
        n = sum(c['Categoria'] == cat for c in tabpfn.values())
        if n != 100:
            raise FileNotFoundError(f'Faltam configurações TabPFN {cat}: encontradas {n}/100 em {avaliacao}. Não será gerado outro catálogo.')
    candidatos.extend(tabpfn.values())
    if len({c['Candidato'] for c in candidatos}) != len(candidatos):
        raise ValueError('IDs duplicados no catálogo de origem.')
    return {'candidatos': candidatos, 'referencias': referencias, 'registro_exp12': registro,
            'fontes': fontes, 'assinaturas': {c['Candidato']: assinatura_candidato(c) for c in candidatos}}


def reutilizar_controle(dados, candidatos, *, pasta_exp12, destino, n_folds=5, seed=42):
    """Copia só checkpoints equivalentes para o Exp. 13; nunca escreve na origem."""
    origem, destino = Path(pasta_exp12).resolve(), Path(destino).resolve()
    if destino == origem or origem in destino.parents:
        raise ValueError('O destino não pode ficar dentro dos checkpoints do Exp. 12.')
    linhas = []
    for c in candidatos:
        for ano in comum.ANOS:
            config, path = anterior._checkpoint(dados[c['Categoria']], c, ano, FEATURES_CONTROLE,
                origem / 'eficiencia' / 'avaliacao', n_folds=n_folds, seed=seed)
            salvo = anterior._ler_checkpoint_compativel(path, config)
            status = 'ausente_ou_nao_equivalente'
            if salvo is not None:
                status = 'reutilizado_' + salvo['status']
                nome = f'corte_{c["Candidato"]}_{ano}_{comum.identidade(salvo["config"])[:16]}.json'
                alvo = destino / nome
                if not alvo.exists():
                    comum.salvar_json(alvo, salvo)
            linhas.append({'Categoria': c['Categoria'], 'Abordagem': c['Abordagem'],
                'Candidato': c['Candidato'], 'Season': ano, 'Status': status})
    return pd.DataFrame(linhas)


def ler_controle(dados, candidatos, *, pasta, n_folds=5, seed=42):
    """Leitura parcial explícita: não completa resultados ausentes com novos fits."""
    avaliacoes, pendentes = [], []
    for c in candidatos:
        for ano in comum.ANOS:
            config, path = anterior._checkpoint(dados[c['Categoria']], c, ano, FEATURES_CONTROLE, pasta,
                                               n_folds=n_folds, seed=seed)
            salvo = anterior._ler_checkpoint_compativel(path, config)
            if salvo is not None:
                avaliacoes.append(salvo)
            else:
                pendentes.append({'Candidato': c['Candidato'], 'Season': ano})
    return {'avaliacoes': avaliacoes, 'pendentes': pendentes}


def avaliar_novo(dados, catalogo, *, pasta, orcamento_tabpfn=14400, repetir_falhas=False):
    assinaturas = {c['Candidato']: assinatura_candidato(c) for c in catalogo}
    resultado = anterior.avaliar_banco(dados, catalogo, features=FEATURES, pasta=pasta,
        n_folds=5, seed=42, orcamento_tabpfn_por_categoria=orcamento_tabpfn, repetir_falhas=repetir_falhas)
    if assinaturas != {c['Candidato']: assinatura_candidato(c) for c in catalogo}:
        raise RuntimeError('O catálogo foi alterado durante a validação.')
    return resultado


def ranquear(avaliacoes, referencias):
    scores, todos = comum.rankings(avaliacoes, referencias)
    scores['RMSE_media'] = scores[[f'RMSE_{ano}' for ano in comum.ANOS]].mean(axis=1)
    ranking = todos.loc[todos.Metrica.eq('rmse')].copy()
    ranking['RMSE_media'] = ranking[[f'RMSE_{ano}' for ano in comum.ANOS]].mean(axis=1)
    return scores, ranking.reset_index(drop=True)


def comparar_pareado(controle, novo, candidatos):
    """Somente candidatos completos nos dois conjuntos; rank pareado no mesmo pool."""
    chaves = ['Categoria', 'Abordagem', 'Candidato']
    pares = controle.merge(novo, on=chaves, suffixes=('_22', '_26'), validate='one_to_one')
    mapa = {c['Candidato']: c['Familia'] for c in candidatos}
    pares['Familia'] = pares.Candidato.map(mapa)
    for metrica in ['RMSE_pond', 'LB_pond', 'RMSE_media']:
        pares['Delta_' + metrica] = pares[metrica + '_26'] - pares[metrica + '_22']
    for conjunto in ['22', '26']:
        ordem = pares.sort_values(['Categoria', 'Abordagem', 'RMSE_pond_' + conjunto, 'Candidato'], kind='stable')
        posicoes = ordem.groupby(['Categoria', 'Abordagem']).cumcount() + 1
        pares['PosicaoPareada_' + conjunto] = posicoes.reindex(pares.index)
    cobertura = []
    for cat in ['M', 'W']:
        for abordagem in ['h2o', 'tabpfn']:
            mascara = lambda t: t.Categoria.eq(cat) & t.Abordagem.eq(abordagem)
            cobertura.append({'Categoria': cat, 'Abordagem': abordagem,
                'Catalogo': sum(c['Categoria'] == cat and c['Abordagem'] == abordagem for c in candidatos),
                'Completos22': int(mascara(controle).sum()), 'Completos26': int(mascara(novo).sum()),
                'ParesCompletos': int(mascara(pares).sum())})
    return pares, pd.DataFrame(cobertura)


def gerar_submissions(dados, candidatos, ranking, confrontos, sample, *, pasta, destino):
    """Seis CSVs Top1/2/3 por RMSE, compartilhando bases OOF e previsões finais."""
    comum.validar_historico(dados, FEATURES)
    if not ranking.Metrica.eq('rmse').all():
        raise ValueError('Exp. 13 seleciona exclusivamente por RMSE ponderado.')
    top = ranking.loc[ranking.Posicao.le(3)].copy()
    esperado = {(a, c, p) for a in ['h2o', 'tabpfn'] for c in ['M', 'W'] for p in [1, 2, 3]}
    if len(top) != 12 or set(map(tuple, top[['Abordagem', 'Categoria', 'Posicao']].to_numpy())) != esperado:
        raise ValueError('São necessários três vencedores válidos por abordagem/categoria.')
    catalogo = {c['Candidato']: c for c in candidatos}
    preds, falhas, tempos = {}, [], []
    for cat, nome in [('M', 'Masculino'), ('W', 'Feminino')]:
        mask = confrontos.Categoria.eq(nome) & confrontos.TemDuasSeeds
        pares = confrontos.loc[mask]
        selecionados = [catalogo[i] for i in top.loc[top.Categoria.eq(cat), 'Candidato'].unique()]
        for abordagem in ['h2o', 'tabpfn']:
            vencedores = [c for c in selecionados if c['Abordagem'] == abordagem]
            pendentes = []
            for c in vencedores:
                config = {'exp13_revisao': REVISAO, 'candidato': c, 'features': FEATURES,
                    'dados': comum.assinatura_dados(dados[cat], FEATURES),
                    'sample': comum.assinatura_dados(pares, FEATURES, target=False),
                    'n_folds': 5, 'seed': 42, 'versoes': comum.versoes()}
                path = Path(pasta) / f'final_{c["Candidato"]}_{comum.identidade(config)[:16]}.json'
                salvo = _ler(path) if path.exists() else None
                if salvo is not None:
                    if salvo['config'] != config or salvo['ID'] != pares.ID.tolist() or not np.isfinite(salvo['Pred']).all():
                        raise ValueError('Checkpoint final inconsistente.')
                    preds[c['Candidato']] = (mask, salvo['Pred'])
                    tempos.append({'Candidato': c['Candidato'], 'Segundos': salvo['segundos']})
                else:
                    pendentes.append((c, config, path))
            if not pendentes:
                continue
            dependencias = {i for c in vencedores if c['Familia'].startswith('StackedEnsemble_') for i in c['Parametros']['Bases']}
            plano = {c['Candidato']: c for c in vencedores}
            plano.update({i: catalogo[i] for i in dependencias})
            contexto = (anterior.SessaoH2O(dados[cat], list(plano.values()), features=FEATURES,
                        ano=2026, n_folds=5, seed=42) if abordagem == 'h2o' else nullcontext(None))
            with contexto as sessao:
                for c, config, path in pendentes:
                    try:
                        inicio = time.perf_counter()
                        ajuste = sessao.adaptador(c) if sessao else anterior._modelo_individual(dados[cat], c, FEATURES, 2026)
                        with ajuste as modelo:
                            pred = np.asarray(modelo.predict(pares[FEATURES]), dtype=float)
                        if pred.shape != (len(pares),) or not np.isfinite(pred).all():
                            raise ValueError('Previsões finais inválidas.')
                        segundos = time.perf_counter() - inicio
                        comum.salvar_json(path, {'config': config, 'ID': pares.ID.tolist(), 'Pred': pred.tolist(), 'segundos': segundos})
                        preds[c['Candidato']] = (mask, pred)
                        tempos.append({'Candidato': c['Candidato'], 'Segundos': segundos})
                    except Exception as exc:
                        if type(exc).__name__ == 'H2OJobCancelled':
                            raise
                        falhas.append({'Candidato': c['Candidato'], 'Erro': f'{type(exc).__name__}: {exc}'})
                if sessao:
                    comum.salvar_json(Path(pasta) / f'auditoria_final_{cat}.json', sessao.audit)
    if falhas:
        comum.salvar_json(Path(pasta) / 'falhas_finais.json', falhas)
        raise RuntimeError('Falhas no treino final; consulte falhas_finais.json. Previsões concluídas preservadas.')
    resumos, hashes = [], {}
    for (abordagem, posicao), grupo in top.groupby(['Abordagem', 'Posicao']):
        saida = confrontos[['ID']].copy()
        saida['Pred'] = 0.
        for ident in grupo.Candidato:
            mask, pred = preds[ident]
            saida.loc[mask, 'Pred'] = pred
        assert saida.loc[~confrontos.TemDuasSeeds, 'Pred'].eq(0).all()
        nome = f'submission_exp13_{abordagem}_rmse_top{int(posicao)}_2026.csv'
        salvar_submission(saida, sample, Path(destino) / nome)
        digest = comum.identidade(saida.Pred.tolist())
        resumos.append({'Arquivo': nome, 'Abordagem': abordagem, 'Posicao': int(posicao),
            'ModeloM': grupo.set_index('Categoria').loc['M', 'Candidato'],
            'ModeloW': grupo.set_index('Categoria').loc['W', 'Candidato'],
            'Linhas': len(saida), 'IdenticaA': hashes.get(digest)})
        hashes.setdefault(digest, nome)
    resultado = {'submissions': resumos, 'falhas': falhas, 'tempos_finais': tempos}
    comum.salvar_json(Path(pasta) / 'submissions.json', resultado)
    return resultado
