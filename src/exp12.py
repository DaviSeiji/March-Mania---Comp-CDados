"""Fluxo do Exp. 12; compartilha validações, métricas, GPU e I/O com o Exp. 11."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import inspect
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from . import exp11 as comum
from . import automl_h2o as backend
from .data_processing import carregar_dados, preparar_seeds, organizar_jogos
from .features import (ESTATISTICAS_DETALHADAS, calcular_estatisticas_agregadas,
    calcular_estatisticas_detalhadas, adicionar_seeds, adicionar_estatisticas,
    adicionar_estatisticas_detalhadas)
from .efficiency_features import ESTATISTICAS_EFICIENCIA, calcular_eficiencia, adicionar_eficiencia
from .exp12_h2o import SessaoH2O
from .metrics import logistic_brier
from .submission import preparar_confrontos, salvar_submission

FEATURES_BASE = ['SeedDif', 'WinRateDiff', 'AvgPointsForDiff', 'AvgPointsAgainstDiff', 'OppWinRateDiff',
                 *[n + 'Diff' for n in ESTATISTICAS_DETALHADAS]]
FEATURES_EFICIENCIA = [*FEATURES_BASE, *[n + 'Diff' for n in ESTATISTICAS_EFICIENCIA]]
REVISAO = 1


def _parametros_tabpfn(parametros):
    """Remove opção obsoleta somente quando o número de estimadores é explícito.

    Nas versões usadas pelo Exp. 12, o inteiro já impede auto-scaling. Não
    normalizamos n_estimators='auto', pois isso poderia mudar o modelo.
    """
    parametros = deepcopy(parametros)
    n = parametros.get('n_estimators')
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('Exp. 12 exige n_estimators inteiro positivo; configuração automática não é equivalente.')
    parametros.pop('auto_scale_n_estimators', None)
    return parametros


@contextmanager
def _modelo_individual(treino, candidato, features, ano):
    """Filtro específico e temporário, cobrindo fit e predict do TabPFN."""
    if candidato['Abordagem'] != 'tabpfn':
        with comum.modelo_ajustado(treino, candidato, features, ano) as modelo:
            yield modelo
        return
    efetivo = deepcopy(candidato)
    efetivo['Parametros'] = _parametros_tabpfn(candidato['Parametros'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning,
            message=r'^auto_scale_n_estimators is deprecated and will be removed in v9\.(?: |$)',
            module=r'^tabpfn(?:\.|$)')
        with comum.modelo_ajustado(treino, efetivo, features, ano) as modelo:
            yield modelo


def _ler_checkpoint_compativel(path, config):
    """Reaproveita o formato legado sem mudar valores ou escrever novo resultado.

    O único alias permitido restaura auto_scale=False com n_estimators inteiro;
    todos os demais dados, versões e hiperparâmetros continuam no fingerprint.
    """
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    candidato = config.get('candidato', {})
    parametros = candidato.get('Parametros', {})
    n = parametros.get('n_estimators')
    if (candidato.get('Abordagem') != 'tabpfn' or 'auto_scale_n_estimators' in parametros
            or not isinstance(n, int) or isinstance(n, bool) or n < 1):
        return None
    legado = deepcopy(config)
    legado['candidato']['Parametros']['auto_scale_n_estimators'] = False
    alias = path.with_name(path.stem.rsplit('_', 1)[0] + '_' + comum.identidade(legado)[:16] + path.suffix)
    if not alias.exists():
        return None
    resultado = json.loads(alias.read_text(encoding='utf-8'))
    if resultado.get('config') != legado:
        raise ValueError('Checkpoint legado com configuração inconsistente.')
    return resultado


def preparar_dados(pasta):
    """Mesmo histórico do Exp. 11, adicionando eficiência e auditando fontes/diferenças."""
    dados, jogos, limites, stats, detalhes, eficiencias, seeds = {}, {}, {}, {}, {}, {}, {}
    auditoria = []
    chaves = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
    for cat, inicio in [('M', 2003), ('W', 2010)]:
        base = dados[cat] = carregar_dados(pasta, cat, incluir_regular=True, incluir_detalhado=True)
        ncaa = base['resultados'].loc[base['resultados'].Season < 2026].copy()
        limites[cat] = {**ncaa.groupby('Season').DayNum.min().to_dict(), 2026: 134}
        raw = base['temporada_regular_detalhada']
        anos = set(raw.Season) & set(limites[cat])
        regular = base['temporada_regular'].loc[base['temporada_regular'].Season.isin(anos)].copy()
        detailed = raw.loc[raw.Season.isin(anos)].copy()
        for tabela in [regular, detailed]:
            if not (tabela.DayNum < tabela.Season.map(limites[cat])).all() or not tabela.merge(ncaa[chaves], on=chaves).empty:
                raise ValueError('Fonte não é exclusivamente pré-NCAA.')
        conferidos = detailed.merge(regular[chaves + ['WScore', 'LScore']], on=chaves,
            suffixes=('', '_compact'), how='left', validate='one_to_one', indicator=True)
        if not conferidos['_merge'].eq('both').all():
            raise ValueError('Detalhado sem partida compact correspondente.')
        for nome in ['WScore', 'LScore']:
            np.testing.assert_array_equal(conferidos[nome], conferidos[nome + '_compact'])
        seeds[cat] = preparar_seeds(base['seeds'])
        stats[cat] = calcular_estatisticas_agregadas(regular, limites_daynum=limites[cat])
        detalhes[cat] = calcular_estatisticas_detalhadas(detailed, limites_daynum=limites[cat])
        eficiencias[cat] = calcular_eficiencia(detailed, limites_daynum=limites[cat])
        tabela = adicionar_seeds(organizar_jogos(ncaa.loc[ncaa.Season.isin(anos)]), seeds[cat])
        tabela = adicionar_estatisticas(tabela, stats[cat])
        tabela = adicionar_estatisticas_detalhadas(tabela, detalhes[cat])
        tabela = adicionar_eficiencia(tabela, eficiencias[cat])
        for agregado, nomes in [(stats[cat], ['WinRate', 'AvgPointsFor', 'AvgPointsAgainst', 'OppWinRate']),
                                  (detalhes[cat], ESTATISTICAS_DETALHADAS), (eficiencias[cat], ESTATISTICAS_EFICIENCIA)]:
            lookup = agregado.set_index(['Season', 'TeamID'])
            for nome in nomes:
                a = lookup[nome].reindex(pd.MultiIndex.from_frame(tabela[['Season', 'TeamA']])).to_numpy()
                b = lookup[nome].reindex(pd.MultiIndex.from_frame(tabela[['Season', 'TeamB']])).to_numpy()
                np.testing.assert_allclose(tabela[nome + 'Diff'], a - b)
        np.testing.assert_allclose(tabela.SeedDif, tabela.SeedA - tabela.SeedB)
        if tabela.Season.min() != inicio or tabela.Season.max() != 2025:
            raise ValueError('Histórico diferente do Exp. 11.')
        jogos[cat] = tabela
        auditoria.append({'Categoria': cat, 'JogosTreinoHistorico': len(tabela),
            'PrimeiroAno': int(tabela.Season.min()), 'UltimoAno': int(tabela.Season.max()),
            'FontesFeatures': 'RegularSeasonCompactResults + RegularSeasonDetailedResults',
            'Limite2026': 134, 'SemSobreposicaoNCAA': True,
            'JogosDetalhados2026': int(detailed.Season.eq(2026).sum()),
            'UltimoDayNum2026': int(detailed.loc[detailed.Season.eq(2026), 'DayNum'].max())})
    comum.validar_historico(jogos, FEATURES_EFICIENCIA)
    sample = dados['M']['sample_submission']
    confrontos = preparar_confrontos(sample, seeds['M'], seeds['W'],
        dados['M']['times'].TeamID, dados['W']['times'].TeamID, temporada=2026)
    for funcao, tabelas in [(adicionar_estatisticas, stats), (adicionar_estatisticas_detalhadas, detalhes),
                            (adicionar_eficiencia, eficiencias)]:
        atuais = pd.concat([t.loc[t.Season.eq(2026)] for t in tabelas.values()], ignore_index=True)
        confrontos = funcao(confrontos, atuais, permitir_ausentes=True)
    assert confrontos.ID.tolist() == sample.ID.tolist()
    assert np.isfinite(confrontos.loc[confrontos.TemDuasSeeds, FEATURES_EFICIENCIA].to_numpy()).all()
    return {'dados': dados, 'jogos': jogos, 'limites': limites, 'eficiencias': eficiencias,
            'confrontos': confrontos, 'sample': sample, 'auditoria': pd.DataFrame(auditoria)}


def candidatos_tabpfn(categoria, *, device='cuda:0', max_candidatos=100, seed=42, verificar_api=True):
    """100 combinações estruturais: 2 versões × 5 ensembles × 5 perfis × 2 temperaturas.

    Uma única seed; não inflamos o catálogo com repetições. Perfis mudam a
    transformação do alvo ou a permutação de colunas. Todos usam o treino inteiro.
    """
    perfis = {'padrao': {}, 'alvo_identidade': {'REGRESSION_Y_PREPROCESS_TRANSFORMS': [None]},
        'alvo_safepower': {'REGRESSION_Y_PREPROCESS_TRANSFORMS': ['safepower']},
        'colunas_rotate': {'FEATURE_SHIFT_METHOD': 'rotate'}, 'colunas_sem_shift': {'FEATURE_SHIFT_METHOD': None}}
    if categoria not in {'M', 'W'} or not isinstance(max_candidatos, int) or not 1 <= max_candidatos <= 100:
        raise ValueError('Use M/W e de 1 a 100 configurações úteis.')
    if verificar_api:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.inference_config import InferenceConfig
        assinatura = inspect.signature(TabPFNRegressor.__init__).parameters
        for nome in ['softmax_temperature', 'inference_config', 'n_estimators']:
            if nome not in assinatura:
                raise RuntimeError(f'API TabPFN sem {nome}; catálogo não gerado.')
        campos = InferenceConfig.__annotations__
        for perfil in perfis.values():
            if not set(perfil) <= set(campos):
                raise RuntimeError('API de pré-processamento TabPFN incompatível.')
        if not all(hasattr(ModelVersion, n) for n in ['V3_5', 'V3_5_FAST']):
            raise RuntimeError('Faltam versões TabPFN 3.5/Fast.')
    banco = []
    for n in [4, 8, 16, 32, 64]:
        for versao in ['V3_5_FAST', 'V3_5']:
            for perfil, config in perfis.items():
                for temperatura in [1., .7]:
                    parametros = {'version': versao, 'n_estimators': n, 'random_state': seed,
                        'device': device, 'ignore_pretraining_limits': False,
                        'softmax_temperature': temperatura, 'inference_config': config}
                    # Preserva IDs históricos: o booleano legado era inerte com n inteiro.
                    identidade_legada = {**parametros, 'auto_scale_n_estimators': False}
                    estrutura = {k: v for k, v in identidade_legada.items() if k not in ['random_state', 'device']}
                    banco.append({'Candidato': f'{categoria}_tabpfn12_{comum.identidade(identidade_legada)[:16]}',
                        'Categoria': categoria, 'Abordagem': 'tabpfn', 'Familia': versao,
                        'Parametros': parametros, 'Perfil': perfil,
                        'Estrutura': comum.identidade(estrutura)[:16], 'SeedRepeticao': seed})
    return banco[:max_candidatos]


def _checkpoint(tabela, candidato, ano, features, pasta, *, n_folds=5, seed=42):
    treino, validacao = tabela.loc[tabela.Season < ano], tabela.loc[tabela.Season == ano]
    config = {'exp12_revisao': REVISAO, 'candidato': candidato, 'ano': ano, 'features': list(features),
        'treino': comum.assinatura_dados(treino, features),
        'validacao': comum.assinatura_dados(validacao, features), 'versoes': comum.versoes(),
        'n_folds': n_folds, 'seed': seed}
    path = Path(pasta) / f'corte_{candidato["Candidato"]}_{ano}_{comum.identidade(config)[:16]}.json'
    return config, path


def auditar_checkpoints(dados, candidatos, *, features, pasta, n_folds=5, seed=42):
    """Distingue catálogo de avaliações realmente salvas e completas, sem treinar."""
    linhas = []
    for cat in ['M', 'W']:
        for abordagem in ['h2o', 'tabpfn']:
            banco = [c for c in candidatos if c['Categoria'] == cat and c['Abordagem'] == abordagem]
            registrados, completos, bases_completas, ensembles_completos, ok, falhas = 0, 0, 0, 0, 0, 0
            for c in banco:
                anos_ok, encontrados = set(), 0
                for ano in comum.ANOS:
                    config, path = _checkpoint(dados[cat], c, ano, features, pasta, n_folds=n_folds, seed=seed)
                    salvo = _ler_checkpoint_compativel(path, config)
                    if salvo is not None:
                        encontrados += 1
                        if salvo['status'] == 'ok':
                            anos_ok.add(ano)
                            ok += 1
                        else:
                            falhas += 1
                registrados += bool(encontrados)
                if anos_ok == set(comum.ANOS):
                    completos += 1
                    if c['Familia'].startswith('StackedEnsemble_'):
                        ensembles_completos += 1
                    else:
                        bases_completas += 1
            linhas.append({'Categoria': cat, 'Abordagem': abordagem, 'CandidatosNoCatalogo': len(banco),
                'CandidatosComCortesRegistrados': registrados, 'CandidatosComTresCortesValidos': completos,
                'BasesComTresCortesValidos': bases_completas, 'EnsemblesComTresCortesValidos': ensembles_completos,
                'CortesValidos': ok, 'CortesComFalha': falhas})
    return pd.DataFrame(linhas)


def _avaliar(tabela, candidato, ano, *, features, pasta, sessao=None, repetir_falhas=False,
             n_folds=5, seed=42):
    config, path = _checkpoint(tabela, candidato, ano, features, pasta, n_folds=n_folds, seed=seed)
    salvo = _ler_checkpoint_compativel(path, config)
    if salvo is not None and (salvo['status'] == 'ok' or not repetir_falhas):
        return salvo
    treino, validacao = tabela.loc[tabela.Season < ano], tabela.loc[tabela.Season == ano]
    resultado = {'config': config, 'Categoria': candidato['Categoria'], 'Abordagem': candidato['Abordagem'],
        'Candidato': candidato['Candidato'], 'Familia': candidato['Familia'], 'Season': ano}
    recursos = {}
    try:
        cuda = candidato['Abordagem'] == 'tabpfn' and candidato['Parametros']['device'].startswith('cuda')
        with comum.medir_recursos(cuda=cuda) as recursos:
            inicio = time.perf_counter()
            contexto = sessao.adaptador(candidato) if sessao else _modelo_individual(treino, candidato, features, ano)
            with contexto as modelo:
                ajuste = time.perf_counter() - inicio
                if sessao and sessao.audit.get('HeapJVMObservadoMB'):
                    recursos['jvm_heap_observado_mb'] = sessao.audit['HeapJVMObservadoMB'][-1]['HeapMB']
                inicio_pred = time.perf_counter()
                pred = np.asarray(modelo.predict(validacao[list(features)]), dtype=float)
                inferencia = time.perf_counter() - inicio_pred
                if pred.shape != (len(validacao),) or not np.isfinite(pred).all():
                    raise ValueError('Previsões não finitas ou tamanho incorreto.')
        resultado.update(status='ok', RMSE=float(np.sqrt(np.mean((validacao.Target.to_numpy() - pred)**2))),
            LB=logistic_brier(validacao.Target, pred), SegundosTreino=ajuste, SegundosInferencia=inferencia,
            JogosTreino=len(treino), JogosValidacao=len(validacao), UltimoAnoTreino=int(treino.Season.max()),
            previsoes=validacao[['Season', 'TeamA', 'TeamB', 'Target']].assign(Pred=pred).to_dict('records'))
    except Exception as exc:
        if type(exc).__name__ == 'H2OJobCancelled':
            raise
        resultado.update(status='falha', erro=f'{type(exc).__name__}: {exc}')
    resultado['recursos'] = recursos
    comum.salvar_json(path, resultado)
    return resultado


def avaliar_banco(dados, candidatos, *, features, pasta, n_folds=5, seed=42,
                  orcamento_tabpfn_por_categoria=14400, repetir_falhas=False):
    comum.validar_historico(dados, features)
    if len({c['Candidato'] for c in candidatos}) != len(candidatos):
        raise ValueError('Candidatos duplicados.')
    if orcamento_tabpfn_por_categoria is not None and (not np.isfinite(orcamento_tabpfn_por_categoria) or orcamento_tabpfn_por_categoria <= 0):
        raise ValueError('Orçamento inválido.')
    linhas, pendentes, gastos, auditorias = [], [], {}, []
    for cat in ['M', 'W']:
        h2o = [c for c in candidatos if c['Categoria'] == cat and c['Abordagem'] == 'h2o']
        for ano in comum.ANOS:
            pendente = []
            for c in h2o:
                config, path = _checkpoint(dados[cat], c, ano, features, pasta, n_folds=n_folds, seed=seed)
                salvo = _ler_checkpoint_compativel(path, config)
                if salvo is not None and (salvo['status'] == 'ok' or not repetir_falhas):
                    linhas.append(salvo)
                else:
                    pendente.append(c)
            if pendente:
                treino = dados[cat].loc[dados[cat].Season < ano]
                with SessaoH2O(treino, h2o, features=features, ano=ano, n_folds=n_folds, seed=seed) as sessao:
                    for c in pendente:
                        linhas.append(_avaliar(dados[cat], c, ano, features=features, pasta=pasta,
                            sessao=sessao, repetir_falhas=repetir_falhas, n_folds=n_folds, seed=seed))
                    auditorias.append(sessao.audit)
                    comum.salvar_json(Path(pasta) / f'auditoria_h2o_{cat}_{ano}.json', sessao.audit)
                print(f'H2O/{cat}/{ano}: {len(pendente)} candidatos registrados.', flush=True)
        gasto = 0.
        for c in [c for c in candidatos if c['Categoria'] == cat and c['Abordagem'] == 'tabpfn']:
            if orcamento_tabpfn_por_categoria is not None and gasto >= orcamento_tabpfn_por_categoria:
                pendentes.append(c['Candidato'])
                continue
            for ano in comum.ANOS:
                resultado = _avaliar(dados[cat], c, ano, features=features, pasta=pasta,
                    repetir_falhas=repetir_falhas, n_folds=n_folds, seed=seed)
                linhas.append(resultado)
                gasto += resultado['recursos'].get('segundos', 0.)
            print(f'TabPFN/{cat}: {c["Candidato"]} registrado.', flush=True)
        gastos[cat] = gasto
    resultado = {'avaliacoes': linhas, 'pendentes': pendentes, 'gastos_tabpfn': gastos,
                 'orcamento': orcamento_tabpfn_por_categoria, 'auditorias_h2o': auditorias}
    comum.salvar_json(Path(pasta) / 'ultima_avaliacao.json', resultado)
    return resultado


def pilotos_tabpfn(dados, candidatos, *, features, pasta):
    """Fast/4 e 3.5/64 por categoria, fixos antes de observar scores."""
    pilotos = []
    for cat in ['M', 'W']:
        for versao, n in [('V3_5_FAST', 4), ('V3_5', 64)]:
            candidatos_piloto = [c for c in candidatos if c['Categoria'] == cat
                and c['Parametros']['version'] == versao and c['Parametros']['n_estimators'] == n
                and c['Perfil'] == 'padrao' and c['Parametros']['softmax_temperature'] == 1.]
            if not candidatos_piloto:
                pilotos.append({'Categoria': cat, 'Familia': versao, 'n_estimators': n,
                                'status': 'nao_planejado', 'erro': 'Configuração fora do catálogo reduzido.'})
                continue
            pilotos.append(_avaliar(dados[cat], candidatos_piloto[0], 2023, features=features,
                                    pasta=pasta, repetir_falhas=True))
    return pilotos


def validar_controle(principal, controle):
    for cat in ['M', 'W']:
        cols = ['Season', 'TeamA', 'TeamB', 'Target', *FEATURES_BASE]
        pd.testing.assert_frame_equal(principal[cat][cols], controle[cat][cols])
    return True


def comparar_rankings(resultados):
    """Mesmos anos/pesos, comparação explícita entre conjuntos e famílias."""
    tabelas = []
    for conjunto, resultado in resultados.items():
        candidatos = {c['Candidato']: c for c in resultado['candidatos']}
        tabela = resultado['ranking'].copy()
        tabela['Conjunto'] = conjunto
        tabela['Familia'] = tabela.Candidato.map(lambda c: candidatos[c]['Familia'])
        tabela['StackedEnsemble'] = tabela.Familia.str.startswith('StackedEnsemble_')
        tabelas.append(tabela)
    return pd.concat(tabelas, ignore_index=True)


def gerar_submissions(dados, candidatos, ranking, confrontos, sample, *, features, pasta, destino,
                       n_folds=5, seed=42):
    if list(features) != FEATURES_EFICIENCIA:
        raise ValueError('As submissions do Exp. 12 exigem as 22 features de eficiência.')
    comum.validar_historico(dados, features)
    top = ranking.loc[ranking.Posicao.eq(1)].copy()
    esperados = {(a, m, c) for a in ['h2o', 'tabpfn'] for m in ['rmse', 'lb'] for c in ['M', 'W']}
    if set(map(tuple, top[['Abordagem', 'Metrica', 'Categoria']].to_numpy())) != esperados or len(top) != 8:
        raise ValueError('São necessários os oito vencedores (M/W × H2O/TabPFN × RMSE/LB).')
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
                config = {'revisao': REVISAO, 'candidato': c, 'features': list(features),
                    'dados': comum.assinatura_dados(dados[cat], features),
                    'sample': comum.assinatura_dados(pares, features, target=False),
                    'n_folds': n_folds, 'seed': seed, 'versoes': comum.versoes()}
                path = Path(pasta) / f'final_{c["Candidato"]}_{comum.identidade(config)[:16]}.json'
                salvo = _ler_checkpoint_compativel(path, config)
                if salvo is not None:
                    if salvo['ID'] != pares.ID.tolist() or not np.isfinite(salvo['Pred']).all():
                        raise ValueError('Checkpoint final inconsistente.')
                    preds[c['Candidato']] = (mask, salvo['Pred'])
                    tempos.append({'Candidato': c['Candidato'], 'Segundos': salvo['segundos']})
                else:
                    pendentes.append((c, config, path))
            if not pendentes:
                continue
            ids_necessarios = {i for c in vencedores if c['Familia'].startswith('StackedEnsemble_') for i in c['Parametros']['Bases']}
            plano = {c['Candidato']: c for c in vencedores}
            plano.update({i: catalogo[i] for i in ids_necessarios})
            contexto = (SessaoH2O(dados[cat], list(plano.values()), features=features, ano=2026,
                                  n_folds=n_folds, seed=seed) if abordagem == 'h2o' else nullcontext(None))
            with contexto as sessao:
                for c, config, path in pendentes:
                    try:
                        inicio = time.perf_counter()
                        ajuste = sessao.adaptador(c) if sessao else _modelo_individual(dados[cat], c, features, 2026)
                        with ajuste as modelo:
                            pred = np.asarray(modelo.predict(pares[list(features)]), dtype=float)
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
        raise RuntimeError('Falhas no treino final; consulte falhas_finais.json. Checkpoints concluídos preservados; quatro CSVs não foram gerados.')
    resumos, hashes = [], {}
    for (abordagem, metrica), grupo in top.groupby(['Abordagem', 'Metrica']):
        saida = confrontos[['ID']].copy()
        saida['Pred'] = 0.
        for ident in grupo.Candidato:
            mask, pred = preds[ident]
            saida.loc[mask, 'Pred'] = pred
        assert saida.loc[~confrontos.TemDuasSeeds, 'Pred'].eq(0).all()
        nome = f'submission_exp12_{abordagem}_{metrica}_top1_2026.csv'
        salvar_submission(saida, sample, Path(destino) / nome)
        digest = comum.identidade(saida.Pred.tolist())
        resumos.append({'Arquivo': nome, 'Abordagem': abordagem, 'Metrica': metrica,
            'ModeloM': grupo.set_index('Categoria').loc['M', 'Candidato'],
            'ModeloW': grupo.set_index('Categoria').loc['W', 'Candidato'],
            'Linhas': len(saida), 'IdenticaA': hashes.get(digest)})
        hashes.setdefault(digest, nome)
    resultado = {'submissions': resumos, 'tempos_finais': tempos, 'falhas': falhas}
    comum.salvar_json(Path(pasta) / 'submissions.json', resultado)
    return resultado
