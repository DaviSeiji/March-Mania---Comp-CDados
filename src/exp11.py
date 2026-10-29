"""Experimento 11: candidatos fixos, validação externa e checkpoints locais.

Imports dos backends são tardios: métricas e testes não exigem TabPFN ou Java.
Os checkpoints contêm dados/versões/configuração e não serializam estimadores.
"""

from contextlib import contextmanager
from copy import deepcopy
import gc
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from . import automl_h2o as h2ob
from .metrics import logistic_brier
from .submission import salvar_submission

ANOS = (2023, 2024, 2025)
REVISAO = 1


def carregar_token_tabpfn(caminho):
    """Carrega a chave local sem retorná-la, imprimi-la ou incluí-la em registros."""
    path = Path(caminho)
    if not path.is_file():
        raise FileNotFoundError('Crie .env.tabpfn na raiz com TABPFN_TOKEN=sua_chave.')
    linhas = [linha.strip() for linha in path.read_text(encoding='utf-8-sig').splitlines()
              if linha.strip() and not linha.lstrip().startswith('#')]
    if len(linhas) != 1 or not linhas[0].startswith('TABPFN_TOKEN='):
        raise ValueError('Use uma única linha no arquivo: TABPFN_TOKEN=sua_chave.')
    chave = linhas[0].split('=', 1)[1].strip()
    if len(chave) >= 2 and chave[0] == chave[-1] and chave[0] in {'"', "'"}:
        chave = chave[1:-1].strip()
    if not chave or chave == 'COLE_SUA_CHAVE_AQUI' or any(c.isspace() for c in chave):
        raise ValueError('Preencha .env.tabpfn com sua API Key válida, sem espaços.')
    os.environ['TABPFN_TOKEN'] = chave


def validar_acesso_tabpfn():
    """Consulta autenticação/licença, sem fit, download de pesos ou exposição da chave."""
    from tabpfn.browser_auth import verify_token, check_license_accepted, _get_license_name
    from tabpfn.settings import settings
    chave = os.environ.get('TABPFN_TOKEN')
    if not chave:
        raise RuntimeError('Carregue .env.tabpfn antes de verificar o acesso.')
    valido = verify_token(chave, settings.tabpfn.auth_api_url)
    if valido is False:
        raise RuntimeError('A Prior Labs rejeitou a chave (inválida ou expirada). Copie uma API Key da página Account e atualize .env.tabpfn.')
    if valido is not True:
        raise RuntimeError('Não foi possível consultar a Prior Labs. Verifique conexão/serviço e tente novamente.')
    licenca = _get_license_name('tabpfn_3_5')
    aceita = check_license_accepted(chave, settings.tabpfn.auth_api_url, licenca)
    if aceita is False:
        raise RuntimeError(f'Chave válida, mas a licença {licenca} ainda não foi aceita nesta conta. Aceite-a em https://ux.priorlabs.ai (Licenses) e reexecute esta célula.')
    if aceita is not True:
        raise RuntimeError('Chave válida, mas o servidor de licenças está indisponível. Tente novamente mais tarde.')
    return {'chave_valida': True, 'licenca': licenca, 'licenca_aceita': True}


def configurar_avisos_h2o():
    """Oculta apenas avisos esperados ao reajustar sem early stopping."""
    warnings.filterwarnings('ignore', category=RuntimeWarning,
        message=r'^Stopping (metric|tolerance) is ignored for _stopping_rounds=0\.$',
        module=r'^h2o\.estimators\.estimator_base$')


def versoes():
    resultado = {}
    for nome in ['h2o', 'tabpfn', 'torch', 'numpy', 'pandas', 'scikit-learn', 'psutil']:
        try:
            resultado[nome] = version(nome)
        except PackageNotFoundError:
            resultado[nome] = None
    return resultado


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Valor não serializável: {type(value)}')


def _parametros_simples(value):
    """Remove wrappers H2OResponse antes de copiar/serializar os parâmetros.

    Modelos recuperados com get_model contêm subclasses de dict cujo construtor
    exige argumentos; deepcopy tenta reconstruí-las e falha. Preservamos valores
    recursivamente em contêineres Python comuns, sem chamar esses construtores.
    """
    if isinstance(value, dict):
        return {k: _parametros_simples(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_parametros_simples(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def identidade(value):
    return sha256(json.dumps(value, sort_keys=True, default=_json_default,
                             allow_nan=False).encode()).hexdigest()


def salvar_json(path, value):
    """Gravação atômica: uma interrupção não deixa um checkpoint parcial válido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporario = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporario.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=_json_default, allow_nan=False), encoding='utf-8')
    temporario.replace(path)


def _normalizar_relatorio_descoberta(resultado):
    """Métricas auxiliares indisponíveis no leaderboard viram null, nunca zero.

    RMSLE, por exemplo, pode ser NaN para margens negativas. Configurações,
    RMSE de seleção e métricas externas continuam sujeitas às validações estritas.
    """
    resultado = deepcopy(resultado)
    leaderboard = pd.DataFrame(resultado['leaderboard'])
    if 'rmse' not in leaderboard or not np.isfinite(leaderboard.rmse.to_numpy(dtype=float)).all():
        raise ValueError('Leaderboard sem RMSE finito para seleção.')
    leaderboard = leaderboard.replace([np.inf, -np.inf], np.nan)
    resultado['leaderboard'] = leaderboard.astype(object).where(leaderboard.notna(), None).to_dict('records')
    return resultado


def recuperar_descoberta_interrompida():
    """Recupera resultado completo retido no traceback do último erro do notebook.

    Não treina modelos e não consulta o cluster, que já pode ter sido limpo.
    Disponível somente enquanto o kernel conserva o traceback da busca.
    """
    traceback = getattr(sys, 'last_traceback', None)
    while traceback is not None:
        frame = traceback.tb_frame
        if (Path(frame.f_code.co_filename).resolve() == Path(__file__).resolve()
                and frame.f_code.co_name == 'descobrir_h2o'):
            resultado, path = frame.f_locals.get('resultado'), frame.f_locals.get('path')
            if resultado is not None and path is not None:
                salvar_json(path, _normalizar_relatorio_descoberta(resultado))
                print('Busca concluída recuperada do erro anterior:', Path(path).name)
                return Path(path)
        traceback = traceback.tb_next
    return None


def assinatura_dados(tabela, features, *, target=True):
    h2ob._validar_tabela(tabela, features, target=target)
    cols = [c for c in ['ID', 'Season', 'TeamA', 'TeamB', *features,
                        *(['Target'] if target else [])] if c in tabela]
    return sha256(pd.util.hash_pandas_object(tabela[cols], index=False).values.tobytes()
                  + str(list(zip(cols, tabela[cols].dtypes.astype(str)))).encode()).hexdigest()


def validar_historico(dados, features):
    if set(dados) != {'M', 'W'}:
        raise ValueError('Informe M e W separados.')
    ids = []
    for tabela in dados.values():
        h2ob._validar_tabela(tabela, features, target=True)
        chaves = tabela[['Season', 'TeamA', 'TeamB']]
        if not np.isfinite(chaves.to_numpy(dtype=float)).all() or not (chaves % 1 == 0).all().all():
            raise ValueError('Identificadores devem ser inteiros e finitos.')
        if not tabela.Season.lt(2026).all() or not tabela.TeamA.lt(tabela.TeamB).all():
            raise ValueError('Histórico deve preceder 2026 e usar TeamA < TeamB.')
        if {'ScoreA', 'ScoreB'} <= set(tabela):
            np.testing.assert_allclose(tabela.Target, tabela.ScoreA - tabela.ScoreB)
        ids.append(set(tabela.TeamA) | set(tabela.TeamB))
    if not ids[0].isdisjoint(ids[1]):
        raise ValueError('Categorias compartilham TeamIDs.')


def detectar_tabpfn(*, exigir_gpu=True):
    """Verifica CUDA e API sem baixar pesos ou ajustar um modelo."""
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    assinatura = inspect.signature(TabPFNRegressor.__init__).parameters
    for nome in ['n_estimators', 'random_state', 'device', 'ignore_pretraining_limits']:
        if nome not in assinatura:
            raise RuntimeError(f'API TabPFN incompatível: falta {nome}.')
    for nome in ['V3_5', 'V3_5_FAST']:
        if not hasattr(ModelVersion, nome):
            raise RuntimeError(f'Instale requirements-tabpfn.txt: versão {nome} indisponível.')
    if not hasattr(TabPFNRegressor, 'create_default_for_version'):
        raise RuntimeError('API de seleção da versão indisponível.')
    cuda = torch.cuda.is_available()
    if exigir_gpu and not cuda:
        raise RuntimeError('CUDA indisponível. Instale PyTorch CUDA no kernel; não houve fallback para CPU.')
    return {'device': 'cuda:0' if cuda else 'cpu', 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if cuda else None,
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None,
            'versoes': versoes()}


def candidatos_tabpfn(categoria, *, device='cuda:0', seed=42):
    if categoria not in {'M', 'W'}:
        raise ValueError('Categoria inválida.')
    # Ordem fixa prioriza configurações menores; reduções não consultam scores.
    return [{'Candidato': f'{categoria}_tabpfn_{v}_{n}', 'Categoria': categoria,
             'Abordagem': 'tabpfn', 'Familia': v,
             'Parametros': {'version': v, 'n_estimators': n, 'random_state': seed,
                            'device': device, 'ignore_pretraining_limits': False}}
            for n in [4, 8, 16, 32] for v in ['V3_5_FAST', 'V3_5']]


def _classe_h2o(familia):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    from h2o.estimators.random_forest import H2ORandomForestEstimator
    from h2o.estimators.xgboost import H2OXGBoostEstimator
    return {'glm': H2OGeneralizedLinearEstimator, 'gbm': H2OGradientBoostingEstimator,
            'drf': H2ORandomForestEstimator, 'xgboost': H2OXGBoostEstimator}[familia]


def extrair_configuracao(modelo, categoria, posicao, rmse):
    """Congela parâmetros efetivos e complexidade escolhida antes de 2023.

    Remove referências aos frames/folds e desliga nova busca/early stopping.
    Árvores usam o número efetivamente construído; GLM usa alpha/lambda vencedores.
    Parâmetros desconhecidos ou referências não portáveis causam falha explícita.
    """
    familia = modelo.algo
    classe = _classe_h2o(familia)
    efetivos = _parametros_simples(modelo.actual_params)
    excluir = {'model_id', 'training_frame', 'validation_frame', 'response_column',
               'ignored_columns', 'fold_column', 'fold_assignment', 'nfolds',
               'keep_cross_validation_models', 'keep_cross_validation_predictions',
               'keep_cross_validation_fold_assignment', 'checkpoint', 'export_checkpoints_dir'}
    parametros = {('lambda_' if k == 'lambda' else k): v for k, v in efetivos.items() if k not in excluir}
    for k in ['offset_column', 'weights_column', 'plug_values', 'beta_constraints',
              'linear_constraints', 'custom_metric_func', 'pretrained_autoencoder']:
        if parametros.get(k) is not None:
            raise ValueError(f'Parâmetro não portável: {k}.')
    ajustes = {'stopping_rounds': 0, 'max_runtime_secs': 0}
    output = modelo._model_json['output']
    if familia == 'glm':
        if output.get('lambda_best') is None or output.get('alpha_best') is None:
            raise ValueError('GLM sem alpha/lambda efetivos para reconstrução.')
        ajustes.update(lambda_=[float(output['lambda_best'])], alpha=[float(output['alpha_best'])],
                       lambda_search=False, early_stopping=False)
    else:
        resumo = output['model_summary'].as_data_frame()
        if 'number_of_trees' not in resumo or resumo.empty:
            raise ValueError('Número efetivo de árvores indisponível.')
        ajustes['ntrees'] = int(resumo['number_of_trees'].iloc[0])
    parametros.update(ajustes)
    permitidos = set(inspect.signature(classe.__init__).parameters) - {'self'}
    # O REST inclui alguns campos nulos que não existem no construtor da família
    # (por exemplo, offset_column em DRF). Nulos não alteram o estimador.
    nulos_omitidos = [k for k, v in parametros.items() if k not in permitidos and v is None]
    for k in nulos_omitidos:
        del parametros[k]
    if set(parametros) - permitidos:
        raise ValueError(f'Parâmetros incompatíveis: {set(parametros) - permitidos}')
    if any(isinstance(v, dict) for v in parametros.values()):
        raise ValueError('Configuração contém referência estruturada não portável.')
    reconstruido = classe(**parametros)
    for k, v in parametros.items():
        salvo = reconstruido._parms.get('lambda' if k == 'lambda_' else k)
        if salvo != v:
            raise ValueError(f'Construtor alterou {k}: {v!r} -> {salvo!r}')
    ident = identidade({'familia': familia, 'parametros': parametros})[:16]
    return {'Candidato': f'{categoria}_h2o_{ident}', 'Categoria': categoria, 'Abordagem': 'h2o',
            'Familia': familia, 'Parametros': parametros, 'ParametrosOriginais': efetivos,
            'ParametrosNulosOmitidos': nulos_omitidos,
            'AjustesReconstrucao': ajustes, 'ModeloOrigem': modelo.model_id,
            'PosicaoLeaderboard': int(posicao), 'RMSEInterno': float(rmse)}


def _snapshot_modelo(modelo, posicao, rmse):
    output = modelo._model_json['output']
    return {'algo': modelo.algo, 'model_id': modelo.model_id,
            'actual_params': _parametros_simples(modelo.actual_params),
            'alpha_best': output.get('alpha_best'), 'lambda_best': output.get('lambda_best'),
            'number_of_trees': (None if modelo.algo == 'glm' else
                int(output['model_summary'].as_data_frame()['number_of_trees'].iloc[0])),
            'posicao': posicao, 'rmse': rmse}


def _extrair_snapshots(snapshots, categoria):
    candidatos, falhas = [], []
    for snapshot in snapshots:
        output = {'alpha_best': snapshot['alpha_best'], 'lambda_best': snapshot['lambda_best'],
                  'model_summary': SimpleNamespace(as_data_frame=lambda s=snapshot:
                      pd.DataFrame({'number_of_trees': [s['number_of_trees']]}))}
        modelo = SimpleNamespace(algo=snapshot['algo'], model_id=snapshot['model_id'],
                                 actual_params=snapshot['actual_params'], _model_json={'output': output})
        try:
            candidato = extrair_configuracao(modelo, categoria, snapshot['posicao'], snapshot['rmse'])
            if candidato['Candidato'] in {c['Candidato'] for c in candidatos}:
                raise ValueError('Configuração efetiva duplicada; não substituída.')
            candidatos.append(candidato)
        except (ValueError, TypeError, KeyError) as exc:
            falhas.append({'Modelo': modelo.model_id, 'Erro': str(exc)})
    return candidatos, falhas


def descobrir_h2o(tabela, categoria, *, features, pasta, max_models=100, top_n=20,
                  seed=42, n_folds=5, exclude_algos=('DeepLearning',)):
    treino = tabela.loc[tabela.Season < 2023].copy()
    config = {'revisao': REVISAO, 'dados': assinatura_dados(treino, features), 'features': list(features),
              'categoria': categoria, 'max_models': max_models, 'top_n': top_n,
              'seed': seed, 'folds': n_folds, 'exclude': list(exclude_algos), 'versoes': versoes()}
    path = Path(pasta) / f'descoberta_{categoria}_{identidade(config)[:16]}.json'
    if path.exists():
        salvo = json.loads(path.read_text(encoding='utf-8'))
        if salvo.get('snapshots') and salvo['falhas']:
            salvo['candidatos'], salvo['falhas'] = _extrair_snapshots(salvo['snapshots'], categoria)
            salvar_json(path, salvo)
        bug_antigo = (not salvo.get('snapshots') and not salvo['candidatos'] and bool(salvo['falhas']) and
                      all('H2OResponse.__new__()' in f['Erro'] for f in salvo['falhas']))
        if not bug_antigo:
            return salvo
        salvar_json(path.with_suffix('.falha_extracao.json'), salvo)
        print(f'{categoria}: checkpoint antigo sem configurações recuperáveis; refazendo a busca.', flush=True)
    inicio = time.perf_counter()
    busca = h2ob.treinar_automl(treino, features=features, ano_previsao=2023, seed=seed,
        max_models=max_models, n_folds=n_folds, exclude_algos=exclude_algos, nome=f'exp11_{categoria}')
    checkpoint_salvo = False
    try:
        leaderboard = busca['leaderboard'].sort_values(['rmse', 'model_id'], kind='stable')
        snapshots, ensembles = [], []
        modelos_base = 0
        for posicao, row in enumerate(leaderboard.to_dict('records'), 1):
            modelo = h2ob._h2o().get_model(row['model_id'])
            if modelo.algo == 'stackedensemble':
                ensembles.append(row['model_id'])
                continue
            if modelos_base >= top_n:
                continue
            modelos_base += 1
            snapshots.append(_snapshot_modelo(modelo, posicao, row['rmse']))
        candidatos, falhas = _extrair_snapshots(snapshots, categoria)
        resultado = {'config': config, 'candidatos': candidatos, 'falhas': falhas, 'snapshots': snapshots,
                     'ensembles_excluidos': ensembles, 'leaderboard': leaderboard.to_dict('records'),
                     'segundos': time.perf_counter() - inicio}
        resultado = _normalizar_relatorio_descoberta(resultado)
        salvar_json(path, resultado)
        checkpoint_salvo = True
        return resultado
    finally:
        if checkpoint_salvo:
            h2ob.limpar_execucao(busca)
        else:
            print('Checkpoint não salvo: modelos mantidos no cluster para recuperação.', flush=True)


@contextmanager
def modelo_ajustado(treino, candidato, features, ano):
    h2ob._validar_tabela(treino, features, target=True)
    if not treino.Season.lt(ano).all() or treino.Season.ge(2026).any():
        raise ValueError('Treino contém o ano previsto/futuro.')
    parametros = deepcopy(candidato['Parametros'])
    if candidato['Abordagem'] == 'h2o':
        h2o = h2ob._h2o()
        antes = h2ob._chaves_h2o()
        try:
            modelo = _classe_h2o(candidato['Familia'])(**parametros,
                model_id=f'exp11_{uuid4().hex}', nfolds=0)
            frame = h2o.H2OFrame(treino[[*features, 'Target']].astype(float),
                               column_types={c: 'real' for c in [*features, 'Target']})
            with warnings.catch_warnings():
                configurar_avisos_h2o()
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                    message=r'Sample rate is 100% and no validation dataset and no cross-validation\.',
                    module=r'h2o\.estimators\.estimator_base')
                modelo.train(x=list(features), y='Target', training_frame=frame)
            if modelo._model_json['output']['model_category'] != 'Regression':
                raise ValueError('Modelo H2O não é regressor.')
            yield h2ob.AdaptadorH2O(modelo, features)
        finally:
            h2ob.limpar_execucao({'chaves_h2o': h2ob._chaves_h2o() - antes})
    elif candidato['Abordagem'] == 'tabpfn':
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        if parametros['device'].startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA indisponível; não substituímos por CPU.')
        modelo = None
        try:
            modelo = TabPFNRegressor.create_default_for_version(
                getattr(ModelVersion, parametros.pop('version')), **parametros)
            modelo.fit(treino[list(features)], treino.Target)
            yield modelo
        finally:
            del modelo
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise ValueError('Abordagem desconhecida.')


@contextmanager
def medir_recursos(cuda=False):
    """RSS do processo Python amostrado; pico CUDA alocado pelo PyTorch (não VRAM total)."""
    import psutil
    processo, parar = psutil.Process(), threading.Event()
    resultado = {'rss_pico_mb': processo.memory_info().rss / 2**20}
    def monitorar():
        while not parar.wait(.05):
            resultado['rss_pico_mb'] = max(resultado['rss_pico_mb'], processo.memory_info().rss / 2**20)
    thread = threading.Thread(target=monitorar, daemon=True)
    if cuda:
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    thread.start()
    try:
        yield resultado
    finally:
        if cuda:
            torch.cuda.synchronize()
            resultado['cuda_pico_mb'] = torch.cuda.max_memory_allocated() / 2**20
        resultado['segundos'] = time.perf_counter() - inicio
        parar.set()
        thread.join()


def avaliar_corte(tabela, candidato, ano, *, features, pasta, repetir_falhas=False):
    if ano not in ANOS:
        raise ValueError('Use 2023, 2024 ou 2025.')
    treino, validacao = tabela.loc[tabela.Season < ano], tabela.loc[tabela.Season == ano]
    config = {'revisao': REVISAO, 'candidato': candidato, 'ano': ano, 'features': list(features),
              'treino': assinatura_dados(treino, features),
              'validacao': assinatura_dados(validacao, features), 'versoes': versoes()}
    path = Path(pasta) / f'corte_{candidato["Candidato"]}_{ano}_{identidade(config)[:16]}.json'
    if path.exists():
        salvo = json.loads(path.read_text(encoding='utf-8'))
        if salvo['status'] == 'ok' or not repetir_falhas:
            return salvo
    resultado = {'config': config, 'Categoria': candidato['Categoria'], 'Abordagem': candidato['Abordagem'],
                 'Candidato': candidato['Candidato'], 'Season': ano}
    recursos = {}
    try:
        cuda = candidato['Abordagem'] == 'tabpfn' and candidato['Parametros']['device'].startswith('cuda')
        with medir_recursos(cuda=cuda) as recursos:
            with modelo_ajustado(treino, candidato, features, ano) as modelo:
                pred = np.asarray(modelo.predict(validacao[list(features)]), dtype=float)
                if pred.shape != (len(validacao),) or not np.isfinite(pred).all():
                    raise ValueError('Previsões inválidas.')
        resultado.update(status='ok', RMSE=float(np.sqrt(np.mean((validacao.Target.to_numpy() - pred)**2))),
            LB=float(logistic_brier(validacao.Target, pred)),
            previsoes=validacao[['Season', 'TeamA', 'TeamB', 'Target']].assign(Pred=pred).to_dict('records'),
            JogosTreino=len(treino), UltimoAnoTreino=int(treino.Season.max()))
    except Exception as exc:
        # Cancelamento do usuário deve parar imediatamente, sem virar candidato inválido.
        if type(exc).__name__ == 'H2OJobCancelled':
            raise
        resultado.update(status='falha', erro=f'{type(exc).__name__}: {exc}')
    resultado['recursos'] = recursos
    salvar_json(path, resultado)
    return resultado


def avaliar_banco(dados, candidatos, *, features, pasta, orcamento_segundos=None, repetir_falhas=False):
    """Orçamento entre candidatos completos, cumulativo incluindo checkpoints.

    Não interrompe fit em andamento nem reduz candidatos com base nos scores.
    Ao acabar o orçamento, preserva resultados e registra os candidatos pendentes.
    """
    validar_historico(dados, features)
    if orcamento_segundos is not None and (not np.isfinite(orcamento_segundos) or orcamento_segundos <= 0):
        raise ValueError('Orçamento deve ser positivo ou None.')
    if len({c['Candidato'] for c in candidatos}) != len(candidatos):
        raise ValueError('Candidatos duplicados.')
    linhas, pendentes, gasto = [], [], 0.
    for candidato in candidatos:
        if orcamento_segundos is not None and gasto >= orcamento_segundos:
            pendentes.append(candidato['Candidato'])
            continue
        for ano in ANOS:
            resultado = avaliar_corte(dados[candidato['Categoria']], candidato, ano,
                features=features, pasta=pasta, repetir_falhas=repetir_falhas)
            linhas.append(resultado)
            gasto += resultado['recursos'].get('segundos', 0.)
        print(f"{candidato['Candidato']}: três cortes registrados.", flush=True)
    resumo = {'avaliacoes': linhas, 'pendentes': pendentes, 'segundos_com_checkpoints': gasto,
              'orcamento_segundos': orcamento_segundos}
    salvar_json(Path(pasta) / 'ultima_avaliacao.json', resumo)
    return resumo


def rankings(avaliacoes, referencias):
    if set(referencias.Season) != set(ANOS) or referencias.Season.duplicated().any():
        raise ValueError('Pesos devem conter exatamente 2023–2025.')
    pesos = referencias.set_index('Season').Peso
    if not np.isfinite(pesos).all() or (pesos < 0).any() or not np.isclose(pesos.sum(), 1):
        raise ValueError('Pesos inválidos.')
    tabela = pd.DataFrame([{k: r.get(k) for k in ['Categoria', 'Abordagem', 'Candidato', 'Season', 'status', 'RMSE', 'LB']}
                           for r in avaliacoes])
    if tabela.empty:
        raise ValueError('Nenhum candidato avaliado.')
    if tabela.duplicated(['Categoria', 'Abordagem', 'Candidato', 'Season']).any():
        raise ValueError('Cortes duplicados.')
    linhas = []
    for (cat, abordagem, candidato), grupo in tabela.groupby(['Categoria', 'Abordagem', 'Candidato']):
        if set(grupo.Season) != set(ANOS) or not grupo.status.eq('ok').all():
            continue
        valores = grupo[['RMSE', 'LB']].to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores < 0).any() or grupo.LB.gt(1).any():
            raise ValueError('Métricas inválidas.')
        linha = {'Categoria': cat, 'Abordagem': abordagem, 'Candidato': candidato}
        for metrica in ['RMSE', 'LB']:
            linha[metrica + '_pond'] = float(sum(r[metrica] * pesos.loc[r.Season] for _, r in grupo.iterrows()))
            linha.update({f'{metrica}_{int(r.Season)}': r[metrica] for _, r in grupo.iterrows()})
        linhas.append(linha)
    if not linhas:
        raise ValueError('Nenhum candidato completou os três anos.')
    scores = pd.DataFrame(linhas)
    tabelas = []
    for metrica in ['RMSE', 'LB']:
        ranking = scores.sort_values(['Abordagem', 'Categoria', metrica + '_pond', 'Candidato'], kind='stable').copy()
        ranking['Metrica'] = metrica.lower()
        ranking['Posicao'] = ranking.groupby(['Abordagem', 'Categoria']).cumcount() + 1
        tabelas.append(ranking)
    return scores, pd.concat(tabelas, ignore_index=True)


def gerar_submissions(dados, candidatos, ranking, confrontos, sample, *, features, pasta, destino):
    """Reajusta uma vez cada vencedor único; salva previsões retomáveis, não modelos."""
    validar_historico(dados, features)
    top = ranking.loc[ranking.Posicao <= 3].copy()
    catalogo = {c['Candidato']: c for c in candidatos}
    preds, falhas, tempos = {}, [], []
    for ident in top.Candidato.unique():
        c = catalogo[ident]
        cat = c['Categoria']
        mask = confrontos.Categoria.eq({'M': 'Masculino', 'W': 'Feminino'}[cat]) & confrontos.TemDuasSeeds
        pares = confrontos.loc[mask]
        config = {'revisao': REVISAO, 'candidato': c, 'features': list(features),
                  'treino': assinatura_dados(dados[cat], features),
                  'sample': assinatura_dados(pares, features, target=False), 'versoes': versoes()}
        path = Path(pasta) / f'final_{ident}_{identidade(config)[:16]}.json'
        try:
            if path.exists():
                salvo = json.loads(path.read_text(encoding='utf-8'))
            else:
                inicio = time.perf_counter()
                with modelo_ajustado(dados[cat], c, features, 2026) as modelo:
                    pred = np.asarray(modelo.predict(pares[list(features)]), dtype=float)
                if pred.shape != (len(pares),) or not np.isfinite(pred).all():
                    raise ValueError('Previsões finais inválidas.')
                salvo = {'config': config, 'ID': pares.ID.tolist(), 'Pred': pred.tolist(),
                         'segundos': time.perf_counter() - inicio}
                salvar_json(path, salvo)
            if salvo['ID'] != pares.ID.tolist() or not np.isfinite(salvo['Pred']).all():
                raise ValueError('Checkpoint final inválido.')
            preds[ident] = (mask, salvo['Pred'])
            tempos.append({'Candidato': ident, 'Segundos': salvo['segundos']})
        except Exception as exc:
            if type(exc).__name__ == 'H2OJobCancelled':
                raise
            falhas.append({'Candidato': ident, 'Erro': f'{type(exc).__name__}: {exc}'})
            salvar_json(Path(pasta) / 'falhas_finais.json', falhas)
    resumos, hashes = [], {}
    for (abordagem, metrica, posicao), grupo in top.groupby(['Abordagem', 'Metrica', 'Posicao']):
        if set(grupo.Categoria) != {'M', 'W'} or len(grupo) != 2 or not set(grupo.Candidato) <= preds.keys():
            falhas.append({'Grupo': [abordagem, metrica, int(posicao)], 'Erro': 'Par M/W incompleto; CSV não gerado.'})
            continue
        saida = confrontos[['ID']].copy()
        saida['Pred'] = 0.
        for ident in grupo.Candidato:
            mask, pred = preds[ident]
            saida.loc[mask, 'Pred'] = pred
        nome = f'submission_exp11_{abordagem}_{metrica}_top{int(posicao)}_2026.csv'
        path = salvar_submission(saida, sample, Path(destino) / nome)
        digest = sha256(saida.Pred.to_numpy(dtype=float).tobytes()).hexdigest()
        resumos.append({'Arquivo': path.name, 'Abordagem': abordagem, 'Metrica': metrica,
            'Posicao': int(posicao), 'ModeloM': grupo.set_index('Categoria').loc['M', 'Candidato'],
            'ModeloW': grupo.set_index('Categoria').loc['W', 'Candidato'],
            'Linhas': len(saida), 'IdenticaA': hashes.get(digest)})
        hashes.setdefault(digest, path.name)
    resultado = {'submissions': resumos, 'falhas': falhas, 'tempos_finais': tempos}
    salvar_json(Path(pasta) / 'submissions.json', resultado)
    return resultado
