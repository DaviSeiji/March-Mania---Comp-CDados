# Experimento 11 — execução local

Notebook: [exp11_automl_tabpfn.ipynb](../notebooks/features_incrementais/exp11_automl_tabpfn.ipynb).

O notebook compara os mesmos 14 atributos do Exp. 8. Há uma busca H2O de até
100 modelos-base por categoria, limitada ao histórico anterior a 2023, com cinco
folds agrupados por temporada e ordenação interna por RMSE. Os 20 primeiros
modelos-base são reconstruídos com configurações fixas em 2023, 2024 e 2025.
Não repetimos AutoML nem CV interna nesses reajustes externos. Stacked Ensembles
ficam fora: seus modelos-base dependentes não são tratados como hiperparâmetros.

TabPFN usa oito configurações por categoria: 3.5/Fast × 4/8/16/32 estimadores,
seed 42. Os estimadores são ensembles de inferência, não treinos de redes do zero.
Não há amostragem dos jogos nem `ignore_pretraining_limits=True`.

## Instalação e GPU

Na inspeção de 29/10/2026, este PC tinha RTX 3060 de 12 GB, driver 576.02,
Python 3.13, H2O 3.46.0.12 e PyTorch 2.13.0+cpu. TabPFN não estava instalado.
Nenhum pacote foi instalado ou alterado pela implementação.

Execute na raiz do repositório, usando o mesmo Python do kernel:

```powershell
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-tabpfn.txt
python -m pip install -r requirements-h2o.txt
python -m pip check
```

O primeiro comando troca o build CPU por um build CUDA 12.8 com wheel publicado
para Windows/Python 3.13. Não é necessário instalar torchvision/torchaudio para
este experimento. Se já os utilizar em outro projeto, suas versões precisam
acompanhar a versão de torch; confira os conflitos indicados por `pip check`.
A instalação do TabPFN está separada da do H2O nos arquivos de requisitos.
Este notebook espera ambos disponíveis no mesmo kernel; resolva os conflitos
indicados pelo pip antes de rodar. Não foi criada uma `.venv`.

Reinicie o kernel depois de instalar. Confira:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

`torch.cuda.is_available()` deve retornar `True`. O notebook usa `device='cuda:0'`
e falha explicitamente se CUDA estiver indisponível, sem fallback silencioso.
O piloto mede o consumo; 12 GB não garante que todos os ensembles caibam na VRAM.
Falhas de memória são registradas sem substituir modelos ou reduzir o dataset.

Na primeira utilização, TabPFN pode solicitar login/aceite da licença da Prior Labs
e baixar os pesos. Complete esse passo conforme a mensagem da biblioteca. Em
ambiente sem navegador, configure `TABPFN_TOKEN` localmente; não salve tokens no
notebook ou no Git. Os pesos ficam no cache padrão do usuário.

Se aparecer `TabPFNLicenseError`, execute a nova célula **Autenticação TabPFN**:
faça login e aceite a licença no site, copie a API Key da página Account e
substitua `COLE_SUA_CHAVE_AQUI` no arquivo `.env.tabpfn` da raiz pela sua chave,
mantendo o formato `TABPFN_TOKEN=sua_chave`. Salve e execute a célula de autenticação:
ela lê o arquivo sem exibir seu conteúdo. O arquivo contém texto local e está
ignorado pelo Git; não é criptografado. Não é necessário reiniciar o kernel ou
refazer H2O. Reexecute o piloto; ele tenta novamente falhas anteriores, mas
continua reutilizando um piloto já concluído com sucesso.

A célula de autenticação também consulta a validade da chave e o aceite da
licença específica do **TabPFN 3.5** (`tabpfn-3-5-license-v1.0`), sem baixar pesos.
Criar a API Key não aceita automaticamente a licença. Se a chave for válida e
a licença estiver pendente, aceite-a em **Licenses** na mesma conta e reexecute
a autenticação antes do piloto. O diagnóstico distingue essa situação de chave
inválida/expirada e de indisponibilidade da conexão.

Avisos de métrica/tolerância de parada ignoradas com `stopping_rounds=0` são
filtrados especificamente: o early stopping está desativado de propósito nos
reajustes. Outros avisos e erros continuam visíveis.

H2O continua na CPU neste Windows: seu XGBoost não está disponível nessa plataforma.
Em ambientes suportados, a detecção existente permite XGBoost e seu backend `auto`
pode escolher GPU. Não trocamos o XGBoost H2O por outra biblioteca neste experimento.

Fontes oficiais consultadas:

- [API e autenticação TabPFN](https://github.com/PriorLabs/TabPFN).
- [Instalações PyTorch por versão](https://pytorch.org/get-started/previous-versions/).
- [Wheels CUDA 12.8](https://download.pytorch.org/whl/cu128/torch/).
- [Suporte GPU e limitações XGBoost H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html).

## Execução e retomada

Execute as células em ordem. Configurações principais ficam no início do notebook:
`MAX_MODELS_H2O=100`, `TOP_BASE_H2O=20`, `N_FOLDS=5`, `SEED=42`.
O piloto TabPFN usa Fast/4, masculino, corte completo de 2023, e é reutilizado
na avaliação. Seu tempo e pico CUDA/RSS são mostrados antes da busca TabPFN.
RSS mede somente o processo Python; não representa a memória da JVM do H2O.

`ORCAMENTO_TABPFN_SEGUNDOS_POR_CATEGORIA=None` usa como orçamento o tempo medido
de descoberta + validação H2O da categoria, com mínimo de uma hora. Pode-se informar
um número explícito de segundos. O orçamento inclui tempos dos checkpoints
reutilizados, é verificado entre candidatos e pode ser excedido pelo último
candidato. Não é um timeout rígido e não interrompe uma operação CUDA em andamento.
O tempo de treinamento final/submission fica fora desse orçamento de avaliação.

A ordem dos candidatos TabPFN é fixa, priorizando n_estimators menores, sem consultar
scores. Pendentes e falhas são mostrados. Se menos de três configurações por
categoria completarem todos os cortes, o notebook para antes das submissions:
revise a causa e aumente o orçamento se necessário. Não escolhemos candidatos
pela média dos anos restantes quando um dos cortes falha.

`checkpoints/exp11/` é ignorado pelo Git e contém JSONs de:

- buscas H2O concluídas, leaderboard, parâmetros originais/efetivos e falhas;
- snapshots dos parâmetros dos modelos-base, para corrigir falhas de extração
  sem repetir a busca;
- cada candidato/ano, métricas, previsões, tempos e memória;
- pesos, rankings e registro de versões/configuração;
- previsões finais por vencedor único e resumo das submissions.

Não salvamos binários dos estimadores. Os arquivos são gravados atomicamente.
Dados, parâmetros, features e versões entram na chave do cache; alterações
nesses itens geram novas avaliações. A preparação do notebook é reexecutada,
mas buscas/cortes completos são reutilizados. Uma busca H2O interrompida antes
de terminar precisa ser refeita. Para repetir falhas já salvas após corrigir
a causa, ajuste `REPETIR_FALHAS=True`.

Checkpoints antigos vazios por erro `H2OResponse.__new__`, sem snapshots, são
arquivados com sufixo `.falha_extracao.json` e a busca é refeita ao executar a
célula de descoberta. Novos checkpoints permitem reprocessar a extração dos
mesmos candidatos sem retreinamento. Se a gravação falhar, os modelos permanecem
no cluster para recuperação; a limpeza só ocorre após salvar o checkpoint.

## Seleção e saídas

Os perfis pré-torneio e a padronização histórica vêm do Exp. 10. Usamos os anos
fixos 2023–2025 e pesos inversos à distância, somando 1, iguais para os dois backends.
As métricas externas são calculadas com os mesmos jogos em cada categoria.
Os empates são resolvidos por ID do candidato em ordem lexicográfica.

Para congelar H2O: removemos referências aos frames/folds, fixamos o número real
de árvores ou alpha/lambda vencedores do GLM e desativamos nova busca/early stopping.
Guardamos os parâmetros originais e essas alterações de reconstrução. Parâmetros
incompatíveis ou não portáveis causam falha registrada, sem substituição silenciosa.

São quatro rankings por categoria (H2O/TabPFN × RMSE/LB ponderados). Vencedores
únicos são reajustados até 2025 e reutilizados entre rankings. Até 12 arquivos
`submissions/submission_exp11_{h2o|tabpfn}_{rmse|lb}_top{1|2|3}_2026.csv`
combinam as mesmas posições M/W. Pred contém margem bruta, sem sigmoid ou clipping.
Pares sem duas seeds mantêm fallback 0. IDs, ordem, quantidade e finitude são
verificados; previsões idênticas são identificadas. Uma falha final impede apenas
os CSVs que dependem daquele candidato; não há troca pelo quarto colocado.

Referência Kaggle Exp. 8: 0.04783. Não usar esse score para selecionar configurações.
O score histórico de seleção não é um teste independente, nem deve ser comparado
diretamente às médias de outros conjuntos de anos.

Testes leves (sem ajustar TabPFN ou executar AutoML):

```powershell
python -B -m unittest discover -s tests -p test_exp11.py
```

Verificação feita na implementação: 73 testes automatizados passaram, incluindo
oito do Exp. 11. Todas as células rodaram com os dados reais e backends simulados,
gerando 12 CSVs apenas em diretório temporário. GLM, GBM e DRF foram reconstruídos
e reajustados em um teste real de 80 linhas (árvores com três árvores cada).
O servidor H2O temporário foi encerrado. A busca AutoML completa e a inferência
TabPFN/GPU não foram executadas; TabPFN ainda requer instalação e aceite da licença.
