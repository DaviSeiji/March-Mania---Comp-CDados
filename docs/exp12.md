# Experimento 12 — eficiência, stacking e TabPFN

[Notebook](../notebooks/features_incrementais/exp12_eficiencia_automl_tabpfn.ipynb).
Os arquivos dos experimentos anteriores não são modificados. O novo fluxo importa
as validações, versões, autenticação, recursos, métricas e serialização do Exp. 11.

## Features e origem dos dados

Conjunto principal: as 14 diferenças do Exp. 11 mais oito diferenças A − B:
OffensiveRating, DefensiveRating, NetRating, PossessionsPerGame,
EffectiveFGPercentage, TurnoverRate, OffensiveReboundRate e FreeThrowRate.
Total: 22 features. `SeedDif` conserva a grafia dos experimentos anteriores.

`src/efficiency_features.py` implementa:

- `calcular_eficiencia`: duas participações por jogo (W/L), agrupadas por
  Season/TeamID; percentuais calculados dos totais acumulados.
- `adicionar_eficiencia`: junção por temporada/equipe e diferenças A − B. Rejeita
  duplicidades e valores inválidos. Ausências explícitas somente em pares do
  sample sem duas seeds, conforme o fallback do projeto.

Fórmulas, com totais sazonais e G = jogos detalhados disponíveis:

| Estatística | Definição |
|---|---|
| Poss próprias | FGA − OR + TO + 0.44 × FTA |
| Poss adversárias | Mesma fórmula no lado adversário de cada partida |
| OffensiveRating | 100 × pontos feitos / Poss próprias |
| DefensiveRating | 100 × pontos sofridos / Poss adversárias |
| NetRating | OffensiveRating − DefensiveRating |
| PossessionsPerGame | (Poss próprias + Poss adversárias) / (2 × G) |
| EffectiveFGPercentage | (FGM + 0.5 × FGM3) / FGA |
| TurnoverRate | TO / (FGA + 0.44 × FTA + TO) |
| OffensiveReboundRate | OR / (OR + DR adversários) |
| FreeThrowRate | FTA / FGA |

O fator 0.44 aproxima posses consumidas por lances livres, incluindo situações
que não encerram uma posse ou vêm em séries. Não é uma contagem exata de posses.
Não há ajuste por prorrogação; NetRating é matematicamente redundante com os dois
ratings, mas é mantido conforme o conjunto solicitado. FTr pode exceder 1; não
é taxa de acerto. Denominadores sazonais zero ou posses não positivas geram erro.

`src/exp12.py:preparar_dados` usa exclusivamente RegularSeasonCompactResults e
RegularSeasonDetailedResults para features, verifica consistência dos placares,
datas e ausência de sobreposição com NCAA. O torneio fornece os alvos e o limite
exclusivo DayNum de cada temporada, nunca os totais das features. Para 2026 usa
DayNum < 134. Preserva M: 2003+, W: 2010+, Target = ScoreA − ScoreB e TeamA < TeamB.
O recorte deve ser respeitado na entrada, sem descarte silencioso de partidas.
O notebook mostra auditoria, estatísticas, NaNs/infinitos e correlação por categoria.

## H2O: descoberta e Stacked Ensembles

Uma busca por categoria e conjunto de features, somente Season < 2023:
max_models=100, seed=42, cinco folds internos agrupados por Season, ordenação RMSE.
DeepLearning é excluído; XGBoost é excluído quando indisponível no backend Windows.
Não há cotas manuais por algoritmo e o AutoML não é repetido nos cortes externos.

`src/exp12_h2o.py:descobrir_h2o` extrai **todos os modelos-base elegíveis** do
leaderboard, com `max_models=100` solicitado ao motor nativo. Configurações efetivas duplicadas são
unificadas; suas identidades originais continuam resolvidas para as dependências
dos ensembles. Campos H2OResponse são convertidos em valores Python simples;
NaNs de métricas auxiliares como RMSLE viram null, sem imputar o RMSE de seleção.
Snapshots são persistidos antes da extração, permitindo corrigir erros sem refazer
o AutoML. Alpha/lambda e número efetivo de árvores são fixados conforme Exp. 11.

Incluímos **duas receitas de ensemble por categoria**: a composição do melhor
BestOfFamily e do melhor AllModels do AutoML, escolhidos por RMSE interno antes
de 2023. Registramos todos os ensembles de origem, seus modelos-base, metalearner
e parâmetros. As receitas concorrem no ranking H2O junto aos modelos individuais.

Reconstrução equivalente, explicitamente diferente da identidade binária do AutoML:

1. Em cada corte, o frame contém somente temporadas anteriores ao ano previsto.
2. Um único mapa Season→FoldInterno, com a mesma ordem de linhas, serve a todas
   as bases. A CV interna é agrupada, não expansiva; somente o corte externo é
   estritamente temporal. Não são usados dados do ano de validação.
3. Bases necessárias ao stacking são treinadas com `fold_column='FoldInterno'`,
   `keep_cross_validation_predictions=True` e sem reter modelos CV individuais.
4. `H2OStackedEnsembleEstimator` usa as previsões OOF dessas bases. Metalearner
   fixo: GLM gaussiano, pesos não negativos, alpha=0, lambda=0, sem padronização,
   até 1000 iterações e sem CV adicional/tuning. O intercepto segue o padrão GLM.
   Não copiamos cegamente os parâmetros de um ensemble para um estimador simples.
5. A validação externa entra apenas na previsão e no cálculo das métricas.

`SessaoH2O` mantém um frame e reutiliza bases OOF entre ensembles e candidatos
individuais no mesmo ano. Bases fora do stacking são reajustadas sem CV e liberadas
após avaliação. Limpeza final remove somente os objetos da sessão. Não persistimos
binários das bases: uma sessão interrompida pode precisar reconstruir dependências,
mas métricas/previsões concluídas e a busca permanecem em checkpoint.

`LIMITE_BASES_ALLMODELS=20` restringe o AllModels às 20 configurações de menor
RMSE interno da **composição original**, antes da avaliação externa. `None` conserva
todas. Registramos a composição completa, a incluída e a excluída. BestOfFamily
preserva sua composição original. Dependências não reconstruíveis são falhas
explícitas; por padrão o notebook exige ambas as receitas antes de avançar.

**Quantidade solicitada:** `max_models=100` para modelos-base, mais duas receitas
de stacking. Isso não é uma regra de truncamento do catálogo: todos os modelos
válidos retornados pelo leaderboard são mantidos, inclusive se o motor exceder
o limite solicitado. O notebook apresenta as contagens efetivas e falhas.

### Auditoria das execuções existentes

Por categoria, os checkpoints atuais contêm **108 bases distintas no leaderboard
nativo**: 100 GBMs do grid, cinco GBMs iniciais, DRF, XRT (ambos `algo='drf'`) e GLM.
Os dois ensembles nativos elevam o leaderboard a 110 linhas. Não há IDs repetidos,
snapshots fora do leaderboard ou bases adicionais inseridas pelo nosso código.
As duas receitas de stacking concorrem com as 108 bases: **110 candidatos únicos
para avaliação externa**, não uma soma repetida de suas dependências.

A busca já utiliza `H2OAutoML` nativo com max_models=100, seed=42 e RMSE, sem
`modeling_plan`, `include_algos`, cotas ou alterações de exploração. A distribuição
desigual por família é a produzida pelo plano nativo e pelas exclusões do ambiente.
O excedente de oito bases está no próprio leaderboard, pela soma dos 100 modelos
do grid com os oito iniciais. A documentação prevê que max_models limite as bases;
estas execuções ultrapassaram esse limite. Não há event_log preservado para
determinar o detalhe interno dessa divergência; não a atribuímos a paralelismo
ou a um ajuste específico do motor sem evidência. Há relatos oficiais de
[max_models excedido no H2O](https://github.com/h2oai/h2o-3/issues/16376).
Não mudamos a busca, não reduzimos max_models nem descartamos modelos para forçar 100.

`auditar_descoberta` separa bases/ensembles nativos, bases externas ao leaderboard
usadas como dependências (zero nestes checkpoints), metalearners de origem (dois),
bases reutilizadas pelas receitas e candidatos únicos. Reajustes OOF e novos
metalearners são trabalhos auxiliares por corte, não configurações novas no
catálogo. Futuras buscas também guardam event_log e o plano executado pelo próprio
H2O para facilitar a investigação; o plano não é reinjetado como uma restrição.
`auditar_checkpoints` apresenta, separadamente, quantos candidatos já têm cortes
salvos e quantos completaram os três anos válidos, distinguindo bases e ensembles.
Um candidato catalogado não é apresentado como se já tivesse sido avaliado.

Referência: [Stacked Ensemble H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html).

## TabPFN: 100 configurações úteis por categoria

O gerador verifica primeiro a API instalada (assinatura de TabPFNRegressor e
campos de InferenceConfig). Espaço fixo, independente de scores:

- Versões: V3_5 e V3_5_FAST.
- n_estimators: 4, 8, 16, 32, 64.
- Cinco perfis: padrão; transformação do alvo apenas identidade; apenas safepower;
  deslocamento de features `rotate`; sem deslocamento de features.
- softmax_temperature: 1.0 e 0.7, aplicada à distribuição preditiva do regressor.

2 × 5 × 5 × 2 = **100 configurações estruturais por categoria**, 200 no principal.
Todas usam random_state=42; não são 100 seeds quase idênticas. O registro distingue
identidade estrutural e seed de repetição. O catálogo pode ser reduzido por
`MAX_CANDIDATOS_TABPFN`, preservando ordem fixa; não troca modelos por resultado.

Mantemos `ignore_pretraining_limits=False`, sem subsampling. Transformações são
ajustadas somente no treino de cada corte pelo TabPFN. `n_estimators` é sempre um
inteiro positivo explícito; não passamos `auto_scale_n_estimators` ao construtor.
GPU CUDA é obrigatória por padrão.
Os modelos são sequenciais, sem execuções concorrentes na RTX 3060 de 12 GB.

O warning obsoleto vinha do `auto_scale_n_estimators=False` antes passado em todas
as configurações. Na versão instalada, esse booleano só muda o comportamento
com `n_estimators='auto'`; com um inteiro, a função devolve o mesmo n, mas ainda
emite FutureWarning. A factory da versão aceita o inteiro explícito e não o troca
por `'auto'`. Removemos apenas o booleano inerte na instanciação.

Preservamos os IDs históricos e lemos também o fingerprint legado dos checkpoints
com esse booleano. O alias só é permitido com n inteiro positivo e todos os demais
dados, versões e hiperparâmetros idênticos. Não há migração das previsões nem
regravação dos cortes válidos. Mudanças reais em n, seed, features, dados, versões
ou outro hiperparâmetro continuam gerando um checkpoint diferente.

Por compatibilidade com versões que ainda emitam o mesmo aviso internamente,
o contexto de execução do TabPFN filtra somente a mensagem
`auto_scale_n_estimators is deprecated and will be removed in v9.` na categoria
FutureWarning e no módulo tabpfn. O filtro termina ao sair de fit/predict. Avisos
de CUDA, memória, cobertura de features, limites e problemas numéricos permanecem.

Pilotos por categoria/conjunto: Fast/4 e 3.5/64, perfil padrão/temperatura 1,
treino anterior a 2023 e validação 2023. Medem tempo, pico de RSS Python e pico
CUDA alocado (não VRAM total do driver). O RSS Python não representa a heap da
JVM do H2O; para H2O registramos também amostras da heap usada após os ajustes,
sem tratá-las como pico contínuo. Não se garante que 64 estimadores caibam em todos os cortes: falhas
ficam registradas. Uma falha do piloto é exibida sem reduzir o dataset.

Referência: [API TabPFNRegressor](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py).

## Validação, controle e comparação

Cada candidato recebe os mesmos jogos em 2023, 2024 e 2025, com Season < ano
no treino. RMSE e Logistic Brier oficiais são calculadas externamente, com tempos
de ajuste/inferência separados e recursos por corte. Modelos masculino e feminino
permanecem separados.

Perfil de similaridade: as 52 dimensões originais do Exp. 10/11, sem estatísticas
novas na distância. Scaler somente em perfis anteriores a 2026. Pesos inversos às
distâncias de 2023–2025, normalizados, iguais para backends e conjuntos.
RMSE_pond = Σ peso × RMSE e LB_pond = Σ peso × LB. Sem os três cortes válidos,
o candidato fica fora. Empate: ID em ordem lexicográfica. Usamos o ranking do
Exp. 11 para essa etapa compartilhada, sem duplicar a implementação.

`RODAR_CONTROLE_14_FEATURES=False` por padrão. Quando True, o mesmo fluxo é executado
com 14 features: mesmos jogos, seed, anos, pesos, max_models, catálogo TabPFN,
folds, limite AllModels e orçamento por categoria. Cada conjunto recebe sua
própria descoberta H2O antes de 2023, pois as features podem mudar o que o AutoML
descobre. A comparação controlada é entre protocolos equivalentes, sem alegar que
os algoritmos/configurações descobertos serão os mesmos.

As tabelas mostram conjunto, abordagem, família, RMSE/LB anuais e ponderados,
vencedores, bases versus stacking, tempos e cobertura efetiva. Se houver falhas
ou orçamento esgotado em apenas um conjunto, considere essa diferença de cobertura
antes de atribuir melhorias às features. O controle não gera submissions extras.

A comparação Exp. 11×12 recalcula os rankings antigos sobre os mesmos anos e
pesos atuais (leitura apenas). O Exp. 12 também amplia a busca e inclui stacking;
diferenças não medem isoladamente efeito de features. Histórico de seleção não
é teste independente. Kaggle Exp. 8 = 0.04783 é só referência, sem uso na seleção.

## Checkpoints e custo

`checkpoints/exp12/` fica ignorado pelo Git. Busca, snapshots, cortes com previsões,
erros, configuração, versões, folds, composições e previsões finais são retomáveis.
Os fingerprints incluem dados/features, versões e candidatos; cortes também
incluem n_folds e seed. Não reutilizamos resultados do Exp. 11 como se fossem
resultados das novas features. `REPETIR_FALHAS=True` refaz somente cortes que
falharam; cortes bem-sucedidos são reaproveitados.

Por conjunto, no catálogo máximo:

- Duas buscas AutoML com max_models=100 solicitado + ensembles, com cinco folds internos.
- Com as contagens observadas: 660 avaliações H2O externas ((108 + 2) × 2 × 3).
- Até 600 avaliações TabPFN (100 × 2 × 3).
- Bases de stacking acrescentam cinco ajustes CV por base/ano/categoria; com
  AllModels=20 e BestOfFamily em torno de quatro bases, seriam até cerca de 720
  ajustes CV adicionais na validação, além dos treinos principais já contados.
- Até oito configurações vencedoras únicas no reajuste final; ensembles ainda
  precisam ajustar suas bases com OOF. O controle quase duplica busca/validação.

O notebook projeta tempo TabPFN usando os pilotos leve/pesado e uma aproximação
linear em n_estimators, indicando quando não é possível estimar. Outros perfis,
anos maiores e downloads podem alterar bastante essa projeção. O H2O ampliado
pode custar várias vezes a avaliação das 20 bases do Exp. 11; não há estimativa
confiável em horas antes dos novos pilotos/treinos. Não foi feito benchmark completo.

Orçamento TabPFN padrão: **4 horas por categoria/conjunto**, verificado entre
candidatos completos, incluindo tempos de checkpoints reutilizados. Pode exceder
o limite por um candidato; não é timeout rígido. O treino final fica fora desse
orçamento. `None` percorre todos os candidatos. Aumentar o orçamento e reexecutar
retoma os pendentes. O H2O é limitado por contagem (max_models), sem limite de tempo.

## Instalação e execução

Não há novas dependências em relação ao Exp. 11, e nenhuma foi instalada pela
implementação. Use o mesmo kernel já configurado. Se precisar instalar:

```powershell
python -m pip install -r requirements-h2o.txt
python -m pip install -r requirements-tabpfn.txt
python -m pip check
```

O PyTorch precisa ser CUDA, conforme já configurado no Exp. 11. Para reinstalar
o conjunto compatível com a RTX 3060/Windows e driver utilizado no projeto:

```powershell
python -m pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
```

Reinicie o kernel se alterar dependências. Preencha `.env.tabpfn` na raiz com
`TABPFN_TOKEN=sua_chave` e aceite a licença TabPFN 3.5 na mesma conta. O notebook
verifica chave/licença sem mostrá-las; não é preciso criar outro arquivo de chave.

1. Abra o notebook do Exp. 12 e execute em ordem.
2. Revise as configurações iniciais; habilite o controle se desejar a comparação
   isolada sob o mesmo protocolo.
3. Confira auditoria, correlações, pesos e pilotos/estimativa de custo.
4. Execute descoberta e avaliação. Revise falhas e os ensembles reconstruídos.
5. Confira rankings/comparações e execute a célula final para gerar quatro CSVs.

Somente o conjunto de 22 features gera:

- `submission_exp12_h2o_rmse_top1_2026.csv`
- `submission_exp12_h2o_lb_top1_2026.csv`
- `submission_exp12_tabpfn_rmse_top1_2026.csv`
- `submission_exp12_tabpfn_lb_top1_2026.csv`

Pred contém margem bruta sem sigmoid/clipping. Sem duas seeds: fallback 0.
Validamos IDs, ordem, unicidade, tamanho, finitude e features. CSVs idênticos são
identificados. Se um vencedor falhar, não há substituição automática e a geração
dos quatro arquivos para, preservando previsões finais concluídas.

## Verificação pequena

```powershell
python -B -m unittest discover -s tests -p test_exp12.py
```

Integração real H2O pequena, em servidor temporário próprio, encerrado ao terminar:

```powershell
$env:MARCH_MANIA_H2O_INTEGRATION="1"
python -B -m unittest discover -s tests -p test_exp12.py
Remove-Item Env:MARCH_MANIA_H2O_INTEGRATION
```

O teste real usa 80 linhas, três bases minúsculas (GLM, GBM e DRF) e dois ensembles.
Confere OOF, reutilização, folds idênticos, previsões finitas e anos do metalearner.
O conjunto completo de treino é sempre anterior ao corte externo.
Também executa um AutoML limitado a três modelos em 80 linhas, extrai o grafo
real dos ensembles e reconstrói uma de suas receitas para prever.

A implementação também validou os dados reais (1.449 jogos M, 961 W, 132.133
pares), as 100 configurações via construtor TabPFN sem fit e o notebook inteiro
com backends simulados, com controle ligado/desligado e quatro CSVs temporários.
Busca completa H2O e avaliação completa TabPFN não foram executadas.

Verificação final após a revisão: **97 testes passaram**, incluindo a integração real habilitada.
O teste pequeno produziu avisos do H2O sobre conversão single-thread e tamanho
insuficiente para uma configuração AutoML com min_rows=100; a busca pequena
prosseguiu e o teste passou. Não foram geradas submissions reais do Exp. 12.

Na revisão, os 154 checkpoints TabPFN já presentes foram validados pelo alias
compatível. Os demais checkpoints de cortes/resultados permaneceram idênticos;
apenas os dois JSONs de descoberta receberam metadados de auditoria. Os testes
também verificam seleção nativa sem cotas, contagens separadas, preservação dos
valores explícitos de n_estimators, filtros locais e rankings RMSE/LB.
