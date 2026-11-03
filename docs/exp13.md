# Experimento 13 — forma recente e Top 3 por RMSE

[Notebook](../notebooks/features_incrementais/exp13_forma_recente.ipynb).
Não altera dados, resultados, código ou checkpoints dos experimentos anteriores.

## Features

Controle: as mesmas 22 features do Exp. 12. Novo: 26 features, acrescentando:

| Diferença A − B | Estatística dos últimos dez jogos por equipe/temporada |
|---|---|
| RecentWinRate10Diff | Vitórias / jogos disponíveis |
| RecentAvgMargin10Diff | Média de pontos feitos menos sofridos |
| RecentAvgPointsFor10Diff | Média de pontos feitos |
| RecentAvgPointsAgainst10Diff | Média de pontos sofridos |

`src/recent_features.py` implementa `calcular_forma_recente` e
`adicionar_forma_recente`: participações W/L de RegularSeasonCompactResults,
ordenação DayNum/OpponentID determinística, janela por Season/TeamID e diferenças.
Se houver menos de dez jogos, utiliza os disponíveis; registra RecentGames10,
RecentFirstDayNum10 e RecentLastDayNum10, sem adicioná-los ao modelo. Não altera
as entradas. Rejeita duplicados, valores inválidos, TeamA >= TeamB e jogos a
partir do limite exclusivo. Ausências são permitidas explicitamente apenas em
pares do sample sem duas seeds, cujo fallback será zero.

`src/exp13.py:preparar_dados` reutiliza o preparo do Exp. 12: mesmo histórico
M:2003+, W:2010+, Target = ScoreA − ScoreB, sem resultados NCAA nas features.
Verifica sobreposição dos IDs de jogos regulares/NCAA e cada diferença A − B.
Cada ano usa somente sua própria temporada pré-torneio; 2026 usa DayNum < 134.
A margem recente é matematicamente redundante com as médias de pontos, mantida
conforme o pedido. O notebook mostra contagens, finitude, estatísticas e correlações
por categoria com todas as features anteriores e o target.

## Catálogo fixo de origem

`carregar_catalogo(checkpoints/exp12)` exige:

- `registro.json` e `similaridade.json`, com as 22 features, seed 42, cinco folds,
  anos 2023–2025 e pesos consistentes. Reutiliza os pesos dos perfis originais;
  não calcula similaridade com as novas features.
- Uma descoberta inequívoca por categoria em
  `eficiencia/descoberta/descoberta_{M,W}_*.json`.
- As 100 configurações TabPFN por categoria em
  `eficiencia/avaliacao/ultima_avaliacao.json` ou nos arquivos `corte_*.json`.

Ausências/ambiguidades causam erro com o caminho necessário, sem gerar um catálogo
substituto. Nenhuma função de descoberta AutoML ou geração TabPFN é chamada.
Os candidatos são preservados integralmente, incluindo seeds efetivas, parâmetros,
dependências e receitas de stacking. `assinatura_candidato` verifica sua identidade;
somente o parâmetro TabPFN inerte auto_scale_n_estimators=False, com n_estimators
inteiro explícito, recebe a compatibilidade já implementada no Exp. 12.

No catálogo disponível na implementação: **110 H2O por categoria (108 bases +
duas receitas de Stacked Ensemble), mais 100 TabPFN por categoria = 420 candidatos**.
O limite 100 foi da busca AutoML original; seus oito modelos iniciais adicionais
continuam elegíveis. Não são impostas cotas ou truncamentos para ajustar a contagem.
As exclusões do ambiente original (DeepLearning e XGBoost indisponível) permanecem.

## Validação, stacking e seleção

Cada candidato sofre três ajustes independentes, com a mesma configuração:
Season < 2023 → validação 2023; Season < 2024 → 2024; Season < 2025 → 2025.
M/W separados. Não há busca dentro dos cortes nem alteração de hiperparâmetros.

`avaliar_novo` delega ao avaliador do Exp. 12 com 26 features, seed 42 e cinco
folds agrupados por temporada. `SessaoH2O` reconstrói as mesmas receitas de
BestOfFamily/AllModels, as configurações-base e o GLM metalearner fixo. As bases
usam folds OOF comuns e apenas dados anteriores ao corte externo. Compartilhar
bases dentro do mesmo corte evita repetir treinos; cortes/categorias usam sessões
independentes. Receitas reconstruídas não garantem identidade binária com os
ensembles originais do AutoML. A composição limitada no Exp. 12 é preservada.

TabPFN usa as configurações salvas V3.5/3.5-Fast, n_estimators explícito, CUDA na
RTX 3060 e execução sequencial. Reutiliza autenticação, medição de recursos e filtro
específico do warning obsoleto no contexto TabPFN, sem ocultar avisos importantes.

Score principal = soma(Peso(ano) × RMSE(ano)). `ranquear` retorna somente o ranking
por RMSE ponderado; desempata pelo ID do candidato. Logistic Brier anual/ponderada
e RMSE médio simples são diagnósticos. Apenas candidatos com três cortes válidos
entram no Top 3 de cada abordagem/categoria. Estes anos foram usados na seleção:
não representam um teste independente.

## Controle, checkpoints e cobertura

`reutilizar_controle` lê checkpoints equivalentes do Exp. 12 e copia os registros
compatíveis para `checkpoints/exp13/controle/avaliacao`. A compatibilidade inclui
candidato completo, dados, versões, features, seed e folds. Nunca escreve na origem.
`ler_controle` não treina ao encontrar ausências. Por padrão,
COMPLETAR_CONTROLE_FALTANTE=False; se habilitado, o controle faltante é ajustado
com os mesmos candidatos e salvo apenas no Exp. 13.

O novo conjunto usa `checkpoints/exp13/recente/avaliacao`. Checkpoints incluem
parâmetros, assinaturas de dados, protocolo, versões, métricas, previsões e recursos.
Mudanças efetivas selecionam outro hash, sem apagar registros válidos. Falhas são
registradas sem substituir candidatos; REPETIR_FALHAS=True permite tentar novamente.

`comparar_pareado` intersecta candidatos completos nos dois conjuntos e recalcula
posições no mesmo pool. Mostra cobertura por categoria/abordagem, deltas anuais e
ponderados, famílias e melhor base vs. stacking. Diferenças entre protocolos ou
candidatos não são atribuídas às novas features. Rankings completos, catálogo,
pesos, comparação, auditorias e recursos ficam exclusivamente no Exp. 13.

## Custo e execução

Não foram executados os treinamentos completos na implementação.
São planejadas **1.260 avaliações externas novas** (660 H2O + 600 TabPFN), além dos
folds internos necessários às bases de stacking. O controle equivalente é reutilizado.
As seis submissions exigem até 12 configurações vencedoras ajustadas novamente,
mais bases/folds necessários aos ensembles. Esses números não são trials novos.

O orçamento inicial TabPFN é 14.400 segundos (4 horas) **por categoria**, cumulativo
incluindo os tempos dos checkpoints. Não é um prazo rígido: termina o candidato
atual antes de verificar o orçamento. H2O e ajuste final ficam fora dele. Para
completar os 100 candidatos por categoria, aumente o orçamento ou use None e rode
novamente; os resultados existentes são reutilizados. Não há estimativa confiável
de duração total sem medir esses dados/configurações.

Abra o notebook com o mesmo kernel/ambiente do Exp. 12, mantenha a chave em
`.env.tabpfn` e execute as células em ordem. Não há nova dependência de pip.
Verifique primeiro o catálogo, os dados e o reuso do controle; depois execute as
células de ambiente e avaliação. A célula final exige três candidatos completos
por abordagem/categoria. O notebook começa sem outputs e só treinará ao executá-lo.

## Submissions

`gerar_submissions` retreina com Season < 2026 usando as 26 features. Gera somente:

- `submission_exp13_h2o_rmse_top{1,2,3}_2026.csv`.
- `submission_exp13_tabpfn_rmse_top{1,2,3}_2026.csv`.

Cada posição combina M/W da mesma posição, sem blending. Margens brutas, sem sigmoid.
Valida quantidade, IDs, ordem, finitude e fallback zero. Previsões finais têm cache
próprio; saídas idênticas aparecem em IdenticaA. Não gera CSV do controle.

## Verificação leve

`python -B -m unittest discover -s tests -p test_exp13.py -v` cobre janelas W/L,
limites, desempate, cópias, diferenças, catálogo sem busca, configurações fixas e
receitas entre cortes, checkpoints, ranking RMSE, pareamento e seis CSVs. Modelos
substitutos determinísticos permitem testar o fluxo sem treinar H2O/TabPFN reais.
O smoke test também deve verificar o preparo com os CSVs locais e a leitura do
catálogo/checkpoints reais, sem alterar os arquivos do Exp. 12.

Verificação na implementação: nove testes novos; suíte com 106 testes, sendo uma
integração H2O desabilitada. Smoke com dados reais (1.449 jogos M, 961 W, sample
com 132.133 pares), 16 candidatos substitutos, 48 cortes e seis CSVs temporários.
Os 1.260 checkpoints históricos reais do controle foram considerados compatíveis;
os 1.284 arquivos do diretório Exp. 12 permaneceram idênticos por SHA256.
