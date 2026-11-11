# Experimento 15 — combinações de classes com TabPFN

Notebook atual: [exp15_busca_independente.ipynb](../notebooks/features_incrementais/exp15_busca_independente.ipynb),
com 20 trials por combinação/categoria e Top 5. Consulte a
[documentação atual](exp15_busca_independente.md).

O notebook antigo foi removido a pedido do usuário; seus checkpoints e submissions
continuam preservados. A descrição abaixo registra a metodologia da versão anterior.

## Referência histórica: versão com três configurações do Exp. 13

## Partição e combinações

`src/exp15.py:CLASSES` define explicitamente A (5 básicas), B (9 detalhadas),
C (8 de eficiência), D (4 recentes). `validar_classes` confere a partição sem
omissões/sobreposições e sua correspondência às constantes implementadas nos
módulos de features e ao conjunto do Exp. 13. `gerar_combinacoes` enumera todas
as uniões não vazias. Cada plano inclui classes, features em sua ordem original,
quantidade de classes/features e assinatura SHA256 determinística.

| Classes | Features | Classes | Features | Classes | Features |
|---|---:|---|---:|---|---:|
| A | 5 | A+C | 13 | A+B+C | 22 |
| B | 9 | A+D | 9 | A+B+D | 18 |
| C | 8 | B+C | 17 | A+C+D | 17 |
| D | 4 | B+D | 13 | B+C+D | 21 |
| A+B | 14 | C+D | 12 | A+B+C+D | 26 |

São as mesmas 15 combinações para M e W. A é opcional: grupos sem seed também
concorrem. A API rejeita subconjuntos que removam features de dentro de uma classe.

## Modelos, dados e validação

Reutiliza `exp14.carregar_origem`, lendo `checkpoints/exp13/catalogo.json`,
`recente26_rankings.json` e `registro.json`: três configurações TabPFN por categoria,
classificadas pelo RMSE ponderado do Exp. 13. Não utiliza os subconjuntos vencedores
do Exp. 14. A ordem Top 1/2/3, versões 3.5/3.5-Fast, parâmetros, seeds e opções de
inferência são preservadas. Ausências/inconsistências de origem geram erro.

O preparo é o mesmo do Exp. 13, com as mesmas linhas históricas para todas as
combinações (M 2003+, W 2010+). Mesmo A isolado não amplia seu histórico, evitando
confundir efeito das classes com tamanho do treino. Features são pré-NCAA, diferenças
A − B, TeamA < TeamB e Target = ScoreA − ScoreB. Para 2026 usa DayNum < 134.

Cada candidato configuração/classes é reajustado independentemente para os cortes:
Season < 2023 → 2023; Season < 2024 → 2024; Season < 2025 → 2025. Os parâmetros
permanecem fixos entre os anos e grupos. Pesos de similaridade são lidos do Exp. 13,
sem recálculo. RMSE ponderado seleciona; LB anual/ponderada, RMSE médio e recursos
são diagnósticos. Não se usam scores Kaggle nas escolhas. Como os anos servem à
seleção, não representam um teste independente de generalização.

TabPFN roda sequencialmente em CUDA, no mesmo ambiente do Exp. 13, com o contexto de
modelo/filtro específico de aviso obsoleto existente. O protocolo registra seed 42
e cinco folds herdados; TabPFN não recebe uma nova validação cruzada interna.
Nenhum servidor H2O é iniciado e nenhum novo algoritmo é introduzido.

## Avaliador, checkpoints e orçamento

`Avaliador` especializa o avaliador temporal do Exp. 14 para aceitar exclusivamente
as 15 uniões. O contexto próprio inclui `exp15_revisao`, classes/planos, candidatos,
dados, versões, pesos e protocolo. Não altera o avaliador antigo. Os registros ficam
somente em `checkpoints/exp15/`, ignorado pelo Git.

Cada avaliação tem uma identidade de configuração/features; cada corte em `cortes/`
usa o hash completo de candidato, features, ano, dados de treino/validação, versões,
seed e folds, reutilizando o formato estável do avaliador do Exp. 12. Reservas são
gravadas antes do fit e resultados são salvos por corte. Uma interrupção retoma
os cortes restantes sem refazer sucessos. Falhas ficam registradas e só são
repetidas se REPETIR_FALHAS=True. Mudanças efetivas selecionam outro contexto/hash.

Somente os resultados de 26 features do Exp. 13 são importados automaticamente,
após comprovar compatibilidade completa. A origem é apenas lida; cópias de registros
compatíveis ficam no Exp. 15. Outros subconjuntos não reutilizam silenciosamente o
Exp. 14. As receitas/limites de seleção individual não participam deste experimento.

MAX_AVALIACOES é um limite **global e cumulativo de candidatos distintos**
(configuração × combinação, com três cortes cada). Inclui controles reutilizados,
falhas e parciais. Não é um limite de segundos nem de chamadas fit. Aumentá-lo ou
usar None permite continuar os mesmos checkpoints. O padrão é None, suficiente
para todo o plano; não se força uma interrupção antes de testar as 15 combinações.

## Rankings e análises

`avaliar_combinacoes` permite escolher grupos e uma a três configurações por
categoria. O modo completo do notebook utiliza todas as 15 e as três configurações:
90 candidatos, 270 cortes. `Avaliador.ranking` inclui apenas candidatos com os três
anos válidos e ordena M/W separadamente por RMSE ponderado, depois ID determinístico.
Empates não usam LB nem quantidade de features. O Top 3 pode repetir as classes,
desde que configuração/features sejam distintos.

`resumir_combinacoes` escolhe a melhor configuração válida de cada grupo e mostra
quantas das três configurações completaram os cortes. O notebook salva ranking
completo, cobertura M/W × 15 grupos, resumo, comparação por número de classes e as
três configurações nos conjuntos de 14/22/26 features. Os Tops apresentam classes,
quantidade e listas expansíveis de features separadas por categoria.

`comparar_adicoes` compara X com X+uma classe, fixando configuração e categoria.
Gera até 28 pares por configuração, 168 no total, incluindo A+B → A+B+C e
A+B+C → A+B+C+D. Delta negativo indica melhora ao adicionar. Registra também
os pares em que menos features obtiveram RMSE menor. As médias dos deltas são
diagnósticos condicionais aos pares disponíveis, sem substituir o critério de seleção.
Classes que pioram isoladamente podem ser complementares em outras uniões; não se
conclui irrelevância a partir do resultado isolado. Comparações M/W devem considerar
que são torneios/dados diferentes, embora o plano de classes seja idêntico.

## Execução manual

Use o mesmo kernel, CUDA/RTX 3060 e `.env.tabpfn` do Exp. 13. Não há novos pip installs
nem necessidade de criar uma `.venv`. Nenhum treinamento real foi executado durante
a implementação.

1. Abra o notebook e execute em ordem com RODAR_PILOTO=True. O padrão testa
   A+B+C+D, A+B e D, na configuração Top 1 de cada categoria: seis candidatos,
   18 cortes; com controles compatíveis, 12 ajustes novos. Altere COMBINACOES_PILOTO
   e N_CONFIGURACOES_PILOTO (1–3) para outro recorte.
2. Depois, habilite RODAR_COMPLETO=True. Pode desativar o piloto ou mantê-lo:
   será reaproveitado. A etapa completa sempre usa 15 grupos e três configurações
   por categoria. Com os seis controles compatíveis, são até 252 ajustes históricos
   novos antes de descontar o piloto. O tempo real depende das configurações/GPU.
3. Confira cobertura, falhas, comparações e Tops. Se definiu um orçamento e ele
   interrompeu o plano, aumente MAX_AVALIACOES ou use None e reexecute.
   MOSTRAR_LOG_AVALIACOES=True permite acompanhar cada avaliação na saída.
4. Habilite GERAR_SUBMISSIONS=True apenas quando quiser retreinar e gerar os CSVs.
   Por padrão está False. As três flags podem ficar True numa execução em ordem.

## Submissions

`gerar_submissions` treina as seis combinações vencedoras com Season < 2026,
utilizando as features próprias de cada candidato M/W. Salva previsões finais em
cache exclusivo com configuração, plano, dados, sample, seed e versões. Gera:

- `submission_exp15_tabpfn_rmse_top1_2026.csv`
- `submission_exp15_tabpfn_rmse_top2_2026.csv`
- `submission_exp15_tabpfn_rmse_top3_2026.csv`

Cada posição combina M/W da mesma posição, com classes possivelmente diferentes.
Preserva IDs/ordem/tamanho, finitude, margens brutas sem sigmoid e fallback zero.
Informa saídas idênticas, se houver. O resumo registra configurações, classes,
features e quantidades M/W. Nunca envia arquivos ao Kaggle. Um ranking parcial
reflete somente os candidatos avaliados; a cobertura aparece explicitamente.

## Testes leves

`python -B -m unittest discover -s tests -p test_exp15.py -v`.
Usa modelos e recursos CUDA simulados, sem treinamento real, autenticação ou
downloads. Cobre partição/planos, protocolo completo de 270 cortes, identidade,
separação temporal/M/W, reuso estrito, interrupção, orçamento, ranking RMSE,
comparações com configuração fixa e três CSVs com classes diferentes e cache final.
