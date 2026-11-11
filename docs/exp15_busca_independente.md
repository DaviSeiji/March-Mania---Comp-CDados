# Experimento 15 — busca independente de 20 trials e Top 5

Notebook atual: [exp15_busca_independente.ipynb](../notebooks/features_incrementais/exp15_busca_independente.ipynb).
O notebook anterior foi removido a pedido do usuário. Seus checkpoints e três
submissions continuam preservados.

## Busca e contagem

`gerar_catalogo_busca` sorteia configurações sem consultar rankings ou vencedores
anteriores. É busca aleatória estratificada, não Optuna adaptativo: reutiliza somente
a definição do espaço de 100 configurações (versões 3.5/Fast, n_estimators 4/8/16/32/64,
cinco perfis de inferência e temperaturas 1.0/0.7). Com 20 trials, cada estrato
versão × n_estimators recebe duas configurações, sem reposição e com seed 42.
Não se escolhem simplesmente as primeiras 20, que privilegiariam n_estimators=4.
O sorteio tem prefixo estável: aumentar N_TRIALS conserva as configurações anteriores.

Os mesmos 20 parâmetros são testados em todos os grupos e categorias para permitir
comparações diretas. Cada combinação tem ajustes e escolha independentes, podendo
selecionar uma configuração diferente. Não se filtra pelo desempenho com 26 features.

| Etapa | Quantidade |
|---|---:|
| Trials por combinação/categoria | 20 |
| Candidatos por categoria: 15 × 20 | 300 |
| Candidatos totais | 600 |
| Ajustes históricos: 600 × três anos | 1.800 |
| Selecionados no final: cinco M + cinco W | 10 |
| Submissions finais | 5 |

`carregar_busca` lê somente `checkpoints/exp13/registro.json` para validar o protocolo
e obter os pesos. Não lê os rankings ou vencedores antigos. `carregar_origem` segue
disponível para compatibilidade com a versão anterior; o notebook novo não a utiliza.

Cada configuração mantém versão, parâmetros, seed e opções de inferência nos três
cortes: Season < 2023 → validação 2023; Season < 2024 → 2024; Season < 2025 → 2025.
São estimadores novos por corte. A seleção usa a média ponderada do RMSE anual com
os pesos originais. LB e RMSE médio são diagnósticos. Não se usam scores Kaggle;
os mesmos anos utilizados na seleção não são um teste independente de generalização.

## Classes e relatórios

Permanecem as quatro classes inteiras A:5, B:9, C:8, D:4 e as 15 uniões não vazias,
sem remoções individuais. Mesmas linhas históricas por grupo, M:2003+, W:2010+,
features pré-NCAA, diferenças A − B, Target = ScoreA − ScoreB e TeamA < TeamB.
2026: DayNum < 134. TabPFN continua sequencial na GPU, com o filtro específico de
warning já existente. O registro dos cinco folds não acrescenta uma CV ao TabPFN.

O Top 5 é global entre os **300 pares configuração + classes de cada categoria**.
Pode repetir classes com configurações distintas ou uma configuração com classes
distintas. Empates usam o ID determinístico. O resumo também mostra o melhor de cada
união e a cobertura dos 20 trials. As listas e quantidades de features aparecem em
detalhes expansíveis, separados por M/W.

O gráfico RMSE por combinação/trial usa uma matriz para evitar 20 linhas sobrepostas.
A análise de adição de classes continua comparando configurações iguais: até 28 pares
por configuração, ou 1.120 pares completos. Uma classe ruim isoladamente pode ser
complementar em outro conjunto; médias de deltas são somente diagnósticos.

## Preservação e checkpoints

Novos resultados: `checkpoints/exp15/busca_independente/`. Os antigos continuam em
`checkpoints/exp15/`. Estados/cortes registram dados, versões, parâmetros, features,
ano e protocolo; reservas antes do fit permitem retomar interrupções. Mudar o limite
ou ampliar o número de trials não exige apagar caches: cortes compatíveis encontrados
na pasta são reutilizados. REPETIR_FALHAS=True tenta novamente somente as falhas.

Os resultados completos de 26 features do Exp. 13 podem ser reutilizados se a
configuração sorteada, dados e protocolo forem equivalentes. Isso não influencia
o sorteio. Até 120 cortes (20 × 2 × 3) podem ser reaproveitados dessa forma;
1.800 avaliações planejadas não significam necessariamente 1.800 fits novos.

MAX_AVALIACOES conta candidatos únicos, incluindo reusos, falhas e parciais,
não segundos ou cortes. None permite concluir o plano. Cobertura parcial fica explícita.

## Execução manual e cinco CSVs

Abra **exp15_busca_independente.ipynb** no mesmo ambiente. N_TRIALS=20, TOP_K=5 e
as três flags True estão configurados como na execução anterior do usuário.
Execute em ordem para piloto, busca completa e cinco submissions. Para rodar só o
piloto, deixe RODAR_COMPLETO=False e GERAR_SUBMISSIONS=False. Não há novos pacotes
nem `.venv`. A implementação executa apenas testes simulados, sem treinar TabPFN real.

Os cinco arquivos `submission_exp15_tabpfn_rmse_top1_2026.csv` até `top5` ficam em
**`submissions/exp15_busca_independente/`**, preservando os três CSVs antigos na raiz.
Cada posição combina M/W da mesma posição, retreinados até 2025 com suas features.
Mantém margens brutas sem sigmoid, fallback zero, IDs, ordem e finitude. Não envia
arquivos ao Kaggle. A geração exige pelo menos cinco candidatos completos por categoria.

## Testes leves

`python -B -m unittest discover -s tests -p test_exp15.py -v`.
Usa dados/modelos e medição CUDA simulados. Cobre sorteio sem rankings, 20 trials,
600 candidatos/1.800 cortes, configuração fixa por ano, retomada e cinco CSVs temporários.
A API anterior de três configurações e Top 3 permanece compatível.
