# Experimento 14 — seleção individual de features com TabPFN

[Notebook](../notebooks/features_incrementais/exp14_selecao_individual_features.ipynb).
Não modifica código, notebooks, resultados ou checkpoints dos experimentos anteriores.
Não faz nova busca AutoML, não gera configurações TabPFN, não usa outros algoritmos.
Combinações de grupos ficam para o Exp. 15 e não são implementadas aqui.

## Origem e protocolo

`src/exp14.py:carregar_origem` lê `checkpoints/exp13/catalogo.json`,
`recente26_rankings.json` e `registro.json`. Recalcula a ordenação a partir das
métricas anuais e dos pesos salvos para conferir o Top 3 publicado. Lê as três
configurações TabPFN masculinas e as três femininas, preservando família, versão
V3.5/V3.5-Fast, parâmetros, seeds e inferência. Arquivos ausentes ou inconsistentes
geram erro, sem substituir candidatos. A seed do protocolo é 42.

Reutiliza `exp13.preparar_dados`: 26 features, histórico M:2003+, W:2010+,
features pré-NCAA, 2026 DayNum < 134, TeamA < TeamB, Target = ScoreA − ScoreB,
diferenças A − B e validação de finitude. Não altera a definição das features.

Cada combinação configuração/subconjunto tem três ajustes independentes:

| Treino | Validação |
|---|---|
| Season < 2023 | 2023 |
| Season < 2024 | 2024 |
| Season < 2025 | 2025 |

O RMSE ponderado é a soma de RMSE(ano) × Peso(ano), com os pesos originais do
Exp. 13. Logistic Brier anual/ponderada e RMSE médio são diagnósticos.
Nenhum peso é recalculado com as features selecionadas. TabPFN roda sequencialmente
em CUDA, com o contexto e filtro específico de warning do Exp. 12. Os cinco folds
fazem parte da identidade do protocolo herdado; TabPFN não ganha uma nova CV interna.

## Etapas e estratégia

1. **Controle de 26 features:** `Avaliador.reutilizar_controles` importa apenas
   controles completos equivalentes, sem fit. A compatibilidade exige candidato
   completo, dados de treino/validação, features, versões, seed e folds iguais.
   O único alias permitido é o parâmetro obsoleto inerte já suportado pelo Exp. 12.
   Controles incompatíveis só são ajustados quando uma etapa habilitada os solicitar.
2. **Piloto:** `avaliar_individuais` aceita uma lista pequena de features e
   configurações. O padrão testa duas features recentes no Top 1 de M e W:
   quatro subconjuntos, 12 ajustes históricos; controles equivalentes são reutilizados.
3. **Análise individual completa:** testa 26 remoções × seis configurações,
   totalizando 156 subconjuntos e 468 ajustes históricos, reutilizando os do piloto.
   Cada remoção parte sempre do controle de 26 features e resulta em 25.
   O ranking de impacto é separado por categoria **e configuração**; não é uma
   importância universal. Delta negativo indica melhora após remover.
4. **Seleção progressiva:** `selecao_progressiva` exige que as 26 avaliações
   individuais da configuração tenham terminado (sucesso ou falha). Começa nas 26
   features, testa cada remoção possível, aceita a maior melhora estritamente
   superior ao mínimo e repete. A primeira rodada reutiliza a análise individual.
   Não há quantidade ideal predefinida, nem exclusão em bloco por grupo.

Cada configuração tem sua própria trajetória, independente das outras configurações
e da outra categoria. Se não há melhora suficiente, conserva o conjunto atual.
Falhas são registradas e não concorrem; a escolha considera as remoções válidas da
rodada concluída, portanto a cobertura das falhas deve ser examinada no relatório.
Se faltar orçamento durante uma rodada, **nenhuma remoção daquela rodada é aceita**.
Aumentar o limite e retomar completa a rodada pelos checkpoints. Empates de remoção
usam o nome da feature; o ranking final desempata pelo ID único da avaliação.

## Configuração e orçamento

| Opção do notebook | Padrão | Efeito |
|---|---|---|
| RODAR_PILOTO | True | Habilita o piloto pequeno |
| RODAR_ANALISE_COMPLETA | False | Habilita as 26 remoções por configuração |
| RODAR_PROGRESSIVA | False | Habilita backward elimination |
| GERAR_SUBMISSIONS | False | Habilita ajustes finais e três CSVs |
| MAX_AVALIACOES_POR_CONFIGURACAO | 100 | Limite cumulativo de subconjuntos distintos, excluindo controle |
| MAX_FEATURES_REMOVIDAS | 10 | Máximo aceito na trajetória de cada configuração |
| MIN_FEATURES | 5 | Piso de quantidade, sem estabelecer o número ideal |
| MELHORA_MINIMA_RMSE | 0.001 | Exige melhora estritamente superior ao valor |
| REPETIR_FALHAS | False | Preserva falhas; True tenta novamente apenas as falhas |

Uma avaliação orçamentária corresponde a **um subconjunto × uma configuração,
com três cortes**. Falhas e parciais contam; controles não contam. O limite é por
configuração em cada categoria e vale conjuntamente para todas as etapas e retomadas.
O padrão permite até 600 subconjuntos / 1.800 ajustes históricos, mais até 18 ajustes
de controle se nenhum puder ser reutilizado. Não há limite de tempo de fit neste
experimento. Para estender a busca, aumente o limite ou use None; não apague os caches.
O custo real depende das configurações e da GPU; não se presume um tempo de execução.
Por padrão, os relatórios evitam logs por avaliação; MOSTRAR_LOG_AVALIACOES=True
permite acompanhar cada subconjunto na saída durante execuções longas.

## Checkpoints e reprodutibilidade

`Avaliador` usa somente `checkpoints/exp14/`. Um estado com hash inclui os dados,
catálogo completo das seis configurações, versões, pesos e protocolo. Guarda todos
os subconjuntos, quantidades, métricas anuais/ponderadas, etapas, recursos e erros.
Reserva a avaliação antes do primeiro fit, permitindo retomar após interrupção.
Os arquivos em `cortes/` identificam categoria/configuração/features/ano por hash e
reutilizam o formato do Exp. 12; cada corte guarda previsões e parâmetros completos.

Os limites da busca não fazem parte do contexto dos modelos: aumentá-los permite
continuar os mesmos resultados. Dados/versões/parametrização diferentes selecionam
outro contexto; arquivos antigos permanecem preservados. Resultados antigos só
entram no ranking atual quando compatíveis. Os resumos nomeados do notebook mostram
a execução mais recente; os estados e históricos progressivos com hash preservam
os resultados anteriores do próprio Exp. 14.

## Relatórios e Tops

`Avaliador.ranking` classifica **todas as combinações válidas** pelo RMSE ponderado,
incluindo controles, piloto, remoções individuais e testes da progressiva. Cada
combinação precisa dos três anos completos. A mesma configuração pode ocupar mais
de uma posição com subconjuntos diferentes; não se força diversidade de modelos.
O ranking é diferente da trajetória aceita pela progressiva, que exige melhora
mínima e uma rodada completa. Testes válidos de uma rodada parcial também são
combinações avaliadas, mas não contam como remoção aceita na trajetória.

O notebook apresenta tabelas compactas por M/W com posição, configuração,
**NFeatures**, NRemovidas, RMSE/LB e deltas. Detalhes expansíveis mostram as listas
de features mantidas/removidas para cada Top, incluindo os Tops de referência.
Os rankings completos com listas ficam em `ranking.json`; o ranking de impacto
fica em `analise_individual.json`, com os 26 testes por configuração em blocos
expansíveis. Os arquivos `progressiva_*.json` guardam cada rodada, todos os testes,
seus efeitos e a composição dos subconjuntos, além da razão de parada.

`comparar_categorias` mostra as features mantidas em ambas, removidas somente em M,
somente em W ou em ambas, para cada posição Top 1/2/3. `comparacao_exp13.json`
compara cada combinação com sua configuração original de 26 features. Se o controle
não for equivalente, os scores antigos aparecem identificados e nenhum delta
causal de features é calculado entre protocolos diferentes. As métricas do controle
reajustado no protocolo atual continuam disponíveis no ranking do Exp. 14.

Os mesmos anos selecionam configuração e features; os resultados são de seleção,
**não um teste independente de generalização**. Mais subconjuntos avaliados podem
favorecer sobreajuste à validação. Não se usam scores Kaggle para escolher atributos.

## Execução manual

Use o mesmo kernel do Exp. 13, CUDA/RTX 3060 e `.env.tabpfn`. Não há novas
dependências nem necessidade de criar `.venv`.

1. Execute as células em ordem com os padrões para o piloto. Confira origem,
   auditoria do controle, NaNs/infinitos e os efeitos das duas remoções.
2. Para a análise completa, configure RODAR_ANALISE_COMPLETA=True; pode desativar
   o piloto. Reexecute em ordem. Seus checkpoints serão reaproveitados.
3. Depois, habilite RODAR_PROGRESSIVA=True. Pode manter a análise completa ligada
   (será reutilizada) ou desligá-la se já terminou; a progressiva verifica os pré-requisitos.
4. Examine orçamento, falhas, histórico, cobertura e os Tops. Aumente limites se
   houver etapas incompletas. Somente depois habilite GERAR_SUBMISSIONS=True.

## Submissions opcionais

`gerar_submissions` treina cada combinação selecionada com Season < 2026 e seu
próprio subconjunto. Previsões finais são retomáveis, com hash de configuração,
features, dados, sample e versões. Gera:

- `submission_exp14_tabpfn_rmse_top1_2026.csv`
- `submission_exp14_tabpfn_rmse_top2_2026.csv`
- `submission_exp14_tabpfn_rmse_top3_2026.csv`

Cada CSV combina M/W da mesma posição, que podem ter quantidades e conjuntos
diferentes. O resumo inclui NFeaturesM/W e as listas FeaturesM/W. Mantém margens
brutas sem sigmoid, fallback zero, IDs/ordem/finitude e informa CSVs idênticos.
Não envia arquivos ao Kaggle e não treina os modelos finais por padrão.

## Verificação leve

`python -B -m unittest discover -s tests -p test_exp14.py -v`.
Os testes substituem modelos e medição CUDA, sem treinamento real, downloads ou
autenticação. Cobrem origem, identidade e protocolo, reuso estrito, orçamento,
interrupção, análise individual, progressiva, ranking RMSE, listas por categoria,
falhas e três CSVs com subconjuntos diferentes e cache final.
