# Experimento 14 — cinco trials por subconjunto

Notebook: [exp14_busca_independente.ipynb](../notebooks/features_incrementais/exp14_busca_independente.ipynb).
Substitui o notebook anterior do Exp. 14. Seus resultados e checkpoints são preservados.

## Configurações e metodologia

`src/exp14_busca.py` reutiliza o avaliador temporal/checkpoints do Exp. 14 e o
sorteio sem scores do Exp. 15. Gera cinco configurações por categoria sem consultar
vencedores ou rankings do Exp. 13. Só reutiliza seus pesos/protocolo e resultados
de controle que forem comprovadamente equivalentes. Não há Optuna adaptativo:
é amostragem reproduzível no espaço já usado, com seed 42. O catálogo é exibido,
incluindo versão, n_estimators e opções; cinco trials não significam cinco epochs.

Os mesmos cinco parâmetros concorrem em cada subconjunto, com ajustes e seleção
independentes. Cada trial mantém parâmetros e seed nos três cortes, sempre com
um estimador novo: Season < 2023 → 2023, < 2024 → 2024, < 2025 → 2025.
M/W separados, média ponderada de RMSE como critério, LB como diagnóstico.
Não usa Kaggle nem apresenta os anos de seleção como um teste independente.

Usa as mesmas 26 features iniciais, mesmas linhas históricas e preparo pré-NCAA,
com 2026 DayNum < 134, TeamA < TeamB, Target = ScoreA − ScoreB e diferenças A − B.
TabPFN roda sequencialmente na GPU com o contexto/filtro de warning existentes.

## Etapas

1. Controle: cinco trials nas 26 features por categoria. Só importa resultados
   compatíveis em dados, parâmetros, versões e protocolo; não escolhe candidatos
   pelo desempenho anterior.
2. Piloto: duas remoções no primeiro trial sorteado de M/W; reaproveitadas depois.
3. Análise individual: 26 remoções, cada uma partindo das 26 features, com os cinco
   trials em M/W. Preserva o ranking pareado por configuração e acrescenta o melhor
   trial de cada remoção comparado com o melhor controle de 26 features.
4. Progressiva: **uma trajetória por categoria**. Para cada remoção possível do
   conjunto atual, avalia os cinco trials e escolhe a melhor configuração. Aceita
   a melhor remoção somente se a melhora de RMSE superar o limiar configurado.

Na progressiva o vencedor pode mudar entre subconjuntos. O ganho desse passo
é da seleção conjunta de features/modelo; o relatório individual com configuração
fixa continua disponível para isolar o efeito da remoção naquele modelo.

Não aceita decisões com pools/rodadas incompletos por orçamento. Falhas são
registradas e somente trials válidos concorrem; cobertura reduzida por falhas deve
ser observada. A progressiva exige que todos os testes individuais terminem
(sucesso ou falha), evitando começar com um ranking parcial do piloto.
Empates usam feature/ID determinísticos. O histórico registra todos os trials,
remoções, métricas, mudanças de configuração e razão de parada.

## Custo e limites

- Controles: 5 × 2 × 3 = até 30 ajustes, com reuso dos compatíveis.
- Remoções individuais: 26 × 5 × 2 × 3 = **780 ajustes históricos**.
- Primeira rodada progressiva: reaproveita a etapa individual.
- Próxima rodada, se M/W continuarem: até 25 × 5 × 2 × 3 = **750 ajustes novos**.
- A rodada seguinte pode acrescentar 720, e assim por diante; não é uma busca de
  custo fixo como as 15 combinações do Exp. 15.

N_TRIALS=5 é configurável. MAX_AVALIACOES_POR_CONFIGURACAO=100 limita subconjuntos
únicos por trial, excluindo controle e compartilhado entre etapas/retomadas.
Com cinco trials por categoria, o teto é 1.000 pares trial/subconjunto, ou até
3.000 ajustes anuais + 30 controles. Não é teto de segundos. Também mantém
MAX_FEATURES_REMOVIDAS, MIN_FEATURES e MELHORA_MINIMA_RMSE configuráveis.

As falhas e os parciais contam no orçamento. Aumentar o limite retoma resultados
sem refazer sucessos; REPETIR_FALHAS permite tentar novamente apenas as falhas.
Nenhum treinamento real foi executado na implementação.

Comparação com o Exp. 15 (um ajuste = um trial/subconjunto/categoria/ano):

| Busca | Ajustes históricos previstos antes do reuso |
| --- | ---: |
| Exp. 15: 15 combinações, 20 trials, M/W, três anos | 1.800 |
| Exp. 14: controles + análise individual; primeira rodada progressiva reaproveitada | 810 |
| Exp. 14: mais uma rodada nova completa em M/W | 1.560 acumulados |
| Exp. 14: mais duas rodadas novas completas em M/W | 2.280 acumulados |
| Exp. 14: teto atual de 100 subconjuntos por configuração, incluindo controles à parte | 3.030 |

As rodadas extras só ocorrem enquanto há melhora; M/W podem parar em momentos
diferentes. Falhas e checkpoints alteram a quantidade efetiva de ajustes novos.
Ambos os notebooks geram Top 5 com até dez ajustes finais adicionais (cinco M,
cinco W), reutilizando previsões finais compatíveis. Selecionar Top 5 não aumenta
o orçamento da busca histórica.

## Rankings e execução

O ranking final mostra Top 5 M/W entre todos os pares configuração/subconjunto
com três anos válidos. Não é limitado aos conjuntos aceitos na
trajetória progressiva: testes válidos também concorrem. Pools parciais não podem
causar uma remoção aceita, embora seus candidatos completos tenham métricas válidas.
Tabelas e detalhes mostram quantidades e listas de features de cada Top.

Execute o notebook novo em ordem, usando o mesmo ambiente e `.env.tabpfn`.
Não há novos pip installs nem `.venv`. As quatro etapas estão habilitadas como na
última execução do usuário; para testar só o piloto, desabilite análise completa,
progressiva e submissions. Confira os limites antes de executar a busca inteira.

Novos checkpoints: `checkpoints/exp14/busca_independente/`.
Novas submissions: `submissions/exp14_busca_independente/` (cinco CSVs).
Mantém margens brutas sem sigmoid, fallback zero e formato/ordem/IDs do sample.
Os resultados originais permanecem fora dessas pastas. Não envia ao Kaggle.

## Testes leves

`python -B -m unittest discover -s tests -p test_exp14_busca.py -v`.
Utiliza modelos e recursos simulados para verificar sorteio, seleção por subconjunto,
mudança de vencedor, protocolo temporal, orçamento, retomada e o notebook.
