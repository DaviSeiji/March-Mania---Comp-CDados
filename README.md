# March-Mania---Comp-CDados
Resolvendo: https://www.kaggle.com/competitions/march-machine-learning-mania-2026-logistic-brier

## Experimento 8 — H2O AutoML

Notebook: [exp8_h2o_automl.ipynb](notebooks/features_incrementais/exp8_h2o_automl.ipynb).
Usa as mesmas 14 features e o histórico principal do Exp. 7 (M: 2003+, W: 2010+).

No ambiente Python usado pelo kernel, instale a dependência opcional:

```sh
python -m pip install -r requirements-h2o.txt
```

É necessário Java disponível (`java -version`). A integração foi testada com
Python 3.13, Java 8 e H2O 3.46.0.12. O ambiente local `.venv` contém essa instalação;
selecione seu Python como kernel ou instale a dependência no kernel que já utiliza.

O notebook inicia o cluster uma vez e executa 10 AutoML de validação e 2 finais,
com seed 42, cinco folds internos agrupados por Season e até 20 modelos base por
busca, além dos ensembles. A seleção interna usa RMSE; a avaliação externa usa
a Logistic Brier existente, com treino anterior a cada ano de 2021–2025.
O ano externo não é enviado ao AutoML. DeepLearning é excluído, e a disponibilidade
do XGBoost no backend H2O é detectada e registrada (indisponível no Windows testado).

As execuções são sequenciais, com limpeza dos modelos/frames de cada busca após o
uso. Não são salvos modelos binários nem uma pasta de resultados. As tabelas ficam
no notebook e a execução completa gera três CSVs em `submissions/`:
`submission_exp8_h2o_automl_top1_2026.csv`, `submission_exp8_h2o_automl_top2_2026.csv`
e `submission_exp8_h2o_automl_top3_2026.csv`, com margens brutas e o mesmo fallback.
Cada arquivo combina a mesma posição dos leaderboards internos M/W por RMSE.
A avaliação oficial continua usando somente o leader; os modelos de posições 2/3
são alternativas para submissão, sem repetir o treinamento. Se alguma categoria
produzir menos de três modelos, o notebook informa e gera apenas os pares disponíveis.

Referência: [documentação do H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html).
