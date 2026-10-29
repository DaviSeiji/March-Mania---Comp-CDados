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

## Experimento 9 — H2O AutoML com Deep Learning

[exp9_h2o_automl_dl.ipynb](notebooks/features_incrementais/exp9_h2o_automl_dl.ipynb)
testa somente DeepLearning, mantendo as 14 features, folds internos por temporada
e validação 2021–2025. Usa até 10 modelos por busca e orçamento de 120 segundos por
modelo, incluindo os treinos finais. GLM, árvores e ensembles ficam fora desta busca.
O padrão das funções continua excluindo DeepLearning para preservar o Exp. 8;
o Exp. 9 usa `include_algos=["DeepLearning"]` e `exclude_algos=None` explicitamente.

A comparação usa os resultados já registrados do Exp. 8. O notebook também
mostra quantos modelos DeepLearning foram treinados e quantos viraram leaders.
DeepLearning e o corte por tempo podem apresentar variação mesmo com seed fixa.
O orçamento difere do Exp. 8 e não é uma garantia de tempo total de execução.
As três submissões seguem `submission_exp9_h2o_automl_top{1,2,3}_2026.csv`,
pareando a mesma posição dos rankings M/W por RMSE interna.

## Experimento 9 — AutoGluon Tabular

[exp9_autogluon.ipynb](notebooks/features_incrementais/exp9_autogluon.ipynb)
continua o Exp. 8 com as mesmas 14 features e histórico (M: 2003+, W: 2010+).
O notebook anterior de H2O/DL permanece separado, com seu nome original.

Dependência opcional, no Python do kernel (sem necessidade de outra `.venv`):

```sh
python -m pip install -r requirements-autogluon.txt
```

AutoGluon Tabular 1.6.3 é compatível com Python 3.10–3.13. A resolução de
`autogluon.tabular[all]` foi verificada no Python 3.13 local, sem instalar ou alterar
as bibliotecas existentes. O extra `all` inclui dependências grandes e pode trocar
versões de backends, como PyTorch e XGBoost; confira a resolução no seu kernel.
O notebook registra versões, famílias efetivamente treinadas e falhas de modelos.

A seleção interna usa a Logistic Brier existente, por um scorer importável.
`score_val` é negativo; as tabelas também mostram o erro positivo. Season controla
cinco folds agrupados via `validation_structure` e é excluída das features.
A validação externa permanece expansiva em 2021–2025, com M/W separados.

Configuração: seed 42, `best_quality`, 300 segundos por busca, uma repetição de
bagging e um nível de stacking. DyStack é desativado para evitar buscas auxiliares
nesse orçamento; weighted ensembles permanecem disponíveis. São dez buscas
históricas e duas finais, com tempo adicional de preparação e exportação.

A execução gera até três `submission_exp9_autogluon_top{1,2,3}_2026.csv`, combinando
as respectivas posições dos rankings internos M/W, com margens brutas e fallback
zero. Modelos ficam em diretórios temporários do sistema e são removidos após uso;
resultados numéricos ficam no notebook. O treino completo deve ser executado pelo usuário.

Os testes rápidos usam mocks e, quando disponível, o scorer/splitter real:

```sh
python -m unittest discover -s tests
```

Referências: [API de treinamento](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)
e [versão 1.6.3](https://pypi.org/project/autogluon.tabular/1.6.3/).

## Experimento 10 — Similaridade de temporadas + H2O

[exp10_similaridade_temporadas.ipynb](notebooks/features_incrementais/exp10_similaridade_temporadas.ipynb)
mantém os dados, 14 features e os estimadores H2O do projeto, mas passa a comparar
**configurações fixas** entre temporadas. São 200 por categoria: 20 GLM, 90 GBM,
50 Random Forest e 40 Extra Trees, amostradas sem reposição com seed 42 em espaços
explícitos. Cada candidato mantém família, parâmetros e seed nos três anos de validação.
Não há AutoML independente por ano nem ensembles automáticos; DeepLearning e XGBoost
continuam fora. O treino completo fica para execução no ambiente do usuário.

O padrão exige 1.200 treinos principais (200 × 3 × 2), mais seis finais. Mantemos
cinco folds internos por Season para diagnóstico de RMSE, acrescentando até 6.000
ajustes de CV à validação. Não há limite de tempo. A seleção usa exclusivamente
Logistic Brier externo ponderado, não RMSE interna.

O perfil de cada temporada usa somente os participantes identificados pelas seeds,
com estatísticas dos arquivos RegularSeason anteriores ao NCAA. São 13 variáveis
de equipe × média/desvio populacional × M/W = 52 dimensões. Seed não participa
da distância, mas permanece como feature dos modelos. Perfis combinados começam
em 2010; o histórico masculino dos modelos continua começando em 2003.

O z-score é ajustado somente nos perfis anteriores ao ano alvo. A distância
Euclidiana escolhe as três referências mais próximas entre 2021–2025 para 2026.
Seus pesos são proporcionais a `1/distância`, normalizados para somar 1. A menor
média ponderada dos Logistic Briers externos de cada configuração determina seu
ranking dentro de M ou W, separadamente. Ponderamos a avaliação, sem alterar os pesos
dos jogos de treino. Cada validação continua treinando apenas em temporadas anteriores.
Em distância zero, o peso é dividido entre as referências com distância zero.
Empates de distância favorecem o ano mais recente; empates de score usam o ID fixo do candidato.
Manhattan é somente sensibilidade, sem trocar a regra após observar resultados.

O backtest 2022–2025 é opcional (`EXECUTAR_BACKTEST=False` por padrão). Habilitá-lo
acrescenta avaliações em 2021/2022, elevando o orçamento a 2.000 treinos principais.
Usa apenas referências e scores anteriores ao alvo e compara candidatos escolhidos
por similaridade ponderada ou média passada uniforme. Em 2022/2023 existem apenas
uma/duas referências; em 2024/2025 já existem três. O score principal ponderado é
um score de seleção entre muitos candidatos, não um teste independente.

`src/season_similarity.py` constrói perfis, normaliza e calcula distâncias/pesos.
`src/h2o_candidates.py` gera o catálogo fixo, treina, avalia e ranqueia candidatos,
além de implementar o novo backtest. Candidatos com falhas em algum ano requerido
ficam fora do ranking; não são usadas médias parciais. Funções e experimentos anteriores
permanecem disponíveis.

A execução retreina as três configurações vencedoras de cada categoria até 2025,
sem nova busca, e salva `submission_exp10_h2o_similarity_top{1,2,3}_2026.csv`
em `submissions/`. Cada CSV combina a mesma posição M/W no ranking ponderado,
com margens brutas, IDs na ordem do sample e fallback zero. O CSV único da versão
anterior não representa este protocolo e não é sobrescrito. Os scores Kaggle aparecem em
markdown no fim do notebook, sem participar de qualquer função de seleção.
Não são criados diretórios de resultados ou arquivos de modelos persistentes.

Referência dos estimadores: [API H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html).
