# 11MLET - Tech Challenge - Fase 2

Pipeline de Engenharia de Machine Learning para estimar a **propensão de compra** de uma sessão de navegação em e-commerce. Usamos o dataset *Online Shoppers Purchasing Intention* (UCI), classificadores clássicos do Scikit-Learn, **DVC** para versionar dados e orquestrar o pipeline, **MLflow** para rastrear experimentos e registrar o melhor modelo, **Poetry** para dependências e **Docker** para execução reprodutível.

## Integrantes

| Nome | GitHub |
| --- | --- |
| Carolina Castro | [@carolinamsc](https://github.com/carolinamsc) |

## Vídeo (método STAR)

Link do vídeo: _a preencher_ — roteiro em [docs/VIDEO_STAR_ROTEIRO.md](docs/VIDEO_STAR_ROTEIRO.md).

## Problema e dados

Cada linha é uma sessão; a coluna `Revenue` indica se a sessão terminou em compra (classificação binária, ~15,5% positivos). São 12.330 sessões e 17 atributos (numéricos de navegação + categóricos como mês, tipo de visitante e tráfego). Fonte: Sakar e Kastro, UCI Machine Learning Repository, DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), licença CC BY 4.0.

## Arquitetura

```text
download_data ─▶ prepare_data ─▶ train ─▶ evaluate ─▶ register
 (UCI → raw)     (split 60/20/20   (LogReg e     (teste no    (Model Registry,
                  estratificado)    RandomForest, run do       alias champion)
                                    1 run MLflow   melhor)
                                    por modelo)
```

| Estágio DVC | Módulo | Saída |
| --- | --- | --- |
| `download_data` | `scripts/download_dataset.py` | `data/raw/online_shoppers_intention.csv` |
| `prepare_data` | `purchase_intent.data.prepare` | `data/interim/{train,validation,test}.csv` |
| `train` | `purchase_intent.training.train` | `models/purchase_intent.joblib`, `models/best_run.json` |
| `evaluate` | `purchase_intent.evaluation.evaluate` | `reports/metrics.json` (métricas DVC) |
| `register` | `purchase_intent.registry` | `reports/registry.json` |

Decisões principais:

- **Pré-processamento dentro do `Pipeline` do Scikit-Learn** (imputação, `StandardScaler`, `OneHotEncoder`), ajustado só no treino — sem vazamento.
- **Candidatos configuráveis** em `params.yaml` (`model.candidates`); cada um vira um run no MLflow com parâmetros, métricas de validação e o modelo com assinatura.
- **Seleção pelo F1 de validação** (classe positiva minoritária); o teste fica isolado e só é usado no modelo escolhido, com as métricas logadas no mesmo run.
- **Model Registry**: o run vencedor vira uma nova versão de `11mlet-purchase-intent` e recebe o alias `champion`, com as métricas de teste como tags.
- **Reprodutibilidade**: seed única (`experiment.random_seed: 42`) em splits e modelos; dependências travadas em `poetry.lock`; configuração de ambiente em `.env`.

## Pré-requisitos

- Python 3.11 e [Poetry](https://python-poetry.org/) 1.8+
- Docker e Docker Compose
- Git

## Executar localmente

```bash
poetry install
cp .env.example .env            # PowerShell: Copy-Item .env.example .env
docker compose up -d mlflow     # MLflow em http://localhost:5000
poetry run dvc repro            # download → prepare → train → evaluate → register
poetry run dvc metrics show
```

Para rodar sem servidor MLflow, defina no `.env` `CHALLENGE_MLFLOW_TRACKING_URI=sqlite:///mlflow.db` e abra a UI com `poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Executar com Docker

```bash
cp .env.example .env
docker compose --profile pipeline up --build --abort-on-container-exit
```

O perfil `pipeline` sobe o MLflow, espera o healthcheck e roda `dvc repro --force` no container (a imagem instala as dependências de produção via Poetry e executa como usuário sem privilégios). Dados, modelos e relatórios ficam montados no workspace; o cache DVC fica num volume nomeado.

## DVC

- `dvc.yaml` define os estágios e `dvc.lock` guarda os hashes de dados, código, parâmetros e saídas.
- `dvc status` mostra o que mudou; `dvc repro` reexecuta só os estágios afetados.
- `dvc dag` exibe o grafo do pipeline; `dvc params diff` e `dvc metrics diff` comparam experimentos.
- O dataset fica no cache local (`.dvc/cache`). Para compartilhar entre máquinas, configure um remote (`dvc remote add -d storage <url>`) e rode `dvc push` / `dvc pull`. Sem remote, o estágio `download_data` baixa novamente da fonte oficial e o hash em `dvc.lock` garante que é o mesmo arquivo.

## Resultados

| Modelo | Val. F1 | Val. ROC AUC | Teste F1 | Teste ROC AUC | Teste Recall | Teste Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Regressão logística | 0.609 | 0.908 | — | — | — | — |
| **Random Forest (champion)** | **0.691** | **0.930** | **0.652** | **0.922** | **0.715** | **0.882** |

O teste é calculado só para o modelo escolhido na validação. Accuracy sozinha engana com classes desbalanceadas, por isso a seleção usa F1. Detalhes e limitações em [MODEL_CARD.md](MODEL_CARD.md); valores versionados em `reports/metrics.json`.

```text
$ dvc metrics show
Path                  accuracy    f1       model_type     precision    recall    roc_auc    test_rows
reports/metrics.json  0.88159     0.65155  random_forest  0.59868      0.71466   0.92227    2466
```

### Evidências (MLflow)

Runs rastreados no experimento `11mlet-purchase-intent`:

![Runs no MLflow](docs/images/mlflow_experimentos.png)

Comparação dos candidatos (parâmetros e métricas de validação/teste):

![Comparação de runs](docs/images/mlflow_comparacao.png)

Métricas do run campeão:

![Métricas do run campeão](docs/images/mlflow_run_metricas.png)

Model Registry com a versão promovida para o alias `champion` e as métricas de teste como tags:

![Model Registry](docs/images/mlflow_model_registry.png)

## Estrutura

```text
src/purchase_intent/
  config.py              settings (.env), params.yaml, logging e caminhos
  data/prepare.py        limpeza do alvo e split estratificado
  models/pipeline.py     pré-processamento + fábrica de classificadores
  training/train.py      treino dos candidatos e tracking MLflow
  evaluation/evaluate.py avaliação no teste
  registry.py            Model Registry e alias champion
  metrics.py             métricas de classificação
scripts/                 download do dataset, validação do ambiente, regra de 20 linhas
configs/logging.yaml     configuração de logging
tests/                   testes unitários (pytest)
data/ models/ reports/   artefatos versionados pelo DVC
```

## Qualidade

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run python scripts/check_function_length.py   # funções com no máximo 20 linhas
poetry run pytest
```

O workflow `.github/workflows/ci.yml` roda essas verificações e o build da imagem Docker a cada push.
