# Roteiro do vídeo (5 minutos, método STAR)

## Situation (~45 s)
- E-commerce quer saber, durante a navegação, quais sessões têm maior propensão de compra.
- Dataset UCI Online Shoppers Purchasing Intention: 12.330 sessões, 17 atributos, alvo `Revenue` com ~15,5% de compras (desbalanceado).

## Task (~45 s)
- Objetivo técnico: classificador clássico em Scikit-Learn com foco em **Engenharia de ML**.
- Restrições do desafio: Clean Code (funções ≤ 20 linhas, type hints, docstrings), Poetry com lock, `.env`, Docker, DVC, MLflow Tracking + Model Registry e seeds fixas.

## Action (~2 min 30 s)
1. **Estrutura** (`src/purchase_intent/`): módulos curtos por responsabilidade (config, data, models, training, evaluation, registry). Mostrar `scripts/check_function_length.py` e `ruff`.
2. **Poetry**: `pyproject.toml` com grupos prod/dev e `poetry.lock`; `.env.example` → `.env`.
3. **DVC**: `dvc dag` mostrando download → prepare → train → evaluate → register; `params.yaml` como fonte única de hiperparâmetros e seed.
4. **MLflow**: um run por candidato (regressão logística e Random Forest), comparação de runs, métricas de teste no run vencedor, versão registrada com alias `champion`.
5. **Predição**: `python -m purchase_intent.predict` carregando o modelo `@champion` do Registry.
6. **Docker**: Dockerfile multi-stage (Poetry só no builder, usuário não-root) e `docker compose --profile pipeline up` subindo MLflow + pipeline.

## Result (~1 min)
- Rodar o pipeline ao vivo (ou mostrar `dvc repro` + `dvc metrics show`).
- Random Forest venceu: F1 de teste 0,646, ROC AUC 0,925, recall 0,783 (regressão logística: F1 de validação 0,609).
- Prints de apoio: `docs/images/`.
- Mostrar o CI verde: lint, testes e o pipeline inteiro rodando no Docker a cada push.
- Aprendizados: seleção por validação mantendo o teste isolado; parâmetros versionados tornam cada experimento reproduzível; métricas além de accuracy em dados desbalanceados; versões travadas no lock e fim de linha LF para os hashes do DVC baterem em qualquer SO.
