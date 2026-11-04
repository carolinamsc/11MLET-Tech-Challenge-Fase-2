# Etapa 1: instala apenas as dependências de produção com Poetry (lock file).
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-cache

# Etapa 2: imagem final enxuta, sem Poetry, executando como usuário sem privilégios.
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY params.yaml dvc.yaml dvc.lock ./
COPY .dvc/config .dvc/config
COPY .dvcignore ./
# O contexto Docker não inclui .git: o DVC roda sem SCM dentro do container.
RUN printf '[core]\n    no_scm = true\n' > .dvc/config.local \
    && mkdir -p data/raw data/interim models reports .dvc/cache \
    && chown -R appuser:appuser /app
USER appuser
CMD ["dvc", "repro"]
