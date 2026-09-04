FROM python:3.11-slim-bookworm AS runtime-dependencies

ARG UV_VERSION=0.8.22

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    GIT_PYTHON_REFRESH=quiet \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM runtime-dependencies AS project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM project AS test

RUN uv sync --frozen

COPY configs ./configs
COPY tests ./tests

CMD ["uv", "run", "python", "-m", "unittest", "discover", "-v"]

FROM project AS research

ARG GIT_REVISION=unknown
ARG GIT_DIRTY=unknown

ENV SINGALIGN_GIT_REVISION=${GIT_REVISION} \
    SINGALIGN_GIT_DIRTY=${GIT_DIRTY}

COPY configs ./configs
COPY data/README.md data/README.md
COPY data/manifests ./data/manifests
COPY experiments ./experiments
COPY reports ./reports
COPY tests ./tests

RUN groupadd --gid 1000 singalign \
    && useradd --uid 1000 --gid singalign --create-home singalign \
    && mkdir -p /mlflow/state /mlflow/artifacts \
    && chown -R singalign:singalign /workspace /mlflow

USER singalign

CMD ["bash"]

FROM project AS api
CMD ["uvicorn", "singalign.api:app", "--host", "0.0.0.0", "--port", "8000"]
