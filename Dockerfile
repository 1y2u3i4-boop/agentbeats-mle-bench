FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Pre-install common ML libraries that generated scripts will use.
# These go into the project venv so sys.executable (used by interpreter.py) finds them.
RUN uv pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    xgboost \
    catboost \
    scipy \
    matplotlib \
    seaborn \
    Pillow \
    opencv-python-headless

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
