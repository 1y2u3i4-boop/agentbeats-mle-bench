# MLE-Bench Purple Agent

An autonomous ML engineering agent for [AgentBeats MLE-Bench](https://agentbeats.dev/agentbeater/mle-bench). Solves Kaggle competitions using AIDE-style tree search with iterative code generation and execution.

## Architecture

The agent uses an **AIDE-style tree search** approach:

1. **Receive** competition data (tar.gz) and instructions from the MLE-Bench green agent
2. **Profile** the dataset to understand its structure
3. **Generate** an initial complete Python solution via LLM
4. **Execute** the solution in an isolated subprocess
5. **Iterate** — select the best-scoring node, ask LLM to improve it
6. **Recover** — if no valid submission is produced, try simplified fallback approaches
7. **Return** the best `submission.csv` as an artifact

### Structural pass@k

Multiple independent solver attempts run with different strategy seeds (quick_baseline, data_first, big_model, ensemble_focus, feature_heavy), and the best result is selected.

## Project Structure

```
src/
├─ server.py      # A2A server setup and agent card
├─ executor.py    # A2A request handling
├─ agent.py       # Main agent: receives data, orchestrates solving
├─ tree.py        # AIDE-style tree search engine
├─ llm.py         # LLM client (OpenAI-compatible)
├─ interpreter.py # Subprocess code execution
├─ strategies.py  # Strategy seeds for diversity
└─ messenger.py   # A2A messaging utilities
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for the LLM provider |
| `OPENAI_MODEL` | `o4-mini` | Model to use for code generation |
| `MAX_ITERATIONS` | `12` | Tree search iterations per attempt |
| `NUM_ATTEMPTS` | `3` | Number of parallel strategy attempts |
| `CODE_TIMEOUT` | `600` | Timeout (seconds) per code execution |

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server (set your API key first)
export OPENAI_API_KEY=sk-...
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build --platform linux/amd64 -t mle-bench-agent .

# Run the container
docker run -p 9009:9009 -e OPENAI_API_KEY=sk-... mle-bench-agent
```

## Testing

```bash
uv sync --extra test
uv run src/server.py &
uv run pytest --agent-url http://localhost:9009
```

## Publishing

Push to `main` or create a version tag (`git tag v1.0.0 && git push origin v1.0.0`) to trigger the CI/CD workflow that builds, tests, and publishes the Docker image to GitHub Container Registry.

Add `OPENAI_API_KEY` in Settings > Secrets and variables > Actions > Repository secrets.
