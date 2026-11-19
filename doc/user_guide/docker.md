# Docker

The repository `Dockerfile` packages AutoREC's CLI on Python 3.10 slim, installs
Graphviz, installs the project, warms the AutoEIS runtime during the image build,
and runs commands as the unprivileged `appuser`. It is a convenient isolated
runtime, not a claim of bitwise reproducibility across machines.

## Build and inspect

From the repository root:

```bash
docker build -t autorec:latest .
docker run --rm autorec:latest --help
```

The warmup step can initialize Julia-backed dependencies and therefore makes
the image build network- and platform-sensitive. Plan for a longer first build.

## Mount runtime data

Do not bake mutable datasets, models, or results into the image. The supplied
Docker configuration uses these container paths:

| Container path | Purpose |
| --- | --- |
| `/app/data` | Input datasets |
| `/app/configs_yaml` | Configuration files |
| `/app/models` | Model exports |
| `/app/runs` | Training outputs |
| `/app/results` | Evaluation outputs |

Train with the container-specific demo configuration:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs_yaml:/app/configs_yaml" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/runs:/app/runs" \
  -v "$PWD/results:/app/results" \
  autorec:latest train \
  --config /app/configs_yaml/examples/demo_docker_pipeline_config.yaml \
  --output-dir /app/runs/demo_train
```

Evaluate the included model:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs_yaml:/app/configs_yaml" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/results:/app/results" \
  autorec:latest infer \
  --config /app/configs_yaml/examples/demo_docker_pipeline_config.yaml \
  --model /app/models/examples/agent_trained.keras \
  --indices 0 \
  --max-actions 24 \
  --use-training-env \
  --output-dir /app/results/inference_demo
```

On SELinux-enabled hosts, bind mounts may require an appropriate `:Z` or `:z`
label. On Windows, use shell-appropriate path syntax. The container user must
have permission to write to the mounted output directories.

## Open a shell

The image entry point is `autorec`. Override it for inspection:

```bash
docker run --rm -it --entrypoint /bin/bash autorec:latest
```

Record the image digest and configuration when using a container for reported
results. Rebuilding the same tag later can resolve newer base-image content.
