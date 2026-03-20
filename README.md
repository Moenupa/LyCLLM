# LyCLLM

A unified Continual Learning library for Language Models, based on [PyTorch Lightning][lightning].

[lightning]: https://github.com/Lightning-AI/pytorch-lightning

## Features

## Quick Start

We highly recommend using `uv` to manage the environment.

```sh
# uv venv                           # create virtual env
uv sync --extra cuda                # for cuda lightning
uv sync --extra npu                 # for npu lightning
```

<details><summary>Alternatively, install via pip or conda:</summary>

```sh
# conda or pip, with optional dependencies for lightning
pip install -e ".[cuda]"            # 'cuda', or 'npu'
```

</details>