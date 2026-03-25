import warnings

import pytest
from PIL import Image

from lycllm.data.datamodule import MultiModalDataModule, get_datasets
from lycllm.hparams.data_args import DataArguments
from lycllm.hparams.model_args import ModelArguments


@pytest.fixture
def model_args(vlm_path: str):
    args = ModelArguments(model_name_or_path=vlm_path, trust_remote_code=True)
    return args


@pytest.fixture
def data_args():
    args = DataArguments(
        dataset_kwargs=[{"path": "lmms-lab/LLaVA-NeXT-Data", "split": "train"}],
        streaming=True,
        buffer_size=1000,
        cutoff_len=512,
    )
    return args


def test_get_datasets(data_args: DataArguments):
    assert data_args.dataset_kwargs is not None
    datasets = get_datasets(*data_args.dataset_kwargs, data_args=data_args)
    assert len(datasets) == 1


@pytest.fixture
def real_datamodule(model_args: ModelArguments, data_args: DataArguments):
    dm = MultiModalDataModule(model_args, data_args, None)  # ty:ignore[invalid-argument-type]
    dm.setup()
    return dm


def test_datamodule_setup(
    real_datamodule: MultiModalDataModule, model_args: ModelArguments
):
    assert real_datamodule.tokenizer is not None
    assert real_datamodule.processor is not None


def test_collate_fn(real_datamodule: MultiModalDataModule):
    mock_image = Image.new("RGB", (32, 32))
    batch = [
        {
            "id": "1",
            "conversations": [
                {"from": "human", "value": "<image>Hello"},
                {"from": "gpt", "value": "Hi there"},
            ],
            "image": mock_image,
        }
    ]
    output = real_datamodule.collate_fn(batch)

    assert "input_ids" in output
    assert "labels" in output
    assert "sample_ids" in output
    assert output["sample_ids"] == ["1"]
    assert output["labels"].shape == output["input_ids"].shape

    warnings.warn(
        "collate log \n"
        + "\n".join(
            f"{k:>25}:{str(type(v).__name__):<15}={str(v)[:20]:<20}...; shape=({v.shape if hasattr(v, 'shape') else len(v)})"
            for k, v in output.items()
        ),
        FutureWarning,
    )
