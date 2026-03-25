# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: str | None = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        },
    )
    dataset_kwargs: list[dict] | None = field(
        default_factory=list,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."
        },
    )
    memory_dataset_kwargs: list[dict] | None = field(
        default_factory=list,
        metadata={
            "help": "The name of dataset(s) to use for memory. Use commas to separate multiple datasets."
        },
    )
    media_dir: str | None = field(
        default=None,
        metadata={
            "help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."
        },
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to mask the history and train on the last turn only."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={
            "help": "Size of the buffer to randomly sample examples from in dataset streaming."
        },
    )
    interleave_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = field(
        default="all_exhausted_without_replacement",
        metadata={
            "help": "Strategy to use in dataset mixing (undersampling/oversampling/sampling w.o. replacement)."
        },
    )
    interleave_probs: str | None = field(
        default=None,
        metadata={
            "help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."
        },
    )
    _interleave_probs: list[float] | None = None
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: int | None = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."
        },
    )
    packing: bool | None = field(
        default=None,
        metadata={
            "help": "Enable sequences packing in training. Will automatically enable in pre-training."
        },
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: str | None = field(
        default=None,
        metadata={
            "help": "Tool format to use for constructing function calling examples."
        },
    )
    default_system: str | None = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: bool | None = field(
        default=True,
        metadata={
            "help": "Whether or not to enable thinking mode for reasoning models."
        },
    )
    tokenized_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use a shared file system for the datasets."
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        if self.interleave_probs is not None:
            self._interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            expected_len = len(self.dataset_kwargs or []) + len(
                self.memory_dataset_kwargs or []
            )
            if expected_len != len(self._interleave_probs):
                raise ValueError(
                    "The length of dataset and interleave probs should be identical."
                )

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
