# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Self

import torch
from omegaconf import OmegaConf


@dataclass
class BaseModelArguments:
    r"""Arguments pertaining to the model."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: str | None = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."
        },
    )
    resize_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to resize the tokenizer vocab and the embedding layers."
        },
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."
        },
    )
    add_tokens: str | None = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: str | None = field(
        default=None,
        metadata={
            "help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    new_special_tokens_config: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to YAML config with special token descriptions for semantic initialization. "
                "If set, this takes precedence over add_special_tokens. "
                "YAML format: {'<token>': 'description text', ...}"
            )
        },
    )
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = (
        field(
            default="noise_init",
            metadata={
                "help": (
                    "Initialization method for new special tokens: "
                    "'noise_init' (default, random noise around mean), "
                    "'desc_init' (semantic initialization from descriptions), "
                    "'desc_init_w_noise' (semantic + random noise). "
                    "Note: 'desc_init' methods require new_special_tokens_config."
                )
            },
        )
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={
            "help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."
        },
    )
    mixture_of_depths: Literal["convert", "load"] | None = field(
        default=None,
        metadata={
            "help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."
        },
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: float | None = field(
        default=None,
        metadata={
            "help": "Coefficient of the auxiliary router loss in mixture-of-experts model."
        },
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(
        default=True,
        metadata={"help": "Whether or not to use reentrant gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_kv_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    use_v1_kernels: bool | None = field(
        default=False,
        metadata={
            "help": "Whether or not to use high-performance kernels in training."
        },
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."
        },
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if (
            self.adapter_name_or_path is not None
        ):  # support merging multiple lora weights
            self.adapter_name_or_path = [  # ty:ignore[invalid-assignment]
                path.strip() for path in self.adapter_name_or_path.split(",")
            ]

        if self.add_tokens is not None:  # support multiple tokens
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",")]  # ty:ignore[invalid-assignment]

        # Process special tokens with priority: new_special_tokens_config > add_special_tokens
        if self.new_special_tokens_config is not None:
            # Priority 1: Load from YAML config (extracts both tokens and descriptions)
            try:
                cfg = OmegaConf.load(self.new_special_tokens_config)
                token_descriptions = OmegaConf.to_container(cfg)

                if not isinstance(token_descriptions, dict):
                    raise ValueError(
                        f"YAML config must be a dictionary mapping tokens to descriptions. "
                        f"Got: {type(token_descriptions)}"
                    )

                # Extract token list from config keys
                extracted_tokens = list(token_descriptions.keys())

                # Override add_special_tokens with extracted tokens (as list)
                self.add_special_tokens = extracted_tokens  # ty:ignore[invalid-assignment]

                # Store descriptions internally for later use (internal attribute)
                self._special_token_descriptions = token_descriptions

            except Exception as e:
                raise e

        elif self.add_special_tokens is not None:
            # Priority 2: Use simple comma-separated string (no descriptions)
            self.add_special_tokens = [  # ty:ignore[invalid-assignment]
                token.strip() for token in self.add_special_tokens.split(",")
            ]
            self._special_token_descriptions = None

        else:
            # No special tokens to add
            self._special_token_descriptions = None

        # Validate init method
        if self.init_special_tokens in ["desc_init", "desc_init_w_noise"]:
            if self._special_token_descriptions is None:
                self.init_special_tokens = "noise_init"


@dataclass
class ProcessorArguments:
    r"""Arguments pertaining to the image processor."""

    image_max_pixels: int = field(
        default=768 * 768,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=32 * 32,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    image_do_pan_and_scan: bool = field(
        default=False,
        metadata={"help": "Use pan and scan to process image for gemma3."},
    )
    crop_to_patches: bool = field(
        default=False,
        metadata={"help": "Whether to crop the image to patches for internvl."},
    )
    video_max_pixels: int = field(
        default=256 * 256,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    video_min_pixels: int = field(
        default=16 * 16,
        metadata={"help": "The minimum number of pixels of video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={"help": "Whether or not to use audio in video inputs."},
    )
    audio_sampling_rate: int = field(
        default=16000,
        metadata={"help": "The sampling rate of audio inputs."},
    )

    def __post_init__(self):
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError(
                "`image_max_pixels` cannot be smaller than `image_min_pixels`."
            )

        if self.video_max_pixels < self.video_min_pixels:
            raise ValueError(
                "`video_max_pixels` cannot be smaller than `video_min_pixels`."
            )


@dataclass
class ModelArguments(
    ProcessorArguments,
    BaseModelArguments,
):
    r"""Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.

    The class on the most right will be displayed first.
    """

    _model_max_length: int | None = field(
        default=None,
        metadata={
            "help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."
        },
    )

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)
        ProcessorArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {
            k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()
        }
        return args
