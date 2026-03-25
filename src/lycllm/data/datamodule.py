import lightning as L
import torch
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

from ..extras.constants import (
    AUDIO_PLACEHOLDER,
    IGNORE_INDEX,
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    get_seed,
)
from ..hparams.data_args import DataArguments
from ..hparams.model_args import ModelArguments
from ..model.loader import load_tokenizer


def get_datasets(
    *load_dataset_kwargs: dict, data_args: DataArguments
) -> list[Dataset | IterableDataset]:
    datasets = []
    shuffle_kwargs = {"seed": get_seed()}
    if data_args.streaming:
        shuffle_kwargs |= {"buffer_size": data_args.buffer_size}

    for kwargs in load_dataset_kwargs:
        kwargs.setdefault("streaming", data_args.streaming)
        kwargs.setdefault("split", "train")

        ds = load_dataset(**kwargs)
        assert isinstance(ds, (Dataset, IterableDataset)), (
            f"Expected a Dataset or IterableDataset, but got {type(ds)}"
        )
        ds = ds.shuffle(**shuffle_kwargs)
        datasets.append(ds)
    return datasets


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        train_args: Seq2SeqTrainingArguments,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_args: ModelArguments = model_args
        self.data_args: DataArguments = data_args
        self.train_args: Seq2SeqTrainingArguments = train_args

        # values to be lazy-init
        self.tokenizer: PreTrainedTokenizer = None  # ty:ignore[invalid-assignment]
        self.processor: ProcessorMixin = None  # ty:ignore[invalid-assignment]

        self.train_dataset: list[Dataset | IterableDataset] | None = None
        self.memory_dataset: list[Dataset | IterableDataset] | None = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.tokenizer is None:
            out = load_tokenizer(self.model_args)
            self.tokenizer = out["tokenizer"]
            self.processor = out["processor"]

        if self.train_dataset is not None:
            return

        if self.data_args.dataset_kwargs is not None:
            self.train_dataset = get_datasets(
                *self.data_args.dataset_kwargs, data_args=self.data_args
            )

        if self.data_args.memory_dataset_kwargs is not None:
            self.memory_dataset = get_datasets(
                *self.data_args.memory_dataset_kwargs, data_args=self.data_args
            )

    def _get_prefix_input_ids(self, messages, image=None):
        """Return input_ids for one message prefix using the training processor path."""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            padding=False,
            truncation=True,
            max_length=self.data_args.cutoff_len,
            return_tensors=None,
        )
        return inputs["input_ids"][0]

    def _build_assistant_labels(
        self, messages, image, input_ids_row, attention_mask_row
    ):
        """Keep labels only on assistant turns."""
        seq_len = int(attention_mask_row.sum().item())
        labels = torch.full_like(input_ids_row, IGNORE_INDEX)

        prefix_lens = [0]
        for end in range(1, len(messages)):
            prefix_ids = self._get_prefix_input_ids(messages[:end], image=image)
            prefix_lens.append(min(len(prefix_ids), seq_len))
        prefix_lens.append(seq_len)

        for turn_idx, (start, end) in enumerate(zip(prefix_lens[:-1], prefix_lens[1:])):
            if messages[turn_idx]["role"] == "assistant" and end > start:
                labels[start:end] = input_ids_row[start:end]

        return labels

    def collate_fn(self, batch):
        processor = self.processor
        texts, images, sample_ids = [], [], []
        batch_messages, batch_image_objs = [], []
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
        }

        for sample in batch:
            image = sample.get("image")
            need_image = image is not None
            image_rgb = None
            messages = []

            for turn in sample.get("conversations", []):
                role = role_map.get(turn.get("from"))
                text = turn.get("value") or ""
                # Remove raw <image>/<video>/<audio> markers from text.
                # They will be added back as structured content by the chat template.
                text = (
                    text.replace(IMAGE_PLACEHOLDER, "")
                    .replace(VIDEO_PLACEHOLDER, "")
                    .replace(AUDIO_PLACEHOLDER, "")
                    .strip()
                )
                content = [{"type": "text", "text": text}]
                # Insert the image only into the first user turn.
                if role == "user" and need_image:
                    content.insert(0, {"type": "image"})
                    need_image = False

                messages.append({"role": role, "content": content})

            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            sample_ids.append(sample.get("id"))
            batch_messages.append(messages)

            if image is not None and not need_image:
                image_rgb = image.convert("RGB")
                images.append(image_rgb)
            batch_image_objs.append(image_rgb)

        model_inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            truncation=True,
            max_length=self.data_args.cutoff_len,
            return_tensors="pt",
        )

        labels = torch.stack(
            [
                self._build_assistant_labels(
                    messages, image_rgb, input_ids_row, attention_mask_row
                )
                for messages, image_rgb, input_ids_row, attention_mask_row in zip(
                    batch_messages,
                    batch_image_objs,
                    model_inputs["input_ids"],
                    model_inputs["attention_mask"],
                )
            ]
        )

        labels.masked_fill_(model_inputs["attention_mask"] == 0, IGNORE_INDEX)

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = sample_ids
        return model_inputs

    @property
    def _dataset(self) -> list[Dataset | IterableDataset]:
        return (self.train_dataset or []) + (self.memory_dataset or [])

    def train_dataloader(self):
        combined_dataset = self._dataset
        if len(combined_dataset) == 0:
            raise RuntimeError("No dataset available for training.")
        elif len(combined_dataset) == 1:
            dataset = combined_dataset[0]
        else:
            dataset = interleave_datasets(
                self._dataset,  # ty:ignore[invalid-argument-type]
                probabilities=self.data_args._interleave_probs,
                seed=get_seed(),
                stopping_strategy=self.data_args.interleave_strategy,
            )

        assert isinstance(dataset, (Dataset, IterableDataset)), (
            f"Expected a Dataset or IterableDataset after interleaving, but got {type(dataset)}"
        )
        if self.data_args.max_samples is not None:
            if isinstance(dataset, IterableDataset) and self.data_args.streaming:
                dataset = dataset.take(self.data_args.max_samples)
            elif (
                isinstance(dataset, Dataset)
                and len(dataset) > self.data_args.max_samples
            ):
                dataset = dataset.select(range(self.data_args.max_samples))
            else:
                raise RuntimeError(
                    f"Failed to apply max_samples={self.data_args.max_samples} to {type(dataset)}."
                )

        return DataLoader(
            dataset,  # ty:ignore[invalid-argument-type]
            batch_size=self.train_args.per_device_train_batch_size,
            num_workers=self.train_args.dataloader_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.train_args.dataloader_num_workers > 0,
            drop_last=False,
        )
