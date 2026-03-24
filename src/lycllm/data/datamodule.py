import lightning as L
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..extras.constants import get_seed
from ..hparams.data_args import DataArguments
from ..hparams.model_args import ModelArguments
from ..model.loader import load_tokenizer


def get_datasets(*load_dataset_kwargs: dict, data_args: DataArguments) -> list[Dataset]:
    datasets = []
    shuffle_kwargs = {"seed": get_seed()}
    if data_args.streaming:
        shuffle_kwargs |= {"buffer_size": data_args.buffer_size}

    for kwargs in load_dataset_kwargs:
        kwargs["streaming"] = data_args.streaming
        ds = load_dataset(**kwargs)
        if isinstance(ds, DatasetDict) and "train" not in ds:
            raise ValueError("DatasetDict should have a 'train' split.")
        elif isinstance(ds, DatasetDict):
            ds = ds["train"]
        else:
            assert isinstance(ds, Dataset), f'Expected a Dataset, but got {type(ds)}'
        ds = ds.shuffle(**shuffle_kwargs)
        datasets.append(ds)
    return datasets


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_args: ModelArguments = model_args
        self.data_args: DataArguments = data_args

        # values to be lazy-init
        self.tokenizer: PreTrainedTokenizer = None  # ty:ignore[invalid-assignment]
        self.processor: ProcessorMixin = None  # ty:ignore[invalid-assignment]

        self.train_dataset: list[Dataset] | None = None
        self.memory_dataset: list[Dataset] | None = None

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
            self.train_dataset = get_datasets(*self.data_args.dataset_kwargs, data_args=self.data_args)

        if self.data_args.memory_dataset_kwargs is not None:
            self.memory_dataset = get_datasets(*self.data_args.memory_dataset_kwargs, data_args=self.data_args)

    def collate_fn(self, batch):
        texts, images, sample_ids = [], [], []

        image_token = getattr(self.processor, "image_token", "<image>")
        video_token = getattr(self.processor, "video_token", "<video>")

        for sample in batch:
            conversations = sample.get("conversations") or []
            if not conversations:
                continue

            messages = []
            used_image = False
            has_assistant = False

            for turn in conversations:
                role = {
                    "human": "user",
                    "user": "user",
                    "gpt": "assistant",
                    "assistant": "assistant",
                }.get(turn.get("from"))
                if role is None:
                    continue

                text = (
                    (turn.get("value") or "")
                    .replace(image_token, "")
                    .replace(video_token, "")
                    .strip()
                )
                if not text:
                    continue

                if role == "user":
                    content = [{"type": "text", "text": text}]
                    if sample.get("image") is not None and not used_image:
                        content.insert(0, {"type": "image"})
                        used_image = True
                    messages.append({"role": "user", "content": content})
                else:
                    has_assistant = True
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        }
                    )

            if not messages or not has_assistant:
                continue

            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            sample_ids.append(sample.get("id"))

            if used_image:
                images.append(sample["image"].convert("RGB"))

        if not texts:
            raise RuntimeError("No valid samples found in batch.")

        model_inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            truncation=True,
            max_length=self.data_args.cutoff_len,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        # TODO: labels 应该去掉 user 输入, 其他 code, 例如 llama factory 怎么处理的？
        labels.masked_fill_(
            model_inputs["attention_mask"] == 0, self.hparams.ignore_index
        )

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels.masked_fill_(
                model_inputs["input_ids"] == image_token_id, self.hparams.ignore_index
            )

        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None:
            labels.masked_fill_(
                model_inputs["input_ids"] == video_token_id, self.hparams.ignore_index
            )

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = sample_ids
        return model_inputs

    @property
    def _dataset(self) -> list[Dataset]:
        return (self.train_dataset or []) + (self.memory_dataset or [])

    def train_dataloader(self):
        combined_dataset = self._dataset
        if len(combined_dataset) == 0:
            raise RuntimeError("No dataset available for training.")
        elif len(combined_dataset) == 1:
            dataset = combined_dataset[0]
        else:
            dataset = (
                interleave_datasets(
                    self._dataset,
                    probabilities=self.data_args._interleave_probs,
                    seed=get_seed(),
                    stopping_strategy=self.data_args.interleave_strategy,
                )
            )

        return DataLoader(
            dataset,  # ty:ignore[invalid-argument-type]
            batch_size=self.data_args.preprocessing_batch_size,
            num_workers=self.data_args.preprocessing_num_workers or 8,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.data_args.preprocessing_num_workers is not None,
            drop_last=False,
        )
