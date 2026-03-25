import lightning as L
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Qwen3VLForConditionalGeneration,
    Seq2SeqTrainingArguments,
    get_scheduler,
)

from ..hparams.finetuning_args import FinetuningArguments
from ..hparams.model_args import ModelArguments


class Qwen3VLSFTModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_args: ModelArguments,
        training_args: Seq2SeqTrainingArguments,
        finetuning_args: FinetuningArguments,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])

        self.model_args = model_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )

        if finetuning_args.finetuning_type == "lora":
            lora_config = LoraConfig(
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                target_modules=finetuning_args.lora_target,
                lora_dropout=finetuning_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        elif finetuning_args.finetuning_type == "freeze":
            for name, param in self.model.named_parameters():
                if not any(
                    module in name
                    for module in finetuning_args.freeze_trainable_modules
                ):
                    param.requires_grad = False
        elif finetuning_args.finetuning_type == "full":
            pass  # keep all parameters trainable

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": self.training_args.weight_decay,
                },
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
        )

        # Learning rate scheduler
        # Default to continuous scheduler if not specified
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_warmup_steps = self.training_args.get_warmup_steps(num_training_steps)

        scheduler = get_scheduler(
            self.training_args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
