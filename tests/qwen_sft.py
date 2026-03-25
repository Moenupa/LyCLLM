import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import Seq2SeqTrainingArguments

from lycllm.data.datamodule import MultiModalDataModule
from lycllm.hparams.data_args import DataArguments
from lycllm.hparams.finetuning_args import FinetuningArguments
from lycllm.hparams.model_args import ModelArguments
from lycllm.learner.qwen3vl import Qwen3VLSFTModule


def test_qwen_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
):
    L.seed_everything(42, workers=True)
    output_dir = "saves/llava_onevision_05b_zero3_sft_1epoch"

    dm = MultiModalDataModule(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    lit_model = Qwen3VLSFTModule(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="epoch{epoch:02d}",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor=None,
    )

    _ = DeepSpeedStrategy(
        config={
            "zero_optimization": {
                "stage": 3,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
            },
            "gradient_clipping": 1.0,
        }
    )

    trainer = L.Trainer(
        accelerator="gpu",
        # strategy=strategy,
        strategy="auto",
        devices="auto",
        max_steps=100,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        logger=WandbLogger(
            name="llava_onevision_05b_zero3_sft",
            project="lycllm",
            log_model=False,
        ),
        default_root_dir=output_dir,
    )

    trainer.fit(lit_model, datamodule=dm)

    hf_save_dir = os.path.join(output_dir, "hf_model")
    os.makedirs(hf_save_dir, exist_ok=True)

    trainer.strategy.barrier()
    if trainer.is_global_zero:
        lit_model.model.save_pretrained(hf_save_dir)
        dm.processor.save_pretrained(hf_save_dir)
    trainer.strategy.barrier()

    print("Training completed and model saved to:", hf_save_dir)


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_class_arguments(ModelArguments, "model")
    parser.add_class_arguments(DataArguments, "data")
    parser.add_class_arguments(Seq2SeqTrainingArguments, "train")
    parser.add_class_arguments(FinetuningArguments, "finetune")
    args = parser.parse_args()

    # convert namespace to dataclass
    args.model = ModelArguments(
        **{k: v for k, v in vars(args.model).items() if not k.startswith("_")}
    )
    args.data = DataArguments(
        **{k: v for k, v in vars(args.data).items() if not k.startswith("_")}
    )
    args.train = Seq2SeqTrainingArguments(
        **{k: v for k, v in vars(args.train).items() if not k.startswith("_")}
    )
    args.finetune = FinetuningArguments(
        **{k: v for k, v in vars(args.finetune).items() if not k.startswith("_")}
    )
    args.model._model_max_length = args.data.cutoff_len

    test_qwen_sft(args.model, args.data, args.train, args.finetune)
