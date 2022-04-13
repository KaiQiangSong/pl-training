import torch
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from deepspeed.ops.adam import FusedAdam
from transformers.optimization import get_linear_schedule_with_warmup

class Summarizer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model)

    def training_step(self, batch_data, batch_idx):
        encoder_input_ids, decoder_input_ids, labels = batch_data
        norm = float((labels.data != self.config.pad).float().sum())
        loss = self.model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )["loss"] / norm

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        # Using validation loss to do the validation
        encoder_input_ids, decoder_input_ids, labels = batch_data
        norm = float((labels.data != self.config.pad).float().sum())
        loss = self.model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )["loss"]
        return float(loss), float(norm)

    def validation_epoch_end(self, batch_outputs):
        total_loss = 0
        total_norm = 0
        for loss, norm in batch_outputs:
            total_loss += loss
            total_norm += norm
        avg_loss = total_loss / total_norm
        self.log("valid_loss", avg_loss)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.config.train_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
