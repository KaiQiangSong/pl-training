import pytorch_lightning as pl
from transformers import BartForConditionalGeneration

class Summarizer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")