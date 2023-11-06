import torch
import pytorch_lightning as pl
from transformers import GPT2Model
from transformers import GPT2Config
from transformers import get_cosine_schedule_with_warmup

import torch.nn as nn

from transformers import GPT2Config, GPT2Model
from ml_collections import ConfigDict



class NRGPredictor(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()

        cfg = ConfigDict()
        cfg.gptcfg = gpt2cfg = ConfigDict()
        gpt2cfg.n_embd = 32
        gpt2cfg.n_layer = 4
        gpt2cfg.n_head = 4
        gpt2cfg.n_positions = 40
        cfg.lr = 1e-3
        cfg.warmup_steps = 100
        
        gpt2cfg = GPT2Config(
            **{k: v for k, v in cfg.gptcfg.items() if k in GPT2Config().to_dict()}
        )
        model = GPT2Model(gpt2cfg)
        
        self.cfg = cfg
        gpt2cfg.vocab_size = tokenizer.vocab_size
        self.save_hyperparameters()
        gpt2cfg = GPT2Config(
            **{k: v for k, v in cfg.gptcfg.items() if k in GPT2Config().to_dict()}
        )
        self.model = GPT2Model(gpt2cfg)
        self.emb2nrg = nn.Linear(32, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, **batch):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        emb = self.model(**inputs)[0][:, 0, :]
        nrg = self.emb2nrg(emb)
        return nrg


    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.loss_fn(outputs, batch["labels"])
        self.log("train_loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        # Call the evaluate function and directly return its result
        return self._evaluate(batch, stage="test")

    def _evaluate(self, batch, stage=None):
        outputs = self(**batch)
        loss = self.loss_fn(outputs, batch["labels"])
        if stage:
            # Log the loss at the end of the epoch, not at each step.
            # Also, ensure the loss is logged only on validation/test stages
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        # Return the loss as part of a dictionary with the key matching your logging
        return {f"{stage}_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.cfg.warmup_steps, self.cfg.warmup_steps * 5
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
