import torch
from pytorch_lightning import LightningModule
from transformers import GPT2Model, GPT2Config
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from ml_collections import ConfigDict


class NRGPredictor(LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer

        # Configuration for GPT-2
        gpt2_config = GPT2Config(
            n_embd=32,
            n_layer=4,
            n_head=4,
            n_positions=40,
            vocab_size=tokenizer.vocab_size,
        )

        # Model and other components based on the configuration
        self.model = GPT2Model(gpt2_config)
        self.emb2nrg = nn.Linear(gpt2_config.n_embd, 1)
        self.loss_fn = nn.MSELoss()

        # Training configuration
        self.learning_rate = 1e-3
        self.warmup_steps = 100

    def forward(self, **batch):
        """Forward pass to compute energy predictions from embeddings."""
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        emb = self.model(**inputs).last_hidden_state[:, 0, :]
        nrg = self.emb2nrg(emb)
        return nrg

    def training_step(self, batch, batch_idx):
        """Processes a single batch of data for training."""
        outputs = self(**batch)
        loss = self.loss_fn(outputs, batch["labels"])
        self.log("train_loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Processes a single batch of data for validation."""
        return self._evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        """Processes a single batch of data for testing."""
        return self._evaluate(batch, stage="test")

    def _evaluate(self, batch, stage=None):
        """Helper method to evaluate the model during validation/testing."""
        outputs = self(**batch)
        loss = self.loss_fn(outputs, batch["labels"])
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return {f"{stage}_loss": loss}

    def configure_optimizers(self):
        """Set up the optimizer and the learning rate scheduler."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, self.warmup_steps * 5
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
