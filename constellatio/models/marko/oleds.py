import torch
from pytorch_lightning import LightningModule
from transformers import GPT2Model, GPT2Config
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from ml_collections import ConfigDict


class MTLNashOLEDs(LightningModule):

    def __init__(self):
        pass