import os
import torch as th
import pytorch_lightning as L  # sorry lightning, pkgs.python311Packages.lightning in my nativeBuildInputs doesnt work
import torch.nn.functional as F

from torch import nn
from torch.optim import adam
from torch.utils.data import DataLoader
from core.utils import prepare_model_and_tokenizer_for_thought_tokens
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer


class ThoughtTokenLM(L.LightningModule):
    def __init__(self, model: PreTrainedModel | str, tokenizer: PreTrainedTokenizer | str, thought_token_embeddings: th.Tensor | int = 4096):
        super().__init__()

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        prepare_model_and_tokenizer_for_thought_tokens(model, tokenizer, thought_token_embeddings)

        self.model = model
        self.tokenizer = tokenizer
