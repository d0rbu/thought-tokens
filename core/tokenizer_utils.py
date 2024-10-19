import torch as th
import torch.nn as nn

from typing import Any
from .logger import logger
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding


LOGGER_NAME = "thought_tokens"
THOUGHT_TOKEN_FORMAT = "[THOUGHT_{i}]"


def add_thought_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, thought_token_embeddings: th.Tensor, thought_token_unembeddings: th.Tensor | None = None) -> None:
    """
    Add thought tokens to the model and tokenizer.

    Args:
        model: The model to which to add the thought tokens.
        tokenizer: The tokenizer to which to add the thought tokens.
        thought_token_embeddings: The thought token embeddings to add to the model, also determines the number of thought tokens to add.
        thought_token_unembeddings: The thought token unembeddings to add to the model, the shape must match the thought token embeddings' shape. If the model has tied embeddings, do not provide this argument.
    """
    logger.info(f"Adding {thought_token_embeddings.shape[0]} thought tokens to the model and tokenizer.")

    new_embeddings = nn.Parameter(th.cat([model.get_input_embeddings().weight, thought_token_embeddings], dim=0))
    model.set_input_embeddings(nn.Embedding.from_pretrained(new_embeddings))

    has_tied_embeddings: bool = getattr(model.config, "tie_word_embeddings", False)
    if thought_token_unembeddings is None:
        assert has_tied_embeddings, "Thought token unembeddings not provided, but the model does not have tied embeddings."
        model.tie_weights()
    else:
        assert not has_tied_embeddings, "Thought token unembeddings provided, but the model has tied embeddings."
        assert thought_token_embeddings.shape == thought_token_unembeddings.shape, "The thought token embeddings and unembeddings' shapes must be equal."
        new_unembeddings = th.cat([model.get_output_embeddings().weight, thought_token_unembeddings], dim=0)
        new_unembedding_linear = nn.Linear(new_unembeddings.shape[1], new_unembeddings.shape[0], bias=False)

        with th.no_grad():
            new_unembedding_linear.weight.copy_(new_unembeddings)

        model.set_output_embeddings(new_unembedding_linear)

    thought_tokens = [THOUGHT_TOKEN_FORMAT.format(i=i) for i in range(thought_token_embeddings.shape[0])]

    tokenizer.add_special_tokens({"additional_special_tokens": thought_tokens}, replace_additional_special_tokens=False)
    tokenizer._thought_tokens = thought_tokens
    tokenizer._thought_token_ids = tokenizer.convert_tokens_to_ids(thought_tokens)


def tokenize_with_thought_token_type_ids(tokenizer: PreTrainedTokenizer, **kwargs: Any) -> BatchEncoding:
    """
    Tokenize a sequence with thought token type IDs. Makes return_token_type_ids default to True.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        kwargs: The keyword arguments to pass to the tokenizer.

    Returns:
        The tokenized sequence with thought token type IDs.
    """
    return_token_type_ids = kwargs.pop("return_token_type_ids", True)
    encoding = tokenizer(**kwargs)

    if not return_token_type_ids:
        return encoding

    thought_token_ids = getattr(tokenizer, "_thought_token_ids", None)
    assert thought_token_ids is not None, "The tokenizer must have thought tokens to tokenize with thought token type IDs."

    token_type_ids = th.zeros_like(encoding["input_ids"])
    thought_token_ids = th.tensor(thought_token_ids, device=token_type_ids.device)

    token_type_ids[encoding["input_ids"].isin(thought_token_ids)] = 1
    encoding["token_type_ids"] = token_type_ids

    return encoding
