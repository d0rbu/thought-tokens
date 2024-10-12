import torch as th

from logger import logger
from typing import Any
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding


THOUGHT_TOKEN_FORMAT = "[THOUGHT_{i}]"


def add_thought_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, thought_token_embeddings: th.Tensor) -> None:
    """
    Add thought tokens to the model and tokenizer.

    Args:
        model: The model to which to add the thought tokens.
        tokenizer: The tokenizer to which to add the thought tokens.
        thought_token_embeddings: The thought token embeddings to add to the model and tokenizer.
    """
    thought_tokens = [THOUGHT_TOKEN_FORMAT.format(i=i) for i in range(thought_token_embeddings.shape[0])]
    logger.info(f"Adding {len(thought_tokens)} thought tokens to the model and tokenizer.")

    tokenizer.add_special_tokens({"additional_special_tokens": thought_tokens}, replace_additional_special_tokens=False)
    tokenizer._thought_tokens = thought_tokens
    tokenizer._thought_token_ids = tokenizer.convert_tokens_to_ids(thought_tokens)
    model.set_input_embeddings(th.cat([model.get_input_embeddings().weight, thought_token_embeddings], dim=0))


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
