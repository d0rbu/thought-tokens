import torch as th

from logger import logger
from functools import partial
from model_utils import THOUGHT_TOKEN_EMBEDDERS
from transformers import PreTrainedModel, PreTrainedTokenizer
from tokenizer_utils import tokenize_with_thought_token_type_ids, add_thought_tokens


def prepare_model_and_tokenizer_for_thought_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, thought_token_embeddings: th.Tensor | int, thought_token_unembeddings: th.Tensor | None = None) -> None:
    """
    Prepare the model and tokenizer for thought tokens.

    Args:
        model: The model to which to add the thought tokens and utility functions.
        tokenizer: The tokenizer to which to add the thought tokens and utility functions.
        thought_token_embeddings: The thought token embeddings to add to the model and tokenizer. If an integer, the thought token embeddings will be randomly initialized with the given size.
        thought_token_unembeddings: The thought token unembeddings to add to the model (if it is not tied), the shape must match the thought token embeddings' shape. If None, they will be randomly initialized.
    """
    if isinstance(thought_token_embeddings, int):
        embedding_dim = model.get_input_embeddings().weight.shape[1]
        thought_token_embeddings = th.empty((thought_token_embeddings, embedding_dim))
        th.nn.init.xavier_uniform_(thought_token_embeddings)

    has_tied_embeddings: bool = getattr(model.config, "tie_word_embeddings", False)
    if thought_token_unembeddings is None:
        if not has_tied_embeddings:
            thought_token_unembeddings = th.empty_like(thought_token_embeddings)
            th.nn.init.xavier_uniform_(thought_token_unembeddings)
    else:
        assert not has_tied_embeddings, "Thought token unembeddings provided, but the model has tied embeddings."
        assert thought_token_embeddings.shape == thought_token_unembeddings.shape, "The thought token embeddings and unembeddings' shapes must be equal."

    add_thought_tokens(model, tokenizer, thought_token_embeddings, thought_token_unembeddings)

    tokenizer.__call__ = partial(tokenize_with_thought_token_type_ids, tokenizer=tokenizer)

    model.thought_token_embeddings = lambda: model.get_input_embeddings().weight[-thought_token_embeddings.shape[0]:]
    model.thought_token_unembeddings = lambda: model.get_output_embeddings().weight[-thought_token_embeddings.shape[0]:]

    for thought_token_embedder in THOUGHT_TOKEN_EMBEDDERS:
        setattr(model, thought_token_embedder.__name__, partial(thought_token_embedder, model=model))

    logger.info("Model and tokenizer prepared for thought tokens.")
    logger.info(f"Available thought token embedders: {', '.join([thought_token_embedder.__name__ for thought_token_embedder in THOUGHT_TOKEN_EMBEDDERS])}.")
    logger.debug(f"Thought tokens: {tokenizer._thought_tokens}.")
