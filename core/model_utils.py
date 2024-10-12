import torch as th

from torch import nn
from typing import Callable, Any
from transformers import PreTrainedModel, BatchEncoding


THOUGHT_TOKEN_EMBEDDERS = []


def dependent_thought_token_embed_modifier(embedding_func: Callable[[th.Tensor, th.LongTensor, th.BoolTensor], th.Tensor]) -> Callable[[PreTrainedModel, BatchEncoding], BatchEncoding]:
    """
    Create a thought token embedder that embeds thought tokens in a sequence with an embedding function instead of using the tokenizer's input embeddings.
    Called dependent_thought_token_embedder because we pass the full sequence, including the standard tokens, to the thought token embedding function.

    Args:
        embedding_func: The thought token embedding function. Should accept at least the original embedded sequence and the thought token mask.

    Returns:
        The thought token embedder. Args and kwargs are passed to the embedding function.
    """

    def embed_thought_tokens(model: PreTrainedModel, encoding: BatchEncoding, *args: Any, **kwargs: Any) -> BatchEncoding:
        """
        Embed thought tokens in a sequence with an embedding function instead of using the tokenizer's input embeddings.

        Args:
            model: The model to use for embedding the encoding.
            encoding: The encoding to embed the thought tokens in.
            args: The arguments to pass to the embedding function.
            kwargs: The keyword arguments to pass to the embedding function.

        Returns:
            The encoding with the embedded sequence.
        """
        token_type_ids: th.LongTensor = encoding.pop("token_type_ids", None)
        assert token_type_ids is not None, "The encoding must have token type IDs to serve as a thought token mask."

        thought_token_mask: th.BoolTensor = token_type_ids > 0
        standard_token_mask: th.BoolTensor = ~thought_token_mask

        input_ids = encoding.pop("input_ids")

        embed: nn.Embedding = model.get_input_embeddings()
        original_embedded_sequence: th.Tensor = embed(input_ids)
        new_embedded_sequence = embedding_func(original_embedded_sequence, input_ids, thought_token_mask, *args, **kwargs)

        assert new_embedded_sequence.shape == original_embedded_sequence.shape, "The new embedded sequence must have the same shape as the original embedded sequence."
        assert new_embedded_sequence.device == original_embedded_sequence.device, "The new embedded sequence must be on the same device as the original embedded sequence."
        assert new_embedded_sequence.dtype == original_embedded_sequence.dtype, "The new embedded sequence must have the same data type as the original embedded sequence."
        assert new_embedded_sequence[standard_token_mask].allclose(original_embedded_sequence[standard_token_mask]), "The embedding function must not change the normal tokens."

        encoding["inputs_embeds"] = new_embedded_sequence

        return encoding

    embed_thought_tokens.__name__ = embedding_func.__name__

    THOUGHT_TOKEN_EMBEDDERS.append(embed_thought_tokens)

    return embed_thought_tokens


def thought_token_embed_modifier(embedding_func: Callable[[th.Tensor], th.Tensor]) -> Callable[[PreTrainedModel, BatchEncoding], BatchEncoding]:
    """
    Create a thought token embedder that embeds thought tokens in a sequence with an embedding function independent of the standard token embeddings.

    Args:
        embedding_func: The thought token embedding function.

    Returns:
        The thought token embedder.
    """

    def dependent_thought_token_embedding_func(embeds: th.Tensor, input_ids: th.LongTensor, thought_token_mask: th.BoolTensor) -> th.Tensor:
        """
        Thought token embedding function wrapper that passes only the thought tokens to the independent embedding function.

        Args:
            embeds: The original embedded sequence.
            thought_token_mask: The thought token mask.

        Returns:
            The new embedded sequence.
        """
        embeds[thought_token_mask] = embedding_func(embeds[thought_token_mask])
        return embeds

    dependent_thought_token_embedding_func.__name__ = embedding_func.__name__

    return dependent_thought_token_embed_modifier(dependent_thought_token_embedding_func)


def thought_token_embedder(embedding_func: Callable[[th.Size], th.Tensor]) -> Callable[[PreTrainedModel, BatchEncoding], BatchEncoding]:
    """
    Create a thought token embedder that only uses the shape to create thought token embeddings.

    Args:
        embedding_func: The thought token embedding function.

    Returns:
        The thought token embedder.
    """

    def embed_thought_tokens(embedded_thought_tokens: th.Tensor) -> th.Tensor:
        """
        Create thought tokens embeddings that adhere to the original embedding shape.

        Args:
            embedded_thought_tokens: The original embedded thought tokens.

        Returns:
            The new embedded thought tokens.
        """
        return embedding_func(embedded_thought_tokens.shape)

    embed_thought_tokens.__name__ = embedding_func.__name__

    return thought_token_embed_modifier(embed_thought_tokens)



@thought_token_embedder
def normal_thought_embedding(embed_shape: th.Size, mean: float = 0.0, std: float = 1.0) -> th.Tensor:
    """
    Initialize thought token embeddings with a normal distribution.

    Args:
        embed_shape: The shape of the thought token embeddings.
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.

    Returns:
        The new embedded thought tokens.
    """

    return th.normal(mean, std, embed_shape)


@thought_token_embedder
def orthogonal_thought_embedding(embed_shape: th.Size) -> th.Tensor:
    """
    Initialize thought token embeddings with orthogonal vectors.

    Args:
        embed_shape: The shape of the thought token embeddings.

    Returns:
        The new embedded thought tokens.
    """

    return th.nn.init.orthogonal_(th.empty(embed_shape))
