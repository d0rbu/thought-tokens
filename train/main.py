import os
import torch as th
import pytorch_lightning as L  # sorry lightning, pkgs.python311Packages.lightning in my nativeBuildInputs doesnt work
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn
from typing import Self, Callable
from functools import partialmethod
from torch.utils.data import DataLoader
from core.utils import prepare_model_and_tokenizer_for_thought_tokens
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, BatchEncoding, AdamW


class InterstitialThoughtTokenLM(L.LightningModule):
    def __init__(
        self: Self,
        model: PreTrainedModel | str,
        tokenizer: PreTrainedTokenizer | str,
        thought_token_embeddings: th.Tensor | int = 1024,
        thought_token_unembeddings: th.Tensor | None = None,
        unembedding_initialization_distance_func: Callable[[th.Tensor, th.Tensor], th.Tensor] = th.cdist,
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        prepare_model_and_tokenizer_for_thought_tokens(model, tokenizer, thought_token_embeddings, thought_token_unembeddings)

        self.model = model
        self.tokenizer = tokenizer
        self.unembedding_initialization_distance_func = unembedding_initialization_distance_func

    def on_fit_start(self: Self) -> None:
        if self.global_step > 0:
            return

        self.model.eval()

        with th.no_grad():
            num_thought_tokens: int = self.model.num_thought_tokens
            current_output_embeddings = self.model.get_output_embeddings().weight

            unembedding_initialization = self._cluster_final_token_embeddings(
                data=self.train_dataloader(),
                non_fixed_centroids=num_thought_tokens,
                fixed_centroids=current_output_embeddings,
                distance_func=self.unembedding_initialization_distance_func
            )
            current_output_embeddings[-num_thought_tokens:] = unembedding_initialization
            self.model.set_output_embeddings(current_output_embeddings)

            # for each unembedding vector, we find the nearest standard token unembedding and set the corresponding embedding accordingly. so k-nearest neighbors
            standard_token_unembeddings = current_output_embeddings[:-num_thought_tokens]
            nearest_embedding_indices = th.argmin(self.unembedding_initialization_distance_func(unembedding_initialization, standard_token_unembeddings), dim=-1)

            current_input_embeddings = self.model.get_input_embeddings().weight
            current_input_embeddings[-num_thought_tokens:] = current_input_embeddings[nearest_embedding_indices]
            self.model.set_input_embeddings(current_input_embeddings)

        self.model.train()

    def _cluster_final_token_embeddings(
        self: Self,
        data: DataLoader,
        non_fixed_centroids: th.Tensor | int,
        fixed_centroids: th.Tensor,
        distance_func: Callable[[th.Tensor, th.Tensor], th.Tensor] = th.cdist,
        max_iter: int = 100,
        distance_threshold: float | None = None
    ) -> th.Tensor:
        """
        Cluster the final token embeddings of the dataset when run through the model. Implemented for a distributed setting.

        Args:
            data: The data to cluster.
            non_fixed_centroids: The number of non-fixed centroids or the non-fixed centroids.
            fixed_centroids: The fixed centroids.
            distance_func: The distance function to use for clustering.
            max_iter: The maximum number of iterations.
            distance_threshold: The distance threshold to filter only data points that are not well fit to existing unembeddings. If None, we do not filter and instead do partial k-means.
        
        Returns:
            The new centroids.
        """
        do_partial_kmeans = distance_threshold is None

        final_token_embeddings = th.empty((0, self.model.config.hidden_size), device=self.model.device)
        for batch in data:
            outputs = self.model(**batch)
            last_hidden_state = outputs.last_hidden_state.detach()

            if not do_partial_kmeans:
                # filter out data points that are too close to the nearest existing unembedding
                unembedding_distances = th.min(distance_func(last_hidden_state, fixed_centroids), dim=-1)
                last_hidden_state = last_hidden_state[unembedding_distances > distance_threshold]

            final_token_embeddings = th.cat([final_token_embeddings, last_hidden_state], dim=0)

        if do_partial_kmeans:
            return self.partial_kmeans(data=final_token_embeddings, non_fixed_centroids=non_fixed_centroids, fixed_centroids=fixed_centroids, distance_func=distance_func, max_iter=max_iter)

        return self.kmeans(data=final_token_embeddings, non_fixed_centroids=non_fixed_centroids, distance_func=distance_func, max_iter=max_iter)

    def partial_kmeans(
        self: Self,
        data: th.Tensor,
        non_fixed_centroids: th.Tensor | int,
        fixed_centroids: th.Tensor | None = None,
        distance_func: Callable[[th.Tensor, th.Tensor], th.Tensor] = th.cdist,
        max_iter: int = 100
    ) -> th.Tensor:
        """
        Perform k-means clustering with a subset of fixed and non-fixed centroids. Implemented for a distributed setting.

        Args:
            data: The data to cluster.
            fixed_centroids: The fixed centroids.
            non_fixed_centroids: The number of non-fixed centroids or the non-fixed centroids.
            max_iter: The maximum number of iterations.

        Returns:
            The new centroids.
        """
        if isinstance(non_fixed_centroids, int):
            if self.global_rank == 0:
                non_fixed_centroids = data[th.randperm(data.shape[0])[:non_fixed_centroids]]
            else:
                non_fixed_centroids = th.empty((non_fixed_centroids, data.shape[1]), device=data.device)

            if dist.is_initialized():
                dist.broadcast(non_fixed_centroids, src=0)

        if fixed_centroids is None:
            fixed_centroids = th.empty((0, data.shape[1]), device=data.device)

        assert data.shape[1] == fixed_centroids.shape[1] == non_fixed_centroids.shape[1], "The data and centroids must have the same dimensionality."

        centroids = th.cat([fixed_centroids, non_fixed_centroids], dim=0)
        cluster_assignments = th.zeros(data.shape[0], dtype=th.long)

        local_data_size = th.tensor([data.shape[0]], device=data.device)
        data_sizes = th.empty((self.trainer.world_size,), device=data.device)

        if dist.is_initialized():
            dist.all_gather_into_tensor(data_sizes, local_data_size)
        else:
            data_sizes[0] = local_data_size

        relative_data_sizes = (data_sizes / data_sizes.sum()).unsqueeze(-1)
        local_means = th.empty((self.trainer.world_size, centroids.shape[1]), device=data.device)

        for _ in range(max_iter):
            new_cluster_assignments = th.argmin(distance_func(data, centroids), dim=-1)

            cluster_assignments_unchanged = (new_cluster_assignments == cluster_assignments).all()

            if dist.is_initialized():
                dist.all_reduce(cluster_assignments_unchanged, op=dist.ReduceOp.PRODUCT)

            if cluster_assignments_unchanged:
                break

            cluster_assignments = new_cluster_assignments

            for i in range(fixed_centroids.shape[0], centroids.shape[0]):
                local_mean = data[cluster_assignments == i].mean(dim=0)

                if dist.is_initialized():
                    dist.all_gather_into_tensor(local_means, local_mean)
                else:
                    local_means[0] = local_mean

                centroids[i] = (local_means * relative_data_sizes).sum(dim=0)

        return centroids[fixed_centroids.shape[0]:]

    kmeans = partialmethod(partial_kmeans, fixed_centroids=None)

    def training_step(
        self: Self,
        batch: BatchEncoding,
        batch_idx: int
    ) -> th.Tensor:
        batch_with_inserted_thought_tokens = self._insert_thought_tokens(batch)
        input_ids = batch_with_inserted_thought_tokens.input_ids
        attention_mask = batch_with_inserted_thought_tokens.attention_mask

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        return outputs.loss[attention_mask].sum()

    def _insert_thought_tokens(
        self: Self,
        batch: BatchEncoding
    ) -> BatchEncoding:
        self.model.eval()
        with th.no_grad():
            outputs = self.model(input_ids=batch.input_ids)
            predicted_token_ids = th.argmax(outputs.logits, dim=-1)  # (B, T)
            thought_token_mask = predicted_token_ids >= self.model.num_standard_tokens

            # expand the length dimension to insert the thought tokens
            max_sequence_expansion = thought_token_mask.sum(dim=-1).max()

            input_ids = th.cat([batch.input_ids, th.empty((batch.input_ids.shape[0], max_sequence_expansion), dtype=th.long, device=batch.input_ids.device)], dim=-1)
            attention_mask = th.cat([batch.attention_mask, th.empty((batch.attention_mask.shape[0], max_sequence_expansion), dtype=th.long, device=batch.attention_mask.device)], dim=-1)
            if "token_type_ids" in batch:
                token_type_ids = th.cat([batch.token_type_ids, th.empty((batch.token_type_ids.shape[0], max_sequence_expansion), dtype=th.long, device=batch.token_type_ids.device)], dim=-1)
            else:
                token_type_ids = th.zeros_like(input_ids)

            # we insert the corresponding thought tokens where the model predicts it should be
            cumulative_thought_tokens_inserted = thought_token_mask.cumsum(dim=-1)  # (B, T)

            # get the new indices of where each token should be, after the thought tokens are inserted
            current_indices = th.arange(input_ids.shape[-1], device=input_ids.device).repeat(input_ids.shape[0], 1)  # (B, T)
            new_indices = current_indices + cumulative_thought_tokens_inserted

            # move the standard tokens to their new indices
            input_ids.scatter_(dim=-1, index=new_indices, src=input_ids)
            attention_mask.scatter_(dim=-1, index=new_indices, src=attention_mask)
            token_type_ids.scatter_(dim=-1, index=new_indices, src=token_type_ids)

            # get indices of where the thought tokens should be inserted
            thought_token_indices = th.where(thought_token_mask)  # (N, 2)

            # offset the thought token indices by the cumulative thought tokens inserted
            thought_token_indices += cumulative_thought_tokens_inserted[thought_token_indices]

            # insert the thought tokens
            input_ids[thought_token_indices] = predicted_token_ids[thought_token_mask]
            attention_mask[thought_token_indices] = 1
            token_type_ids[thought_token_indices] = 1

        self.model.train()

        return BatchEncoding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
