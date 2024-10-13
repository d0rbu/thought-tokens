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
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, AdamW


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
            num_thought_tokens: int = self.model.thought_token_embeddings().shape[0]
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
        batch: th.Tensor,
        batch_idx: int
    ) -> th.Tensor:
        # TODO: we need to come up with a good training setup that allows us to both optimize free thought embeddings on a given batch and also optimize the thought embeddings themselves across batches
        pass
