import torch as th
import torch.nn as nn
import pytorch_lightning as L  # sorry lightning, pkgs.python311Packages.lightning in my nativeBuildInputs doesnt work
import torch.distributed as dist

from itertools import chain
from functools import partialmethod
from typing import Self, Callable, Any
from torch.utils.data import DataLoader
from core.utils import prepare_model_and_tokenizer_for_thought_tokens
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, BatchEncoding, AdamW


class InterstitialThoughtTokenLM(L.LightningModule):
    def __init__(
        self: Self,
        model: PreTrainedModel | str,
        tokenizer: PreTrainedTokenizer | str,
        thought_token_embeddings: th.Tensor | int = 1024,
        thought_token_unembeddings: th.Tensor | int | None = None,
        unembedding_initialization_distance_func: Callable[[th.Tensor, th.Tensor], th.Tensor] = th.cdist,
        warmup_steps: int = 0,
        lr: float = 5e-5,
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
        self.warmup_steps = warmup_steps
        self.lr = lr

    def on_fit_start(self: Self) -> None:
        if self.global_step > 0:
            return

        self.model.eval()
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with th.no_grad():
            num_thought_tokens: int = self.model.num_thought_tokens
            output_head = self.model.get_output_embeddings()
            current_output_embeddings = output_head.weight.detach()

            unembedding_initialization = self._cluster_final_token_embeddings(
                data=self.trainer.datamodule.init_dataloader(),
                non_fixed_centroids=num_thought_tokens,
                fixed_centroids=current_output_embeddings,
                distance_func=self.unembedding_initialization_distance_func
            )
            current_output_embeddings[-num_thought_tokens:] = unembedding_initialization
            output_head.weight.copy_(current_output_embeddings)

            has_tied_embeddings: bool = getattr(self.model.config, "tie_word_embeddings", False)

            if not has_tied_embeddings:
                # for each unembedding vector, we find the nearest standard token unembedding and set the corresponding embedding accordingly. so k-nearest neighbors
                standard_token_unembeddings = current_output_embeddings[:-num_thought_tokens]
                nearest_embedding_indices = th.argmin(self.unembedding_initialization_distance_func(unembedding_initialization, standard_token_unembeddings), dim=-1)

                input_embed = self.model.get_input_embeddings()
                current_input_embeddings = input_embed.weight.detach()
                current_input_embeddings[-num_thought_tokens:] = current_input_embeddings[nearest_embedding_indices]
                input_embed.weight.copy_(current_input_embeddings)

                # free up memory
                del current_input_embeddings, standard_token_unembeddings, nearest_embedding_indices
            del current_output_embeddings, unembedding_initialization
            th.cuda.empty_cache()

        self.model.config.use_cache = use_cache
        self.model.train()

    def training_step(
        self: Self,
        batch: dict[str, Any],
        batch_idx: int
    ) -> th.Tensor:
        batch: BatchEncoding = self._batch_encode(batch)
        batch_with_inserted_thought_tokens = self._insert_thought_tokens(batch)
        input_ids, attention_mask, labels = batch_with_inserted_thought_tokens["input_ids"], batch_with_inserted_thought_tokens["attention_mask"], batch_with_inserted_thought_tokens["labels"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return outputs.loss

    def validation_step(
        self: Self,
        batch: dict[str, Any],
        batch_idx: int
    ) -> th.Tensor:
        batch: BatchEncoding = self._batch_encode(batch)
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        batch_with_inserted_thought_tokens = self._insert_thought_tokens(batch)
        long_input_ids, long_attention_mask, long_labels = batch_with_inserted_thought_tokens["input_ids"], batch_with_inserted_thought_tokens["attention_mask"], batch_with_inserted_thought_tokens["labels"]

        th.cuda.empty_cache()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        th.cuda.empty_cache()
        long_outputs = self.model(input_ids=long_input_ids, attention_mask=long_attention_mask, labels=long_labels)

        loss = outputs.loss
        perplexity = th.exp(loss)

        long_loss = long_outputs.loss
        long_perplexity = th.exp(long_loss)

        import pdb; pdb.set_trace()
        return {
            "val_loss": loss,
            "val_perplexity": perplexity,
            "val_long_loss": long_loss,
            "val_long_perplexity": long_perplexity
        }

    def configure_optimizers(self: Self) -> th.optim.Optimizer:
        # only optimize the embeddings and unembeddings
        optimizer = AdamW(chain(self.model.get_input_embeddings().parameters(), self.model.get_output_embeddings().parameters()), lr=self.lr)

        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: min(1.0, step / self.warmup_steps)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def _batch_encode(self: Self, batch: dict[str, Any]) -> BatchEncoding:
        batch_encoded = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)

        input_ids = batch_encoded["input_ids"]
        target = input_ids.clone()
        target[target == self.tokenizer.pad_token_id] = -100
        batch_encoded["labels"] = target

        # move to gpu
        return {k: v.to(self.model.device) for k, v in batch_encoded.items()}

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
        for batch in map(self._batch_encode, data):
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].detach()

            if not do_partial_kmeans:
                # filter out data points that are too close to the nearest existing unembedding
                unembedding_distances = th.min(distance_func(last_hidden_state, fixed_centroids), dim=-1)
                last_hidden_state = last_hidden_state[unembedding_distances > distance_threshold]
                del unembedding_distances

            final_token_embeddings = th.cat([final_token_embeddings, last_hidden_state[batch["attention_mask"].bool()]], dim=0)

        # free up memory
        del last_hidden_state, outputs
        th.cuda.empty_cache()

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
        cluster_assignments = th.zeros(data.shape[0], dtype=th.long, device=data.device)

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

        # free up memory
        del local_means, local_mean, new_cluster_assignments, cluster_assignments_unchanged, relative_data_sizes, data_sizes, local_data_size, cluster_assignments, data
        th.cuda.empty_cache()

        return centroids[fixed_centroids.shape[0]:]

    kmeans = partialmethod(partial_kmeans, fixed_centroids=None)

    def _insert_thought_tokens(
        self: Self,
        batch: BatchEncoding
    ) -> BatchEncoding:
        self.model.eval()
        pad_token_id = self.tokenizer.pad_token_id

        with th.no_grad():
            input_ids, attention_mask, labels, token_type_ids = batch["input_ids"], batch["attention_mask"], batch["labels"], batch.get("token_type_ids", None)
            if token_type_ids is None:
                token_type_ids = th.zeros_like(input_ids)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_token_ids = th.argmax(outputs.logits, dim=-1)  # (B, T)
            thought_token_mask = (predicted_token_ids >= self.model.num_standard_tokens) & (labels != -100)
            thought_token_mask = thought_token_mask.roll(shifts=1, dims=-1)
            thought_token_mask[:, 0] = 0

            # expand the length dimension to insert the thought tokens
            max_sequence_expansion = thought_token_mask.sum(dim=-1).max()

            long_input_ids = th.full((input_ids.shape[0], input_ids.shape[1] + max_sequence_expansion), pad_token_id, dtype=th.long, device=input_ids.device)
            long_attention_mask = th.zeros((attention_mask.shape[0], attention_mask.shape[1] + max_sequence_expansion), dtype=th.long, device=attention_mask.device)
            long_token_type_ids = th.zeros((token_type_ids.shape[0], token_type_ids.shape[1] + max_sequence_expansion), dtype=th.long, device=token_type_ids.device)
            long_labels = th.full((labels.shape[0], labels.shape[1] + max_sequence_expansion), -100, dtype=th.long, device=labels.device)

            # we insert the corresponding thought tokens where the model predicts it should be
            cumulative_thought_tokens_inserted = thought_token_mask.cumsum(dim=-1)  # (B, T)

            # get the new indices of where each token should be, after the thought tokens are inserted
            current_indices = th.arange(input_ids.shape[-1], device=input_ids.device).repeat(input_ids.shape[0], 1)  # (B, T)
            new_indices = current_indices + cumulative_thought_tokens_inserted

            # move the standard tokens to their new indices
            long_input_ids.scatter_(dim=-1, index=new_indices, src=input_ids)
            long_attention_mask.scatter_(dim=-1, index=new_indices, src=attention_mask)
            long_token_type_ids.scatter_(dim=-1, index=new_indices, src=token_type_ids)
            long_labels.scatter_(dim=-1, index=new_indices, src=labels)

            # get indices of where the thought tokens should be inserted
            thought_token_indices = th.nonzero(thought_token_mask)  # (N, 2)

            # offset the thought token indices by the cumulative thought tokens inserted
            thought_token_indices[:, 1] += cumulative_thought_tokens_inserted[thought_token_mask]
            indices_0 = thought_token_indices[:, 0]
            indices_1 = thought_token_indices[:, 1]

            # insert the thought tokens
            long_input_ids[indices_0, indices_1] = predicted_token_ids[thought_token_mask]
            long_attention_mask[indices_0, indices_1] = 1
            long_token_type_ids[indices_0, indices_1] = 1
            long_labels[indices_0, indices_1] = predicted_token_ids[thought_token_mask]

        self.model.train()

        return BatchEncoding(data={
            "input_ids": long_input_ids,
            "attention_mask": long_attention_mask,
            "token_type_ids": long_token_type_ids,
            "labels": long_labels,
        })
