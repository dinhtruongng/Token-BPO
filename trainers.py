import torch

torch.backends.cuda.matmul.allow_tf32 = True
import contextlib
import functools
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensor_parallel as tp
import torch.distributed as dist
import torch.nn as nn
import tqdm
import transformers
import wandb
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from loss.h_function import make_h
from loss.loss import (
    bregman_loss,
    preference_loss,
    tdpo_loss,
    tisdpo_loss,
)
from loss.loss_utils import (
    Q_tbpo_get_batch_logps,
    _get_batch_logps,
    _get_batch_logps_tisdpo,
    _tdpo_get_batch_logps,
)
from preference_datasets import get_batch_iterator
from utils import (
    all_gather_if_needed,
    compute_tbpo_loss_mask,
    concatenated_inputs,
    formatted_dict,
    get_block_class_from_model,
    pad_to_length,
    rank0_print,
    slice_and_move_batch_for_device,
)


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir
        self.clip_hits = 0

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            hf_dataset_repo_names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            sft_mode=config.loss.name == "sft",
            seed=seed,
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split="train",
            n_epochs=config.n_epochs,
            n_examples=config.n_examples,
            batch_size=config.batch_size,
            silent=rank != 0,
        )
        rank0_print("Loaded train data iterator")
        self.eval_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split="test",
            n_examples=config.n_eval_examples,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            ctx = lambda: (
                FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False)
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            reference_output = pad_to_length(
                reference_output, self.config.max_length, self.tokenizer.pad_token_id
            )
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        concatenated_batch = concatenated_inputs(batch)
        # dict_keys(['concatenated_weight', 'concatenated_input_ids', 'concatenated_attention_mask', 'concatenated_labels'])
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            token_level=self.config.loss.token_level,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def tisdpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
            ).logits.to(torch.float32)

        all_logps_margin, all_position_kl, all_logps = _get_batch_logps_tisdpo(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            concatenated_batch["concatenated_weight"],
            average_log_prob=False,
        )

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]
        chosen_position_kl = all_position_kl[: batch["chosen_input_ids"].shape[0]]
        rejected_position_kl = all_position_kl[batch["chosen_input_ids"].shape[0] :]

        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]].detach()
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :].detach()

        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
        )

    def tdpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
            ).logits.to(torch.float32)
        all_logps_margin, all_position_kl, all_logps = _tdpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]
        chosen_position_kl = all_position_kl[: batch["chosen_input_ids"].shape[0]]
        rejected_position_kl = all_position_kl[batch["chosen_input_ids"].shape[0] :]

        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]].detach()
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :].detach()

        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
        )

    def Q_tbpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """
        Compute R_theta
        """
        concatenated_batch = concatenated_inputs(batch)
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            output_hidden_states=True,
        )
        all_logits = outputs.logits.to(torch.float32)
        all_last_hidden_states = outputs.hidden_states[-1].to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
            ).logits.to(torch.float32)

        labels = concatenated_batch["concatenated_labels"][:, 1:].clone()
        per_sample_mask = labels != -100
        loss_mask = compute_tbpo_loss_mask(batch, concatenated_batch)

        all_logps_margin, all_logps = Q_tbpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
        )

        all_logps = all_logps * per_sample_mask

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]

        all_last_hidden_states_detached = all_last_hidden_states.detach()
        all_baselines = self.policy.baseline_head(all_last_hidden_states_detached)
        chosen_baselines = all_baselines[: batch["chosen_input_ids"].shape[0]]
        rejected_baselines = all_baselines[batch["chosen_input_ids"].shape[0] :]
        # align to token-prediction positions: logits[:, :-1] predicts labels[:, 1:]
        b_chosen = chosen_baselines[:, :-1]
        b_rejected = rejected_baselines[:, :-1]
        log_w = b_rejected - b_chosen
        mean_logw_per_prompt = (log_w * loss_mask).sum(-1) / loss_mask.sum(-1)
        # center w
        log_w = log_w - mean_logw_per_prompt.unsqueeze(-1)
        log_w = log_w.clamp(self.config.model.baseline_l, self.config.model.baseline_u)

        # R_theta
        beta = self.config.loss.beta
        log_R = beta * (rejected_logps_margin - chosen_logps_margin + log_w)

        chosen_logps = beta * (all_logps[: batch["chosen_input_ids"].shape[0]].detach()).sum(-1)
        rejected_logps = beta * (all_logps[batch["chosen_input_ids"].shape[0] :].detach()).sum(-1)

        return log_R, chosen_logps, rejected_logps, loss_mask

    def A_tbpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute R_theta"""

        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
            ).logits.to(torch.float32)

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True
    ):
        metrics = {}
        train_test = "train" if train else "eval"

        if loss_config.name in {"dpo", "ipo"}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.reference_model, batch
                )

            if loss_config.name == "dpo":
                loss_kwargs = {
                    "beta": loss_config.beta,
                    "reference_free": loss_config.reference_free,
                    "label_smoothing": loss_config.label_smoothing,
                    "ipo": False,
                }
            elif loss_config.name == "ipo":
                loss_kwargs = {"beta": loss_config.beta, "ipo": True}
            else:
                raise ValueError(f"unknown loss {loss_config.name}")

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_kwargs,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == "tdpo":
            (
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                policy_chosen_logps,
                policy_rejected_logps,
            ) = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                if_tdpo2=loss_config.if_tdpo2,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            all_device_chosen_position_kl = all_gather_if_needed(
                chosen_position_kl.detach(), self.rank, self.world_size
            )
            all_device_rejected_position_kl = all_gather_if_needed(
                rejected_position_kl.detach(), self.rank, self.world_size
            )

            metrics[f"kl_{train_test}/chosen"] = (
                all_device_chosen_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/rejected"] = (
                all_device_rejected_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/margin"] = (
                (all_device_chosen_position_kl - all_device_rejected_position_kl)
                .cpu()
                .numpy()
                .tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == "tisdpo":
            (
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                policy_chosen_logps,
                policy_rejected_logps,
            ) = self.tisdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tisdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                token_level=loss_config.token_level,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            all_device_chosen_position_kl = all_gather_if_needed(
                chosen_position_kl.detach(), self.rank, self.world_size
            )
            all_device_rejected_position_kl = all_gather_if_needed(
                rejected_position_kl.detach(), self.rank, self.world_size
            )

            metrics[f"kl_{train_test}/chosen"] = (
                all_device_chosen_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/rejected"] = (
                all_device_rejected_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/margin"] = (
                (all_device_chosen_position_kl - all_device_rejected_position_kl)
                .cpu()
                .numpy()
                .tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == "sft":
            policy_chosen_logits = self.policy(
                batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits,
                batch["chosen_labels"],
                average_log_prob=False,
                token_level=False,
            )

            losses = -policy_chosen_logps

        elif loss_config.name == "Q_tbpo":
            log_R, policy_chosen_logps, policy_rejected_logps, loss_mask = (
                self.Q_tbpo_concatenated_forward(self.policy, self.reference_model, batch)
            )
            losses = bregman_loss(log_R, loss_mask, h_func=make_h(**loss_config.bregman_loss))

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == "A_tbpo":
            pass

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/chosen"] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        if self.config.scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.total_steps,
            )
        elif self.config.scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.total_steps,
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "tbpo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(f"Running evaluation after {self.example_counter} train examples")
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, self.config.loss, train=False
                        )

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(
                            f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts."
                        )
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = (
                            self.config.n_eval_model_samples // self.config.eval_batch_size
                        )
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (
                        tqdm.tqdm(sample_batches, desc="Generating samples...")
                        if self.rank == 0
                        else sample_batches
                    ):
                        local_eval_batch = slice_and_move_batch_for_device(
                            eval_batch, self.rank, self.world_size, self.rank
                        )
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch["prompt"], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                            for prompt, sample in zip(eval_batch["prompt"], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )
                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                            wandb.log(
                                {"reference_samples": reference_text_table},
                                step=self.example_counter,
                            )

            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(
                    batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)
            batch_metrics["clip_frac"].append(self.clip_hits / self.config.total_steps)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )
            #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy and tokenizer to disk."""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, "LATEST")
        else:
            model_save_dir = output_dir

        os.makedirs(model_save_dir, exist_ok=True)

        # Save model using transformers save_pretrained
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")

        # Save tokenizer alongside the model
        self.tokenizer.save_pretrained(model_save_dir)

        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
        transform_config=None,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

        This trainer will shard both the policy and reference model across all available GPUs.
        Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(
            policy,
            config,
            seed,
            run_dir,
            reference_model,
            rank,
            world_size,
        )
        assert config.model.block_name is not None, (
            "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        )

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        norm_val = self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
        if norm_val > self.config.max_grad_norm and dist.get_rank() == 0:
            self.clip_hits += 1
        return norm_val

    def save(self, output_dir=None, metrics=None):
        """Save policy and tokenizer state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = self.run_dir
            else:
                model_save_dir = output_dir

            os.makedirs(model_save_dir, exist_ok=True)

            # Get the original model class and instantiate it directly
            from transformers import AutoModelForCausalLM

            model_name = self.config.model.name_or_path
            unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
            unwrapped_model.load_state_dict(policy_state_dict)

            # Save using transformers save_pretrained
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
            del unwrapped_model

            # Save tokenizer alongside the model
            self.tokenizer.save_pretrained(model_save_dir)

            # Save metrics separately
            if metrics is not None:
                metrics_file = os.path.join(model_save_dir, "training_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)

        del policy_state_dict
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(
        self,
        policy,
        config,
        seed,
        run_dir,
        reference_model=None,
        rank=0,
        world_size=1,
        transform_config=None,
    ):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

        Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
           see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(
            policy,
            config,
            seed,
            run_dir,
            reference_model,
            rank,
            world_size,
            transform_config=transform_config,
        )

        rank0_print("Sharding policy...")
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            rank0_print("Sharding reference model...")
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        # Save model using transformers save_pretrained
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, "LATEST")
        else:
            model_save_dir = output_dir

        os.makedirs(model_save_dir, exist_ok=True)

        # Get the original model class and instantiate it directly
        from transformers import AutoModelForCausalLM

        model_name = self.config.model.name_or_path
        unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
        unwrapped_model.load_state_dict(policy_state_dict)

        # Save using transformers save_pretrained
        unwrapped_model.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        del unwrapped_model

        # Save tokenizer alongside the model
        self.tokenizer.save_pretrained(model_save_dir)

        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)

        del policy_state_dict
