import torch

torch.backends.cuda.matmul.allow_tf32 = True
import os
import resource
import socket
from typing import Optional, Set

import hydra
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf

import trainers
from utils import (
    build_exp_name,
    disable_dropout,
    get_local_dir,
    get_local_run_dir,
    get_open_port,
    init_distributed,
)

OmegaConf.register_new_resolver(
    "get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir)
)
OmegaConf.register_new_resolver(
    "build_exp_name",
    lambda loss_name, model_name, datasets: build_exp_name(
        loss_name,
        model_name,
        datasets,
    ),
)


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if "FSDP" in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Now resolve hydra references with the updated transform config
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print("Setting eval_every to", config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if "FSDP" in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.output_dir)
    print("building policy")
    model_kwargs = {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs
    )
    disable_dropout(policy)

    if config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "tbpo"}:
        print("building reference model")
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            **model_kwargs,
        )
        disable_dropout(reference_model)
    else:
        reference_model = None

    if "FSDP" in config.trainer:
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy, reference_model),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
