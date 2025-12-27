import random
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Union

import datasets
import numpy as np
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

from utils import TemporarilySeededRandom


def get_dataset_from_hf(
    hf_dataset_repo_name: str,
    split: str,
    silent: bool = False,
    cache_dir: Optional[str] = None,
):
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    data_iter: Iterator[Dict]

    print(f"Loading {hf_dataset_repo_name} dataset ({split} split) from HF...")
    data_iter = datasets.load_dataset(hf_dataset_repo_name, split=split, cache_dir=cache_dir)

    for example in tqdm.tqdm(data_iter, desc=f"Processing {hf_dataset_repo_name}", disable=silent):
        assert example["chosen"][:-1] == example["rejected"][:-1], (
            f"Prompt in chosen and rejected do not match: "
            f"{example['chosen'][:-1]} vs {example['rejected'][:-1]}"
        )
        prompt = example["chosen"][:-1]
        chosen = example["chosen"][-1]
        rejected = example["rejected"][-1]
        assert type(prompt) is dict and prompt["role"] == "user"
        assert type(chosen) is dict and chosen["role"] == "assistant"
        assert type(rejected) is dict and rejected["role"] == "assistant"

        responses = [[chosen], [rejected]]
        prompt_str = prompt["content"]

        n_responses = len(data[prompt_str]["responses"])
        data[prompt_str]["prompt_dict"] = [prompt]
        data[prompt_str]["pairs"].append((n_responses, n_responses + 1))
        data[prompt_str]["responses"].extend(responses)
        data[prompt_str]["sft_target"] = [chosen]

    return data


def tokenize_batch_element(
    prompt: list[dict],
    chosen: list[dict],
    rejected: list[dict],
    tokenizer,
    max_length: int,
) -> Optional[Dict]:
    """Tokenize a single batch element"""
    assert len(prompt) == 1 and len(chosen) == 1 and len(rejected) == 1
    # Data quality check: we don't want EOS appear at the middle of the prompt or response
    raw_prompt_tokens = tokenizer(prompt[0]["content"], add_special_tokens=False)
    raw_chosen_tokens = tokenizer(chosen[0]["content"], add_special_tokens=False)
    raw_rejected_tokens = tokenizer(rejected[0]["content"], add_special_tokens=False)

    assert tokenizer.eos_token_id not in raw_prompt_tokens["input_ids"], (
        f"Prompt contains EOS token: {prompt}"
    )
    assert tokenizer.eos_token_id not in raw_chosen_tokens["input_ids"], (
        f"Chosen response contains EOS token: {chosen}"
    )
    assert tokenizer.eos_token_id not in raw_rejected_tokens["input_ids"], (
        f"Rejected response contains EOS token: {rejected}"
    )

    chosen_message = [prompt] + [chosen]
    rejected_message = [prompt] + [rejected]

    chosen_template_message = tokenizer.apply_chat_template(
        chosen_message, add_generation_prompt=False, tokenize=False
    )
    rejected_template_message = tokenizer.apply_chat_template(
        rejected_message, add_generation_prompt=False, tokenize=False
    )
    prompt_template_message = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )

    prompt_sequence_tokens = tokenizer(prompt_template_message, add_special_tokens=False)
    chosen_sequence_tokens = tokenizer(chosen_template_message, add_special_tokens=False)
    rejected_sequence_tokens = tokenizer(rejected_template_message, add_special_tokens=False)

    # discard the sample if too long
    longer_response_length = max(
        len(chosen_sequence_tokens["input_ids"]), len(rejected_sequence_tokens["input_ids"])
    )
    if longer_response_length > max_length:
        return None

    # Create labels (we don't want to compute loss on prompt tokens)
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_sequence_tokens["input_ids"])] = [-100] * len(
        prompt_sequence_tokens["input_ids"]
    )
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_sequence_tokens["input_ids"])] = [-100] * len(
        prompt_sequence_tokens["input_ids"]
    )

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_sequence_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
      ints [tokens] or strings [the original texts]) and returns a batch of examples,
      PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if "prompt" in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if "prompt" in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        # import ipdb; ipdb.set_trace()

        return padded_batch

    return collate_fn


def get_batch_iterator(
    hf_dataset_repo_names,
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
) -> Iterator[Dict]:
    assert n_epochs is not None or n_examples is not None, (
        "Must specify either n_epochs or n_examples"
    )
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []

        for prompt, data in get_dataset_from_hf(
            hf_dataset_repo_names,
            split,
            silent=silent,
            cache_dir=cache_dir,
        ).items():
            flat_data.append(
                (
                    prompt,
                    data["prompt_dict"],
                    data["responses"],
                    data["pairs"],
                    data["sft_target"],
                )
            )

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for (
            prompt,
            prompt_dict,
            responses,
            pairs,
            sft_target,
        ) in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(
                    prompt_dict,
                    sft_target,
                    sft_target,
                    tokenizer,
                    max_length,
                )
                if batch_element is None:
                    continue
                batch_element = {k: v for k, v in batch_element.items() if "rejected" not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f"Finished generating {n_examples} examples on {split} split")
                        done = True

                    batch = []
            else:
                for index, p in enumerate(pairs):
                    if done:
                        break
                    batch_element = tokenize_batch_element(
                        prompt,
                        responses[p[0]],
                        responses[p[1]],
                        tokenizer,
                        max_length,
                    )
                    if batch_element is None:
                        continue
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1
