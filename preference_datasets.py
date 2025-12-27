import warnings

import datasets
import torch

# Suppress torchvision warning about image loading
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import random
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils import TemporarilySeededRandom


def binary_weight_transform(nums, top_percent=100):
    sorted_nums = sorted(nums, reverse=True)

    threshold_index = int(len(nums) * top_percent / 100)

    num_to_value = {num: 1 if i < threshold_index else 0 for i, num in enumerate(sorted_nums)}

    transformed_list = [num_to_value[num] for num in nums]

    return transformed_list


def threshold_weight_transform(nums, upper_threshold=1, lower_threshold=-1):
    result = []
    for num in nums:
        if num > upper_threshold:
            result.append(upper_threshold)
        elif num < lower_threshold:
            result.append(lower_threshold)
        else:
            result.append(num)
    return result


def threshold_and_scale_transform(nums, min_val=-1.5, max_val=1.5, min_scale=0.7, max_scale=1.3):
    result = []
    for num in nums:
        num = max(min_val, min(max_val, num))
        scaled = min_scale + (num - min_val) * (max_scale - min_scale) / (max_val - min_val)
        result.append(scaled)
    return result


def random_weight_transform(nums, min_val=0.7, max_val=1.3):
    return [random.uniform(min_val, max_val) for _ in nums]


def rank_based_transform(nums, min_scale=0.7, max_scale=1.3):
    sorted_indices = sorted(range(len(nums)), key=lambda k: nums[k])
    result = []
    for i, idx in enumerate(sorted_indices):
        if len(nums) == 1:
            result.append(1.0)
        else:
            scaled = min_scale + (i / (len(nums) - 1)) * (max_scale - min_scale)
            result.append(scaled)
    final_result = [result[sorted_indices.index(i)] for i in range(len(nums))]
    return final_result


weight_transform_methods = {
    "origin": lambda x: x,
    "binary": binary_weight_transform,
    "threshold": threshold_weight_transform,
    "threshold_and_scale": threshold_and_scale_transform,
    "random": random_weight_transform,
    "rank_based": rank_based_transform,
}


def get_dataset_from_hf(
    hf_dataset_repo_name: str,
    split: str,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    base_data_dir: Optional[str] = None,
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
        assert chosen["role"] == "assistant"
        assert rejected["role"] == "assistant"

        responses = [chosen, rejected]

        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def tokenize_batch_element(
    prompt: dict,
    chosen: dict,
    rejected: dict,
    tokenizer,
    max_length: int,
) -> Optional[Dict]:
    """ """
    # Data quality check: we don't want EOS appear at the middle of the prompt or response
    raw_prompt_tokens = tokenizer(prompt["content"], add_special_tokens=False)
    raw_chosen_tokens = tokenizer(chosen["content"], add_special_tokens=False)
    raw_rejected_tokens = tokenizer(rejected["content"], add_special_tokens=False)

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
        [prompt], add_generation_prompt=True, tokenize=False
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

    batch["prompt"] = [prompt]
    batch["chosen"] = [prompt] + [chosen]
    batch["rejected"] = [prompt] + [rejected]
    batch["chosen_response_only"] = [chosen]
    batch["rejected_response_only"] = [rejected]

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
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
                or k.endswith("_weight")
            ):
                if "prompt" in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    if k.endswith("_weight"):
                        to_pad = [torch.FloatTensor(ex[k]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask") or k.endswith("_weight"):
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
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    transform_config=None,
    base_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    reverse_dataset: bool = False,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        transform_config: Configuration for weight transformation. Can be a string (method name) or
                          a dict with a 'method' field and parameters for that method.
        base_data_dir: Base directory for the dataset.
        cache_dir: Directory to cache the datasets in.
        reverse_dataset: Whether to reverse the dataset.
    """
    assert n_epochs is not None or n_examples is not None, (
        "Must specify either n_epochs or n_examples"
    )
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            for prompt, data in get_dataset(
                name,
                split,
                silent=silent,
                cache_dir=cache_dir,
                transform_config=transform_config,
                base_data_dir=base_data_dir,
                reverse_dataset=reverse_dataset,
            ).items():
                flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        data["rejected_weight"],
                        data["chosen_weight"],
                        truncation_mode,
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
            responses,
            pairs,
            sft_target,
            rejected_weight,
            chosen_weight,
            truncation_mode,
        ) in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(
                    prompt,
                    sft_target,
                    sft_target,
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
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
                    rejected_weight_item = rejected_weight[index] if rejected_weight else None
                    chosen_weight_item = chosen_weight[index] if chosen_weight else None
                    batch_element = tokenize_batch_element(
                        prompt,
                        responses[p[0]],
                        responses[p[1]],
                        truncation_mode,
                        tokenizer,
                        max_length,
                        max_prompt_length,
                        rejected_weight_item,
                        chosen_weight_item,
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


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != " " and str_b[idx] != " ":
                return False
            else:
                if str_a[idx] == " ":
                    str_a = str_a[:idx] + str_a[idx + 1 :]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1 :]

    return True
