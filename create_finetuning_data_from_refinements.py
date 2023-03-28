import argparse
import logging
import re

from datasets import Dataset, load_dataset, concatenate_datasets


def format_prompt(mbpp, task_id):
    idx = mbpp["task_id"].index(task_id)
    text = mbpp["text"][idx]
    tests = mbpp["test_list"][idx]
    sample_code = mbpp["code"][idx]

    # Create prompt from scratch
    prompt = f'"""\n{text}\n\n'
    # Add the first unit test as an input-output example
    example = tests[0].split("assert ")[-1].replace("==", "=")
    prompt += f">>> Example: {example}\n"

    # Add code prefix
    fn_name = tests[0].split("assert ")[-1].split("(")[0]
    fn_search = re.search(f"def {fn_name}\(.*\):", sample_code)
    if fn_search is None:
        raise ValueError(
            f"Could not find 'def {fn_name}\(.*\):' in code for task {task_id}."
        )
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'{prompt}"""\n\n{code_prefix}\n'
    return prompt


def load_scored_data(feedback_path):
    d = load_dataset("json", data_files={"train": feedback_path})["train"].map(
        lambda _, idx: {"row_id": idx},
        with_indices=True,
    )
    print(f"Initial length of d: {len(d)}")
    d = d.filter(lambda example: example["passed"])
    print(f"Length of d after filtering for passed: {len(d)}")
    return d


def dedupe_dataset(dataset):
    cols = dataset.column_names
    row_set = set()
    for ex in dataset:
        ex_tuple = tuple(ex[col] for col in cols)
        row_set.add(ex_tuple)
    deduped = {k: [row[i] for row in row_set] for i, k in enumerate(cols)}
    return Dataset.from_dict(deduped)


def remove_prefix_and_func_sig(code, func_sig):
    if f"{func_sig}\r\n" in code:
        return code[code.rfind(f"{func_sig}\r\n") + len(f"{func_sig}\r\n") :]
    elif f"{func_sig} \r\n" in code:
        return code[code.rfind(f"{func_sig} \r\n") + len(f"{func_sig} \r\n") :]
    elif f"{func_sig}\n" in code:
        return code[code.rfind(f"{func_sig}\n") + len(f"{func_sig}\n") :]
    elif f"{func_sig}" in code:
        return code[code.rfind(f"{func_sig}") + len(f"{func_sig}") :]
    else:
        return code


def get_completion(prompt, completion):
    """If 'REFINEMENT:' is in the completion, remove it. Also remove prompt prefix if present."""
    ref_str = "REFINEMENT:"
    if ref_str in completion:
        idx = completion.rfind(ref_str)
        completion = completion[idx + len(ref_str) :]
    if prompt in completion:
        idx = completion.rfind(prompt)
        completion = completion[idx + len(prompt) :]
    return completion


def create_prompts(args):
    mbpp = load_dataset("mbpp")
    mbpp = concatenate_datasets([mbpp[k] for k in mbpp.keys()])
    ref_data = load_scored_data(args.refinement_file)
    print(f"Length of scored data: {len(ref_data)}")

    # Get unique pairs of (task ID, prompt) from the scored refinements.
    tasks = set([(example["task_id"], example["prompt"]) for example in ref_data])

    if not args.no_output_gold_data:
        mbpp_ft_data = {
            "finetuning_prompt": [],
            "finetuning_completion": [],
            "task_id": [],
        }
        task_id_to_func_sig = {}
        for task_id, prompt in tasks:
            mbpp_idx = mbpp["task_id"].index(task_id)

            # Get the original reformatted MBPP prompt
            orig_prompt = format_prompt(mbpp, task_id)

            # Remove method signature prefix
            gold_code = mbpp["code"][mbpp_idx]
            sig_idx = prompt.rfind("def ")
            colon_idx = prompt.rfind(":")
            func_sig = prompt[sig_idx : colon_idx + 1]
            task_id_to_func_sig[task_id] = func_sig
            gold_code = remove_prefix_and_func_sig(gold_code, func_sig)
            if gold_code is None:
                logging.warning(
                    f"Could not find function signature {func_sig} in gold code.\nGold code:\n{gold_code}"
                )
                continue
            mbpp_ft_data["finetuning_prompt"].append(orig_prompt)
            mbpp_ft_data["finetuning_completion"].append(gold_code)
            mbpp_ft_data["task_id"].append(task_id)
        mbpp_ft_data = Dataset.from_dict(mbpp_ft_data)

        if args.sample_size is not None:
            n = min(len(mbpp_ft_data), args.sample_size)
            mbpp_ft_data = mbpp_ft_data.shuffle().select(range(n))
        mbpp_ft_data.to_json(
            f"{args.output_dir}/finetuning_prompts_mbpp_gold_{args.output_file_suffix}.jsonl"
        )

    refs_ft_data = ref_data.map(
        lambda ex: {
            "finetuning_prompt": format_prompt(mbpp, ex["task_id"]),
        }
    ).map(
        lambda ex: {
            "finetuning_completion": get_completion(
                ex["finetuning_prompt"], ex["completion"]
            )
        }
    )
    cols_to_remove = list(
        set(refs_ft_data.column_names)
        - set(["task_id", "finetuning_prompt", "finetuning_completion"])
    )
    refs_ft_data = refs_ft_data.remove_columns(cols_to_remove)
    refs_ft_data = dedupe_dataset(refs_ft_data)
    if args.one_per_task:
        df = refs_ft_data.shuffle().to_pandas()
        df = df.groupby("task_id").first()
        refs_ft_data = Dataset.from_pandas(df)

    if args.sample_size is not None:
        n = min(len(refs_ft_data), args.sample_size)
        refs_ft_data = refs_ft_data.shuffle().select(range(n))
    refs_ft_data.to_json(
        f"{args.output_dir}/finetuning_prompts_mbpp_refinements_{args.output_file_suffix}.jsonl"
    )


def parse_args(input_args):
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning prompts from model-generated refinements. Also generate FT prompts for those same task IDs from the original MBPP dataset using gold code."
    )
    parser.add_argument(
        "--refinement-file",
        type=str,
        help="Path to file containing evaluated refinements. Needs to have the following columns: passed, task_id, prompt, completion.",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to output data files in."
    )
    parser.add_argument(
        "--no-output-gold-data",
        action="store_true",
        help="If set, will not output finetuning files for gold completions.",
    )
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument(
        "-n",
        "--sample-size",
        default=None,
        type=int,
        help="If set, will limit the number of outputs to this value.",
    )
    parser.add_argument(
        "--one-per-task",
        action="store_true",
        help="If set, will randomly select one correct refinement per task.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args(None)
    create_prompts(args)


if __name__ == "__main__":
    main()
