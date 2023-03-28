from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from jaxformer.hf.codegen import modeling_codegen
from jaxformer.hf import sample
import torch
import pprint
import os
import logging
import json
import csv
import argparse
import re


def load_jsonl(filepath):
    data = [json.loads(line) for line in open(filepath).readlines()]
    fields = data[0].keys()
    data_dict = {k: [x[k] for x in data] for k in fields}
    ds = Dataset.from_dict(data_dict)
    return ds


def load_csv(filepath):
    data = list(csv.DictReader(open(filepath)))
    fields = data[0].keys()
    data_dict = {k: [x[k] for x in data] for k in fields}
    ds = Dataset.from_dict(data_dict)
    return ds


def load_feedback(feedback_path):
    extension = "csv" if feedback_path.endswith("csv") else "json"
    if extension == "json":
        d = load_jsonl(feedback_path)
    else:
        d = load_csv(feedback_path)
    d = d.map(
        lambda _, idx: {"row_id": idx},
        with_indices=True,
    )
    d = d.filter(
        lambda example: example["Refinement"] is not None and example["Refinement"]
    )
    return d


def sample_code_from_codegen(args, prompt, model, tokenizer):
    device = "cuda:0"
    completions = []
    print(f"Tokenizing input: {prompt}")
    input_ids = tokenizer(
        prompt, truncation=True, max_length=1024, return_tensors="pt"
    ).input_ids.cuda()
    if args.temperature == 0.0:
        args.num_samples = 1
    for i in range(args.num_samples):
        try:
            # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
            if args.temperature > 0:
                tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=1,
                    max_length=input_ids.shape[1] + 1024,
                    temperature=args.temperature,
                    use_cache=True,
                )
            else:
                tokens = model.generate(
                    input_ids,
                    num_return_sequences=1,
                    max_length=input_ids.shape[1] + 1024,
                    use_cache=True,
                )
            text = tokenizer.decode(tokens[0])
            if "<|endoftext|>" in text:
                text = text[: text.find("<|endoftext|>")]
            completions.append(text)
        except RuntimeError as e:
            logging.error(f"Could not sample from model: {e}")
    return completions


def truncate(ex, tokenizer, max_length):
    return tokenizer.decode(
        tokenizer(ex, max_length=max_length, truncation=True).input_ids
    )


def format_mbpp_prompt(mbpp, task_id):
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


def gen_refinement_prompt(args, example, tokenizer, mbpp):
    prompt = (
        f"OLD CODE:\n{truncate(example[args.completion_column], tokenizer, 512)}"
        f"\n\nFEEDBACK:\n{example['Feedback']}\n\n"
        f"REFINEMENT:\n{format_mbpp_prompt(mbpp, example['task_id'])}"
    )
    return prompt


def gen_code(args, data, model, tokenizer):
    mbpp = load_dataset("mbpp")
    mbpp = concatenate_datasets([mbpp[k] for k in mbpp.keys()])
    output = data.map(
        lambda ex: {"input_str": gen_refinement_prompt(args, ex, tokenizer, mbpp)}
    )
    output = output.map(
        lambda ex: {
            "output_strs": sample_code_from_codegen(
                args, ex["input_str"], model, tokenizer
            )
        },
        desc="Sampling code from codegen...",
    )
    return output


def generate_code_for_problems(args):
    data = load_feedback(args.feedback_file)

    if args.model_path is None:
        model = modeling_codegen.CodeGenForCausalLM.from_pretrained(
            f"{args.codegen_model_dir}/{args.arch}-mono",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
    else:
        model = modeling_codegen.CodeGenForCausalLM.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32
        ).cuda()
    tokenizer = sample.create_custom_gpt2_tokenizer()
    tokenizer.pad_token = 50256
    val = gen_code(args, data, model, tokenizer)

    output = []
    for row in tqdm(val):
        for completion in row["output_strs"]:
            output.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["input_str"],
                    "feedback": row["Feedback"],
                    "old_completion": row[args.completion_column],
                    "completion": completion,
                }
            )
    return output


def write_jsonl(data, output_filepath):
    with open(output_filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch", default="codegen-6B", choices=["codegen-16B", "codegen-6B"]
    )
    parser.add_argument(
        "--codegen-model-dir",
        default="checkpoints",
        help="Directory where pre-trained CodeGen model checkpoints are saved.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        required=True,
        help="Directory to load model checkpoint from. If None, will load a pre-trained "
        "CodeGen model using the --arch argument instead.",
    )
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--feedback-file",
        default=None,
        required=True,
        help="CSV file containing feedback and past completions.",
    )
    parser.add_argument("--completion-column", default="completion")
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    completions = generate_code_for_problems(args)

    if args.model_path is None:
        output_filepath = os.path.join(
            args.output_dir,
            f"refinements_{args.arch}_temp{args.temperature}_{args.output_file_suffix}.jsonl",
        )
    else:
        output_filepath = os.path.join(
            args.model_path,
            f"refinements_{args.arch}_temp{args.temperature}_{args.output_file_suffix}.jsonl",
        )
    write_jsonl(completions, output_filepath)


if __name__ == "__main__":
    main(parse_args())
