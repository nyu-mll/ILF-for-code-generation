import argparse
import json
import logging
import openai
import os
import pprint
import re
import time
import torch

from jaxformer.hf import sample
from jaxformer.hf.codegen import modeling_codegen
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def format_prompt(task_id, text, tests, sample_code, num_prompts):
    # Create prompt from scratch
    prompt = f'"""\n{text}\n\n'
    if num_prompts > 0:
        for i in range(num_prompts):
            example = tests[i].split("assert ")[-1].replace("==", "=")
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


# GPT-J
def sample_code_from_gpt_models(args, prompt, model, tokenizer):
    output_strs = []
    num_samples = args.num_samples
    temperature = args.temperature
    debug = args.debug
    try:
        with torch.no_grad():
            input_ids = (
                torch.LongTensor(tokenizer.encode(prompt, verbose=False))
                .unsqueeze(0)
                .cuda()
            )
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,  # 0.2, 0.8
                max_length=1024 - len(input_ids),
                num_return_sequences=num_samples,
            )
            output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if debug:
                print(f"Input: {prompt}")
                print(f"Outputs: {output_strs}")
    except Exception as e:
        if (
            isinstance(e, UnboundLocalError)
            and str(e) == "local variable 'next_tokens' referenced before assignment"
        ):
            # See https://github.com/huggingface/transformers/issues/5118
            if debug:
                print("Problem text was > 1024 tokens, so cannot do generation")
                print(e)
        print(e)
    return output_strs


def sample_code_from_codegen(args, prompt, model, tokenizer):
    device = "cuda:0"
    completions = []
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


def initialize_openai(args):
    api_key = open(f"{args.openai_creds_dir}/openai_api_key.txt").read()
    openai.organization = open(
        f"{args.openai_creds_dir}/openai_organization_id.txt"
    ).read()
    openai.api_key = api_key


def sample_code_from_openai_model(args, prompt_text):
    output_strs = []
    start = time.time()

    arch_mapping = {
        "codex": "code-davinci-002",
        "gpt3": "text-davinci-001",
        "davinci-002": "text-davinci-002",
        "davinci-003": "text-davinci-003",
        "ada": "text-ada-001",
        "babbage": "text-babbage-001",
        "curie": "text-curie-001",
    }
    engine_name = arch_mapping[args.arch]

    for i in range(args.num_samples):
        while time.time() - start < args.max_request_time:
            try:
                response = openai.Completion.create(
                    engine=engine_name,
                    prompt=prompt_text,
                    max_tokens=1024,
                    n=1,
                    temperature=args.temperature,
                )
                output_strs += [
                    prompt_text + choice["text"] for choice in response["choices"]
                ]
                break
            except Exception as e:
                print(
                    f"Unexpected exception in generating solution. Sleeping again: {e}"
                )
                time.sleep(args.sleep_time)
    return output_strs


def write_jsonl(data, output_filepath):
    with open(output_filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def generate_code_for_problems(args):
    mbpp = load_dataset("mbpp")
    mbpp = concatenate_datasets([mbpp[k] for k in mbpp.keys()])

    output = []
    if args.arch in ["gpt3", "codex"]:
        initialize_openai(args)
        generate_code_fn = sample_code_from_openai_model
    elif args.arch in ["codegen-6B", "codegen-16B"]:
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
        tokenizer = sample.create_custom_gpt2_tokenizer(truncation_side="left")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = 50256
        generate_code_fn = lambda args, prompt: sample_code_from_codegen(
            args, prompt, model, tokenizer
        )

    task_ids_range = set(range(args.start, args.end))
    for i in tqdm(range(len(mbpp))):
        if mbpp["task_id"][i] not in task_ids_range:
            continue
        try:
            prompt = format_prompt(
                mbpp["task_id"][i],
                mbpp["text"][i],
                mbpp["test_list"][i],
                mbpp["code"][i],
                args.num_shots,
            )
        except ValueError as e:
            logging.error(e)
            continue

        task_id = mbpp["task_id"][i]
        for completion in generate_code_fn(args, prompt):
            output.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion,
                }
            )
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch",
        default="gptj",
        choices=[
            "gptj",
            "codex",
            "gpt3",
            "codegen-16B",
            "codegen-6B",
            "davinci-002",
            "davinci-003",
            "ada",
            "babbage",
            "curie",
        ],
    )
    parser.add_argument(
        "--codegen-model-dir",
        default="checkpoints",
        help="Directory where pre-trained CodeGen model checkpoints are saved.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Directory to load model checkpoint from. If None, will load a pre-trained "
        "CodeGen model using the --arch argument instead.",
    )
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Which MBPP split to use. In datasets v1.16.1, MBPP only has the split 'test'.",
    )
    parser.add_argument(
        "-s", "--start", default=1, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=975, type=int, help="Task ID to end with (exclusive)."
    )
    parser.add_argument(
        "-n",
        "--num-shots",
        default=0,
        type=int,
        help="Number of assert (test examples) to give in the task description.",
    )
    parser.add_argument(
        "--max-request-time",
        type=int,
        default=80,
        help="Max. time to wait for a successful GPT-3 request.",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="Time to sleep (in seconds) between each GPT-3 call.",
    )
    parser.add_argument(
        "--openai-creds-dir",
        type=str,
        default=None,
        help="Directory where OpenAI API credentials are stored. Assumes the presence of "
        "openai_api_key.txt and openai_organization_id.txt files.",
    )
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    completions = generate_code_for_problems(args)
    output_filepath = os.path.join(
        args.output_dir,
        f"samples_{args.split}_{args.arch}_{args.num_shots}shot_temp{args.temperature}_{args.start}-{args.end}{args.output_file_suffix}.jsonl",
    )
    write_jsonl(completions, output_filepath)


if __name__ == "__main__":
    main(parse_args())
