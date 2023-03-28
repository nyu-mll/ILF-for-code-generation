import argparse
import gzip
import io
import itertools
import json
import pprint
import numpy as np
import re
import sys
import timeout_decorator
import traceback


from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from multiprocessing import Process, Queue
from tqdm import tqdm
from typing import Dict, List, Union


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model completions on the MBPP benchmark."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File containing columns <args.prompt_column_name>, 'completion', and 'task_id'.",
    )
    parser.add_argument("--k", default="1,10")
    parser.add_argument("--file-suffix", default="results")
    parser.add_argument(
        "--prompt-column-name", default="prompt", help="Name of prompt column."
    )
    args = parser.parse_args()
    return args


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Taken from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py#L13.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_results(eval_results):
    results = defaultdict(list)
    for row in eval_results:
        ti = row["task_id"]
        passed = row["passed"]
        results[ti].append(passed)
    outputs = {
        ti: {"num_correct": np.sum(r), "num_total": len(r)} for ti, r in results.items()
    }
    return outputs


def compute_at_least_one_pass_per_task(results):
    total = 0
    task_ids = []
    for task_id, results_dict in results.items():
        if results_dict["num_correct"] > 0:
            total += 1
            task_ids.append(task_id)
    return total, task_ids


def compute_pass_at_ks(results, ks):
    output = {
        k: estimate_pass_at_k(
            [x["num_total"] for _, x in results.items()],
            [x["num_correct"] for _, x in results.items()],
            k,
        ).mean()
        for k in ks
    }
    return output


@timeout_decorator.timeout(3)
def eval_code(q, src, test, entry_point):
    all_src = f"{src}\n{test}\ncheck({entry_point})\n"
    try:
        exec(all_src, {})
    except Exception:
        with io.StringIO() as f:
            traceback.print_exception(*sys.exc_info(), file=f)
            q.put((False, f.getvalue()))
        return
    q.put((True, None))


def eval_code_wrapper(src, test, entry_point):
    queue = Queue()
    p = Process(target=eval_code, args=(queue, src, test, entry_point))
    p.start()
    p.join(3)
    if p.is_alive():
        p.kill()
    if not queue.empty():
        return queue.get()
    else:
        return False, f"Exit code: {p.exitcode}"


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def format_test(mbpp, entrypoint, task_id):
    idx = mbpp["task_id"].index(task_id)
    test_list = mbpp["test_list"][idx]

    test_str = "def check(candidate):\n"

    # use pytest.approx() for float results
    if is_float(test_list[0].split("==")[-1]):
        test_str = "from pytest import approx\n\n" + test_str
        for i in range(len(test_list)):
            split = test_list[i].split("==")
            split[-1] = f"approx({split[-1]})"
            test_list[i] = "==".join(split)

    for test in test_list:
        test_str += f"\t{test}\n"
    test_str += "\n"

    if entrypoint != "check":
        test_str = test_str.replace(entrypoint, "candidate")
    else:
        test_str = test_str.replace(f"assert {entrypoint}", "assert candidate")
    return test_str


def get_entry_point(mbpp, task_id):
    idx = mbpp["task_id"].index(task_id)
    assert_statement = mbpp["test_list"][idx][0]
    assert_statement = assert_statement[len("assert ") :]
    lparen_idx = assert_statement.index("(")
    entrypoint = assert_statement[:lparen_idx]
    return entrypoint


def get_dict_list(filename: str) -> List[Dict]:
    output_list = []
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        output_list.append(json.loads(line))
    elif filename.endswith(".jsonl"):
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    output_list.append(json.loads(line))
    elif filename.endswith(".csv"):
        d = load_dataset("csv", data_files={"train": filename})["train"]
        for i in range(len(d[d.column_names[0]])):
            output_list.append({col: d[col][i] for col in d.column_names})
    else:
        raise ValueError(f"Unrecognized file extension type for file {filename}!")
    return output_list


def truncate_code(completion, prompt):
    if isinstance(completion, list):
        completion = completion[0]

    # if code is refinement, remove everything else before it.
    if "REFINEMENT:" in completion or "Refinement:\n" in completion:
        refinement_str = (
            "REFINEMENT:" if "REFINEMENT:" in completion else "Refinement:\n"
        )
        ref_end_idx = completion.rfind(refinement_str) + len(refinement_str)
        completion = completion[ref_end_idx:]

        if not completion.startswith(prompt):
            # completion doesn't start with exact prompt for some reason, even though it should
            # return early
            return completion

    # Remove prompt first so that we can fix the indentation of the completion.
    code = completion[len(prompt) :]

    # sometimes indentation on the first line is messed up
    if not code.startswith("    "):
        # find the first line
        eo_fl_idx = code.find("\n")
        first_line = code[:eo_fl_idx].strip()
        first_line = "    " + first_line
        code = first_line + code[eo_fl_idx:]

    # Find end of function and truncate there
    eof_m = re.search(r'\n[A-Za-z#"]+?', code)
    if eof_m is not None:
        code = code[: eof_m.start() + 1]

    # Now re-add the prompt
    code = prompt + code
    completion = code
    return completion


def eval_samples(args):
    ks = [int(elem) for elem in args.k.split(",")]
    output_file_prefix = args.input_file + f"_{args.file_suffix}"
    ext = args.input_file.split(".")[-1]
    output_file = f"{output_file_prefix}.{ext}"
    output_summ_file = f"{output_file_prefix}_summary.{ext}"

    mbpp = load_dataset("mbpp")
    mbpp = concatenate_datasets([mbpp[k] for k in mbpp.keys()])
    samples = get_dict_list(args.input_file)
    for sample_dict in tqdm(samples, desc="Evaluating and scoring..."):
        completion = sample_dict["completion"]
        prompt = sample_dict[args.prompt_column_name]
        completion = truncate_code(completion, prompt)
        entrypoint = get_entry_point(mbpp, sample_dict["task_id"])
        test_str = format_test(mbpp, entrypoint, sample_dict["task_id"])
        try:
            p, r = eval_code_wrapper(completion, test_str, entrypoint)
        except Exception as e:
            with io.StringIO() as f:
                traceback.print_exception(*sys.exc_info(), file=f)
                r = f.getvalue()
            p = False
            print(f"Caught exception from eval_code: {e}\n{r}")
        sample_dict["passed"] = p
        sample_dict["result"] = r
    num_corr_results = compute_results(samples)
    pass_at_k_results = compute_pass_at_ks(num_corr_results, ks)
    at_least_one_correct, _ = compute_at_least_one_pass_per_task(num_corr_results)
    pc_one_correct = at_least_one_correct / len(num_corr_results.keys())
    pass_at_k_results["% tasks with at least one passed completion"] = pc_one_correct
    print(pass_at_k_results)

    with open(output_file, "w") as f:
        for d in samples:
            f.write(json.dumps(d) + "\n")
    with open(output_summ_file, "w") as f:
        f.write(json.dumps(pass_at_k_results))


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    eval_samples(args)


if __name__ == "__main__":
    main(parse_args())
