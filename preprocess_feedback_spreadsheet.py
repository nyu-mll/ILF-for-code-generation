import argparse
from datasets import Dataset, load_dataset


def group_by_and_select_one(ds, group_by_col):
    df = ds.shuffle().to_pandas()
    df = df.groupby(group_by_col).first()
    ds = Dataset.from_pandas(df)
    return ds


def truncate_completion(src):
    ref_str = "Refinement:\n"
    if ref_str in src:
        src = src[src.rfind(ref_str) + len(ref_str) :]
    return src


def preprocess_data(args):
    orig_ext = args.input_file.split(".")[-1]
    if orig_ext not in ["csv", "json", "jsonl"]:
        raise ValueError(f"{ext} is not a supported file extension.")
    if orig_ext == "jsonl":
        ext = "json"
    else:
        ext = orig_ext
    d = load_dataset(ext, data_files={"train": args.input_file})["train"].filter(
        lambda ex: ex[args.feedback_column] is not None and ex[args.feedback_column]
    )

    if args.old_refinement_column is not None:
        d = d.map(
            lambda ex: {args.refinement_column: ex[args.old_refinement_column]},
            remove_columns=[args.old_refinement_column],
        )

    d = d.map(
        lambda ex: {"completion": ex[args.model_completion_column]},
    )

    d = d.filter(
        lambda ex: ex[args.refinement_column] is not None and ex[args.refinement_column]
    ).map(
        lambda ex: {
            args.refinement_column: truncate_completion(ex[args.refinement_column])
        }
    )

    if args.filter_for_correct and "passed" in d.column_names:
        # Filter for correct ones only, if the column exists in the spreadsheet
        d = d.filter(lambda ex: ex["passed"])

    if args.one_per_task:
        # Filter for just one sample per task ID.
        d = group_by_and_select_one(d, args.id_col)

    # Split data and print out filenames
    output_file_prefix = ".".join(args.input_file.split(".")[:-1])
    if args.output_dir is not None:
        fname_prefix = output_file_prefix.split("/")[-1]
        output_file_prefix = f"{args.output_dir}/{fname_prefix}"

    df = d.to_pandas().set_index(args.id_col)
    train_df = df[
        (df.index >= args.training_start_id) & (df.index <= args.training_end_id)
    ]
    train_n = min(len(train_df), args.training_n)
    train_df = train_df.sample(n=train_n)
    train_output_filepath = f"{output_file_prefix}-train.jsonl"
    train_df.reset_index().to_json(train_output_filepath, orient="records", lines=True)
    val_df = df[(df.index >= args.val_start_id) & (df.index <= args.val_end_id)]
    val_n = min(len(val_df), args.val_n)
    val_df = val_df.sample(n=val_n)
    val_output_filepath = f"{output_file_prefix}-val.jsonl"
    val_df.reset_index().to_json(val_output_filepath, orient="records", lines=True)
    print("\n".join([train_output_filepath, val_output_filepath]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and pre-process CSV or JSONL input file containing feedback and refinements."
    )
    parser.add_argument(
        "--input_file",
        default="",
        required=True,
        help="Input CSV or JSONL file containing feedback and refinements.",
    )
    parser.add_argument(
        "--feedback_column", default="Feedback", help="Name of feedback column."
    )
    parser.add_argument(
        "--old_refinement_column",
        default=None,
        help="If set, will change the column with this name to --refinement_column.",
    )
    parser.add_argument(
        "--refinement_column", default="Refinement", help="Name of refinement column."
    )
    parser.add_argument(
        "--model_completion_column", default="original_model_completion"
    )
    parser.add_argument(
        "--training_n",
        default=None,
        type=int,
        help="Number of examples to be used for training data. If None, does not split data into train/val.",
    )
    parser.add_argument(
        "--val_n",
        default=None,
        type=int,
        help="Number of examples to be used for validation data. If None, just uses all non-training examples as validation data.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="task_id",
        help="Which column to index on and to split data by.",
    )
    parser.add_argument(
        "--one_per_task",
        action="store_true",
        help="If set, then will filter only one sample per task.",
    )
    parser.add_argument(
        "--filter_for_correct",
        action="store_true",
        help="Filter for only the rows for which passed=True. "
        + "(May want to keep off for feedback spreadsheets where the 'passed' column corresponds to the original model completion instead of the Refinement.)",
    )
    parser.add_argument(
        "--training_start_id",
        type=int,
        default=601,
    )
    parser.add_argument("--training_end_id", type=int, default=974)
    parser.add_argument("--val_start_id", type=int, default=511)
    parser.add_argument("--val_end_id", type=int, default=600)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If None, outputs to the same directory that the input file is already in.",
    )
    args = parser.parse_args()

    # if training_n is set, then val_n must also be set.
    assert (args.training_n is None) or (
        args.val_n is not None
    ), "Error: if --training_n is set, then --val_n must also be set."
    return args


def main(args):
    argsdict = vars(args)
    preprocess_data(args)


if __name__ == "__main__":
    main(parse_args())
