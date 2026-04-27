from datasets import load_dataset
import argparse


def get_question(hf_datasets, ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


def process_and_update(example, hf_datasets):
    ds_name, ds_split, ds_index = example["dataset"], example["split"], int(example["index"])
    question = get_question(hf_datasets, ds_name, ds_split, ds_index)
    if question is not None:
        example["question"] = question
    return example


def main():
    parser = argparse.ArgumentParser(description="Download and save OpenCodeReasoning-2 dataset")
    parser.add_argument("--output_dir", type=str, default="data/OpenCodeReasoning-2-updated",
                        help="Directory to save the updated dataset")
    parser.add_argument("--languages", type=str, nargs="+", default=["python", "cpp"],
                        choices=["python", "cpp"], help="Languages to process and save")
    args = parser.parse_args()

    hf_datasets = {
        "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
        "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
        "code_contests": load_dataset("deepmind/code_contests"),
        "open-r1/codeforces": load_dataset("open-r1/codeforces")
    }
    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # 使用 map 函数实质性地更新数据集中的 question 字段
    for language in args.languages:
        ocr2_dataset[language] = ocr2_dataset[language].map(
            process_and_update,
            fn_kwargs={"hf_datasets": hf_datasets}
        )

        # map 之后，去掉不需要的列（若存在）
        cols_to_remove = ["r1_generation", "qwq_critique", "solution", "judgement"]
        existing_cols = [c for c in cols_to_remove if c in ocr2_dataset[language].column_names]
        if existing_cols:
            ocr2_dataset[language] = ocr2_dataset[language].remove_columns(existing_cols)

    ocr2_dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()