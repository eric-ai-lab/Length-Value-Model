import argparse

from evaluate.evaluate_all_results import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./result",
        required=False,
        help='Directory containing the data to be evaluated. Default: ./result'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluate_result",
        required=False,
        help='Directory to save the evaluation results. Default: ./evaluate_result'
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        required=False,
        help='Optional explicit path to the evaluation summary CSV. Overrides --output_dir/evaluate_result.csv'
    )
    parser.add_argument(
        "--length_metric",
        type=str,
        default="word",
        choices=["word", "token"],
        required=False,
        help="Length metric used during evaluation. Choices: word, token. Default: word"
    )
    args = parser.parse_args()

    # Run the evaluation process
    evaluate(args.data_dir, args.output_dir, args.output_csv, args.length_metric)
    if args.output_csv:
        print(f"Evaluation completed. Results have been saved to: {args.output_csv}")
    else:
        print(f"Evaluation completed. Results have been saved to: {args.output_dir}")
