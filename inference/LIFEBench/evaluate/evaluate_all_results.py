import os.path

import numpy as np
import pandas as pd

from evaluate.collect_data import collect_data
from evaluate.filters import illegal_filter


def _resolve_length_field(length_metric):
    metric = str(length_metric or "word").strip().lower()
    if metric == "word":
        return "word_count"
    if metric == "token":
        return "token_count"
    raise ValueError(f"Unsupported length_metric: {length_metric}. Use 'word' or 'token'.")


def calculate_deviation(row, length_field):
    L_output = row.get(length_field)
    if L_output is None:
        raise ValueError(
            f"Missing '{length_field}' in evaluation record for index={row.get('index')}. "
            "Please regenerate results with the matching length metric."
        )
    L_required = row['length_constraint']
    deviation = (L_output - L_required) / L_required * 100
    return deviation


def calculate_deviation_for_data(collected_data, length_field):
    results = []

    for model_name, length_types in collected_data.items():
        for control_method, files in length_types.items():
            row = {'model_name': model_name, 'control_method': control_method}

            scores_per_file = {}

            for jsonl_file, entries in files.items():
                deviations = []

                for entry in entries:
                    deviation = calculate_deviation(entry, length_field)

                    if deviation is not None:
                        deviations.append(deviation)

                if deviations:
                    scores_per_file[jsonl_file] = np.mean(np.abs(deviations))
                else:
                    scores_per_file[jsonl_file] = None

            valid_scores = [s for s in scores_per_file.values() if s is not None]
            row.update(scores_per_file)
            if control_method == 'equal to':
                row["AVG"] = np.mean(np.abs(valid_scores)) if valid_scores else None
            else:
                row["AVG"] = np.mean(valid_scores) if valid_scores else None
            for key in scores_per_file:
                if scores_per_file[key] is not None:
                    scores_per_file[key] = f"{scores_per_file[key]:.0f}%"
            row.update(scores_per_file)

            if row["AVG"] is not None:
                row["AVG"] = f"{row['AVG']:.0f}%"

            results.append(row)

    df = pd.DataFrame(results)
    return df


def exp_asymmetric(deviation, k1=5, k2=2):
    if deviation < 0:
        return 100 * np.exp(k1 * deviation)
    return 100 * np.exp(-k2 * deviation)


def calculate_scores_eq(entry, length_field):
    deviation = calculate_deviation(entry, length_field)
    score_e = exp_asymmetric(deviation / 100)
    return score_e


def calculate_scores_at_most(entry, length_field):
    deviation = calculate_deviation(entry, length_field)
    if deviation < 0:
        return 100
    return exp_asymmetric(deviation / 100)


def calculate_scores_at_least(entry, length_field):
    deviation = calculate_deviation(entry, length_field)
    if deviation > 0:
        return 100
    return exp_asymmetric(deviation / 100)


def calculate_scores_for_data(collected_data, length_field):
    results = []

    for model_name, control_methods in collected_data.items():
        for control_method, files in control_methods.items():
            row = {'model_name': model_name, 'control_method': control_method}

            scores_per_file = {}

            for jsonl_file, entries in files.items():
                scores = []

                for entry in entries:
                    if control_method == 'equal to':
                        score = calculate_scores_eq(entry, length_field)
                    elif control_method == 'at most':
                        score = calculate_scores_at_most(entry, length_field)
                    elif control_method == 'at least':
                        score = calculate_scores_at_least(entry, length_field)
                    else:
                        score = None

                    if score is not None:
                        scores.append(score)

                if scores:
                    scores_per_file[jsonl_file] = round(np.mean(scores), 1)
                else:
                    scores_per_file[jsonl_file] = None

            valid_scores = [s for s in scores_per_file.values() if s is not None]
            row.update(scores_per_file)
            row["AVG"] = round(np.mean(valid_scores), 1) if valid_scores else None

            results.append(row)

    df = pd.DataFrame(results)
    return df


def save_scores_to_csv(score_df, deviation_df, output_path):
    model_names = score_df['model_name'].unique()
    columns = ['model_name', 'equal_to_deviation', 'equal_to_score',
               'at_most_deviation', 'at_most_score',
               'at_least_deviation', 'at_least_score']
    formatted_df = pd.DataFrame(columns=columns)

    for model_name in model_names:
        row = {'model_name': model_name}

        for control_method, dev_col, score_col in zip(
                ['equal to', 'at most', 'at least'],
                ['equal_to_deviation', 'at_most_deviation', 'at_least_deviation'],
                ['equal_to_score', 'at_most_score', 'at_least_score']):
            deviation_value = \
                deviation_df[
                    (deviation_df['model_name'] == model_name) & (deviation_df['control_method'] == control_method)][
                    "AVG"].values
            score_value = \
                score_df[(score_df['model_name'] == model_name) & (score_df['control_method'] == control_method)][
                    "AVG"].values

            row[dev_col] = deviation_value[0] if len(deviation_value) > 0 else None
            row[score_col] = score_value[0] if len(score_value) > 0 else None

        formatted_df = pd.concat([formatted_df, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    formatted_df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")


def evaluate(data_dir, output_dir, output_csv=None, length_metric="word"):
    output_path = output_csv or os.path.join(output_dir, "evaluate_result.csv")
    length_field = _resolve_length_field(length_metric)
    collected_data = collect_data(data_dir, length_metric=length_metric)
    collected_data, reject_filtered_data = illegal_filter(collected_data, length_field=length_field)
    score_df = calculate_scores_for_data(collected_data, length_field)
    deviation_df = calculate_deviation_for_data(collected_data, length_field)
    save_scores_to_csv(score_df, deviation_df, output_path)
