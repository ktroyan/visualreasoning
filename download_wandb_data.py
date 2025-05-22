import os

import pandas as pd
import wandb
import warnings
import json
import ast
import numpy as np
import math
import itertools

# warnings.filterwarnings("ignore")

api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

EXTRACTED_METRICS = ['test_acc_epoch', 'test_acc_grid_epoch', 'test_acc_grid_no_pad_epoch', 'test_acc_step',
                     'test_acc_grid_step', 'test_acc_grid_no_pad_step', 'metrics/val_acc_epoch',
                     'metrics/val_acc_grid_epoch', 'metrics/val_acc_grid_no_pad_epoch', 'gen_test_acc_epoch',
                     'gen_test_acc_grid_epoch', 'gen_test_acc_grid_no_pad_epoch', 'gen_test_acc_step',
                     'gen_test_acc_grid_step', 'gen_test_acc_grid_no_pad_step', 'metrics/gen_val_acc_epoch',
                     'metrics/gen_val_acc_grid_epoch', 'metrics/gen_val_acc_grid_no_pad_epoch']


EXTRACTED_METRICS = ['test_acc_grid_no_pad_epoch', 'gen_test_acc_grid_no_pad_epoch']


def download_data(entity: str, project: str) -> pd.DataFrame:
    output = []
    runs = api.runs(entity + "/" + project)

    print("Number of runs:", len(runs))

    for run in runs:
        if run.state != "finished":
            continue

        try:
            run_infos = {'id': run.id, 'name': run.name, 'state': run.state}

            # Load required config
            config = json.loads(run.json_config)
            run_infos['data_env'] = ast.literal_eval(config["base_config"]["value"])["data_env"]
            run_infos['seed'] = ast.literal_eval(config["base_config"]["value"])["seed"]
            has_gen_test_data = ast.literal_eval(config["data_config"]["value"])["use_gen_test_set"]
            has_gen_val_data = ast.literal_eval(config["data_config"]["value"])["validate_in_and_out_domain"]
            if "experiment" in config:
                run_infos['study'] = config["experiment"]["value"]["study"]
                run_infos['setting'] = config["experiment"]["value"]["setting"]
                run_infos['name'] = config["experiment"]["value"]["name"]
            else:
                exp_infos = ast.literal_eval(config["data_config"]["value"])['dataset_dir'].split("/")
                run_infos['study'] = exp_infos[-3]
                run_infos['setting'] =exp_infos[-2]
                run_infos['name'] = exp_infos[-1]

            if "backbone_network_config" in config:
                run_infos['backbone'] = ast.literal_eval(config["backbone_network_config"]["value"])["name"]
            else:
                run_infos['backbone'] = config["model"]["value"]["backbone"]

            if "backbone_network_config" in config:
                run_infos['head'] = ast.literal_eval(config["head_network_config"]["value"])["name"]
            else:
                run_infos['head'] = config["model"]["value"]["head"]

            # Get required results
            summary = run.summary._json_dict
            for key in EXTRACTED_METRICS:

                if "gen" in key and "test" in key and not has_gen_test_data:
                    continue

                if "gen" in key and "val" in key and not has_gen_val_data:
                    continue

                r = summary.get(key, None)
                if r is None:
                    print(f"Key {key} not found in run {run.id}")
                    print("It might be worth checking the slurm output of job",
                          run.metadata.get('slurm', {}).get('job_id', None))

                run_infos[key] = r

            output.append(run_infos)

        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue

    return pd.DataFrame.from_dict(output)


import itertools
import pandas as pd

def check_completeness(run_data_df: pd.DataFrame) -> None:
    # Definitions
    compositionality_settings = ['exp_setting_1', 'exp_setting_2', 'exp_setting_3']
    compositionality_names = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']

    sysgen_settings = ['exp_setting_1', 'exp_setting_2', 'exp_setting_3', 'exp_setting_4', 'exp_setting_5']
    sysgen_names = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']

    compositionality_combos = list(itertools.product(['compositionality'], compositionality_settings, compositionality_names))
    sysgen_combos = list(itertools.product(['sys-gen'], sysgen_settings, sysgen_names))
    target_combos = compositionality_combos + sysgen_combos
    required_seeds = {1997, 2025, 4269}

    # === Deduplicate and average metric columns ===
    unique_cols = ["data_env", "model", "study", "setting", "name", "seed"]
    metrics = ["test_acc_grid_no_pad_epoch", "gen_test_acc_grid_no_pad_epoch"]

    dedup_df = (
        run_data_df
        .groupby(unique_cols, dropna=False)[metrics]
        .mean()
        .reset_index()
    )

    # === Filter relevant rows ===
    filtered_df = dedup_df[dedup_df[['study', 'setting', 'name']].apply(tuple, axis=1).isin(target_combos)]

    # === Check for missing (model, study, setting, name) combinations ===
    all_models = dedup_df['model'].unique()
    all_expected = pd.MultiIndex.from_tuples(
        [(model, study, setting, name) for model in all_models for (study, setting, name) in target_combos],
        names=['model', 'study', 'setting', 'name']
    )
    actual = pd.MultiIndex.from_frame(filtered_df[['model', 'study', 'setting', 'name']])
    missing_combinations = all_expected.difference(actual)

    # === Check for missing seeds per combination ===
    seed_grouped = filtered_df.groupby(['model', 'study', 'setting', 'name'])['seed'].apply(set).reset_index()
    missing_seeds_rows = seed_grouped[seed_grouped['seed'].apply(lambda s: not required_seeds.issubset(s))]

    # === Check for NaNs ===
    nan_rows = filtered_df[
        filtered_df['test_acc_grid_no_pad_epoch'].isna() |
        filtered_df['gen_test_acc_grid_no_pad_epoch'].isna()
    ]
    nan_info = nan_rows[["model", "study", "setting", "name", "seed", "test_acc_grid_no_pad_epoch", "gen_test_acc_grid_no_pad_epoch"]]

    # === Output ===
    print("=== Missing combinations ===")
    if not missing_combinations.empty:
        for entry in missing_combinations:
            print(entry)
    else:
        print("✅ All (model, study, setting, name) combinations are present.")

    print("\n=== Missing seeds ===")
    if not missing_seeds_rows.empty:
        for _, row in missing_seeds_rows.iterrows():
            missing_seeds = required_seeds - row['seed']
            print(f"{tuple(row[['model', 'study', 'setting', 'name']])}: missing seeds {sorted(missing_seeds)}")
    else:
        print("✅ All combinations have required seeds (1997, 2025, 4269).")

    print("\n=== Models with NaN in performance metrics ===")
    if not nan_info.empty:
        print(nan_info)
    else:
        print("✅ No missing values in key metrics.")



def calc_table_averages(run_data_df: pd.DataFrame) -> None:
    metrics = ["test_acc_grid_no_pad_epoch", "gen_test_acc_grid_no_pad_epoch"]
    grouping = ["data_env", "model", "study", "setting"]

    # Model names
    run_data_df['model'] = run_data_df['backbone'] + '+' + run_data_df['head']

    # Filter runs
    run_data_df = run_data_df[(run_data_df.state == "finished") & (run_data_df["data_env"] == "BEFOREARC")]

    # if we have same experiment twice, average them
    unique_groups = ["data_env", "model", "study", "setting", "name", "seed"]
    run_data_df = run_data_df.groupby(unique_groups)[metrics].mean().reset_index()

    check_completeness(run_data_df)


    # Output values -> main table
    aggregated = run_data_df.groupby(grouping)[metrics].mean().reset_index()
    print(aggregated)
    aggregated_std = run_data_df.groupby(grouping)[metrics].std().reset_index()
    print(aggregated_std)

    # Output values -> per experiment table / appendix
    grouping = grouping + ["name"]
    aggregated = run_data_df.groupby(grouping)[metrics].mean().reset_index()
    print(aggregated)
    aggregated_std = run_data_df.groupby(grouping)[metrics].std().reset_index()
    print(aggregated_std)




def main():
    beforearc_llada_df = download_data(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada-final")
    # pretty_print_run_data(beforearc_llada_df)

    #klim_df = download_data(entity="VisReas-ETHZ", project="VisReas-project")
    # pretty_print_run_data(klim_df)

    data_df = pd.concat([beforearc_llada_df])
    # create_latex_tables(data_df)
    calc_table_averages(data_df)


if __name__ == "__main__":
    main()
