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


def pretty_print_run_data(run_data_df: pd.DataFrame) -> None:
    # Drop entries missing required metrics
    filtered_df = run_data_df.dropna(subset=EXTRACTED_METRICS, how='all')

    # Sort by the hierarchy
    filtered_df = filtered_df.sort_values(by=['study', 'setting', 'name', 'use_task_embeddings'])

    # Hierarchical print including task embedding flag
    for study in ['comp', 'sysgen']: # filtered_df['study'].unique():
        print(f"Study: {study}")
        study_df = filtered_df[filtered_df['study'] == study]

        for setting in study_df['setting'].unique():
            print(f"  Setting: {setting}")
            setting_df = study_df[study_df['setting'] == setting]

            for use_te in [True, False]:
                te_df = setting_df[setting_df['use_task_embeddings'] == use_te]
                if te_df.empty:
                    continue
                print(f"    Use Task Embeddings: {use_te}")

                for _, row in te_df.iterrows():
                    print(f"      Name: {row['name']}")
                    for metric in EXTRACTED_METRICS:
                        val = row.get(metric)
                        if val is not None:
                            print(f"        {metric}: {val}")


def create_latex_tables(run_data_df: pd.DataFrame) -> None:
    metrics = [
        {'key': 'test_acc_grid_no_pad_epoch',
         'col': 'IID',
         },
        {'key': 'gen_test_acc_grid_no_pad_epoch',
         'col': 'OOD',
         },
    ]

    models = [
        {'name': 'LlADA',
         'conditions': {
             'backbone': 'llada',
             'head': 'mlp',
         },
         },
        {'name': 'Looped ViT',
         'conditions': {
             'backbone': 'looped_vit',
             'head': 'mlp',
         },
         },
        {'name': 'ResNet',
         'conditions': {
             'backbone': 'resnet',
             'head': 'mlp',
         },
         },
    ]

    for data_env in run_data_df['data_env'].unique():
        data_df = run_data_df[run_data_df['data_env'] == data_env]
        for study in data_df['study'].unique():
            for metric in metrics:
                table = []
                table.append(r"\begin{table}[ht]")
                table.append(r"\centering")
                table.append(f"\\caption{{{data_env}: {metric['col']} for Study: {study}}}")
                table.append(r"\begin{tabular}{l" + "|c" * len(models) + "}")
                table.append(r"\toprule")
                header = ["Experiment"] + [model["name"] for model in models]
                header = [f"\\textbf{{{h}}}" for h in header]
                table.append(" & ".join(header) + r" \\")
                table.append(r"\midrule")

                data_df2 = data_df[data_df['study'] == study]
                settings = data_df2['setting'].unique()
                for setting in sorted(settings):
                    table.append(f"\\multicolumn{{{len(models)+1}}}{{c}}{{\\textbf{{\\emph{{{setting.replace('exp_setting_', 'Exp. Setting ')}}}}}}} \\\\")
                    table.append(r"\midrule")
                    setting_rows = []

                    data_df3 = data_df2[data_df2['setting'] == setting]
                    experiments = data_df3['name'].unique()
                    for experiment in sorted(experiments):
                        row = [experiment.replace('experiment_', 'Exp. ')]
                        for model in models:
                            cond = model['conditions']
                            base_conditions = (
                                    (data_df3['study'] == study) &
                                    (data_df3['setting'] == setting) &
                                    (data_df3['name'] == experiment)
                            )

                            # Build additional conditions from `cond` keys
                            additional_conditions = [data_df3[key] == value for key, value in cond.items()]
                            mask = base_conditions & np.logical_and.reduce(additional_conditions)
                            value = data_df3.loc[mask, metric['key']]
                            if len(value.values) > 1:
                                warnings.warn(f"Multiple values found for {metric['key']} with {study}, {setting}, {experiment}, {model['name']}")
                                warnings.warn(f"Found these values {str(value.values)}")
                            if not value.empty:
                                if math.isnan(value.values[0]):
                                    warnings.warn(f"No data found for {metric['key']}")
                                row.append(f"{value.values[0]:.2f}")
                            else:
                                row.append("N/A")
                        setting_rows.append(" & ".join(row) + r" \\")
                    table.extend(setting_rows)
                    table.append(r"\midrule")

                table.append(r"\bottomrule")
                table.append(r"\end{tabular}")
                table.append(r"\end{table}")
                table.append("\n")

                # Output the LaTeX table (e.g., print or save to file)
                print("\n".join(table))




def check_completeness(run_data_df: pd.DataFrame) -> None:
    compositionality_settings = ['exp_setting_1', 'exp_setting_2', 'exp_setting_3']
    compositionality_names = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']

    sysgen_settings = ['exp_setting_1', 'exp_setting_2', 'exp_setting_3', 'exp_setting_4', 'exp_setting_5']
    sysgen_names = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']

    compositionality_combos = list(itertools.product(['compositionality'], compositionality_settings, compositionality_names))
    sysgen_combos = list(itertools.product(['sys-gen'], sysgen_settings, sysgen_names))

    target_combos = compositionality_combos + sysgen_combos

    # Filter the DataFrame
    filtered_df = run_data_df[run_data_df[['study', 'setting', 'name']].apply(tuple, axis=1).isin(target_combos)]
    grouped = filtered_df.groupby(['model', 'study', 'setting', 'name']).size().reset_index(name='count')

    # Identify issues
    # a) Missing combinations
    all_models = run_data_df['model'].unique()
    all_expected = pd.MultiIndex.from_tuples(
        [(model, study, setting, name) for model in all_models for (study, setting, name) in target_combos],
        names=['model', 'study', 'setting', 'name']
    )
    actual = pd.MultiIndex.from_frame(grouped[['model', 'study', 'setting', 'name']])
    missing = all_expected.difference(actual)

    # Duplicate runs
    duplicates = grouped[grouped['count'] > 1]

    # Just keep model-identifying columns for NaN rows
    nan_rows = filtered_df[
        filtered_df['test_acc_grid_no_pad_epoch'].isna() |
        filtered_df['gen_test_acc_grid_no_pad_epoch'].isna()
        ]
    nan_models_info = nan_rows[['id', 'model', 'study', 'setting', 'name', 'test_acc_grid_no_pad_epoch', 'gen_test_acc_grid_no_pad_epoch']]

    # Output results
    print("=== Missing combinations ===")
    print(missing)

    print("\n=== Duplicate runs ===")
    print(duplicates)

    print("\n=== Models with NaN values in performance columns ===")
    print(nan_models_info)


def calc_table_averages(run_data_df: pd.DataFrame) -> None:
    metrics = ["test_acc_grid_no_pad_epoch", "gen_test_acc_grid_no_pad_epoch"]
    grouping = ["data_env", "model", "study", "setting"]

    # Model names
    run_data_df['model'] = run_data_df['backbone'] + '+' + run_data_df['head']

    # Filter runs
    run_data_df = run_data_df[(run_data_df.state == "finished") & (run_data_df["data_env"] == "BEFOREARC")]
    check_completeness(run_data_df)

    # Output values
    aggregated = run_data_df.groupby(grouping)[metrics].mean().reset_index()
    print(aggregated)



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
