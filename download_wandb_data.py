import os

import pandas as pd
import wandb
import warnings
import json
import ast
import numpy as np
import math

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
        if run.state == "running":
            print(run.name, "running")
            continue

        try:
            run_infos = {'id': run.id, 'name': run.name, 'state': run.state}

            # Load required config
            config = json.loads(run.json_config)
            run_infos['data_env'] = config["base"]["value"]["data_env"]
            run_infos['study'] = config["experiment"]["value"]["study"]
            run_infos['name'] = config["experiment"]["value"]["name"]
            run_infos['setting'] = config["experiment"]["value"]["setting"]
            has_gen_test_data = ast.literal_eval(config["data_config"]["value"])["use_gen_test_set"]
            has_gen_val_data = ast.literal_eval(config["data_config"]["value"])["validate_in_and_out_domain"]

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






def calc_table_averages(run_data_df: pd.DataFrame) -> None:
    metrics = ["test_acc_grid_no_pad_epoch", "gen_test_acc_grid_no_pad_epoch"]
    grouping = ["data_env", "study", "setting", "backbone", "head"]

    # Filter runs
    run_data_df = run_data_df[(run_data_df.state == "finished") & (run_data_df["data_env"] == "BEFOREARC")]

    # Check Completeness
    expected_runs = 23
    counts = run_data_df.groupby(["backbone", "head"]).count()["id"]
    for key, count in counts.items():
        if count != expected_runs:
            print(f"Warning: {key} has {count} runs instead of {expected_runs}")

    # TODO; Check for NaN values

    # TODO; Check for duplicate runs

    # Output values
    aggregated = run_data_df.groupby(grouping)[metrics].mean().reset_index()

    print(aggregated)

    # TODO; make it nicer


def main():
    rearc_llada_df = download_data(entity="VisReas-ETHZ", project="VisReas-project-REARC-llada")
    # pretty_print_run_data(rearc_llada_df)

    beforearc_llada_df = download_data(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada")
    # pretty_print_run_data(beforearc_llada_df)

    klim_df = download_data(entity="VisReas-ETHZ", project="VisReas-project")
    # pretty_print_run_data(klim_df)

    data_df = pd.concat([rearc_llada_df, beforearc_llada_df, klim_df])
    # create_latex_tables(data_df)
    calc_table_averages(data_df)


if __name__ == "__main__":
    main()
