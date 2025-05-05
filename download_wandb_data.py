import os

import pandas as pd
import wandb
import warnings
import json
import ast

warnings.filterwarnings("ignore")

api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

EXTRACTED_METRICS = ['test_acc_epoch', 'test_acc_grid_epoch', 'test_acc_grid_no_pad_epoch', 'test_acc_step',
                     'test_acc_grid_step', 'test_acc_grid_no_pad_step', 'metrics/val_acc_epoch',
                     'metrics/val_acc_grid_epoch', 'metrics/val_acc_grid_no_pad_epoch', 'gen_test_acc_epoch',
                     'gen_test_acc_grid_epoch', 'gen_test_acc_grid_no_pad_epoch', 'gen_test_acc_step',
                     'gen_test_acc_grid_step', 'gen_test_acc_grid_no_pad_step', 'metrics/gen_val_acc_epoch',
                     'metrics/gen_val_acc_grid_epoch', 'metrics/gen_val_acc_grid_no_pad_epoch']


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
            run_infos['study'] = config["experiment"]["value"]["study"]
            run_infos['name'] = config["experiment"]["value"]["name"]
            run_infos['setting'] = config["experiment"]["value"]["setting"]
            run_infos['backbone'] = ast.literal_eval(config["backbone_network_config"]["value"])["name"]
            run_infos['head'] = ast.literal_eval(config["head_network_config"]["value"])["name"]
            run_infos['use_task_embeddings'] = ast.literal_eval(config['model_config']['value'])['task_embedding'][
                'enabled']

            # Get required results
            summary = run.summary._json_dict
            for key in EXTRACTED_METRICS:

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
    for study in filtered_df['study'].unique():
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
         'col': 'Test Acc. Grid',
         },
        {'key': 'metrics/val_acc_grid_no_pad_epoch',
         'col': 'Eval Acc. Grid',
         },
        {'key': 'gen_test_acc_grid_no_pad_epoch',
         'col': 'Gen. Test Acc. Grid',
         },
    ]

    models = [
        {'name': 'LlADA + MLP w/o TE',
         'conditions': {
             'backbone': 'llada',
             'head': 'mlp',
             'use_task_embeddings': False,
         },
         },
        {'name': 'LlADA + MLP w/ TE',
         'conditions': {
             'backbone': 'llada',
             'head': 'mlp',
             'use_task_embeddings': True,
         },
         },
    ]

    for study in run_data_df['study'].unique():
        for metric in metrics:
            table = []
            table.append(r"\begin{table}[ht]")
            table.append(r"\centering")
            table.append(f"\\caption{{{metric['col']} for Study: {study}}}")
            table.append(r"\begin{tabular}{l" + "c" * len(models) + "}")
            table.append(r"\toprule")
            header = ["Experiment"] + [model["name"] for model in models]
            table.append(" & ".join(header) + r" \\")
            table.append(r"\midrule")

            settings = run_data_df[run_data_df['study'] == study]['setting'].unique()
            for setting in sorted(settings):
                table.append(f"{setting.replace('exp_setting_', 'Exp. Setting ')}")
                table.append(r"\midrule")
                setting_rows = []

                experiments = run_data_df[(run_data_df['study'] == study) & (run_data_df['setting'] == setting)]['name'].unique()
                for experiment in sorted(experiments):
                    row = [experiment]
                    for model in models:
                        cond = model['conditions']
                        mask = (
                                (run_data_df['study'] == study) &
                                (run_data_df['setting'] == setting) &
                                (run_data_df['name'] == experiment) &
                                (run_data_df['backbone'] == cond['backbone']) &
                                (run_data_df['head'] == cond['head']) &
                                (run_data_df['use_task_embeddings'] == cond['use_task_embeddings'])
                        )
                        value = run_data_df.loc[mask, metric['key']]
                        if len(value.values) > 1:
                            warnings.warn(f"Multiple values found for {metric['key']} with {study}, {setting}, {experiment}, {model['name']}")
                            warnings.warn(f"Found these values {str(value.values)}")
                        if not value.empty:
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


def main():
    data_beforearc_df = download_data(entity="sagerpascal", project="VisReas-project-BEFOREARC-sweep")
    data_beforearc_df['dataset'] = ["BEFOREARC"] * len(data_beforearc_df)
    # pretty_print_run_data(run_data_df)

    data_rearc_df = download_data(entity="sagerpascal", project="VisReas-project-REARC-sweep")
    data_rearc_df['dataset'] = ["REARC"] * len(data_rearc_df)
    # pretty_print_run_data(run_data_df)

    data_df = pd.concat([data_beforearc_df, data_rearc_df])
    create_latex_tables(data_df)


if __name__ == "__main__":
    main()
