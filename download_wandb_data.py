import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import wandb
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
import ast


warnings.filterwarnings("ignore")

api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
entity = "sagerpascal"
project = "VisReas-project-BEFOREARC-sweep"


def download_data() -> pd.DataFrame:
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
            run_infos['use_task_embeddings'] = ast.literal_eval(config['model_config']['value'])['task_embedding']['enabled']

            # Get required results
            summary = run.summary._json_dict
            # for key in ['best_gen_val_acc', 'best_val_acc', 'best_val_grid_acc', 'gen_test_acc_epoch',
            #             'gen_test_acc_grid_epoch', 'gen_test_acc_grid_no_pad_epoch', 'gen_test_acc_grid_no_pad_step',
            #             'gen_test_acc_grid_step', 'gen_test_acc_no_pad_epoch', 'gen_test_acc_no_pad_step',
            #             'gen_test_acc_step', 'gen_val_acc_best_epoch', 'metrics/gen_val_acc_epoch',
            #             'metrics/gen_val_acc_grid_epoch', 'metrics/gen_val_acc_grid_no_pad_epoch',
            #             'metrics/gen_val_acc_grid_no_pad_step', 'metrics/gen_val_acc_grid_step',
            #             'metrics/gen_val_acc_no_pad_epoch', 'metrics/gen_val_acc_no_pad_step',
            #             'metrics/gen_val_acc_step',
            #             'metrics/val_acc_epoch', 'metrics/val_acc_grid_epoch', 'metrics/val_acc_grid_no_pad_epoch',
            #             'metrics/val_acc_grid_no_pad_step', 'metrics/val_acc_grid_step', 'metrics/val_acc_no_pad_epoch',
            #             'metrics/val_acc_no_pad_step', 'metrics/val_acc_step', 'test_acc_epoch', 'test_acc_grid_epoch',
            #             'test_acc_grid_no_pad_epoch', 'test_acc_grid_no_pad_step', 'test_acc_grid_step',
            #             'test_acc_no_pad_epoch', 'test_acc_no_pad_step', 'test_acc_step', 'val_acc_best_epoch',
            #             'val_grid_acc_best_epoch']:

            for key in ['test_acc_epoch', 'test_acc_grid_epoch', 'test_acc_grid_no_pad_epoch', 'test_acc_step', 'test_acc_grid_step', 'test_acc_grid_no_pad_step', 'metrics/val_acc_epoch', 'metrics/val_acc_grid_epoch', 'metrics/val_acc_grid_no_pad_epoch', 'gen_test_acc_epoch', 'gen_test_acc_grid_epoch', 'gen_test_acc_grid_no_pad_epoch', 'gen_test_acc_step', 'gen_test_acc_grid_step', 'gen_test_acc_grid_no_pad_step', 'metrics/gen_val_acc_epoch', 'metrics/gen_val_acc_grid_epoch', 'metrics/gen_val_acc_grid_no_pad_epoch']:

                r = summary.get(key, None)
                if r is None:
                    print(f"Key {key} not found in run {run.id}")
                    print("It might be worth checking the slurm output of job", run.metadata.get('slurm', {}).get('job_id', None))

                run_infos[key] = r

            output.append(run_infos)

        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue

    return pd.DataFrame.from_dict(output)

def pretty_print_run_data(run_data_df: pd.DataFrame) -> None:
    selected_metrics =  ['test_acc_epoch', 'test_acc_grid_epoch', 'test_acc_grid_no_pad_epoch', 'test_acc_step', 'test_acc_grid_step', 'test_acc_grid_no_pad_step', 'metrics/val_acc_epoch', 'metrics/val_acc_grid_epoch', 'metrics/val_acc_grid_no_pad_epoch', 'gen_test_acc_epoch', 'gen_test_acc_grid_epoch', 'gen_test_acc_grid_no_pad_epoch', 'gen_test_acc_step', 'gen_test_acc_grid_step', 'gen_test_acc_grid_no_pad_step', 'metrics/gen_val_acc_epoch', 'metrics/gen_val_acc_grid_epoch', 'metrics/gen_val_acc_grid_no_pad_epoch']

    # Drop entries missing required metrics
    filtered_df = run_data_df.dropna(subset=selected_metrics, how='all')

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
                    for metric in selected_metrics:
                        val = row.get(metric)
                        if val is not None:
                            print(f"        {metric}: {val}")

        continue

        # Melt for visualization
        melted_df = filtered_df.melt(
            id_vars=["study", "setting", "name", "use_task_embeddings"],
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Accuracy"
        ).dropna()

        # Plot per study and metric
        for study in melted_df['study'].unique():
            study_df = melted_df[melted_df['study'] == study]

            for metric in study_df['Metric'].unique():
                metric_df = study_df[study_df['Metric'] == metric]

                settings = metric_df['setting'].unique()
                n_settings = len(settings)

                # Set up the figure with one subplot per setting
                fig, axes = plt.subplots(nrows=1, ncols=n_settings, figsize=(6 * n_settings, 6), sharey=True)
                if n_settings == 1:
                    axes = [axes]  # Make iterable if only one subplot

                for ax, setting in zip(axes, settings):
                    setting_df = metric_df[metric_df['setting'] == setting]
                    sns.barplot(
                        data=setting_df,
                        x='name',
                        y='Accuracy',
                        hue='use_task_embeddings',
                        ax=ax
                    )
                    ax.set_title(f"Setting: {setting}")
                    ax.set_xlabel("Run Name")
                    ax.set_ylabel("Accuracy")
                    ax.tick_params(axis='x', rotation=45)

                fig.suptitle(f"Study: {study} | Metric: {metric}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


def main():
    run_data_df = download_data()
    pretty_print_run_data(run_data_df)


if __name__ == "__main__":
    main()