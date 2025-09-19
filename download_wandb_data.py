import os
import wandb
import json
import ast
import pandas as pd
import itertools

# warnings.filterwarnings("ignore")

api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
EXTRACTED_METRICS = ['test_acc_grid_epoch', 'gen_test_acc_grid_epoch']

CACHE_DIR = "cache_wandb"
os.makedirs(CACHE_DIR, exist_ok=True)


def cached_download(entity: str, project: str, refresh: bool = False) -> pd.DataFrame:
    """
    Downloads dataframe from HuggingFace and caches it locally as a pickle.
    If refresh=False and cache exists, load from cache instead of re-downloading.
    """
    cache_file = os.path.join(CACHE_DIR, f"{entity}_{project}.pkl")

    if not refresh and os.path.exists(cache_file):
        print(f"Loading cached dataframe: {cache_file}")
        return pd.read_pickle(cache_file)

    print(f"Downloading dataframe from HF: entity={entity}, project={project}")
    df = download_data(entity=entity, project=project)  # <-- your existing function
    df.to_pickle(cache_file)
    return df


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

                if 'exp_specifics' in config["experiment"]["value"]:
                    run_infos['exp_specifics'] = config["experiment"]["value"]["exp_specifics"]

            else:
                exp_infos = ast.literal_eval(config["data_config"]["value"])['dataset_dir'].split("/")
                run_infos['study'] = exp_infos[-3]
                run_infos['setting'] = exp_infos[-2]
                run_infos['name'] = exp_infos[-1]

            if "backbone_network_config" in config:
                run_infos['backbone'] = ast.literal_eval(config["backbone_network_config"]["value"])["name"]
            else:
                run_infos['backbone'] = config["model"]["value"]["backbone"]

            if "backbone_network_config" in config:
                run_infos['head'] = ast.literal_eval(config["head_network_config"]["value"])["name"]
            else:
                run_infos['head'] = config["model"]["value"]["head"]

            if not run_infos['backbone'] == "llada":
                continue

            # Define model size
            backbone = ast.literal_eval(config["backbone_network_config"]["value"])
            run_infos['model_definitions'] = f"{backbone['embed_dim']}-{backbone['mlp_hidden_size']}-{backbone['mlp_ratio']}-{backbone['n_heads']}-{backbone['n_kv_heads']}-{backbone['n_layers']}"

            # TODO: Remove this temporary fix
            if run_infos['seed'] == 42:
                run_infos['seed'] = 4269

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

        do_update = True

        if do_update:
            if not run_infos['backbone'] == "llada" or "study" in run_infos or "setting" in run_infos or "experiment_name" in run_infos or "model_name" in run_infos or "seed" in run_infos:
                print(f"Skipping 'do_update' for run {run.id}: {run_infos}")
                continue

            assert run_infos['backbone'] == "llada"
            assert "study" not in run.summary.keys()
            assert "setting" not in run.summary.keys()
            assert "experiment_name" not in run.summary.keys()
            assert "model_name" not in run.summary.keys()
            assert "seed" not in run.summary.keys()

            run.summary["study"] = run_infos['study']
            run.summary["setting"] = run_infos['setting']
            run.summary["experiment_name"] = run_infos['name']
            run.summary["model_name"] = "LLaDA"
            run.summary["seed"] = run_infos['seed']
            if "test_acc_grid_epoch" not in run.summary.keys():
                run.summary["test_acc_grid_epoch"] = run_infos['test_acc_grid_epoch'] if "test_acc_grid_epoch" in run_infos else None
            if "gen_test_acc_grid_epoch" not in run.summary.keys():
                run.summary["gen_test_acc_grid_epoch"] = run_infos['gen_test_acc_grid_epoch'] if "gen_test_acc_grid_epoch" in run_infos else None
            run.summary.update()

    return pd.DataFrame.from_dict(output)


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
    if 'exp_specifics' in run_data_df.columns:
        unique_cols.append("exp_specifics")
    if 'model_definitions' in run_data_df.columns:
        unique_cols.append("model_definitions")

    # Filter metrics to only those that exist in the dataframe
    available_metrics = [m for m in EXTRACTED_METRICS if m in run_data_df.columns]

    if available_metrics:
        dedup_df = (
            run_data_df
            .groupby(unique_cols, dropna=False)[available_metrics]
            .mean()
            .reset_index()
        )
    else:
        dedup_df = run_data_df.groupby(unique_cols, dropna=False).first().reset_index()

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
    if available_metrics:
        nan_condition = filtered_df[available_metrics[0]].isna()
        for metric in available_metrics[1:]:
            nan_condition |= filtered_df[metric].isna()

        nan_rows = filtered_df[nan_condition]
        nan_info = nan_rows[["model", "study", "setting", "name", "seed"] + available_metrics]
    else:
        nan_rows = pd.DataFrame()  # Empty dataframe if no metrics available
        nan_info = pd.DataFrame()

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
    metrics = EXTRACTED_METRICS
    grouping = ["data_env", "model", "study", "setting"]

    # Model names
    run_data_df['model'] = run_data_df['backbone'] + '+' + run_data_df['head']

    # Filter runs
    run_data_df = run_data_df[(run_data_df.state == "finished") & (run_data_df["data_env"] == "BEFOREARC")]

    # if we have same experiment twice, average them
    unique_groups = ["data_env", "model", "study", "setting", "name", "seed"]
    if 'exp_specifics' in run_data_df.columns:
        unique_groups.append("exp_specifics")
    if 'model_definitions' in run_data_df.columns:
        unique_groups.append("model_definitions")

    # Filter metrics to only those that exist in the dataframe
    available_metrics = [m for m in metrics if m in run_data_df.columns]
    if available_metrics:
        run_data_df = run_data_df.groupby(unique_groups)[available_metrics].mean().reset_index()
    else:
        run_data_df = run_data_df.groupby(unique_groups).first().reset_index()

    check_completeness(run_data_df)

    # Output values -> main table
    if available_metrics:
        aggregated = run_data_df.groupby(grouping)[available_metrics].mean().reset_index()
        aggregated_std = run_data_df.groupby(grouping)[available_metrics].std().reset_index()

        # Add model_definitions and exp_specifics to grouping for display if available
        display_grouping = grouping.copy()
        if 'model_definitions' in run_data_df.columns and 'model_definitions' not in display_grouping:
            display_grouping.append('model_definitions')
        if 'exp_specifics' in run_data_df.columns and 'exp_specifics' not in display_grouping:
            display_grouping.append('exp_specifics')

        # Re-aggregate with display grouping if it's different
        if display_grouping != grouping:
            aggregated = run_data_df.groupby(display_grouping)[available_metrics].mean().reset_index()
            aggregated_std = run_data_df.groupby(display_grouping)[available_metrics].std().reset_index()

        # Sort by the specified order
        sort_cols = ['model']
        if 'model_definitions' in aggregated.columns:
            sort_cols.append('model_definitions')
        sort_cols.extend(['study', 'setting'])
        if 'name' in aggregated.columns:
            sort_cols.append('name')
        if 'exp_specifics' in aggregated.columns:
            sort_cols.append('exp_specifics')

        aggregated = aggregated.sort_values(sort_cols).reset_index(drop=True)
        aggregated_std = aggregated_std.sort_values(sort_cols).reset_index(drop=True)

        # Create combined table with mean ± std format
        combined_table = aggregated.copy()
        for metric in available_metrics:
            combined_table[metric] = aggregated[metric].apply(lambda x: f"{x * 100:.1f}") + " ± " + aggregated_std[
                metric].apply(lambda x: f"{x * 100:.1f}")

        # Reorder columns: data_env, model, model_definitions, study, setting, name, exp_specifics, then metrics
        column_order = []
        if 'data_env' in combined_table.columns:
            column_order.append('data_env')
        if 'model' in combined_table.columns:
            column_order.append('model')
        if 'model_definitions' in combined_table.columns:
            column_order.append('model_definitions')
        if 'study' in combined_table.columns:
            column_order.append('study')
        if 'setting' in combined_table.columns:
            column_order.append('setting')
        if 'name' in combined_table.columns:
            column_order.append('name')
        if 'exp_specifics' in combined_table.columns:
            column_order.append('exp_specifics')
        # Add metrics at the end
        column_order.extend(available_metrics)

        combined_table = combined_table[column_order]

        # Set pandas options to display all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print("\n=== Main Table ===")
        print(combined_table.to_string(index=False))

        # Output values -> per experiment table / appendix
        grouping = grouping + ["name"]

        # Add model_definitions and exp_specifics to grouping for display if available
        display_grouping_exp = grouping.copy()
        if 'model_definitions' in run_data_df.columns and 'model_definitions' not in display_grouping_exp:
            display_grouping_exp.append('model_definitions')
        if 'exp_specifics' in run_data_df.columns and 'exp_specifics' not in display_grouping_exp:
            display_grouping_exp.append('exp_specifics')

        aggregated = run_data_df.groupby(display_grouping_exp)[available_metrics].mean().reset_index()
        aggregated_std = run_data_df.groupby(display_grouping_exp)[available_metrics].std().reset_index()

        # Sort by the specified order
        aggregated = aggregated.sort_values(sort_cols).reset_index(drop=True)
        aggregated_std = aggregated_std.sort_values(sort_cols).reset_index(drop=True)

        # Create combined table with mean ± std format
        combined_table = aggregated.copy()
        for metric in available_metrics:
            combined_table[metric] = aggregated[metric].apply(lambda x: f"{x * 100:.1f}") + " ± " + aggregated_std[
                metric].apply(lambda x: f"{x * 100:.1f}")

        # Reorder columns: data_env, model, model_definitions, study, setting, name, exp_specifics, then metrics
        column_order = []
        if 'data_env' in combined_table.columns:
            column_order.append('data_env')
        if 'model' in combined_table.columns:
            column_order.append('model')
        if 'model_definitions' in combined_table.columns:
            column_order.append('model_definitions')
        if 'study' in combined_table.columns:
            column_order.append('study')
        if 'setting' in combined_table.columns:
            column_order.append('setting')
        if 'name' in combined_table.columns:
            column_order.append('name')
        if 'exp_specifics' in combined_table.columns:
            column_order.append('exp_specifics')
        # Add metrics at the end
        column_order.extend(available_metrics)

        combined_table = combined_table[column_order]

        print("\n=== Per Experiment Table ===")
        print(combined_table.to_string(index=False))
    else:
        print("No available metrics found in dataframe")


def main():
    refresh = False

    ## BASE RUNS

    # Base Runs with seed 1997
    df_1 = cached_download(entity="VisReas-ETHZ", project="VisReas-project", refresh=refresh)
    # Base Runs with seed 2025
    df_2 = cached_download(entity="VisReas-ETHZ", project="VisReas-project-seed2025", refresh=refresh)
    # Base Runs with seed 42
    df_3 = cached_download(entity="VisReas-ETHZ", project="VisReas-project-seed42", refresh=refresh)
    print("BASE RUNS")
    data_df = pd.concat([df_1, df_2, df_3])
    calc_table_averages(data_df)

    ## NEW COMP-GEN RUNS
    data_df = cached_download(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada-new-comgen", refresh=refresh)
    print("NEW COMP-GEN RUNS")
    calc_table_averages(data_df)

    # TODO: Here, replace Base Runs with new comp-gen runs for final table

    ## SAMPLE EFFICIENCY RUNS
    data_df = cached_download(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada-se", refresh=refresh)
    print("SAMPLE EFFICIENCY RUNS")
    calc_table_averages(data_df)

    ## GRID-SIZES RUNS
    data_df = cached_download(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada-grid-size", refresh=refresh)
    print("GRID-SIZES RUNS")
    calc_table_averages(data_df)

    ## MODEL SIZES RUNS
    data_df = cached_download(entity="VisReas-ETHZ", project="VisReas-project-BEFOREARC-llada-model-size", refresh=refresh)
    print("MODEL SIZES RUNS")
    calc_table_averages(data_df)


if __name__ == "__main__":
    main()