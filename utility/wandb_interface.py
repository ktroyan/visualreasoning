import glob
import os
import wandb
import pandas as pd
import numpy as np

def get_table_results(api, project_path, table_keys):
    """
    Get table results of all the runs from a W&B project.

    NOTE:
    To get a value from different components of the run object, we can use:
    value = run.summary.get(key)    # get from summary (histogram icon)
    value = run.config.get(key)     # get from config (wheel icon)
    """

    # Get all runs from the project
    runs = api.runs(project_path)

    # Collect results
    results = []
    for run in runs:
        row = {}
        for key in table_keys:
            value = run.summary.get(key)

            if value is None:
                # Handle the 'VisReas-project' that was the first project where the keys were not normalized...
                if key == "study":
                    print(f"Key '{key}' not found in summary of project {project_path}. Trying to get the same key with another name from the config.")
                    value = run.config.get("paper_study")   # for ViT and Vanilla ViT

                elif key == "setting":
                    print(f"Key '{key}' not found in summary of project {project_path}. Trying to get the same key with another name from the config.")
                    value = run.config.get("paper_setting")    # for ViT and Vanilla ViT

                elif key == "experiment_name":
                    print(f"Key '{key}' not found in summary of project {project_path}. Trying to get the same key with another name from the config.")
                    value = run.config.get("paper_experiment")   # for ViT and Vanilla ViT
                        
                elif key == "model_name":
                    value = run.summary.get("paper_model_name")   # for all models

            # Normalization of study "sys-gen"
            if key == "study" and value == "sysgen":
                value = "sys-gen"
            
            row[key] = value
        
        results.append(row)

    # Save results as a pandas DataFrame
    df = pd.DataFrame(results)

    # Order the columns by "model_name", "study", "setting", "experiment_name" and then the rest of the keys
    table_keys = ["model_name", "study", "setting", "experiment_name"] + [key for key in table_keys if key not in ["model_name", "study", "setting", "experiment_name"]]
    df = df[table_keys]

    # Order the rows by the model name and then the 'study' and then the setting and then the experiment name
    df = df.sort_values(by=["model_name", "study", "setting", "experiment_name"])

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    return df

def get_all_projects_results(project_paths, table_keys):
    # Get table results for each project
    tables_results = {}
    for project_path in project_paths:
        print(f"Processing project: {project_path}")
        table_results = get_table_results(api, project_path, table_keys)
        print(f"Results for project {project_path}:")
        print(table_results)
        print("\n")

        tables_results[project_path] = table_results

    # Save results to CSV
    for project_path, table_results in tables_results.items():
        csv_path = f"{project_path.replace('/', '_')}.csv"
        table_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return tables_results

def compute_seed_averaged_results(csv_paths, output_csv="seed_averaged_results.csv"):
    """
    Computes seed-averaged results for multiple W&B project result CSVs.
    """
    # Load and combine all project result tables
    df_list = [pd.read_csv(path) for path in csv_paths]
    full_df = pd.concat(df_list, ignore_index=True)

    # Groupby and compute mean for numeric columns
    # group_keys = ["model_name", "study", "setting"] # to average across seeds for each same setting
    group_keys = ["model_name", "study", "setting", "experiment_name"]  # to average across seeds for each same experiment
    
    averaged_df = full_df.groupby(group_keys).mean(numeric_only=True).reset_index()

    # Reorder columns: grouped keys first, then metric columns
    metric_cols = [col for col in averaged_df.columns if col not in group_keys]
    averaged_df = averaged_df[group_keys + metric_cols]

    # Save to CSV
    averaged_df.to_csv(output_csv, index=False)
    print(f"Seed-averaged results saved to {output_csv}")

    return averaged_df

def generate_per_study_latex_tables(df, id_col="test_acc_grid_epoch", ood_col="gen_test_acc_grid_epoch"):
    """
    Given a DataFrame with columns:
    ["study", "setting", "experiment_name", "model_name", id_col, ood_col],
    produce two LaTeX tables for the studies (Compositionality, Sys-Gen),
    showing ID & OOD performance for the four models:
    [ResNet, Vanilla-ViT, Grid-ViT, LLaDA]
    
    Missing values become empty cells {}.
    """

    # Fixed order of models
    model_order = ["ResNet", "ViT-vanilla", "ViT", "LLaDA"]
    display_names = {'ResNet': "ResNet", "ViT-vanilla": "Vanilla-ViT", "ViT": "Grid-ViT", "LLaDA": "LLaDA"}

    # Map study key --> (caption, label)
    study_map = {
        "compositionality": ("Compositionality (Comp)", "compositionality"),
        "sys-gen":           ("Systematic Generalization (Gen)", "sys-gen"),
    }

    tables = {}

    for key, (caption, label_key) in study_map.items():
        sub = df[df["study"].str.lower().str.contains(key)]
        if sub.empty:
            continue

        # Organize into nested[setting][experiment][model] = (id, ood)
        nested = {}
        for _, row in sub.iterrows():
            es = row["setting"]
            ex = row["experiment_name"]
            m  = row["model_name"]
            idv  = row.get(id_col, np.nan)
            oodv = row.get(ood_col, np.nan)
            nested.setdefault(es, {}).setdefault(ex, {})[m] = (idv, oodv)

        # Build LaTeX template
        num_models = len(model_order)
        col_spec = "l l" + " S[table-format=2.1]" * (2 * num_models)

        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            fr"  \caption{{ID and OOD grid accuracy for {caption}}}",
            fr"  \label{{tab:{label_key}_grid_acc}}",
            "",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            # Header row
            "    & & " + " & ".join(
                fr"\multicolumn{{2}}{{c}}{{\textbf{{{display_names[m]}}}}}"
                for m in model_order
            ) + r" \\",
        ]

        # cmidrules: place each under its two numeric columns
        non_num_cols = 2
        for idx in range(len(model_order)):
            start = non_num_cols + idx*2 + 1
            end   = non_num_cols + idx*2 + 2
            lines.append(f"    \\cmidrule(lr){{{start}-{end}}}")

        # Column names (note the extra & before the first ID, otherwise it would be misaligned)
        lines.append(
            r"    \textbf{Setting} & \textbf{Experiment} & "
            + " & ".join(r"\textbf{ID} & \textbf{OOD}" for _ in model_order)
            + r" \\"
        )

        lines.append(r"    \midrule")

        # Table Body
        for es, exps in nested.items():
            n_ex = len(exps)
            for i, (ex, models) in enumerate(exps.items()):
                row = []
                # Setting only on first row
                if key == "compositionality":
                    setting_prefix = "C"
                elif key == "sys-gen":
                    setting_prefix = "G"
                row.append(
                    fr"\multirow{{{n_ex}}}{{*}}{{\textbf{{{es.replace('exp_setting_', setting_prefix)}}}}}"
                    if i == 0 else ""
                )
                row.append(ex.replace("experiment_", "experiment-"))  
                for m in model_order:
                    idv, oodv = models.get(m, (np.nan, np.nan))
                    print(f"Model: {m}, ID: {idv}, OOD: {oodv}")
                    idv = idv * 100
                    oodv = oodv * 100
                    id_str  = f"{idv:.1f}" if pd.notna(idv) else "{}"
                    ood_str = f"{oodv:.1f}" if pd.notna(oodv) else "{}"
                    row.extend([id_str, ood_str])

                lines.append("    " + " & ".join(row) + r" \\")
                if i < n_ex - 1:
                    right = 2 + 2 * num_models
                    lines.append(f"    \\cmidrule(lr){{2-{right}}}")
            # End experiments, separate settings
            lines.append(r"    \midrule")

        # Replace last \midrule with \bottomrule
        for idx in range(len(lines)-1, -1, -1):
            if lines[idx].strip() == r"\midrule":
                lines[idx] = r"    \bottomrule"
                break

        lines.extend([
            r"  \end{tabular}",
            r"\end{table}"
        ])

        tables[label_key] = "\n".join(lines)

    return tables

def generate_per_study_latex_tables_with_sem(df, sem_df, id_col="test_acc_grid_epoch", ood_col="gen_test_acc_grid_epoch"):
    """
    Generates LaTeX tables with mean ± SEM format for ID and OOD performance.
    Requires a second DataFrame sem_df with SEM values already computed.

    Given a DataFrame with columns:
    ["study", "setting", "experiment_name", "model_name", id_col, ood_col],
    produce two LaTeX tables for the studies (Compositionality, Sys-Gen),
    showing ID & OOD performance for the four models:
    [ResNet, Vanilla-ViT, Grid-ViT, LLaDA]
    
    Missing values become empty cells {}.
    """

    # Fixed order of models
    # We should uncomment the models we want to include in the table keeping in mind that
    # more than 3 models will not fit within the NeurIPS page width.
    # Therefore, for four models, we can create two tables and run the program twice.
    # model_order = ["ResNet", "ViT-vanilla", "ViT", "LLaDA"]
    # model_order = ["ResNet", "ViT-vanilla"]
    model_order = ["ViT", "LLaDA"]
    display_names = {'ResNet': "ResNet", "ViT-vanilla": "Vanilla-ViT", "ViT": "Grid-ViT", "LLaDA": "LLaDA"}

    study_map = {
        "compositionality": ("Compositionality (Comp)", "compositionality"),
        "sys-gen":           ("Systematic Generalization (Gen)", "sys-gen"),
    }

    # Rename and prepare SEM dataframe
    sem_df = sem_df.rename(columns={
        "test_acc_grid_epoch_mean": "id_mean",
        "test_acc_grid_epoch_sem":  "id_sem",
        "gen_test_acc_grid_epoch_mean": "ood_mean",
        "gen_test_acc_grid_epoch_sem":  "ood_sem"
    })
    # Use "::" as a separator to easily retrieve the values later
    sem_df["key"] = sem_df["model_name"] + "::" + sem_df["study"] + "::" + sem_df["setting"] + "::" + sem_df["experiment_name"]
    sem_lookup = sem_df.set_index("key")[["id_mean", "id_sem", "ood_mean", "ood_sem"]].to_dict("index")

    tables = {}

    for key, (caption, label_key) in study_map.items():
        sub = df[df["study"].str.lower().str.contains(key)]
        if sub.empty:
            continue

        nested = {}
        for _, row in sub.iterrows():
            es = row["setting"]
            ex = row["experiment_name"]
            m  = row["model_name"]
            nested.setdefault(es, {}).setdefault(ex, {})[m] = row

        num_models = len(model_order)
        # Each model contributes 4 columns: ID (mean ± sem), OOD (mean ± sem)
        # @{${}\mathbin{\pm}{}$} makes it appear as two wider columns so that the ± sign can fit
        col_spec = r"l l" + r" " + " ".join(
            [r"S[table-format=2.1]@{${}\mathbin{\pm}{}$}S[table-format=1.1] "
             r"S[table-format=2.1]@{${}\mathbin{\pm}{}$}S[table-format=1.1]"
             for _ in model_order]
        )

        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            fr"  \caption{{ID and OOD grid accuracy for {caption}}}",
            fr"  \label{{tab:{label_key}_grid_acc}}",
            "",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    & & " + " & ".join(
                fr"\multicolumn{{4}}{{c}}{{\textbf{{{display_names[m]}}}}}"
                for m in model_order
            ) + r" \\",
        ]

        for idx in range(len(model_order)):
            start = 3 + idx * 4
            end = start + 3
            lines.append(f"    \\cmidrule(lr){{{start}-{end}}}")

        lines.append(
            r"    \textbf{Setting} & \textbf{Experiment} & "
            + " & ".join(r"\multicolumn{2}{c}{\textbf{ID}} & \multicolumn{2}{c}{\textbf{OOD}}" for _ in model_order)
            + r" \\"
        )
        lines.append(r"    \midrule")

        for es, exps in nested.items():
            n_ex = len(exps)
            for i, (ex, models) in enumerate(exps.items()):
                row = []
                setting_prefix = "C" if key == "compositionality" else "G"
                row.append(fr"\multirow{{{n_ex}}}{{*}}{{\textbf{{{es.replace('exp_setting_', setting_prefix)}}}}}" if i == 0 else "")
                row.append(ex.replace("experiment_", "experiment-"))

                for m in model_order:
                    r = models.get(m)
                    if r is None:
                        row.extend(["{}", "{}", "{}", "{}"])
                        continue

                    sem_key = f"{m}::{r['study']}::{r['setting']}::{r['experiment_name']}"
                    sem = sem_lookup.get(sem_key)

                    if sem:
                        id_mean, id_sem = sem["id_mean"] * 100, sem["id_sem"] * 100
                        ood_mean, ood_sem = sem["ood_mean"] * 100, sem["ood_sem"] * 100
                        row.extend([
                            f"{id_mean:.1f}", f"{id_sem:.1f}",
                            f"{ood_mean:.1f}", f"{ood_sem:.1f}"
                        ])
                    else:
                        row.extend(["{}", "{}", "{}", "{}"])

                lines.append("    " + " & ".join(row) + r" \\")
                if i < n_ex - 1:
                    right = 2 + 4 * num_models
                    lines.append(f"    \\cmidrule(lr){{2-{right}}}")
            lines.append(r"    \midrule")

        for idx in range(len(lines)-1, -1, -1):
            if lines[idx].strip() == r"\midrule":
                lines[idx] = r"    \bottomrule"
                break

        lines.extend([
            r"  \end{tabular}",
            r"\end{table}"
        ])

        tables[label_key] = "\n".join(lines)

    return tables


if __name__ == "__main__":

    # Initialize API
    api = wandb.Api()

    # Set project details
    entity = "VisReas-ETHZ"     # username or team name
    projects = ["VisReas-project", "VisReas-project-seed42", "VisReas-project-seed2025"]    # names of the projects
    project_paths = [os.path.join(entity, project) for project in projects]     # paths to the projects

    # Table keys to extract
    table_keys = [
        "study",
        "setting",
        "experiment_name",
        "model_name",
        "test_acc_grid_epoch",
        "gen_test_acc_grid_epoch",
        "test_acc_epoch",
        "gen_test_acc_epoch"
    ]

    ## Create CSV files with the results for all the relevant projects
    tables_results = get_all_projects_results(project_paths, table_keys)

    ## Create seed-averaged results table (i.e., average across the projects that only differ in their seed)
    # Get all CSV files created above
    csv_paths = glob.glob("VisReas-ETHZ_VisReas-project*.csv")

    # Compute seed-averaged results
    final_avg_df = compute_seed_averaged_results(csv_paths)

    ## Generate LaTeX table from the averaged results
    # Load the averaged results
    final_avg_df = pd.read_csv("seed_averaged_results.csv")
    latex_tables = generate_per_study_latex_tables(final_avg_df)

    # Write to file
    for study_key, latex_table in latex_tables.items():
        with open(f"all_{study_key}_results.tex", "w") as f:
            f.write(latex_table)


    ## Compute error bars and generate LaTeX tables
    # Load all seed CSVs containing the results
    paths = [
        "VisReas-ETHZ_VisReas-project.csv",
        "VisReas-ETHZ_VisReas-project-seed42.csv",
        "VisReas-ETHZ_VisReas-project-seed2025.csv",
    ]

    dfs = [pd.read_csv(p) for p in paths]
    all_data = pd.concat(dfs)

    # Group by experiment
    group_cols = ["model_name", "study", "setting", "experiment_name"]

    # Metrics of interest for SEM
    metrics = ["test_acc_grid_epoch", "gen_test_acc_grid_epoch"]

    # Compute mean and SEM
    summary = (
        all_data.groupby(group_cols)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Calculate SEM
    for metric in metrics:
        summary[(metric, "sem")] = summary[(metric, "std")] / np.sqrt(len(paths))   # divide by the number of seeds over which we average

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    # Save to CSV
    summary.to_csv("results_with_sem.csv", index=False)

    ## Generate tables with SEM for each study 
    # Load data
    df_main = pd.read_csv("seed_averaged_results.csv")
    df_sem = pd.read_csv("results_with_sem.csv")
    
    # Generate LaTeX tables with SEM for each study
    latex_tables = generate_per_study_latex_tables_with_sem(df_main, df_sem)

    # Write to file
    for study_key, latex_table in latex_tables.items():
        with open(f"all_{study_key}_results_with_SEM.tex", "w") as f:
            f.write(latex_table)