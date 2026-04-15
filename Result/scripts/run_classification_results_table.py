"""Run multiple classification baselines and save a results CSV."""

from pathlib import Path
import pandas as pd

from tdc_single_scaffold_cv_baseline import run_tdc_single_scaffold_cv_baseline


def run_classification_results_table(
    datasets: str = "BBB_Martins,Pgp_Broccatelli,CYP2D6_Substrate_CarbonMangels,hERG_Karim,PAMPA_NCATS,SARSCoV2_3CLPro_Diamond,SARSCoV2_Vitro_Touret",
    models: str = "logistic_regression,random_forest,gradient_boosting",
    dataset_source: str = "prepared",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    save_dir: Path = Path("repro_data/tdc_single"),
    output_csv: Path = Path("repro_results/classification_results_long.csv"),
    output_table_csv: Path = Path("repro_results/classification_results_table.csv"),
    output_split_csv: Path = Path("repro_results/classification_results_splits.csv"),
    feature_type: str = "rdkit_descriptors",
    seed: int = 42,
    do_grid_search: bool = False,
    score_scale: float = 100.0,
) -> None:
    dataset_list = [x.strip() for x in datasets.split(",") if x.strip()]
    model_list = [x.strip() for x in models.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for dataset in dataset_list:
        for model in model_list:
            print(f"Running dataset={dataset}, model={model}")
            out = run_tdc_single_scaffold_cv_baseline(
                dataset=dataset,
                dataset_source=dataset_source,
                prepared_dir=prepared_dir,
                prepared_extra_dir=prepared_extra_dir,
                prepared_src_dir=prepared_src_dir,
                save_dir=save_dir,
                feature_type=feature_type,
                classifier=model,
                seed=seed,
                do_grid_search=do_grid_search,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": out["metric_name"],
                    "mean": out["mean"],
                    "std": out["std"],
                    "mean_std": f"{out['mean']:.4f} +/- {out['std']:.4f}",
                    "mean_pct": out["mean"] * score_scale,
                    "std_pct": out["std"] * score_scale,
                    "mean_std_pct": f"{out['mean'] * score_scale:.1f} +/- {out['std'] * score_scale:.1f}",
                    "seed": seed,
                    "feature_type": feature_type,
                    "dataset_source": dataset_source,
                    "n_molecules": out.get("n_molecules"),
                    "n_features": out.get("n_features"),
                    "n_splits": out.get("n_splits"),
                }
            )
            for split_i, score in enumerate(out.get("fold_scores", []), start=1):
                split_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": out["metric_name"],
                        "split": split_i,
                        "score": score,
                        "score_pct": score * score_scale if pd.notna(score) else score,
                        "seed": seed,
                        "feature_type": feature_type,
                        "dataset_source": dataset_source,
                    }
                )

    df = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved long-format results to {output_csv}")

    if split_rows:
        split_df = pd.DataFrame(split_rows).sort_values(["dataset", "model", "split"]).reset_index(drop=True)
        output_split_csv.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_split_csv, index=False)
        print(f"Saved split-level results to {output_split_csv}")

    table = (
        df.pivot(index="model", columns="dataset", values="mean_std_pct")
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    avg_df = (
        df.groupby("model", as_index=False)["mean_pct"]
        .mean()
        .rename(columns={"model": "Model", "mean_pct": "Average"})
    )
    avg_df["Average"] = avg_df["Average"].map(lambda x: f"{x:.1f}")
    table = table.merge(avg_df, on="Model", how="left")
    table = table.sort_values("Average", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    table.to_csv(output_table_csv, index=False)
    print(f"Saved table-format results to {output_table_csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    from tap import tapify

    tapify(run_classification_results_table)
