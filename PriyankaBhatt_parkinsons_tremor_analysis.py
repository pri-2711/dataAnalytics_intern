from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def load_and_clean():
    df = pd.read_csv(DATA_PATH, header=1)
    df = df.dropna(how="all").copy()

    # Keep only the actual tremor and relevant clinical columns.
    useful_cols = [
        "Participant code",
        "Age (years)",
        "Gender",
        "Duration of disease from first symptoms (years)",
        "20. Tremor at Rest - head",
        "20. Tremor at Rest - RUE",
        "20. Tremor at Rest - LUE",
        "20. Tremor at Rest - RLE",
        "20. Tremor at Rest - LLE",
        "21. Action or Postural Tremor - RUE",
        "21. Action or Postural Tremor - LUE",
        "UPDRS III total (-)",
    ]
    df = df[[col for col in useful_cols if col in df.columns]].copy()

    for col in df.columns:
        if col == "Participant code":
            df[col] = df[col].astype(str).str.strip()
        elif col == "Gender":
            df[col] = df[col].astype(str).str.strip().str.upper()
        else:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace({"-": np.nan, "": np.nan}), errors="coerce")

    return df


def engineer_features(df):
    df = df.copy()
    df["total_rest_tremor"] = df[[
        "20. Tremor at Rest - head",
        "20. Tremor at Rest - RUE",
        "20. Tremor at Rest - LUE",
        "20. Tremor at Rest - RLE",
        "20. Tremor at Rest - LLE",
    ]].sum(axis=1, skipna=True)

    df["total_action_tremor"] = df[[
        "21. Action or Postural Tremor - RUE",
        "21. Action or Postural Tremor - LUE",
    ]].sum(axis=1, skipna=True)

    df["upper_limb_rest_tremor"] = df["20. Tremor at Rest - RUE"] + df["20. Tremor at Rest - LUE"]
    df["lower_limb_rest_tremor"] = df["20. Tremor at Rest - RLE"] + df["20. Tremor at Rest - LLE"]
    df["right_left_rest_asymmetry"] = (df["20. Tremor at Rest - RUE"] - df["20. Tremor at Rest - LUE"]).abs()
    return df


def print_dataset_summary(df):
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:\n", df.isna().sum().sort_values(ascending=False))


def summarize_main_features(df):
    cols = [
        "total_rest_tremor",
        "total_action_tremor",
        "upper_limb_rest_tremor",
        "lower_limb_rest_tremor",
        "right_left_rest_asymmetry",
    ]
    print("\nMain feature summary:")
    for col in cols:
        s = df[col].dropna()
        print(f"{col}: mean={s.mean():.2f}, median={s.median():.2f}, min={s.min():.2f}, max={s.max():.2f}")


def plot_tremor_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    melt_df = df[[
        "20. Tremor at Rest - head",
        "20. Tremor at Rest - RUE",
        "20. Tremor at Rest - LUE",
        "20. Tremor at Rest - RLE",
        "20. Tremor at Rest - LLE",
        "21. Action or Postural Tremor - RUE",
        "21. Action or Postural Tremor - LUE",
    ]].melt()
    sns.boxplot(data=melt_df, x="variable", y="value", ax=ax)
    ax.set_title("Tremor score distribution")
    ax.set_xlabel("Tremor variable")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tremor_distribution.png", dpi=200)
    plt.close(fig)


def plot_tremor_by_region(df):
    region_means = pd.DataFrame({
        "Rest tremor": [
            df["20. Tremor at Rest - head"].mean(),
            df["20. Tremor at Rest - RUE"].mean(),
            df["20. Tremor at Rest - LUE"].mean(),
            df["20. Tremor at Rest - RLE"].mean(),
            df["20. Tremor at Rest - LLE"].mean(),
        ],
        "Action tremor": [
            np.nan,
            df["21. Action or Postural Tremor - RUE"].mean(),
            df["21. Action or Postural Tremor - LUE"].mean(),
            np.nan,
            np.nan,
        ],
    }, index=["Head", "Right UE", "Left UE", "Right LE", "Left LE"])

    fig, ax = plt.subplots(figsize=(8, 5))
    region_means.plot(kind="bar", ax=ax)
    ax.set_title("Mean tremor by body region")
    ax.set_xlabel("Body region")
    ax.set_ylabel("Mean score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tremor_by_region.png", dpi=200)
    plt.close(fig)


def plot_rest_vs_action(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [df["20. Tremor at Rest - RUE"].mean(), df["20. Tremor at Rest - LUE"].mean(), df["20. Tremor at Rest - head"].mean()]
    y = [df["21. Action or Postural Tremor - RUE"].mean(), df["21. Action or Postural Tremor - LUE"].mean(), np.nan]
    ax.bar(["Right UE", "Left UE", "Head"], x, label="Rest tremor")
    ax.bar(["Right UE", "Left UE", "Head"], y, label="Action tremor", alpha=0.7)
    ax.set_title("Rest tremor vs action/postural tremor")
    ax.set_ylabel("Mean score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rest_vs_action.png", dpi=200)
    plt.close(fig)


def plot_tremor_vs_updrs(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df[["total_rest_tremor", "UPDRS III total (-)"]].dropna(), x="total_rest_tremor", y="UPDRS III total (-)", ax=ax)
    ax.set_title("Tremor burden vs UPDRS III total")
    ax.set_xlabel("Total rest tremor")
    ax.set_ylabel("UPDRS III total")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tremor_vs_updrs.png", dpi=200)
    plt.close(fig)


def correlation_analysis(df):
    corr_rest = df[["total_rest_tremor", "UPDRS III total (-)"]].dropna().corr(method="spearman").iloc[0, 1]
    corr_action = df[["total_action_tremor", "UPDRS III total (-)"]].dropna().corr(method="spearman").iloc[0, 1]
    print("\nSpearman correlations:")
    print(f"Total rest tremor vs UPDRS III total: {corr_rest:.3f}")
    print(f"Total action tremor vs UPDRS III total: {corr_action:.3f}")
    return corr_rest, corr_action


def key_findings(df, corr_rest, corr_action):
    findings = []
    mean_rest = df["total_rest_tremor"].mean()
    mean_action = df["total_action_tremor"].mean()
    findings.append(f"Average rest tremor was {mean_rest:.2f}, which was higher than average action tremor {mean_action:.2f}.")

    region_means = {
        "Head": df["20. Tremor at Rest - head"].mean(),
        "Right UE": df["20. Tremor at Rest - RUE"].mean(),
        "Left UE": df["20. Tremor at Rest - LUE"].mean(),
        "Right LE": df["20. Tremor at Rest - RLE"].mean(),
        "Left LE": df["20. Tremor at Rest - LLE"].mean(),
    }
    strongest_region = max(region_means, key=region_means.get)
    findings.append(f"The highest average tremor score was observed in the {strongest_region} region.")

    if corr_rest > corr_action:
        findings.append(f"Total rest tremor showed a stronger positive relationship with UPDRS III total (rho={corr_rest:.3f}) than action tremor (rho={corr_action:.3f}).")
    else:
        findings.append(f"Total action tremor showed a slightly stronger positive relationship with UPDRS III total (rho={corr_action:.3f}) than rest tremor (rho={corr_rest:.3f}).")

    asymmetry_mean = df["right_left_rest_asymmetry"].dropna().mean()
    findings.append(f"Average right-left rest tremor asymmetry was {asymmetry_mean:.2f}, suggesting side-to-side differences are relevant in the cohort.")
    return findings


def main():
    df = load_and_clean()
    df = engineer_features(df)

    print("=== Dataset overview ===")
    print_dataset_summary(df)

    print("\n=== Descriptive statistics ===")
    summarize_main_features(df)

    plot_tremor_distribution(df)
    plot_tremor_by_region(df)
    plot_rest_vs_action(df)
    plot_tremor_vs_updrs(df)

    corr_rest, corr_action = correlation_analysis(df)
    findings = key_findings(df, corr_rest, corr_action)

    print("\n=== Key findings ===")
    for i, item in enumerate(findings, start=1):
        print(f"{i}. {item}")

    print("\nConclusion: This project is an academic data analytics study of tremor patterns in a Parkinson's cohort and is not a medical diagnostic system.")


if __name__ == "__main__":
    main()
