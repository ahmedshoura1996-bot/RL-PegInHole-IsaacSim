import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "diagnostics/ppo_10k/ppo_10k_all_metrics.csv"
OUTPUT_DIR = "diagnostics/ppo_10k"

WINDOW_SIZE = 500
TOTAL_ITERATIONS = 10000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print(" PPO 10K - 500 ITERATION WINDOW ANALYSIS")
print("=" * 70)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print(f"\nInput file: {INPUT_CSV}")
print(f"Rows: {len(df)}")
print(f"First iteration: {df['iteration'].min()}")
print(f"Last iteration:  {df['iteration'].max()}")


# ============================================================
# VALIDATE ITERATIONS
# ============================================================

expected = set(range(TOTAL_ITERATIONS))
actual = set(df["iteration"].astype(int))

missing = sorted(expected - actual)
extra = sorted(actual - expected)

if missing:
    print("\nWARNING: Missing iterations:")
    print(missing[:50])

if extra:
    print("\nWARNING: Unexpected iterations:")
    print(extra[:50])

if not missing and not extra:
    print("\nIteration coverage: 0 -> 9999 COMPLETE")


# ============================================================
# METRICS
# ============================================================

metrics = [
    "Train/mean_reward",
    "Train/mean_episode_length",

    "Episode_Reward/action_rate",
    "Episode_Reward/joint_vel",
    "Episode_Reward/peg_hole_xy_alignment",
    "Episode_Reward/peg_insertion_progress",

    "Metrics/object_pose/position_error",
    "Metrics/object_pose/orientation_error",

    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",

    "Policy/mean_noise_std",

    "Perf/total_fps",
    "Perf/collection time",
    "Perf/learning_time",
]


available_metrics = [
    m for m in metrics
    if m in df.columns
]

missing_metrics = [
    m for m in metrics
    if m not in df.columns
]

if missing_metrics:
    print("\nMissing metrics:")
    for m in missing_metrics:
        print("  -", m)


# ============================================================
# CREATE 500-ITERATION WINDOWS
# ============================================================

window_results = []

for start in range(0, TOTAL_ITERATIONS, WINDOW_SIZE):

    end = start + WINDOW_SIZE - 1

    window = df[
        (df["iteration"] >= start)
        & (df["iteration"] <= end)
    ]

    if len(window) == 0:
        continue

    row = {
        "window_start": start,
        "window_end": end,
        "iterations_recorded": len(window),
    }

    for metric in available_metrics:
        row[metric] = window[metric].mean()

    window_results.append(row)


windows_df = pd.DataFrame(window_results)


# ============================================================
# SAVE WINDOW AVERAGES
# ============================================================

window_csv = os.path.join(
    OUTPUT_DIR,
    "ppo_10k_500_iteration_windows.csv"
)

windows_df.to_csv(
    window_csv,
    index=False,
)

print("\n" + "=" * 70)
print("500-ITERATION WINDOW AVERAGES")
print("=" * 70)

print(
    f"\nNumber of windows: {len(windows_df)}"
)

print(
    f"Window size: {WINDOW_SIZE} iterations"
)

print(
    f"\nWrote:\n{window_csv}"
)


# ============================================================
# PRINT MAIN PERFORMANCE TABLE
# ============================================================

display_metrics = [
    "Train/mean_reward",
    "Episode_Reward/peg_hole_xy_alignment",
    "Episode_Reward/peg_insertion_progress",
    "Metrics/object_pose/position_error",
    "Metrics/object_pose/orientation_error",
    "Policy/mean_noise_std",
]

display_metrics = [
    m for m in display_metrics
    if m in windows_df.columns
]

print("\n" + "=" * 70)
print("MAIN PERFORMANCE BY 500-ITERATION WINDOW")
print("=" * 70)

for _, row in windows_df.iterrows():

    print(
        f"\n[{int(row['window_start']):4d} - "
        f"{int(row['window_end']):4d}]"
    )

    for metric in display_metrics:

        print(
            f"  {metric:50s} "
            f"{row[metric]: .6f}"
        )


# ============================================================
# FIRST 500 vs LAST 500
# ============================================================

first_window = windows_df.iloc[0]
last_window = windows_df.iloc[-1]

comparison_rows = []

for metric in available_metrics:

    first_value = first_window[metric]
    last_value = last_window[metric]

    if pd.isna(first_value) or pd.isna(last_value):
        continue

    absolute_change = last_value - first_value

    if abs(first_value) > 1e-12:
        percent_change = (
            absolute_change
            / abs(first_value)
            * 100.0
        )
    else:
        percent_change = np.nan

    comparison_rows.append({
        "metric": metric,
        "first_500_mean": first_value,
        "last_500_mean": last_value,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
    })


comparison_df = pd.DataFrame(comparison_rows)


# ============================================================
# SAVE FIRST/LAST COMPARISON
# ============================================================

comparison_csv = os.path.join(
    OUTPUT_DIR,
    "ppo_10k_first500_vs_last500.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False,
)

print("\n" + "=" * 70)
print("FIRST 500 vs LAST 500")
print("=" * 70)

for _, row in comparison_df.iterrows():

    pct = row["percent_change"]

    if pd.isna(pct):
        pct_text = "N/A"
    else:
        pct_text = f"{pct:+.2f}%"

    print(
        f"{row['metric']:55s} "
        f"{row['first_500_mean']: .6f} -> "
        f"{row['last_500_mean']: .6f} "
        f"({pct_text})"
    )


# ============================================================
# SELECT IMPORTANT METRICS
# ============================================================

important = [
    "Train/mean_reward",
    "Episode_Reward/peg_hole_xy_alignment",
    "Episode_Reward/peg_insertion_progress",
    "Metrics/object_pose/position_error",
    "Metrics/object_pose/orientation_error",
    "Policy/mean_noise_std",
]


important = [
    m for m in important
    if m in comparison_df["metric"].values
]


important_df = comparison_df[
    comparison_df["metric"].isin(important)
].copy()


# ============================================================
# SAVE IMPORTANT SUMMARY
# ============================================================

summary_csv = os.path.join(
    OUTPUT_DIR,
    "ppo_10k_performance_summary.csv"
)

important_df.to_csv(
    summary_csv,
    index=False,
)


# ============================================================
# PRINT COMPACT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(" PPO 10K SUMMARY")
print("=" * 70)

print(
    f"\nFirst window : iterations 0-499"
)

print(
    f"Last window  : iterations 9500-9999"
)

for _, row in important_df.iterrows():

    pct = row["percent_change"]

    if pd.isna(pct):
        pct_text = "N/A"
    else:
        pct_text = f"{pct:+.2f}%"

    print(
        f"\n{row['metric']}"
    )

    print(
        f"  First 500 : {row['first_500_mean']:.6f}"
    )

    print(
        f"  Last 500  : {row['last_500_mean']:.6f}"
    )

    print(
        f"  Change    : {row['absolute_change']:+.6f}"
    )

    print(
        f"  Change %  : {pct_text}"
    )


# ============================================================
# FIND BEST WINDOW FOR REWARD
# ============================================================

if "Train/mean_reward" in windows_df.columns:

    best_reward_idx = windows_df[
        "Train/mean_reward"
    ].idxmax()

    best_reward = windows_df.loc[
        best_reward_idx
    ]

    print("\n" + "=" * 70)
    print("BEST 500-ITERATION WINDOW BY MEAN REWARD")
    print("=" * 70)

    print(
        f"\nWindow: "
        f"{int(best_reward['window_start'])} - "
        f"{int(best_reward['window_end'])}"
    )

    print(
        f"Mean reward: "
        f"{best_reward['Train/mean_reward']:.6f}"
    )


# ============================================================
# BEST INSERTION PROGRESS WINDOW
# ============================================================

if "Episode_Reward/peg_insertion_progress" in windows_df.columns:

    best_idx = windows_df[
        "Episode_Reward/peg_insertion_progress"
    ].idxmax()

    best = windows_df.loc[best_idx]

    print("\n" + "=" * 70)
    print("BEST INSERTION-PROGRESS WINDOW")
    print("=" * 70)

    print(
        f"\nWindow: "
        f"{int(best['window_start'])} - "
        f"{int(best['window_end'])}"
    )

    print(
        f"Mean insertion progress: "
        f"{best['Episode_Reward/peg_insertion_progress']:.6f}"
    )


# ============================================================
# BEST XY ALIGNMENT WINDOW
# ============================================================

if "Episode_Reward/peg_hole_xy_alignment" in windows_df.columns:

    best_idx = windows_df[
        "Episode_Reward/peg_hole_xy_alignment"
    ].idxmax()

    best = windows_df.loc[best_idx]

    print("\n" + "=" * 70)
    print("BEST XY-ALIGNMENT WINDOW")
    print("=" * 70)

    print(
        f"\nWindow: "
        f"{int(best['window_start'])} - "
        f"{int(best['window_end'])}"
    )

    print(
        f"Mean XY alignment reward: "
        f"{best['Episode_Reward/peg_hole_xy_alignment']:.6f}"
    )


# ============================================================
# FINAL FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\n1. {window_csv}"
)

print(
    f"2. {comparison_csv}"
)

print(
    f"3. {summary_csv}"
)

print("\nAnalysis complete.")
print("=" * 70)
