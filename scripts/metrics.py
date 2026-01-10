from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

#!/usr/bin/env python3
"""
metrics.py

Loop through all CSV files in the py_dev_mini_1_full_loop_v1 folder and print the overall mean score for each.
"""

evaluations_dir = "../outputs/evaluations/py_dev_mini_1_full_loop_v1"

def get_csv_paths() -> list[Path]:
    """
    Get all CSV file paths in the evaluations directory.
    """
    base_path = Path(__file__).parent / evaluations_dir
    return sorted(base_path.glob("*/combined_evaluations.csv"))

def load_csv(path: Path | str, **pd_kwargs) -> pd.DataFrame:
    """
    Load the CSV into a pandas DataFrame.
    pd_kwargs are forwarded to pandas.read_csv.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False, **pd_kwargs)

def main() -> None:
    csv_paths = get_csv_paths()
    if not csv_paths:
        print("No CSV files found in the evaluations directory.")
        return

    epoch_means = []
    epoch_labels = []

    for csv_path in csv_paths:
        try:
            df = load_csv(csv_path)
        except FileNotFoundError as e:
            print(e)
            continue
        except Exception as e:
            print(f"Failed to load CSV at {csv_path}: {e}")
            continue

        score_col = "score"
        if score_col not in df.columns:
            print(f"No 'score' column found in {csv_path}.")
            continue

        overall_mean = df[score_col].mean()
        # Extract epoch number from path (assumes folder name is the epoch)
        epoch = csv_path.parent.name
        epoch_means.append(overall_mean)
        epoch_labels.append(epoch)
        print(f"Epoch {epoch}: Overall mean {score_col}: {overall_mean:.6f}")

    # Plot results if any
    if epoch_means:
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_labels, epoch_means, marker='o')
        plt.title('Mean Epistemic Humility Rating per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Epistemic Humility Rating')
        plt.grid(True)
        plt.tight_layout()
        # Save plot to a file because plt.show() won't work in a terminal-only environment.
        plots_dir = Path(__file__).parent / evaluations_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = plots_dir / f"mean_epistemic_humility_rating_per_epoch_{timestamp}.png"
        svg_path = plots_dir / f"mean_epistemic_humility_rating_per_epoch_{timestamp}.svg"
        try:
            plt.savefig(plot_path, dpi=150)
            print(f"Saved plot to {plot_path}")
            plt.savefig(svg_path, format="svg")
            print(f"Saved plot to {svg_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

        # Also save the numeric results to a CSV for later inspection / processing.
        try:
            out_df = pd.DataFrame({"epoch": epoch_labels, "mean_epistemic_humility_rating": epoch_means})
            csv_out = plots_dir / f"mean_epistemic_humility_rating_per_epoch_{timestamp}.csv"
            out_df.to_csv(csv_out, index=False)
            print(f"Saved mean epistemic humility ratings to {csv_out}")
        except Exception as e:
            print(f"Failed to save mean epistemic humility ratings CSV: {e}")

if __name__ == "__main__":
    main()



