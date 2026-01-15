from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

#!/usr/bin/env python3
"""
metrics.py

Loop through all CSV files in the py_dev_mini_1_full_loop_v1 folder and print the overall mean score for each.
"""

evaluations_dir = "../outputs/evaluations/py_dev_mini_1_full_loop_v3_lr5e4"

def get_csv_paths() -> list[Path]:
    """
    Get all CSV file paths in the evaluations directory.
    """
    base_path = Path(__file__).parent / evaluations_dir
    filenames = base_path.glob("*/combined_evaluations.csv")

    ## getting filenames in order
    epoch_map = {}
    for file in os.listdir(evaluations_dir):
        epoch = file.split("py_dev_mini_1_full_loop_v1/")[-1]
        print("epoch", epoch)
        # initialize a map to collect CSV paths keyed by epoch (first iteration)

        

        epoch_name = file  # directory name under evaluations_dir should be the epoch
        candidate = Path(evaluations_dir) / epoch_name / "combined_evaluations.csv"

        if candidate.exists():
            try:
                key = int(epoch_name)
            except ValueError:
                key = epoch_name
            epoch_map[key] = candidate
            print(f"registered epoch {key} -> {candidate}")

        # rebuild `filenames` as a sorted list of Paths keyed by epoch
        filenames = [epoch_map[k] for k in sorted(epoch_map, key=lambda k: (k if isinstance(k, int) else str(k)))]
    return filenames

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

    print(csv_paths)

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


        # Convert epoch labels to integers for plotting logic
    try:
        epoch_ints = [int(e) for e in epoch_labels]
    except ValueError:
        print("Epoch labels are not numeric; cannot distinguish base vs LoRA.")
        epoch_ints = []


    if epoch_means and epoch_ints:
        # Split base model (epoch 0) vs LoRA (epochs 1+)
        base_epochs = [e for e, i in zip(epoch_labels, epoch_ints) if i == 0]
        base_means = [m for m, i in zip(epoch_means, epoch_ints) if i == 0]

        lora_epochs = [e for e, i in zip(epoch_labels, epoch_ints) if i >= 1]
        lora_means = [m for m, i in zip(epoch_means, epoch_ints) if i >= 1]

        plt.figure(figsize=(8, 5))

        # Optional dotted transition from base -> first LoRA epoch
        transition_x = []
        transition_y = []

        if base_epochs and lora_epochs:
            # Assumes epoch 1 is the first LoRA epoch
            transition_x = [base_epochs[-1], lora_epochs[0]]
            transition_y = [base_means[-1], lora_means[0]]


        # Base model (epoch 0)
        if base_epochs:
            plt.plot(
                base_epochs,
                base_means,
                marker='o',
                linestyle='',
                color='tab:gray',
                label='Base model (epoch 0)'
            )

        # Dotted transition indicating LoRA attachment / training step
        if transition_x:
            plt.plot(
                transition_x,
                transition_y,
                linestyle=':',
                color='tab:blue',
                linewidth=1.5,
                label='Base → LoRA transition'
            )


        # PEFT / LoRA model (epochs 1+)
        if lora_epochs:
            plt.plot(
                lora_epochs,
                lora_means,
                marker='o',
                linestyle='-',
                color='tab:blue',
                label='PEFT model w/ LoRA (epochs 1+)'
            )

        plt.suptitle('Mean Epistemic Humility Rating per Epoch')
        plt.title('Scores range from 0–5', fontsize=10)

        plt.xlabel('Epoch')
        plt.ylabel('Mean Epistemic Humility Rating')
        plt.legend()
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



