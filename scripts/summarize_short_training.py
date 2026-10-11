"""Summarize actual short-run trainer state and sampled GPU telemetry."""
import csv
import ast
import json
from pathlib import Path
import re
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
state = json.loads((root / "checkpoints/lift_pot_short/checkpoint-100/trainer_state.json").read_text())
assert state["global_step"] == 100
history = [row for row in state["log_history"] if "loss" in row]
log = (root / "logs/lift-pot-short-training.log").read_text()
progress = {}
for step, minutes, seconds in re.findall(r"(\d+)/100 \[(\d+):(\d+)<", log):
    progress[int(step)] = int(minutes)*60+int(seconds)
gaps = [progress[step]-progress[step-1] for step in range(3,100)
        if step != 51 and step in progress and step-1 in progress]
gpu = list(csv.DictReader((root / "logs/lift-pot-short-gpu.csv").open()))
run = json.loads((root / "configs/lift_pot_short_run.json").read_text())
first_checkpoint_overhead = progress.get(51, 0)-progress.get(50, 0)
trainer_summary = state["log_history"][-1] if "train_runtime" in state["log_history"][-1] else {}
for line in log.replace("\r", "\n").splitlines():
    if line.startswith("{'train_runtime':"):
        trainer_summary = ast.literal_eval(line)
result = dict(steps=100, first_logged_loss=history[0]["loss"], last_logged_loss=history[-1]["loss"],
    peak_sampled_gpu_memory_mib=max(float(row["memory_used_mib"]) for row in gpu),
    median_progress_step_seconds=statistics.median(gaps),
    mean_progress_step_seconds=statistics.mean(gaps),
    checkpoint_50_pause_seconds=first_checkpoint_overhead,
    progress_at_step_100_seconds=progress.get(100), trainer_summary=trainer_summary,
    run=run, note="GPU samples every 5 seconds; progress times rounded to seconds; step 51 and final checkpoint pause excluded from mean step time. Train loss is not validation loss or task success.")
with (root / "results/short_training_loss.csv").open("w") as f:
    writer=csv.DictWriter(f, fieldnames=["step","loss","grad_norm","learning_rate"])
    writer.writeheader()
    for row in history: writer.writerow({key:row.get(key) for key in writer.fieldnames})
(root / "results/short_training_summary.json").write_text(json.dumps(result, indent=2))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot([r["step"] for r in history], [r["loss"] for r in history], marker="o")
axes[0].set(xlabel="Optimizer step", ylabel="Training loss", title="Lift Pot: 100-step GR00T N1.7 test")
axes[1].plot([float(r["elapsed_seconds"])/60 for r in gpu], [float(r["memory_used_mib"])/1024 for r in gpu])
axes[1].set(xlabel="Monitoring time (minutes)",ylabel="GPU memory (GiB)",title="RTX A6000 sampled memory")
fig.tight_layout(); fig.savefig(root / "results/training_curves/lift_pot_short.png", dpi=160)
print(json.dumps(result, indent=2))
