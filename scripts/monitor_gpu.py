"""Record GPU telemetry while a specified training process exists."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
start = time.monotonic()
with open(args.output, "w") as output:
    writer = csv.writer(output)
    writer.writerow(["utc", "elapsed_seconds", "memory_used_mib", "gpu_util_percent", "temperature_c", "power_w"])
    while Path(f"/proc/{args.pid}").exists():
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if result.returncode == 0:
            writer.writerow([datetime.now(timezone.utc).isoformat(), round(time.monotonic()-start, 3),
                             *[item.strip() for item in result.stdout.strip().split(",")]])
            output.flush()
        time.sleep(5)
