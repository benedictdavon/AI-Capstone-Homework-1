from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run([python, "scripts/generate_data.py"])
    run([python, "scripts/run_supervised.py"])
    run([python, "scripts/run_clustering.py"])


if __name__ == "__main__":
    main()
