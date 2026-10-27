"""Run the continuous checkpoint experiment in a deterministic child process."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONHASHSEED="0",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    subprocess.run(
        [
            sys.executable,
            str(directory / "main_factory.py"),
            "--config",
            str(directory / "continuous_config.yaml"),
        ],
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
