"""Run the continuous checkpoint experiment in a deterministic child process."""

import argparse
from pathlib import Path
import subprocess
import sys

from ..checkpoint_runtime import REPOSITORY_ROOT, deterministic_subprocess_environment


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=directory / "config.yaml",
        help="Full configuration file for the continuous training process.",
    )
    args = parser.parse_args()

    environment = deterministic_subprocess_environment()
    environment.setdefault("AUTOREC_ROOT", str(REPOSITORY_ROOT))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "example.checkpoint.train_session",
            "--config",
            str(args.config.resolve()),
        ],
        env=environment,
        cwd=REPOSITORY_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
