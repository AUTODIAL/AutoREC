"""Worker process for the AutoREC Streamlit GUI."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True)

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_status(config: dict[str, Any], state: str, message: str, **extra: Any) -> None:
    status_path = Path(config["status_path"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "state": state,
                "message": message,
                "results_csv": config["results_csv"],
                **extra,
            },
            indent=2,
        )
    )


def _artifact_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("artifact_dir") or config["output_dir"])


def _artifact_paths(config: dict[str, Any], eis_index: int) -> dict[str, Path]:
    artifact_dir = _artifact_dir(config)
    return {
        "circuit_plot": artifact_dir / "circuit_plots" / f"circuit_EIS_i_{eis_index}.png",
        "fit_plot": artifact_dir / "fit_plots" / f"fit_EIS_i_{eis_index + 1}.png",
        "circuit_gif": artifact_dir
        / "generated_circuits"
        / f"circuit_evolution_EIS_{eis_index}.gif",
        "final_circuit_frame": (
            artifact_dir
            / "generated_circuits"
            / "final_circuits"
            / f"final_circuit_EIS_{eis_index}.png"
        ),
    }


def _write_circuit_plot(config: dict[str, Any], result: dict[str, Any]) -> Path | None:
    if not bool(config.get("circuit_plots", True)):
        return None

    circuit = result.get("best_circuit")
    eis_index = result.get("EIS_i")
    if not circuit or eis_index is None:
        return None

    paths = _artifact_paths(config, int(eis_index))
    try:
        from autorec.utils import draw_circuit_png

        draw_circuit_png(str(circuit), paths["circuit_plot"], dpi=150)
        return paths["circuit_plot"]
    except Exception as exc:
        print(f"Could not draw best circuit {circuit!r}: {exc}", flush=True)
        return None


def _flatten_result(
    result: dict[str, Any],
    config: dict[str, Any],
    circuit_plot: Path | None,
) -> dict[str, Any]:
    metrics = result.get("best_metrics") or {}
    eis_index = result.get("EIS_i")
    paths = _artifact_paths(config, int(eis_index)) if eis_index is not None else {}
    circuit_plot_path = circuit_plot or paths.get("circuit_plot")
    return {
        "EIS_i": result.get("EIS_i"),
        "ground_truth_circuit": result.get("ground_truth_circuit"),
        "found_solution": result.get("found_solution"),
        "best_reward": result.get("best_reward"),
        "best_action_number": result.get("best_action_number"),
        "best_circuit": result.get("best_circuit"),
        "best_coding": result.get("best_coding"),
        "total_actions": result.get("total_actions_taken"),
        "chi_square": metrics.get("chi_square"),
        "r2_score": metrics.get("r2_score"),
        "r2_mag": metrics.get("r2_mag"),
        "r2_phase": metrics.get("r2_phase"),
        "circuit_plot": (
            str(circuit_plot_path)
            if circuit_plot_path is not None and circuit_plot_path.exists()
            else None
        ),
        "fit_plot": str(paths["fit_plot"])
        if paths.get("fit_plot") and paths["fit_plot"].exists()
        else None,
        "circuit_gif": (
            str(paths["circuit_gif"])
            if paths.get("circuit_gif") and paths["circuit_gif"].exists()
            else None
        ),
        "final_circuit_frame": (
            str(paths["final_circuit_frame"])
            if paths.get("final_circuit_frame") and paths["final_circuit_frame"].exists()
            else None
        ),
    }


def _load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    from autorec.data_preparation import EISDataPrep

    data_path = Path(config["data_path"])
    if config["data_mode"] == "Raw CSV folder":
        prep = EISDataPrep(path=data_path, mode="process", evaluation=False)
        dataset = prep.load()
        processed_output = config.get("processed_output")
        if processed_output:
            prep.save(Path(processed_output), file_type="pickle")
    else:
        prep = EISDataPrep(path=data_path, mode="load", evaluation=False)
        dataset = prep.load()
    dataset = dataset.reset_index(drop=True).copy()
    if "true_circuit" not in dataset.columns:
        dataset["true_circuit"] = "unknown"
    return dataset


def _build_agent(config: dict[str, Any], dataset: pd.DataFrame) -> Any:
    from autorec.agent import DDQN_ECM
    from autorec.environment import EIS_ECM_Env
    from autorec.utils import set_global_seed

    seed = int(config["seed"])
    set_global_seed(seed, deterministic_ops=True)
    env = EIS_ECM_Env(
        dataset=dataset,
        seed=seed,
        chromosome_HEAD_len=int(config["head_len"]),
        cache_enabled=bool(config["cache_enabled"]),
        cache_capacity=int(config["cache_capacity"]),
    )
    agent = DDQN_ECM(
        training_env=env,
        eval_env=env,
        save_dir=_artifact_dir(config),
        num_trials=1,
        action_cap=int(config["max_actions"]),
        random_seed=seed,
    )
    agent.load_model(Path(config["model_path"]))
    return agent


def run(job_path: Path) -> None:
    config = json.loads(job_path.read_text())
    try:
        _write_status(config, "running", "Loading dataset...", progress=0.02)
        dataset = _load_dataset(config)
        _write_status(config, "running", "Loading trained agent...", progress=0.08)
        agent = _build_agent(config, dataset)

        indices = [int(index) for index in config["indices"]]
        results: list[dict[str, Any]] = []
        for row_number, index in enumerate(indices):
            base_progress = 0.1 + 0.85 * (row_number / max(len(indices), 1))
            progress_span = 0.85 / max(len(indices), 1)

            def update_action_status(event: dict[str, Any]) -> None:
                action_number = int(event.get("action_number", 0) or 0)
                max_actions = int(event.get("max_actions", config["max_actions"]) or 1)
                action_progress = min(action_number / max(max_actions, 1), 1.0)
                progress = min(base_progress + progress_span * action_progress, 0.95)

                phase = event.get("phase")
                if phase == "action_fit":
                    message = (
                        f"Evaluating EIS {row_number + 1}/{len(indices)} "
                        f"(index {index}), action {action_number}/{max_actions}: "
                        f"{event.get('action_type')} at position {event.get('action_position')}"
                    )
                elif phase == "action_complete":
                    message = (
                        f"Finished action {action_number}/{max_actions} for EIS index {index}. "
                        f"Best results will appear when this EIS completes."
                    )
                else:
                    message = (
                        f"Evaluating EIS {row_number + 1}/{len(indices)} "
                        f"(index {index}), action {action_number}/{max_actions}"
                    )
                _write_status(
                    config,
                    "running",
                    message,
                    progress=progress,
                    current_event=_json_safe(event),
                )

            _write_status(
                config,
                "running",
                f"Evaluating EIS {row_number + 1}/{len(indices)}: index {index}",
                progress=base_progress,
            )
            result = agent.generate_ECM(
                EIS_i=index,
                max_actions=int(config["max_actions"]),
                verbose=False,
                get_fit_plots=bool(config["fit_plots"]),
                gif_generation=bool(config["gif_generation"]),
                use_eval_env=True,
                progress_callback=update_action_status,
            )
            results.append(result)

            circuit_plot = _write_circuit_plot(config, result)
            rows = [
                _flatten_result(
                    item,
                    config,
                    circuit_plot if item is result else None,
                )
                for item in results
            ]
            Path(config["results_csv"]).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(config["results_csv"], index=False)

        _write_status(
            config,
            "complete",
            f"Complete. Evaluated {len(results)} EIS sample(s).",
            progress=1.0,
        )
    except Exception as exc:
        _write_status(config, "failed", f"Failed: {exc}", progress=0.0)
        raise


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m autorec.gui_worker JOB_JSON")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
