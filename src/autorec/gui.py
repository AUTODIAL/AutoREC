"""Streamlit web UI for launching AutoREC ECM generation jobs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_BUILD = "AutoREC Streamlit GUI - process-based evaluator"


st.set_page_config(
    page_title="AutoREC",
    page_icon="AutoREC",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _path_or_none(raw_value: str) -> Path | None:
    value = raw_value.strip()
    if not value:
        return None
    return Path(value).expanduser()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _repo_root() -> Path:
    return Path.cwd()


def _parse_indices(raw_indices: str, dataset_size: int | None = None) -> list[int]:
    indices: list[int] = []
    for item in raw_indices.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("Indices must be non-negative.")
        if dataset_size is not None and index >= dataset_size:
            raise ValueError(f"Index {index} is outside 0-{dataset_size - 1}.")
        indices.append(index)
    if not indices:
        raise ValueError("Provide at least one EIS index.")
    return indices


def _job_paths(output_dir: Path) -> dict[str, Path]:
    job_root = output_dir / "gui_jobs" / time.strftime("%Y%m%d_%H%M%S")
    return {
        "root": job_root,
        "job": job_root / "job.json",
        "status": job_root / "status.json",
        "log": job_root / "worker.log",
        "results": job_root / "results.csv",
    }


def _start_job(config: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["job"].write_text(json.dumps(config, indent=2))
    paths["status"].write_text(
        json.dumps(
            {
                "state": "queued",
                "message": "Job queued.",
                "results_csv": str(paths["results"]),
            },
            indent=2,
        )
    )

    log_file = paths["log"].open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "autorec.gui_worker", str(paths["job"])],
        cwd=_repo_root(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    st.session_state["active_job"] = {
        "pid": process.pid,
        "root": str(paths["root"]),
        "job": str(paths["job"]),
        "status": str(paths["status"]),
        "log": str(paths["log"]),
        "results": str(paths["results"]),
    }


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Settings")
        settings = {
            "head_len": int(
                st.number_input(
                    "Chromosome head length",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                )
            ),
            "max_actions": int(
                st.number_input(
                    "Max actions",
                    min_value=1,
                    max_value=500,
                    value=5,
                    step=1,
                )
            ),
            "seed": int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=1_000_000,
                    value=42,
                    step=1,
                )
            ),
            "cache_enabled": st.toggle("Circuit cache", value=True),
        }
        settings["cache_capacity"] = int(
            st.number_input(
                "Cache capacity",
                min_value=100,
                max_value=500_000,
                value=20_000,
                step=100,
                disabled=not settings["cache_enabled"],
            )
        )
        settings["fit_plots"] = st.toggle("Generate fit plots", value=False)
        settings["circuit_plots"] = st.toggle("Generate circuit plots", value=True)
        settings["gif_generation"] = st.toggle("Generate circuit GIFs", value=False)
    return settings


def _render_inputs(settings: dict[str, Any]) -> None:
    st.subheader("Inputs")
    model_path = _path_or_none(
        st.text_input("Trained agent model", value="models/examples/agent_trained.keras")
    )
    data_mode = st.radio(
        "Dataset source",
        ["Processed dataset file", "Raw CSV folder"],
        horizontal=True,
    )
    default_data_path = (
        "data/examples/training_dataset.pkl"
        if data_mode == "Processed dataset file"
        else "tutorials/EIS_raw_demo"
    )
    data_path = _path_or_none(st.text_input("Dataset path", value=default_data_path))
    output_dir = _path_or_none(st.text_input("Output folder", value="results/gui"))
    processed_output = None
    if data_mode == "Raw CSV folder":
        processed_output = _path_or_none(
            st.text_input("Save processed dataset as", value="data/gui_processed.pkl")
        )

    st.subheader("Evaluation")
    mode = st.radio(
        "Evaluation mode",
        ["Single EIS", "Specified indices"],
        horizontal=True,
    )
    if mode == "Single EIS":
        indices = [int(st.number_input("EIS index", min_value=0, value=0, step=1))]
    else:
        raw_indices = st.text_input("EIS indices", value="0")
        try:
            indices = _parse_indices(raw_indices)
        except ValueError as exc:
            st.warning(str(exc))
            return

    missing = [
        label
        for label, path in [
            ("model", model_path),
            ("dataset", data_path),
            ("output folder", output_dir),
        ]
        if path is None
    ]
    if missing:
        st.info(f"Provide {', '.join(missing)} to continue.")
        return

    assert model_path is not None
    assert data_path is not None
    assert output_dir is not None
    if not model_path.exists():
        st.error(f"Model does not exist: {model_path}")
        return
    if not data_path.exists():
        st.error(f"Dataset path does not exist: {data_path}")
        return

    if st.button("Generate ECM", type="primary"):
        paths = _job_paths(output_dir)
        config = {
            "model_path": str(model_path),
            "data_mode": data_mode,
            "data_path": str(data_path),
            "processed_output": str(processed_output) if processed_output else None,
            "output_dir": str(output_dir),
            "artifact_dir": str(paths["root"]),
            "results_csv": str(paths["results"]),
            "status_path": str(paths["status"]),
            "indices": indices,
            **settings,
        }
        _start_job(config, paths)
        st.success("ECM generation job started.")
        st.rerun()


def _render_best_circuits(results: pd.DataFrame) -> None:
    if results.empty or "best_circuit" not in results.columns:
        return

    st.subheader("Best Circuits")
    display_columns = [
        column
        for column in [
            "EIS_i",
            "best_circuit",
            "found_solution",
            "best_reward",
            "best_action_number",
            "chi_square",
            "r2_score",
        ]
        if column in results.columns
    ]
    summary = results[display_columns].copy()
    rename_map = {
        "EIS_i": "EIS index",
        "best_circuit": "Best circuit",
        "found_solution": "Good fit",
        "best_reward": "Reward",
        "best_action_number": "Action",
        "chi_square": "Chi-square",
        "r2_score": "R2",
    }
    summary = summary.rename(columns=rename_map)
    st.dataframe(summary, width="stretch", hide_index=True)

    latest = results.iloc[-1]
    best_circuit = latest.get("best_circuit")
    if pd.notna(best_circuit) and str(best_circuit).strip():
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Latest best circuit", str(best_circuit))
        metric_b.metric("Reward", _format_metric(latest.get("best_reward")))
        metric_c.metric("Action", _format_metric(latest.get("best_action_number"), decimals=0))


def _format_metric(value: Any, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _existing_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if path.exists():
        return path
    return None


def _render_generated_visuals(results: pd.DataFrame, config: dict[str, Any]) -> None:
    show_circuit_plot = bool(config.get("circuit_plots")) and "circuit_plot" in results.columns
    show_fit_plot = bool(config.get("fit_plots")) and "fit_plot" in results.columns
    show_gif_output = bool(config.get("gif_generation")) and (
        "final_circuit_frame" in results.columns or "circuit_gif" in results.columns
    )
    if not (show_circuit_plot or show_fit_plot or show_gif_output):
        return

    st.subheader("Generated Visuals")
    for _, row in results.iterrows():
        eis_index = row.get("EIS_i", "unknown")
        with st.expander(f"EIS {eis_index}", expanded=len(results) == 1):
            displayed_any = False

            if show_circuit_plot:
                circuit_plot = _existing_path(row.get("circuit_plot"))
                if circuit_plot is not None:
                    st.caption("Best circuit")
                    st.image(str(circuit_plot))
                    st.caption(str(circuit_plot))
                    displayed_any = True

            if show_fit_plot:
                fit_plot = _existing_path(row.get("fit_plot"))
                if fit_plot is not None:
                    st.caption("Fit plot")
                    st.image(str(fit_plot))
                    st.caption(str(fit_plot))
                    displayed_any = True

            if show_gif_output:
                final_frame = _existing_path(row.get("final_circuit_frame"))
                if final_frame is not None:
                    st.caption("Final circuit frame")
                    st.image(str(final_frame))
                    st.caption(str(final_frame))
                    displayed_any = True

                gif_path = _existing_path(row.get("circuit_gif"))
                if gif_path is not None:
                    st.info(f"Circuit evolution GIF saved at: {gif_path}")
                    displayed_any = True

            if not displayed_any:
                st.info(
                    "No visual artifact was saved for this EIS yet. Fit plots are only "
                    "created for terminal good fits."
                )


def _render_active_job() -> None:
    active_job = st.session_state.get("active_job")
    if not active_job:
        return

    st.subheader("ECM Job")
    status_path = Path(active_job["status"])
    status = _read_json(status_path)
    if status is None:
        st.info("Waiting for worker status...")
        return

    state = status.get("state", "unknown")
    message = status.get("message", "")
    if state in {"queued", "running"}:
        st.info(message)
    elif state == "complete":
        st.success(message)
    elif state == "failed":
        st.error(message)
    else:
        st.write(message)

    progress = status.get("progress")
    if isinstance(progress, (int, float)):
        st.progress(min(max(float(progress), 0.0), 1.0))

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("State", state)
    metric_b.metric("PID", active_job["pid"])
    metric_c.metric("Job folder", Path(active_job["root"]).name)

    button_a, button_b = st.columns([0.2, 0.2])
    if button_a.button("Refresh Status"):
        st.rerun()
    if button_b.button("Clear Job"):
        st.session_state.pop("active_job", None)
        st.rerun()

    results_path = Path(active_job["results"])
    if results_path.exists():
        st.subheader("Results")
        results = pd.read_csv(results_path)
        job_config = _read_json(
            Path(active_job.get("job", Path(active_job["root"]) / "job.json"))
        )
        job_config = job_config or {}
        _render_best_circuits(results)
        _render_generated_visuals(results, job_config)
        st.dataframe(results, width="stretch")
        st.caption(f"Results saved at {results_path}")

    log_path = Path(active_job["log"])
    if log_path.exists():
        with st.expander("Worker Log", expanded=state == "failed"):
            st.code(log_path.read_text()[-8000:], language=None)


def main() -> None:
    st.title("AutoREC")
    st.caption(APP_BUILD)
    settings = _render_sidebar()
    _render_inputs(settings)
    _render_active_job()


if __name__ == "__main__":
    main()
