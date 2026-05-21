"""Streamlit GUI for loading a trained AutoREC agent and generating ECMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=False)

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AutoREC GUI", layout="wide")


def _log_event(message: str) -> None:
    print(f"[AutoREC GUI] {message}", flush=True)
    events = st.session_state.setdefault("run_events", [])
    events.append(message)
    del events[:-30]


def _set_session_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def _choose_path(text_key: str, browser_open_key: str, path: Path) -> None:
    st.session_state[text_key] = str(path)
    st.session_state[browser_open_key] = False


def _initial_browser_dir(raw_value: str, kind: str) -> Path:
    if raw_value.strip():
        path = Path(raw_value).expanduser()
        if kind in {"file", "save"} and not path.is_dir():
            path = path.parent
    else:
        path = Path.cwd()

    if not path.exists() or not path.is_dir():
        path = Path.cwd()
    return path.resolve()


def _list_directory(path: Path) -> tuple[list[Path], list[Path], str | None]:
    try:
        entries = list(path.iterdir())
    except Exception as exc:
        return [], [], str(exc)

    directories = sorted(
        [entry for entry in entries if entry.is_dir()],
        key=lambda item: item.name.lower(),
    )
    files = sorted(
        [entry for entry in entries if entry.is_file()],
        key=lambda item: item.name.lower(),
    )
    return directories, files, None


def _render_path_browser(
    label: str,
    key: str,
    kind: str,
    text_key: str,
) -> None:
    browser_open_key = f"{key}_browser_open"
    browser_dir_key = f"{key}_browser_dir"
    raw_value = st.session_state.get(text_key, "")

    if browser_dir_key not in st.session_state:
        st.session_state[browser_dir_key] = str(_initial_browser_dir(raw_value, kind))

    current_dir = Path(st.session_state[browser_dir_key]).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = Path.cwd().resolve()
        st.session_state[browser_dir_key] = str(current_dir)

    with st.container(border=True):
        st.caption(label)
        st.code(str(current_dir), language=None)

        nav_col_a, nav_col_b, nav_col_c, nav_col_d = st.columns(4)
        nav_col_a.button(
            "Up",
            key=f"{key}_up",
            width="stretch",
            on_click=_set_session_value,
            args=(browser_dir_key, str(current_dir.parent)),
        )
        nav_col_b.button(
            "Home",
            key=f"{key}_home",
            width="stretch",
            on_click=_set_session_value,
            args=(browser_dir_key, str(Path.home())),
        )
        nav_col_c.button(
            "Repo",
            key=f"{key}_repo",
            width="stretch",
            on_click=_set_session_value,
            args=(browser_dir_key, str(Path.cwd())),
        )
        nav_col_d.button(
            "Close",
            key=f"{key}_close",
            width="stretch",
            on_click=_set_session_value,
            args=(browser_open_key, False),
        )

        directories, files, error = _list_directory(current_dir)
        if error:
            st.error(f"Cannot read folder: {error}")
            return

        if kind == "save":
            directory_options = {f"[dir] {item.name}": item for item in directories}
            if directory_options:
                selected_dir_label = st.selectbox(
                    "Folders",
                    list(directory_options),
                    key=f"{key}_dir_selection",
                )
                selected_dir = directory_options[selected_dir_label]
                st.button(
                    "Open Selected Folder",
                    key=f"{key}_open_selected_dir",
                    width="stretch",
                    on_click=_set_session_value,
                    args=(browser_dir_key, str(selected_dir)),
                )
            else:
                st.caption("No folders in this location.")

            current_name = Path(raw_value).name if raw_value.strip() else "processed_data.pkl"
            file_name = st.text_input(
                "File name",
                key=f"{key}_save_name",
                value=current_name,
            )
            if file_name.strip():
                st.button(
                    "Use This Save Path",
                    key=f"{key}_choose_save_path",
                    type="primary",
                    width="stretch",
                    on_click=_choose_path,
                    args=(text_key, browser_open_key, current_dir / file_name.strip()),
                )
            return

        path_options = {f"[dir] {item.name}": item for item in directories}
        if kind == "file":
            path_options.update({f"[file] {item.name}": item for item in files})

        if not path_options:
            st.caption("No selectable items in this location.")
        else:
            selected_label = st.selectbox(
                "Contents",
                list(path_options),
                key=f"{key}_selection",
            )
            selected_path = path_options[selected_label]
            if selected_path.is_dir():
                open_col, choose_col = st.columns(2)
                open_col.button(
                    "Open Folder",
                    key=f"{key}_open_folder",
                    width="stretch",
                    on_click=_set_session_value,
                    args=(browser_dir_key, str(selected_path)),
                )
                if kind == "directory":
                    choose_col.button(
                        "Use This Folder",
                        key=f"{key}_choose_selected_folder",
                        type="primary",
                        width="stretch",
                        on_click=_choose_path,
                        args=(text_key, browser_open_key, selected_path),
                    )
            elif kind == "file":
                st.button(
                    "Use This File",
                    key=f"{key}_choose_file",
                    type="primary",
                    width="stretch",
                    on_click=_choose_path,
                    args=(text_key, browser_open_key, selected_path),
                )

        if kind == "directory":
            st.button(
                "Use Current Folder",
                key=f"{key}_choose_current_folder",
                type="primary",
                width="stretch",
                on_click=_choose_path,
                args=(text_key, browser_open_key, current_dir),
            )


def _path_input(
    label: str,
    key: str,
    kind: str,
    help_text: str | None = None,
) -> Path | None:
    text_key = f"{key}_text"
    browser_open_key = f"{key}_browser_open"
    if key in st.session_state and text_key not in st.session_state:
        st.session_state[text_key] = st.session_state[key]

    col_path, col_button = st.columns([0.78, 0.22], vertical_alignment="bottom")
    with col_path:
        value = st.text_input(label, key=text_key, help=help_text)
    with col_button:
        st.button(
            "Browse",
            key=f"{key}_browse",
            width="stretch",
            on_click=_set_session_value,
            args=(browser_open_key, not st.session_state.get(browser_open_key, False)),
        )

    if st.session_state.get(browser_open_key, False):
        _render_path_browser(label, key, kind, text_key)

    if not value.strip():
        return None
    return Path(value).expanduser()


def _load_or_process_dataset(
    data_mode: str,
    data_path: Path,
    processed_output_path: Path | None,
) -> pd.DataFrame:
    _log_event(f"Loading dataset: mode={data_mode}, path={data_path}")
    configure_autorec_runtime(thread_count=1, warmup_autoeis=True)
    from autorec.data_preparation import EISDataPrep

    if data_mode == "Raw EIS folder":
        prep = EISDataPrep(path=data_path, mode="process", evaluation=False)
        dataset = prep.load()
        if processed_output_path is not None:
            processed_output_path.parent.mkdir(parents=True, exist_ok=True)
            prep.save(processed_output_path, file_type="pickle")
            _log_event(f"Saved processed dataset: {processed_output_path}")
        return dataset

    prep = EISDataPrep(path=data_path, mode="load", evaluation=False)
    dataset = prep.load()
    _log_event(f"Loaded processed dataset with {len(dataset)} sample(s).")
    return dataset


def _prepare_dataset_for_environment(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.reset_index(drop=True).copy()
    if "true_circuit" not in dataset.columns:
        dataset["true_circuit"] = "unknown"
    return dataset


def _build_agent(
    model_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    chromosome_head_len: int,
    seed: int,
    cache_enabled: bool,
    cache_capacity: int,
    action_cap: int | None,
) -> Any:
    _log_event(f"Building agent from model: {model_path}")
    configure_autorec_runtime(thread_count=1, warmup_autoeis=True)
    from autorec.agent import DDQN_ECM
    from autorec.environment import EIS_ECM_Env
    from autorec.utils import set_global_seed

    set_global_seed(seed, deterministic_ops=True)
    env = EIS_ECM_Env(
        dataset=dataset,
        seed=seed,
        chromosome_HEAD_len=chromosome_head_len,
        cache_enabled=cache_enabled,
        cache_capacity=cache_capacity,
    )
    agent = DDQN_ECM(
        training_env=env,
        eval_env=env,
        save_dir=output_dir,
        num_trials=1,
        action_cap=action_cap,
        random_seed=seed,
    )
    agent.load_model(model_path)
    _log_event("Agent loaded.")
    return agent


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _display_safe_value(value: Any) -> Any:
    safe_value = _json_safe(value)
    if isinstance(safe_value, (dict, list)):
        return json.dumps(safe_value)
    return safe_value


def _display_safe_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_df = dataframe.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(_display_safe_value)
    return display_df


def _flatten_result_for_table(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("best_metrics") or {}
    return {
        "EIS_i": result.get("EIS_i"),
        "ground_truth_circuit": result.get("ground_truth_circuit"),
        "found_solution": result.get("found_solution"),
        "best_reward": result.get("best_reward"),
        "best_action_number": result.get("best_action_number"),
        "best_circuit": result.get("best_circuit"),
        "best_coding": result.get("best_coding"),
        "best_state": result.get("best_state"),
        "total_actions": result.get("total_actions_taken"),
        "chi_square": metrics.get("chi_square"),
        "r2_score": metrics.get("r2_score"),
        "r2_mag": metrics.get("r2_mag"),
        "r2_phase": metrics.get("r2_phase"),
    }


def _save_single_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ecm_result_eis_{result['EIS_i']}.json"
    csv_path = output_dir / f"ecm_result_eis_{result['EIS_i']}.csv"
    json_path.write_text(json.dumps(_json_safe(result), indent=2))
    table = pd.DataFrame([_flatten_result_for_table(result)])
    _display_safe_dataframe(table).to_csv(csv_path, index=False)


def _save_batch_result(results: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ecm_batch_results.csv"
    serializable = results.copy()
    for column in serializable.columns:
        serializable[column] = serializable[column].map(_json_safe)
    serializable.to_csv(csv_path, index=False)
    return csv_path


def _artifact_paths(output_dir: Path, eis_index: int) -> dict[str, Path]:
    return {
        "fit_plot": output_dir / "fit_plots" / f"fit_EIS_i_{eis_index + 1}.png",
        "gif": output_dir / "generated_circuits" / f"circuit_evolution_EIS_{eis_index}.gif",
    }


def _result_eis_indices(results: pd.DataFrame) -> list[int]:
    if "EIS_i" not in results.columns:
        return []

    indices: list[int] = []
    for value in results["EIS_i"].tolist():
        safe_value = _json_safe(value)
        try:
            indices.append(int(safe_value))
        except (TypeError, ValueError):
            continue
    return indices


def _display_generated_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    show_fit_plots: bool,
    show_gifs: bool,
) -> None:
    if not show_fit_plots and not show_gifs:
        return

    eis_indices = _result_eis_indices(results)
    if not eis_indices:
        return

    st.subheader("Generated Visuals")

    for eis_index in eis_indices:
        paths = _artifact_paths(output_dir, eis_index)
        with st.expander(f"EIS {eis_index}", expanded=len(eis_indices) == 1):
            if show_fit_plots:
                fit_plot = paths["fit_plot"]
                if fit_plot.exists():
                    st.image(str(fit_plot), caption=f"Fit plot: {fit_plot}")
                else:
                    st.info(
                        "No fit plot was generated for this EIS. "
                        "AutoREC only saves a fit plot when a terminal fit is found."
                    )

            if show_gifs:
                gif_path = paths["gif"]
                if gif_path.exists():
                    st.image(str(gif_path), caption=f"Circuit evolution GIF: {gif_path}")
                else:
                    st.info(
                        "No GIF was generated for this EIS. Check the Run Log for GIF "
                        "generation warnings."
                    )


def _parse_indices(raw_indices: str, dataset_size: int) -> list[int]:
    indices = []
    for item in raw_indices.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= dataset_size:
            raise ValueError(
                f"Index {index} is outside the dataset range 0-{dataset_size - 1}."
            )
        indices.append(index)
    if not indices:
        raise ValueError("Enter at least one index.")
    return indices


def _summary_metrics(results: pd.DataFrame) -> None:
    if results.empty:
        return
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("EIS evaluated", len(results))
    found_solution = results["found_solution"].map(_json_safe).fillna(False)
    col_b.metric("Solutions found", int(found_solution.sum()))
    rewards = pd.to_numeric(results["best_reward"].map(_json_safe), errors="coerce")
    mean_reward = rewards.replace(-np.inf, np.nan).mean()
    col_c.metric("Mean reward", f"{mean_reward:.4f}")


def _batch_result_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "EIS_i": result["EIS_i"],
        "environment_type": result["environment_type"],
        "ground_truth_circuit": result["ground_truth_circuit"],
        "found_solution": result["found_solution"],
        "best_reward": result["best_reward"],
        "best_action_number": result["best_action_number"],
        "best_circuit": result["best_circuit"],
        "best_coding": result["best_coding"],
        "best_metrics": result["best_metrics"],
        "best_param": result["best_param"],
        "total_actions": result["total_actions_taken"],
    }


def _run_single_eis(
    agent: Any,
    eis_index: int,
    max_actions: int,
    fit_plots: bool,
    gif_generation: bool,
    output_dir: Path,
) -> pd.DataFrame:
    _log_event(f"Starting single EIS evaluation: index={eis_index}, max_actions={max_actions}")
    action_progress = st.progress(0.0)
    action_status = st.empty()

    def update_action_progress(event: dict[str, Any]) -> None:
        event_name = event["event"]
        total = max(int(event["max_actions"]), 1)
        if event_name == "evaluation_started":
            action_status.write(
                f"EIS {event['EIS_i']}: evaluation started. "
                "Selecting first action..."
            )
            action_progress.progress(0.0)
            _log_event(f"EIS {event['EIS_i']}: evaluation started.")
        elif event_name == "action_selecting":
            action = int(event["action"])
            action_status.write(
                f"EIS {event['EIS_i']}: selecting action {action + 1}/{total}..."
            )
            action_progress.progress(action / total)
            _log_event(f"EIS {event['EIS_i']}: selecting action {action + 1}/{total}.")
        elif event_name == "action_selected":
            action = int(event["action"])
            action_status.write(
                f"EIS {event['EIS_i']}: selected action {action + 1}/{total} "
                f"({event['action_type']} at position {event['action_position']})."
            )
            _log_event(
                f"EIS {event['EIS_i']}: selected action {action + 1}/{total} "
                f"({event['action_type']} at position {event['action_position']})."
            )
        elif event_name == "action_started":
            action = int(event["action"])
            action_status.write(
                f"EIS {event['EIS_i']}: fitting action {action + 1}/{total} "
                f"({event['action_type']} at position {event['action_position']})"
            )
            action_progress.progress(action / total)
            _log_event(f"EIS {event['EIS_i']}: fitting action {action + 1}/{total}.")
        elif event_name == "action_finished":
            action = int(event["action"])
            action_progress.progress((action + 1) / total)
            reward = event.get("reward")
            reward_text = f"{reward:.4f}" if isinstance(reward, (int, float)) else "n/a"
            action_status.write(
                f"EIS {event['EIS_i']}: completed action {action + 1}/{total}; "
                f"reward {reward_text}"
            )
            _log_event(
                f"EIS {event['EIS_i']}: completed action {action + 1}/{total}; "
                f"reward {reward_text}."
            )
        elif event_name == "evaluation_finished":
            action_status.write(
                f"EIS {event['EIS_i']}: evaluation finished; "
                f"best circuit {event.get('best_circuit') or 'none'}."
            )
            _log_event(
                f"EIS {event['EIS_i']}: evaluation finished; "
                f"found_solution={event.get('found_solution')}."
            )

    result = agent.generate_ECM(
        EIS_i=eis_index,
        max_actions=max_actions,
        verbose=False,
        get_fit_plots=fit_plots,
        gif_generation=gif_generation,
        use_eval_env=True,
        progress_callback=update_action_progress,
    )
    _save_single_result(result, output_dir)
    action_progress.progress(1.0)
    action_status.write(f"EIS {eis_index}: complete.")
    _log_event(f"Single EIS evaluation complete: index={eis_index}")
    return pd.DataFrame([_flatten_result_for_table(result)])


def _run_batch_eis(
    agent: Any,
    eis_indices: list[int],
    max_actions: int,
    fit_plots: bool,
    gif_generation: bool,
    output_dir: Path,
) -> pd.DataFrame:
    _log_event(
        f"Starting batch evaluation: count={len(eis_indices)}, max_actions={max_actions}, "
        f"indices={eis_indices}"
    )
    overall_progress = st.progress(0.0)
    action_progress = st.progress(0.0)
    overall_status = st.empty()
    action_status = st.empty()
    results: list[dict[str, Any]] = []
    total_eis = len(eis_indices)

    for row_number, eis_index in enumerate(eis_indices):
        overall_status.write(
            f"Evaluating EIS {row_number + 1}/{total_eis} (index {eis_index})."
        )
        action_progress.progress(0.0)

        def update_action_progress(event: dict[str, Any]) -> None:
            event_name = event["event"]
            total_actions = max(int(event["max_actions"]), 1)
            if event_name == "evaluation_started":
                action_status.write(
                    f"EIS {event['EIS_i']}: evaluation started. "
                    "Selecting first action..."
                )
                _log_event(f"EIS {event['EIS_i']}: evaluation started.")
            elif event_name == "action_selecting":
                action = int(event["action"])
                action_status.write(
                    f"EIS {event['EIS_i']}: selecting action {action + 1}/{total_actions}..."
                )
                action_progress.progress(action / total_actions)
                _log_event(
                    f"EIS {event['EIS_i']}: selecting action {action + 1}/{total_actions}."
                )
            elif event_name == "action_selected":
                action = int(event["action"])
                action_status.write(
                    f"EIS {event['EIS_i']}: selected action {action + 1}/{total_actions} "
                    f"({event['action_type']} at position {event['action_position']})."
                )
                _log_event(
                    f"EIS {event['EIS_i']}: selected action {action + 1}/{total_actions} "
                    f"({event['action_type']} at position {event['action_position']})."
                )
            elif event_name == "action_started":
                action = int(event["action"])
                action_status.write(
                    f"EIS {event['EIS_i']}: fitting action {action + 1}/{total_actions} "
                    f"({event['action_type']} at position {event['action_position']})"
                )
                action_progress.progress(action / total_actions)
                _log_event(
                    f"EIS {event['EIS_i']}: fitting action {action + 1}/{total_actions}."
                )
            elif event_name == "action_finished":
                action = int(event["action"])
                action_progress.progress((action + 1) / total_actions)
                _log_event(
                    f"EIS {event['EIS_i']}: completed action {action + 1}/{total_actions}."
                )
            elif event_name == "evaluation_finished":
                action_status.write(
                    f"EIS {event['EIS_i']}: evaluation finished; "
                    f"best circuit {event.get('best_circuit') or 'none'}."
                )
                _log_event(
                    f"EIS {event['EIS_i']}: evaluation finished; "
                    f"found_solution={event.get('found_solution')}."
                )

        try:
            result = agent.generate_ECM(
                EIS_i=eis_index,
                max_actions=max_actions,
                verbose=False,
                get_fit_plots=fit_plots,
                gif_generation=gif_generation,
                use_eval_env=True,
                progress_callback=update_action_progress,
            )
            results.append(_batch_result_row(result))
        except Exception as exc:
            _log_event(f"EIS {eis_index}: evaluation failed: {exc}")
            results.append(
                {
                    "EIS_i": eis_index,
                    "environment_type": agent.get_current_env_type(),
                    "ground_truth_circuit": None,
                    "found_solution": False,
                    "best_reward": -np.inf,
                    "best_action_number": -1,
                    "best_circuit": None,
                    "best_coding": None,
                    "best_metrics": None,
                    "best_param": None,
                    "total_actions": 0,
                    "error": str(exc),
                }
            )

        overall_progress.progress((row_number + 1) / total_eis)

    results_df = pd.DataFrame(results)
    _save_batch_result(results_df, output_dir)
    action_status.write("Batch evaluation complete.")
    overall_status.write(f"Evaluated {total_eis} EIS sample(s).")
    _log_event(f"Batch evaluation complete: count={total_eis}")
    return results_df


def main() -> None:
    st.title("AutoREC")

    with st.sidebar:
        st.header("Evaluation Settings")
        chromosome_head_len = st.number_input(
            "Chromosome head length",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Use the same value that was used when the agent was trained.",
        )
        max_actions = st.number_input(
            "Max actions",
            min_value=1,
            max_value=500,
            value=24,
            step=1,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
        )
        cache_enabled = st.toggle("Circuit cache", value=True)
        cache_capacity = st.number_input(
            "Cache capacity",
            min_value=100,
            max_value=500_000,
            value=20_000,
            step=100,
            disabled=not cache_enabled,
        )
        gif_generation = st.toggle("Generate circuit GIFs", value=False)
        fit_plots = st.toggle("Generate fit plots", value=True)

    st.subheader("Inputs")
    model_path = _path_input(
        "Trained agent model file",
        key="model_path",
        kind="file",
        help_text="Select a .keras or .h5 model saved by DDQN_ECM.save_model().",
    )

    data_mode = st.radio(
        "Dataset type",
        ["Raw EIS folder", "Processed dataset file"],
        horizontal=True,
    )
    data_path = _path_input(
        "Raw EIS folder" if data_mode == "Raw EIS folder" else "Processed dataset file",
        key="data_path",
        kind="directory" if data_mode == "Raw EIS folder" else "file",
    )
    output_dir = _path_input(
        "Output folder",
        key="output_dir",
        kind="directory",
        help_text="Processed datasets, evaluation tables, plots, and GIFs are saved here.",
    )

    processed_output_path = None
    if data_mode == "Raw EIS folder":
        processed_output_path = _path_input(
            "Save processed dataset as",
            key="processed_output_path",
            kind="save",
            help_text="The raw EIS folder will be processed and saved before evaluation.",
        )

    missing = [
        name
        for name, value in [
            ("trained agent model", model_path),
            ("dataset path", data_path),
            ("output folder", output_dir),
        ]
        if value is None
    ]
    if data_mode == "Raw EIS folder" and processed_output_path is None:
        missing.append("processed dataset output path")

    if missing:
        st.info(f"Select {', '.join(missing)} to continue.")
        return

    assert model_path is not None
    assert data_path is not None
    assert output_dir is not None

    if not model_path.exists():
        st.error(f"Model path does not exist: {model_path}")
        return
    if not data_path.exists():
        st.error(f"Dataset path does not exist: {data_path}")
        return

    if st.button("Load Data And Agent", type="primary"):
        try:
            with st.spinner("Loading dataset and trained agent..."):
                dataset = _load_or_process_dataset(
                    data_mode,
                    data_path,
                    processed_output_path,
                )
                dataset = _prepare_dataset_for_environment(dataset)
                agent = _build_agent(
                    model_path=model_path,
                    dataset=dataset,
                    output_dir=output_dir,
                    chromosome_head_len=int(chromosome_head_len),
                    seed=int(seed),
                    cache_enabled=cache_enabled,
                    cache_capacity=int(cache_capacity),
                    action_cap=int(max_actions),
                )
                st.session_state["dataset"] = dataset
                st.session_state["agent"] = agent
                st.session_state["loaded_output_dir"] = output_dir
            st.success(f"Loaded {len(dataset)} EIS sample(s).")
        except Exception as exc:
            _log_event(f"Load failed: {exc}")
            st.exception(exc)

    dataset = st.session_state.get("dataset")
    agent = st.session_state.get("agent")
    loaded_output_dir = st.session_state.get("loaded_output_dir", output_dir)

    if dataset is None or agent is None:
        return

    st.subheader("Dataset")
    st.write(f"{len(dataset)} EIS sample(s), columns: {', '.join(dataset.columns)}")
    preview_columns = [
        column
        for column in ["sub_id", "true_circuit", "chi_thresh", "r2_thresh"]
        if column in dataset.columns
    ]
    if preview_columns:
        preview = _display_safe_dataframe(dataset[preview_columns].head(20))
        st.dataframe(preview, width="stretch")

    st.subheader("Circuit Generation")
    if len(dataset) == 1:
        run_mode = "Single EIS"
    else:
        run_mode = st.radio(
            "Evaluation mode",
            [
                "Single EIS",
                "Batch evaluation (random)",
                "Batch evaluation using specified indices",
                "Full dataset evaluation",
            ],
        )

    if run_mode == "Single EIS":
        eis_index = st.number_input(
            "EIS index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
        )
    elif run_mode == "Batch evaluation (random)":
        random_count = st.number_input(
            "Number of random EIS samples",
            min_value=1,
            max_value=len(dataset),
            value=min(10, len(dataset)),
            step=1,
        )
    elif run_mode == "Batch evaluation using specified indices":
        raw_indices = st.text_input(
            "EIS indices",
            value="0",
            help=f"Comma-separated row positions between 0 and {len(dataset) - 1}.",
        )

    if st.button("Generate ECM", type="primary"):
        _log_event(f"Generate ECM button clicked: mode={run_mode}")
        try:
            with st.spinner("Generating ECM results..."):
                if run_mode == "Single EIS":
                    table = _run_single_eis(
                        agent=agent,
                        eis_index=int(eis_index),
                        max_actions=int(max_actions),
                        fit_plots=fit_plots,
                        gif_generation=gif_generation,
                        output_dir=loaded_output_dir,
                    )
                    st.session_state["last_results"] = table
                    st.session_state["last_fit_plots_requested"] = fit_plots
                    st.session_state["last_gifs_requested"] = gif_generation
                    st.session_state["last_artifact_output_dir"] = loaded_output_dir
                elif run_mode == "Batch evaluation (random)":
                    all_indices = list(range(len(dataset)))
                    sample_size = min(int(random_count), len(all_indices))
                    indices = np.random.choice(
                        all_indices,
                        size=sample_size,
                        replace=False,
                    ).tolist()
                    results = _run_batch_eis(
                        agent=agent,
                        eis_indices=indices,
                        max_actions=int(max_actions),
                        fit_plots=fit_plots,
                        gif_generation=gif_generation,
                        output_dir=loaded_output_dir,
                    )
                    st.session_state["last_results"] = results
                    st.session_state["last_fit_plots_requested"] = fit_plots
                    st.session_state["last_gifs_requested"] = gif_generation
                    st.session_state["last_artifact_output_dir"] = loaded_output_dir
                elif run_mode == "Batch evaluation using specified indices":
                    indices = _parse_indices(raw_indices, len(dataset))
                    results = _run_batch_eis(
                        agent=agent,
                        eis_indices=indices,
                        max_actions=int(max_actions),
                        fit_plots=fit_plots,
                        gif_generation=gif_generation,
                        output_dir=loaded_output_dir,
                    )
                    st.session_state["last_results"] = results
                    st.session_state["last_fit_plots_requested"] = fit_plots
                    st.session_state["last_gifs_requested"] = gif_generation
                    st.session_state["last_artifact_output_dir"] = loaded_output_dir
                else:
                    results = _run_batch_eis(
                        agent=agent,
                        eis_indices=list(range(len(dataset))),
                        max_actions=int(max_actions),
                        fit_plots=fit_plots,
                        gif_generation=gif_generation,
                        output_dir=loaded_output_dir,
                    )
                    st.session_state["last_results"] = results
                    st.session_state["last_fit_plots_requested"] = fit_plots
                    st.session_state["last_gifs_requested"] = gif_generation
                    st.session_state["last_artifact_output_dir"] = loaded_output_dir
        except Exception as exc:
            _log_event(f"Generate ECM failed: {exc}")
            st.exception(exc)

    results = st.session_state.get("last_results")
    if results is not None:
        st.subheader("Results")
        _summary_metrics(results)
        st.dataframe(_display_safe_dataframe(results), width="stretch")
        st.caption(f"Outputs saved in {loaded_output_dir}")
        _display_generated_artifacts(
            results=results,
            output_dir=st.session_state.get("last_artifact_output_dir", loaded_output_dir),
            show_fit_plots=st.session_state.get("last_fit_plots_requested", False),
            show_gifs=st.session_state.get("last_gifs_requested", False),
        )

    run_events = st.session_state.get("run_events", [])
    if run_events:
        st.subheader("Run Log")
        st.code("\n".join(run_events[-30:]), language=None)


if __name__ == "__main__":
    main()
