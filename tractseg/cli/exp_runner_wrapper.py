from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Iterable, List


def _has_option(arguments: Iterable[str], option_name: str) -> bool:
    return any(argument == option_name or argument.startswith(option_name + "=") for argument in arguments)


def _append_default(arguments: List[str], option_name: str, option_value: str | None = None) -> None:
    if _has_option(arguments, option_name):
        return
    arguments.append(option_name)
    if option_value is not None:
        arguments.append(option_value)


def _run_exp_runner(arguments: List[str]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_runner_path = repo_root / "ExpRunner"
    if not exp_runner_path.exists():
        raise FileNotFoundError(
            "Could not find the repository ExpRunner script. Run these wrappers from an editable checkout."
        )

    sys.argv = [str(exp_runner_path)] + arguments
    runpy.run_path(str(exp_runner_path), run_name="__main__")


def exp_runner_main() -> None:
    _run_exp_runner(sys.argv[1:])


def train_main() -> None:
    arguments = list(sys.argv[1:])
    _append_default(arguments, "--train", "True")
    _append_default(arguments, "--test", "True")
    _run_exp_runner(arguments)


def inference_main() -> None:
    arguments = list(sys.argv[1:])
    if _has_option(arguments, "-h") or _has_option(arguments, "--help"):
        _run_exp_runner(arguments)
        return
    _append_default(arguments, "--train", "False")
    _append_default(arguments, "--test", "False")
    if not _has_option(arguments, "--lw"):
        arguments.append("--lw")
    if not _has_option(arguments, "--weights_path"):
        raise SystemExit("RunInference requires --weights_path so replication uses an explicit checkpoint.")
    if not _has_option(arguments, "--seg") and not _has_option(arguments, "--probs"):
        arguments.append("--seg")
    _run_exp_runner(arguments)
