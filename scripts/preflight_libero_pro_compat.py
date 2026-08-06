#!/usr/bin/env python
# LIBERO-PRO compatibility preflight.
# 1. Validate config paths before model/Ray startup.
# 2. Fail fast on robosuite versions that removed LIBERO's legacy import path.
# 3. Check the LIBERO-PRO env import used by rollout and init-state generation.

import argparse
import os
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def _require_path(label: str, path: str, *, is_dir: bool = False) -> str:
    normalized = os.path.abspath(os.path.expanduser(path))
    exists = os.path.isdir(normalized) if is_dir else os.path.exists(normalized)
    if not exists:
        raise SystemExit(f"[preflight] missing {label}: {normalized}")
    return normalized


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _version_tuple(raw: str) -> tuple[int, int, int]:
    parts = []
    for item in raw.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


def _require_numpy_1x(install_hint: str) -> None:
    numpy_version = _package_version("numpy")
    if numpy_version == "not-installed" or _version_tuple(numpy_version) >= (2, 0, 0):
        raise SystemExit(
            f"[preflight] incompatible numpy=={numpy_version}; "
            "LIBERO/OpenVLA/tensorflow-cpu require NumPy 1.x in this repo.\n"
            f"[preflight] install/fix environment with: {install_hint}"
        )


def _require_import(module_name: str, install_hint: str) -> None:
    try:
        import_module(module_name)
    except Exception as exc:
        raise SystemExit(
            f"[preflight] cannot import {module_name}: {exc}\n"
            f"[preflight] install/fix environment with: {install_hint}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/evaluation_config.yaml")
    args = parser.parse_args()

    cfg_path = _require_path("LIBERO_PRO_EVAL_CONFIG_PATH", args.config)
    cfg = _load_yaml(cfg_path)
    libero_pro_root = str(
        cfg.get("libero_pro_root") or os.environ.get("LIBERO_PRO_ROOT") or ""
    ).strip()
    if not libero_pro_root or libero_pro_root.startswith("/path/to/"):
        raise SystemExit(
            "[preflight] libero_pro_root is not configured. Set "
            "configs/evaluation_config.yaml:libero_pro_root or LIBERO_PRO_ROOT."
        )

    libero_pro_root = _require_path("LIBERO_PRO root", libero_pro_root, is_dir=True)
    _require_path(
        "LIBERO_PRO benchmark root",
        os.path.join(libero_pro_root, "libero", "libero"),
        is_dir=True,
    )
    _require_path(
        "LIBERO_PRO bddl_files",
        os.path.join(libero_pro_root, "libero", "libero", "bddl_files"),
        is_dir=True,
    )
    _require_path(
        "LIBERO_PRO generate_init_states.py",
        os.path.join(libero_pro_root, "notebooks", "generate_init_states.py"),
    )

    repo_root = str(Path(__file__).resolve().parents[1])
    for path in (repo_root, libero_pro_root):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, repo_root)
    sys.path.insert(1, libero_pro_root)

    robosuite_version = _package_version("robosuite")
    install_hint = (
        "python -m pip uninstall -y mink && python -m pip install --force-reinstall "
        "'numpy==1.26.4' 'opencv-python>=4.8,<4.12' 'robosuite==1.4.1' "
        "'mujoco>=2.3.7,<3.0' bddl easydict cloudpickle 'gym>=0.23,<0.26'"
    )
    _require_numpy_1x(install_hint)
    _require_import(
        "robosuite.environments.manipulation.single_arm_env",
        install_hint,
    )

    # Activate a non-interactive LIBERO runtime config before importing
    # libre modules, otherwise first-time imports may prompt for dataset path.
    try:
        from verl.utils.libero_path import ensure_libero_pro_root

        ensure_libero_pro_root(
            evaluation_config_path=cfg_path,
            explicit_root=libero_pro_root,
        )
        from libero.libero import get_libero_path

        init_states_path = _require_path(
            "LIBERO_PRO init_states runtime path",
            get_libero_path("init_states"),
            is_dir=True,
        )
        _require_path(
            "LIBERO_PRO bddl_files runtime path",
            get_libero_path("bddl_files"),
            is_dir=True,
        )
        runtime_config_dir = os.environ.get("LIBERO_CONFIG_PATH", "")
        runtime_config_file = os.path.join(runtime_config_dir, "config.yaml")
        runtime_cfg = _load_yaml(runtime_config_file)
        if not runtime_cfg:
            raise RuntimeError(
                f"LIBERO runtime config is empty: {runtime_config_file}"
            )
        if os.path.abspath(runtime_cfg.get("init_states", "")) != os.path.abspath(
            init_states_path
        ):
            raise RuntimeError(
                "LIBERO runtime config init_states mismatch: "
                f"{runtime_cfg.get('init_states')} vs {init_states_path}"
            )
    except Exception as exc:
        raise SystemExit(
            "[preflight] failed to activate LIBERO runtime config: "
            f"{exc}\n"
            "[preflight] check configs/evaluation_config.yaml and make sure "
            "libero_pro_root points to a valid LIBERO_PRO clone."
        ) from exc

    _require_import("libero.libero.envs", install_hint)

    try:
        perturbation = import_module("modules.libero_pro.perturbation")
        has_create_env = callable(getattr(perturbation, "create_env", None))
        has_fallback_api = all(
            hasattr(perturbation, name)
            for name in ("PerturbFlags", "process_bddl_file_mixed", "EvalEnvCreator")
        )
        if not has_create_env and not has_fallback_api:
            raise RuntimeError(
                "missing create_env() and fallback asset-generation components"
            )
    except Exception as exc:
        raise SystemExit(
            "[preflight] cannot use modules.libero_pro.perturbation for "
            f"LIBERO-PRO suite asset generation: {exc}"
        ) from exc

    print(
        "[preflight] LIBERO-PRO compatibility OK "
        f"(root={libero_pro_root}, robosuite={robosuite_version})"
    )


if __name__ == "__main__":
    main()
