import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


_LIBERO_PRO_BDDL_RELATIVE_PATH = ("libero", "libero", "bddl_files")
_LIBERO_BENCHMARK_RELATIVE_PATH = ("libero", "libero")
_LIBERO_PRO_SCRIPT_RELATIVE_PATH = ("notebooks", "generate_init_states.py")
_LIBERO_PRO_INIT_RELATIVE_PATH = ("libero", "libero", "init_files")
_DEFAULT_OFFLINE_WM_CONFIG_RELATIVE_PATH = ("configs", "wm_offline_config.py")
_LIBERO_PRO_OOD_RELATIVE_PATHS = {
    "environment": ("ood_environment.yaml",),
    "swap": ("ood_spatial_relation.yaml",),
    "object": ("ood_object.yaml",),
    "language": ("ood_language.yaml",),
    "task": ("ood_task.yaml",),
}
_ROBOSUITE_INSTALL_HINT = (
    "python -m pip uninstall -y mink && python -m pip install --force-reinstall "
    "'numpy==1.26.4' 'opencv-python>=4.8,<4.12' 'robosuite==1.4.1' "
    "'mujoco>=2.3.7,<3.0' bddl easydict cloudpickle 'gym>=0.23,<0.26'"
)


def _normalize_existing_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isdir(normalized):
        return normalized
    return None


def _join_path(root: Optional[str], suffix_parts: Tuple[str, ...]) -> str:
    root_text = str(root or "").strip().rstrip("/\\")
    if not root_text:
        return ""
    if "/" in root_text and "\\" not in root_text:
        return "/".join([root_text, *suffix_parts])
    return os.path.join(root_text, *suffix_parts)


def _prefer_explicit_or_derived(
    explicit_path: Optional[str], root: Optional[str], suffix_parts: Tuple[str, ...]
) -> str:
    explicit_text = str(explicit_path or "").strip()
    if explicit_text:
        return explicit_text
    return _join_path(root, suffix_parts)


def _infer_root_from_suffix_str(
    path: Optional[str], suffix_parts: Tuple[str, ...]
) -> Optional[str]:
    if not path:
        return None

    normalized = os.path.abspath(os.path.expanduser(str(path).rstrip("/\\")))
    path_obj = Path(normalized)
    path_parts = [part.lower() for part in path_obj.parts]
    suffix_lower = [part.lower() for part in suffix_parts]
    if len(path_parts) < len(suffix_lower):
        return None
    if path_parts[-len(suffix_lower) :] != suffix_lower:
        return None

    root = Path(*path_obj.parts[: -len(suffix_lower)])
    return str(root)


def _infer_root_from_suffix(
    path: Optional[str], suffix_parts: Tuple[str, ...]
) -> Optional[str]:
    return _normalize_existing_dir(_infer_root_from_suffix_str(path, suffix_parts))


def _ensure_sys_path(path: Optional[str]) -> Optional[str]:
    normalized = _normalize_existing_dir(path)
    if normalized:
        repo_root = str(Path(__file__).resolve().parents[2])
        for candidate in (repo_root, normalized):
            while candidate in sys.path:
                sys.path.remove(candidate)
        sys.path.insert(0, repo_root)
        if normalized != repo_root:
            sys.path.insert(1, normalized)
    return normalized


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _version_tuple(raw: str) -> Tuple[int, int, int]:
    parts = []
    for item in raw.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


def _ensure_numpy_1x_compat() -> None:
    numpy_version = _package_version("numpy")
    if numpy_version == "not-installed" or _version_tuple(numpy_version) >= (2, 0, 0):
        raise RuntimeError(
            "[LIBERO_PRO] incompatible numpy environment. "
            f"LIBERO/OpenVLA/tensorflow-cpu require NumPy 1.x, but numpy=={numpy_version}. "
            f"Fix the environment with: {_ROBOSUITE_INSTALL_HINT}"
        )


def _ensure_libero_pro_robosuite_compat() -> None:
    _ensure_numpy_1x_compat()
    module_name = "robosuite.environments.manipulation.single_arm_env"
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "[LIBERO_PRO] incompatible robosuite environment. "
            f"LIBERO-PRO requires the legacy module '{module_name}', "
            f"but robosuite=={_package_version('robosuite')} cannot import it. "
            f"Fix the environment with: {_ROBOSUITE_INSTALL_HINT}"
        ) from exc


def _parent_dir_if_named(path: Optional[str], leaf_name: str) -> Optional[str]:
    normalized = _normalize_existing_dir(path)
    if normalized is None:
        return None
    path_obj = Path(normalized)
    if path_obj.name.lower() != leaf_name.lower():
        return None
    return str(path_obj.parent)


def _resolve_libero_benchmark_root(
    root: Optional[str],
    bddl_files_path: Optional[str] = None,
    init_file_dir: Optional[str] = None,
) -> Optional[str]:
    candidates = []

    root_text = str(root or "").strip().rstrip("/\\")
    if root_text:
        candidates.append(_join_path(root_text, _LIBERO_BENCHMARK_RELATIVE_PATH))
        candidates.append(root_text)

    candidates.extend(
        [
            _parent_dir_if_named(bddl_files_path, "bddl_files"),
            _parent_dir_if_named(init_file_dir, "init_files"),
        ]
    )

    for candidate in candidates:
        normalized = _normalize_existing_dir(candidate)
        if normalized:
            return normalized
    return None


def _build_libero_runtime_path_mapping(
    benchmark_root: str,
    bddl_files_path: Optional[str] = None,
    init_file_dir: Optional[str] = None,
) -> Dict[str, str]:
    normalized_benchmark_root = os.path.abspath(os.path.expanduser(str(benchmark_root)))
    normalized_bddl_files_path = os.path.abspath(
        os.path.expanduser(
            str(
                bddl_files_path or os.path.join(normalized_benchmark_root, "bddl_files")
            )
        )
    )
    normalized_init_file_dir = os.path.abspath(
        os.path.expanduser(
            str(init_file_dir or os.path.join(normalized_benchmark_root, "init_files"))
        )
    )
    normalized_datasets_path = os.path.abspath(
        os.path.join(normalized_benchmark_root, "..", "datasets")
    )
    normalized_assets_path = os.path.abspath(
        os.path.join(normalized_benchmark_root, "assets")
    )

    return {
        "benchmark_root": normalized_benchmark_root,
        "bddl_files": normalized_bddl_files_path,
        "init_states": normalized_init_file_dir,
        "init_files": normalized_init_file_dir,
        "datasets": normalized_datasets_path,
        "assets": normalized_assets_path,
    }


def _resolve_libero_runtime_config_dir(
    runtime_name: str, config_path: Optional[str] = None
) -> str:
    normalized_config_path = str(config_path or "").strip()
    if normalized_config_path:
        config_path_obj = Path(
            os.path.abspath(os.path.expanduser(normalized_config_path))
        )
        base_dir = config_path_obj.parent.parent
        return str(base_dir / "tmp_files" / "libero_runtime" / runtime_name)
    return str(Path(tempfile.gettempdir()) / "merl_libero_runtime" / runtime_name)


def _activate_libero_runtime_config(
    runtime_name: str,
    path_mapping: Mapping[str, str],
    config_path: Optional[str] = None,
) -> str:
    config_dir = _resolve_libero_runtime_config_dir(
        runtime_name=runtime_name,
        config_path=config_path,
    )
    os.makedirs(config_dir, exist_ok=True)

    runtime_config_file = os.path.join(config_dir, "config.yaml")
    _atomic_write_yaml_mapping(runtime_config_file, path_mapping)

    os.environ["LIBERO_CONFIG_PATH"] = config_dir

    for module_name in ("libero.libero", "libero.libero.utils"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        setattr(module, "libero_config_path", config_dir)
        setattr(module, "config_file", runtime_config_file)

    return config_dir


def _atomic_write_yaml_mapping(path: str, mapping: Mapping[str, str]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".config.", suffix=".tmp", dir=directory, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(dict(mapping), file_obj, sort_keys=True)
            file_obj.flush()
            os.fsync(file_obj.fileno())

        with open(tmp_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj) or {}
        if not isinstance(loaded, dict) or not loaded:
            raise RuntimeError(
                f"Refusing to activate empty LIBERO runtime config: {tmp_path}"
            )

        os.replace(tmp_path, path)
        if hasattr(os, "O_DIRECTORY"):
            dir_fd = None
            try:
                dir_fd = os.open(directory, os.O_DIRECTORY)
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                if dir_fd is not None:
                    os.close(dir_fd)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _load_yaml_mapping(config_path: Optional[str]) -> Mapping[str, Any]:
    if not config_path:
        return {}
    normalized = os.path.abspath(os.path.expanduser(str(config_path)))
    if not os.path.isfile(normalized):
        return {}
    with open(normalized, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    if isinstance(config, dict):
        return config
    return {}


def _resolve_default_offline_wm_config_path() -> str:
    return str(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath(*_DEFAULT_OFFLINE_WM_CONFIG_RELATIVE_PATH)
    )


def _load_offline_wm_root_text(config_path: Optional[str] = None) -> str:
    normalized = os.path.abspath(
        os.path.expanduser(
            str(config_path or _resolve_default_offline_wm_config_path())
        )
    )
    if not os.path.isfile(normalized):
        return ""

    try:
        spec = importlib.util.spec_from_file_location(
            "_merl_wm_offline_config", normalized
        )
        if spec is None or spec.loader is None:
            return ""
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return ""

    wm_args = getattr(module, "wm_args", None)
    return str(getattr(wm_args, "libero_root", "") or "").strip().rstrip("/\\")


def _resolve_libero_pro_root_text(evaluation_cfg: Mapping[str, Any]) -> str:
    explicit_root = str(evaluation_cfg.get("libero_pro_root") or "").strip()
    if explicit_root:
        return explicit_root.rstrip("/\\")

    inferred_root = (
        _infer_root_from_suffix_str(
            evaluation_cfg.get("bddl_files_path"), _LIBERO_PRO_BDDL_RELATIVE_PATH
        )
        or _infer_root_from_suffix_str(
            evaluation_cfg.get("init_file_dir"), _LIBERO_PRO_INIT_RELATIVE_PATH
        )
        or _infer_root_from_suffix_str(
            evaluation_cfg.get("script_path"), _LIBERO_PRO_SCRIPT_RELATIVE_PATH
        )
    )
    return str(inferred_root or "").rstrip("/\\")


def load_libero_pro_config(config_path: Optional[str]) -> Dict[str, Any]:
    evaluation_cfg = dict(_load_yaml_mapping(config_path))
    libero_pro_root = _resolve_libero_pro_root_text(evaluation_cfg)
    if not libero_pro_root:
        libero_pro_root = (
            str(os.environ.get("LIBERO_PRO_ROOT") or "").strip().rstrip("/\\")
        )

    normalized_cfg = dict(evaluation_cfg)
    normalized_cfg["libero_pro_root"] = libero_pro_root
    normalized_cfg["bddl_files_path"] = _prefer_explicit_or_derived(
        evaluation_cfg.get("bddl_files_path"),
        libero_pro_root,
        _LIBERO_PRO_BDDL_RELATIVE_PATH,
    )
    normalized_cfg["script_path"] = _prefer_explicit_or_derived(
        evaluation_cfg.get("script_path"),
        libero_pro_root,
        _LIBERO_PRO_SCRIPT_RELATIVE_PATH,
    )
    normalized_cfg["init_file_dir"] = _prefer_explicit_or_derived(
        evaluation_cfg.get("init_file_dir"),
        libero_pro_root,
        _LIBERO_PRO_INIT_RELATIVE_PATH,
    )

    libero_ood_root = str(evaluation_cfg.get("libero_ood_root") or "").strip()
    if not libero_ood_root:
        libero_ood_root = _join_path(libero_pro_root, ("libero_ood",))
    normalized_cfg["libero_ood_root"] = libero_ood_root

    explicit_ood_cfg = dict(evaluation_cfg.get("ood_task_configs") or {})
    normalized_ood_cfg = {}
    for key, suffix_parts in _LIBERO_PRO_OOD_RELATIVE_PATHS.items():
        normalized_ood_cfg[key] = _prefer_explicit_or_derived(
            explicit_ood_cfg.get(key), libero_ood_root, suffix_parts
        )
    normalized_cfg["ood_task_configs"] = normalized_ood_cfg
    return normalized_cfg


def resolve_libero_pro_root(
    evaluation_config_path: Optional[str] = None,
    explicit_root: Optional[str] = None,
) -> Optional[str]:
    candidates = [
        explicit_root,
        os.environ.get("LIBERO_PRO_ROOT"),
    ]
    for candidate in candidates:
        normalized = _normalize_existing_dir(candidate)
        if normalized:
            return normalized

    evaluation_cfg = load_libero_pro_config(evaluation_config_path)
    candidates = [
        evaluation_cfg.get("libero_pro_root"),
        _infer_root_from_suffix(
            evaluation_cfg.get("bddl_files_path"),
            _LIBERO_PRO_BDDL_RELATIVE_PATH,
        ),
        _infer_root_from_suffix(
            evaluation_cfg.get("init_file_dir"),
            _LIBERO_PRO_INIT_RELATIVE_PATH,
        ),
        _infer_root_from_suffix(
            evaluation_cfg.get("script_path"),
            _LIBERO_PRO_SCRIPT_RELATIVE_PATH,
        ),
    ]
    for candidate in candidates:
        normalized = _normalize_existing_dir(candidate)
        if normalized:
            return normalized

    return None


def ensure_libero_pro_root(
    evaluation_config_path: Optional[str] = None,
    explicit_root: Optional[str] = None,
) -> Optional[str]:
    normalized_root = _ensure_sys_path(
        resolve_libero_pro_root(
            evaluation_config_path=evaluation_config_path,
            explicit_root=explicit_root,
        )
    )
    if normalized_root:
        os.environ["LIBERO_PRO_ROOT"] = normalized_root
    _ensure_libero_pro_robosuite_compat()

    evaluation_cfg = load_libero_pro_config(evaluation_config_path)
    benchmark_root = _resolve_libero_benchmark_root(
        root=normalized_root or evaluation_cfg.get("libero_pro_root"),
        bddl_files_path=evaluation_cfg.get("bddl_files_path"),
        init_file_dir=evaluation_cfg.get("init_file_dir"),
    )
    if benchmark_root:
        _activate_libero_runtime_config(
            runtime_name="libero_pro",
            path_mapping=_build_libero_runtime_path_mapping(
                benchmark_root=benchmark_root,
                bddl_files_path=evaluation_cfg.get("bddl_files_path"),
                init_file_dir=evaluation_cfg.get("init_file_dir"),
            ),
            config_path=evaluation_config_path,
        )

    return normalized_root


def resolve_libero_root(
    explicit_root: Optional[str] = None,
    offline_config_path: Optional[str] = None,
) -> Optional[str]:
    candidates = [
        explicit_root,
        os.environ.get("LIBERO_ROOT"),
        _load_offline_wm_root_text(offline_config_path),
    ]
    for candidate in candidates:
        normalized = _normalize_existing_dir(candidate)
        if normalized:
            return normalized

    return None


def ensure_libero_root(
    explicit_root: Optional[str] = None,
    offline_config_path: Optional[str] = None,
) -> Optional[str]:
    normalized_root = _ensure_sys_path(
        resolve_libero_root(
            explicit_root=explicit_root,
            offline_config_path=offline_config_path,
        )
    )
    if normalized_root:
        os.environ["LIBERO_ROOT"] = normalized_root

    benchmark_root = _resolve_libero_benchmark_root(root=normalized_root)
    if benchmark_root:
        _activate_libero_runtime_config(
            runtime_name="libero",
            path_mapping=_build_libero_runtime_path_mapping(
                benchmark_root=benchmark_root,
            ),
            config_path=offline_config_path,
        )

    return normalized_root
