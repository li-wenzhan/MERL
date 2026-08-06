import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


_SAFE_TENSOR_SUFFIXES = {".safetensors"}
_PICKLE_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_CACHE_ROOT = _REPO_ROOT / "tmp_files" / "hf_safetensors_cache"
_HF_SINGLE_FILE_SAFE_NAMES = {
    "pytorch_model.bin": "model.safetensors",
    "diffusion_pytorch_model.bin": "diffusion_pytorch_model.safetensors",
    "adapter_model.bin": "adapter_model.safetensors",
}
_HF_INDEX_SAFE_NAMES = {
    "pytorch_model.bin.index.json": "model.safetensors.index.json",
    "diffusion_pytorch_model.bin.index.json": "diffusion_pytorch_model.safetensors.index.json",
    "adapter_model.bin.index.json": "adapter_model.safetensors.index.json",
}
_HF_SHARDED_WEIGHT_RE = re.compile(
    r"^(pytorch_model|diffusion_pytorch_model|adapter_model)-(\d{5})-of-(\d{5})\.bin$"
)
_HF_SAFE_SHARD_PREFIX = {
    "pytorch_model": "model",
    "diffusion_pytorch_model": "diffusion_pytorch_model",
    "adapter_model": "adapter_model",
}


def _parse_torch_version(raw_version: str) -> Tuple[int, int]:
    base = str(raw_version or "").split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)", base)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def torch_supports_safe_pickle_load() -> bool:
    return _parse_torch_version(torch.__version__) >= (2, 6)


def is_placeholder_path(path: Optional[str]) -> bool:
    normalized = str(path or "").strip()
    return not normalized or normalized.startswith("/path/to/")


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _safe_basename_for_pickle(basename: str) -> Optional[str]:
    if basename in _HF_SINGLE_FILE_SAFE_NAMES:
        return _HF_SINGLE_FILE_SAFE_NAMES[basename]
    if basename in _HF_INDEX_SAFE_NAMES:
        return _HF_INDEX_SAFE_NAMES[basename]

    match = _HF_SHARDED_WEIGHT_RE.match(basename)
    if match is not None:
        prefix = _HF_SAFE_SHARD_PREFIX[match.group(1)]
        return f"{prefix}-{match.group(2)}-of-{match.group(3)}.safetensors"

    return None


def _safe_relative_path_for_pickle(rel_path: Path) -> Optional[Path]:
    safe_basename = _safe_basename_for_pickle(rel_path.name)
    if safe_basename is None:
        return None
    return rel_path.with_name(safe_basename)


def inspect_local_weight_layout(model_path: str) -> Dict[str, Any]:
    normalized = _expand_path(model_path)
    root = Path(normalized)
    safe_files = []
    safe_index_files = []
    pickle_files = []
    hf_pickle_files = []
    missing_safe_equivalents = []

    if root.is_dir():
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            rel_path = file_path.relative_to(root)
            safe_rel_path = _safe_relative_path_for_pickle(rel_path)
            if suffix in _SAFE_TENSOR_SUFFIXES:
                safe_files.append(str(file_path))
            elif file_path.name.endswith(".safetensors.index.json"):
                safe_index_files.append(str(file_path))
            elif suffix in _PICKLE_WEIGHT_SUFFIXES:
                pickle_files.append(str(file_path))
                if safe_rel_path is not None:
                    hf_pickle_files.append(str(file_path))
                    if not (root / safe_rel_path).is_file():
                        missing_safe_equivalents.append(str(file_path))
            elif file_path.name.endswith(".bin.index.json") and safe_rel_path is not None:
                hf_pickle_files.append(str(file_path))
                if not (root / safe_rel_path).is_file():
                    missing_safe_equivalents.append(str(file_path))

    return {
        "path": normalized,
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "safetensors_files": sorted(safe_files),
        "safe_index_files": sorted(safe_index_files),
        "pickle_files": sorted(pickle_files),
        "hf_pickle_files": sorted(hf_pickle_files),
        "missing_safe_equivalents": sorted(missing_safe_equivalents),
    }


def _is_hf_safe_ready(layout: Dict[str, Any]) -> bool:
    return not layout["missing_safe_equivalents"]


def _cache_dir_for_model(model_path: str) -> Path:
    normalized = _expand_path(model_path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return _SAFE_CACHE_ROOT / f"{Path(normalized).name}-{digest}"


def _copy_file_if_needed(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_file() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
        return
    shutil.copy2(src_path, dst_path)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _coerce_tensor_state_dict(payload: Any, *, source_path: str) -> Dict[str, torch.Tensor]:
    state_dict = payload
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported checkpoint payload in {source_path}: expected dict, got {type(state_dict)}")

    normalized_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(
                f"unsupported checkpoint entry in {source_path}: key={key!r}, value_type={type(value)}"
            )
        normalized_state_dict[key] = value.detach().cpu().contiguous()
    return normalized_state_dict


def _save_safetensors_state_dict(state_dict: Dict[str, torch.Tensor], target_path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to materialize trusted Hugging Face pickle weights "
            "into a safe local cache. Install the safetensors package in the training environment."
        ) from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    save_file(state_dict, str(tmp_path))
    os.replace(tmp_path, target_path)


def _materialize_safe_cache(source_root: Path, target_root: Path, *, component_name: str) -> None:
    target_root.mkdir(parents=True, exist_ok=True)

    for file_path in source_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(source_root)
        if _safe_relative_path_for_pickle(rel_path) is not None:
            continue
        if file_path.name.endswith(".bin.index.json") and _safe_relative_path_for_pickle(rel_path) is not None:
            continue
        _copy_file_if_needed(file_path, target_root / rel_path)

    for file_path in source_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(source_root)
        safe_rel_path = _safe_relative_path_for_pickle(rel_path)
        if safe_rel_path is None:
            continue

        target_path = target_root / safe_rel_path
        if target_path.is_file() and target_path.stat().st_mtime >= file_path.stat().st_mtime:
            continue

        if file_path.name.endswith(".bin.index.json"):
            with open(file_path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload.get("weight_map"), dict):
                payload["weight_map"] = {
                    key: _safe_basename_for_pickle(value) or value
                    for key, value in payload["weight_map"].items()
                }
            _write_json_atomic(target_path, payload)
            continue

        state_dict = load_trusted_state_dict(str(file_path), map_location="cpu")
        safe_state_dict = _coerce_tensor_state_dict(
            state_dict,
            source_path=str(file_path),
        )
        _save_safetensors_state_dict(safe_state_dict, target_path)

    print(
        f"[hf-safe-cache] {component_name}: materialized safetensors cache at {target_root} "
        f"from {source_root}"
    )


def _render_weight_examples(paths: List[str], *, limit: int = 3) -> str:
    if not paths:
        return "none"
    head = [os.path.basename(path) for path in paths[:limit]]
    rendered = ", ".join(head)
    if len(paths) > limit:
        rendered += f", ... (+{len(paths) - limit} more)"
    return rendered


def validate_local_hf_model_dir(model_path: str, *, component_name: str) -> Dict[str, Any]:
    if is_placeholder_path(model_path):
        raise ValueError(
            f"{component_name} path is not configured: {model_path!r}. "
            "Set the local backbone directory in the world-model config first."
        )

    layout = inspect_local_weight_layout(model_path)
    normalized = layout["path"]
    if not layout["exists"]:
        raise FileNotFoundError(f"missing {component_name}: {normalized}")
    if not layout["is_dir"]:
        raise ValueError(f"{component_name} must be a directory, got file: {normalized}")

    if not layout["hf_pickle_files"] and not layout["safetensors_files"]:
        raise ValueError(
            f"{component_name} at {normalized} does not expose recognizable Hugging Face weight files."
        )

    if _is_hf_safe_ready(layout):
        return layout

    if layout["hf_pickle_files"] and not torch_supports_safe_pickle_load():
        raise ValueError(
            f"{component_name} at {normalized} still requires pickle weights "
            f"({_render_weight_examples(layout['missing_safe_equivalents'])}), but the current "
            f"torch=={torch.__version__} is below 2.6. Recent transformers blocks "
            "torch.load-based Hugging Face loading in that environment. The repo should "
            "auto-materialize a safetensors cache for trusted local backbones before runtime; "
            "if you still see this error, cache generation failed."
        )

    return layout


def prepare_local_hf_model_dir(
    model_path: str,
    *,
    component_name: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if is_placeholder_path(model_path):
        raise ValueError(
            f"{component_name} path is not configured: {model_path!r}. "
            "Set the local backbone directory in the world-model config first."
        )

    normalized = _expand_path(model_path)
    if not os.path.isdir(normalized):
        if os.path.exists(normalized):
            raise ValueError(f"{component_name} must point to a directory, got file: {normalized}")
        if torch_supports_safe_pickle_load():
            return model_path, {}, inspect_local_weight_layout(model_path)
        return model_path, {"use_safetensors": True}, inspect_local_weight_layout(model_path)

    layout = inspect_local_weight_layout(normalized)
    if not layout["exists"]:
        raise FileNotFoundError(f"missing {component_name}: {normalized}")
    if not layout["is_dir"]:
        raise ValueError(f"{component_name} must be a directory, got file: {normalized}")
    if not layout["hf_pickle_files"] and not layout["safetensors_files"]:
        raise ValueError(
            f"{component_name} at {normalized} does not expose recognizable Hugging Face weight files."
        )

    if _is_hf_safe_ready(layout):
        return normalized, {"use_safetensors": True}, layout

    if torch_supports_safe_pickle_load():
        return normalized, {}, layout

    cache_dir = _cache_dir_for_model(normalized)
    _materialize_safe_cache(Path(normalized), cache_dir, component_name=component_name)
    cache_layout = inspect_local_weight_layout(str(cache_dir))
    if not _is_hf_safe_ready(cache_layout):
        raise RuntimeError(
            f"failed to materialize safetensors cache for {component_name}: {cache_layout['path']}"
        )
    return str(cache_dir), {"use_safetensors": True}, cache_layout


def build_hf_pretrained_kwargs(model_path: str, *, component_name: str) -> Dict[str, Any]:
    _, kwargs, _ = prepare_local_hf_model_dir(
        model_path,
        component_name=component_name,
    )
    return kwargs


def rethrow_hf_loading_error(exc: Exception, *, component_name: str, model_path: str) -> None:
    message = str(exc)
    if "require users to upgrade torch to at least v2.6" in message or "check_torch_load_is_safe" in message:
        raise RuntimeError(
            f"{component_name} cannot be loaded from {model_path!r}. "
            f"Current torch=={torch.__version__}. The checkpoint resolved to a "
            "torch.load-based Hugging Face weight file, which transformers now blocks "
            "on torch < 2.6. Use a safetensors-exported backbone directory or upgrade torch."
        ) from exc
    raise exc


def resolve_ctrl_world_ckpt_path(config: Any) -> Optional[str]:
    if not bool(getattr(config, "load_from_ckpt", False)):
        return None

    ckpt_path = str(getattr(config, "ckpt_path", "") or "").strip()
    if is_placeholder_path(ckpt_path):
        raise ValueError(
            "load_from_ckpt=True but ckpt_path is empty or still points at the placeholder /path/to/... entry."
        )

    normalized = _expand_path(ckpt_path)
    if not os.path.isfile(normalized):
        raise FileNotFoundError(f"missing Ctrl-World checkpoint: {normalized}")
    return normalized


def _normalize_safetensors_device(map_location: Any) -> str:
    if isinstance(map_location, torch.device):
        return str(map_location)
    return str(map_location or "cpu")


def load_trusted_state_dict(path: str, *, map_location: Any = "cpu") -> Dict[str, Any]:
    normalized = _expand_path(path)
    suffix = Path(normalized).suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load a .safetensors checkpoint. "
                "Install the safetensors package in the training environment."
            ) from exc

        return load_file(normalized, device=_normalize_safetensors_device(map_location))

    try:
        return torch.load(normalized, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(normalized, map_location=map_location)


def summarize_weight_layout(layout: Dict[str, Any]) -> str:
    return (
        f"{layout['path']} "
        f"(safetensors={len(layout['safetensors_files'])}, "
        f"hf_pickle={len(layout['hf_pickle_files'])}, "
        f"missing_safe={len(layout['missing_safe_equivalents'])})"
    )