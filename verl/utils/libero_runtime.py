import os
from typing import Any, Dict, MutableMapping, Optional

_STABLE_HEADLESS_BACKENDS = {"egl", "osmesa"}


def resolve_egl_device_id(requested="auto", visible_devices=None) -> str:
    """Use a visible device identifier, not a rank-local CUDA ordinal.

    robosuite checks this identifier against CUDA_VISIBLE_DEVICES before loading
    its EGL context. In particular, a Ray worker with visibility '2' needs '2'.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible_devices is None else visible_devices
    devices = [part.strip() for part in str(visible or "").split(",") if part.strip()]
    text = str(requested or "auto").strip().lower()
    if text in {"none", "unset", "disable", "disabled"}:
        return ""
    if text in {"auto", "default", ""}:
        text = devices[0] if devices else "0"
    if not text.isdigit() or (devices and text not in devices):
        raise ValueError(f"EGL device {text!r} is not a numeric device in CUDA_VISIBLE_DEVICES={visible!r}; "
                         "use auto with numeric visibility or configure a compatible rendering device explicitly")
    return text


def _normalize(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _choose_backend(
    mujoco_gl: str,
    pyopengl_platform: str,
    *,
    force_headless: bool,
    has_display: bool,
) -> str:
    if pyopengl_platform == "osmesa" or mujoco_gl == "osmesa":
        return "osmesa"

    # Respect an explicit GLX request when an X display is available.
    # Our training launchers provide Xvfb on headless servers, so glx is a
    # valid and user-tested backend in that deployment mode.
    if has_display and mujoco_gl == "glx" and pyopengl_platform in {"", "glx"}:
        return "glx"

    if force_headless or not has_display:
        return "egl"

    return "egl"


def configure_libero_runtime_env(
    env: Optional[MutableMapping[str, str]] = None,
    *,
    force_headless: bool = True,
    force_spawn: bool = True,
) -> Dict[str, Any]:
    target_env = os.environ if env is None else env

    display = str(target_env.get("DISPLAY") or "").strip()
    original_mujoco_gl = _normalize(target_env.get("MUJOCO_GL"))
    original_pyopengl_platform = _normalize(target_env.get("PYOPENGL_PLATFORM"))
    backend = _choose_backend(
        original_mujoco_gl,
        original_pyopengl_platform,
        force_headless=force_headless,
        has_display=bool(display),
    )
    pyopengl_platform = "glx" if backend == "glx" else backend

    changes: Dict[str, Dict[str, str]] = {}

    def _set(key: str, value: str) -> None:
        previous = str(target_env.get(key) or "")
        if previous != value:
            target_env[key] = value
            changes[key] = {"old": previous, "new": value}

    _set("MUJOCO_GL", backend)
    _set("PYOPENGL_PLATFORM", pyopengl_platform)
    if backend == "glx" and _normalize(target_env.get("MERL_GLX_SOFTWARE")) in {
        "1",
        "true",
        "yes",
        "on",
    }:
        _set("LIBGL_ALWAYS_SOFTWARE", "1")
        _set("LIBGL_DRI3_DISABLE", "1")
        _set("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    if backend != "egl":
        target_env.pop("MUJOCO_EGL_DEVICE_ID", None)

    if force_spawn:
        _set("MERL_ENV_MP_START_METHOD", "spawn")

    return {
        "display": display,
        "headless": bool(force_headless or not display),
        "backend": backend,
        "pyopengl_platform": pyopengl_platform,
        "start_method": str(target_env.get("MERL_ENV_MP_START_METHOD") or "spawn"),
        "changes": changes,
        "stable_headless": backend in _STABLE_HEADLESS_BACKENDS,
    }


def format_libero_runtime_env_summary(
    summary: Dict[str, Any],
    *,
    prefix: str = "[libero runtime]",
) -> str:
    parts = [
        f"{prefix} MUJOCO_GL={summary['backend']}",
        f"PYOPENGL_PLATFORM={summary['pyopengl_platform']}",
        f"MERL_ENV_MP_START_METHOD={summary['start_method']}",
    ]

    if summary.get("display"):
        parts.append(f"DISPLAY={summary['display']}")
    else:
        parts.append("DISPLAY=<unset>")

    changes = summary.get("changes") or {}
    if changes:
        rendered_changes = []
        for key, value in changes.items():
            old = value.get("old") or "<unset>"
            rendered_changes.append(f"{key}:{old}->{value.get('new')}")
        parts.append("changes=" + ", ".join(rendered_changes))

    return " | ".join(parts)
