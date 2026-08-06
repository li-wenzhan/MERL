from __future__ import annotations

from typing import Iterable, List, Optional


def normalize_task_descriptions(
    task_descriptions: Optional[Iterable[object]],
    batch_size: int,
    *,
    context: str,
) -> List[str]:
    if batch_size < 0:
        raise ValueError(f"[{context}] batch_size must be non-negative, got {batch_size}.")

    if task_descriptions is None:
        return [""] * batch_size

    normalized = [str(item or "") for item in list(task_descriptions)]
    if len(normalized) == 0:
        return [""] * batch_size
    if len(normalized) != batch_size:
        raise ValueError(
            f"[{context}] task_descriptions length mismatch: got {len(normalized)}, expected {batch_size}."
        )
    return normalized