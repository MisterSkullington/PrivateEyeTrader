"""
Model artifact checkpointing for Phase 6 online learning.

Before each online-learning promotion, the current model's artifact files are
copied to a versioned checkpoint directory. This allows rollback if a newly
trained model regresses on validation loss.

Layout:
  artifacts/checkpoints/{model_name}_{YYYYMMDD_HHMMSS}/
    <all artifact files>
    meta.json  ← {"model_name", "bars_seen", "saved_at", "artifact_files": [...]}
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from privateye.utils.logging import get_logger

log = get_logger()


def save_checkpoint(
    model: object,
    bars_seen: int,
    checkpoint_dir: Path,
) -> Path:
    """
    Copy the model's artifact files into a new versioned checkpoint directory.

    Parameters
    ----------
    model       : any object with .artifacts_dir: Path and optionally .__class__.__name__
    bars_seen   : number of bars the model was trained on (recorded in meta.json)
    checkpoint_dir : parent directory for all checkpoints

    Returns
    -------
    Path to the newly created checkpoint directory.
    """
    artifacts_dir: Path = model.artifacts_dir  # type: ignore[attr-defined]
    model_name = model.__class__.__name__

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_path = Path(checkpoint_dir) / f"{model_name}_{ts}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    artifact_files: list[str] = []
    for src in artifacts_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, ckpt_path / src.name)
            artifact_files.append(src.name)

    meta = {
        "model_name": model_name,
        "bars_seen": bars_seen,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_files": sorted(artifact_files),
    }
    (ckpt_path / "meta.json").write_text(json.dumps(meta, indent=2))

    log.debug(f"[Checkpoint] Saved {model_name} → {ckpt_path} ({len(artifact_files)} files)")
    return ckpt_path


def load_checkpoint(
    model: object,
    checkpoint_dir: Path,
    latest: bool = True,
    timestamp: str | None = None,
) -> None:
    """
    Restore a model's artifacts from a checkpoint, then call model.load().

    Parameters
    ----------
    model          : model instance (artifacts_dir must point to correct location)
    checkpoint_dir : parent directory containing checkpoint subdirectories
    latest         : if True, load the most recent checkpoint (default)
    timestamp      : ISO prefix string to select a specific checkpoint (e.g. "20240101_120000")
                     Ignored when latest=True.

    Raises
    ------
    FileNotFoundError if no matching checkpoints exist.
    """
    model_name = model.__class__.__name__  # type: ignore[attr-defined]
    checkpoints = list_checkpoints(model_name, Path(checkpoint_dir))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for {model_name} in {checkpoint_dir}"
        )

    if latest:
        chosen = checkpoints[0]  # sorted newest-first
    else:
        if timestamp is None:
            chosen = checkpoints[0]
        else:
            matches = [c for c in checkpoints if timestamp in c["saved_at"]]
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint matching timestamp '{timestamp}' for {model_name}"
                )
            chosen = matches[0]

    ckpt_path = Path(chosen["path"])
    artifacts_dir: Path = model.artifacts_dir  # type: ignore[attr-defined]

    for src in ckpt_path.iterdir():
        if src.is_file() and src.name != "meta.json":
            shutil.copy2(src, artifacts_dir / src.name)

    model.load()  # type: ignore[attr-defined]
    log.info(
        f"[Checkpoint] Restored {model_name} from {ckpt_path.name} "
        f"(bars_seen={chosen['bars_seen']})"
    )


def list_checkpoints(model_name: str, checkpoint_dir: Path) -> list[dict]:
    """
    Return checkpoint metadata dicts for a given model, sorted newest-first.

    Each dict has keys: path (str), model_name, bars_seen, saved_at.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    results: list[dict] = []
    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("model_name") != model_name:
            continue
        results.append({
            "path": str(entry),
            "model_name": meta["model_name"],
            "bars_seen": meta.get("bars_seen", 0),
            "saved_at": meta.get("saved_at", ""),
            "artifact_files": meta.get("artifact_files", []),
        })

    # Sort newest-first by saved_at (ISO string lexicographic order works fine)
    results.sort(key=lambda c: c["saved_at"], reverse=True)
    return results


def prune_old_checkpoints(
    model_name: str,
    checkpoint_dir: Path,
    keep_n: int = 3,
) -> int:
    """
    Delete all but the most recent `keep_n` checkpoints for `model_name`.

    Returns the number of checkpoint directories deleted.
    """
    checkpoints = list_checkpoints(model_name, Path(checkpoint_dir))
    to_delete = checkpoints[keep_n:]  # oldest entries (sorted newest-first)

    deleted = 0
    for ckpt in to_delete:
        path = Path(ckpt["path"])
        try:
            shutil.rmtree(path)
            deleted += 1
            log.debug(f"[Checkpoint] Pruned {path.name}")
        except OSError as e:
            log.warning(f"[Checkpoint] Failed to prune {path}: {e}")

    return deleted
