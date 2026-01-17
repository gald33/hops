import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple


REGISTRY_DIR = os.path.join("runs", "_registry")
REGISTRY_PATH = os.path.join(REGISTRY_DIR, "hops_registry.jsonl")
PACKAGE_INDEX_PATH = os.path.join(REGISTRY_DIR, "package_index.jsonl")
LATEST_PACKAGE_PATH = os.path.join(REGISTRY_DIR, "latest_package.txt")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_registry_dir() -> None:
    os.makedirs(REGISTRY_DIR, exist_ok=True)


def append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_latest_package(package_name: str, package_dir: str) -> None:
    with open(LATEST_PACKAGE_PATH, "w", encoding="utf-8") as f:
        f.write(f"{package_name}\t{package_dir}\t{utc_now_iso()}\n")


def git_info(repo_root: str) -> Tuple[str, Optional[bool]]:
    commit = "unknown"
    dirty = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip() or "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return commit, dirty
    return commit, dirty


def write_registry_records(records: Iterable[Dict]) -> None:
    ensure_registry_dir()
    for record in records:
        append_jsonl(REGISTRY_PATH, record)


def write_package_record(record: Dict) -> None:
    ensure_registry_dir()
    append_jsonl(PACKAGE_INDEX_PATH, record)
    write_latest_package(record.get("package_name", "unknown"), record.get("package_dir", ""))


def write_registry_csv(records: Iterable[Dict], path: str) -> None:
    rows = list(records)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
