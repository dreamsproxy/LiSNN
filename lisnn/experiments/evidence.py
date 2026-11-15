"""Append-only accounting for planned experiments, without learning claims.

This stores JSON metadata and hashes caller-supplied artifacts. It never runs
an experiment or deserializes a neural snapshot. Failed runs remain first-class
outcomes; completing a plan says nothing about whether adaptation occurred.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def _json_bytes(value):
    """Canonical, finite JSON; reject unsupported objects rather than coerce."""
    def validate(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError("evidence metadata must contain only JSON values")
    validate(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected an ordinary JSON file: {path.name}")
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                       parse_constant=reject_constant)
    _json_bytes(value)  # Also catches numeric overflow such as 1e10000.
    return value


def _write_exclusive(path, value):
    """Publish a complete JSON file atomically, without replacing any file."""
    payload = _json_bytes(value)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".receipt-",
                                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic creation, unlike replace().
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_ids(protocol):
    if not isinstance(protocol, dict):
        raise TypeError("protocol must be a JSON object")
    ids = protocol.get("run_ids")
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(item, str) or not _RUN_ID.fullmatch(item) for item in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError("protocol run_ids must be a nonempty list of unique safe identifiers")
    return ids


def _metrics(value, *, required):
    if value is None:
        value = {}
    if not isinstance(value, dict) or (required and not value):
        raise ValueError("completed runs need a nonempty numeric metrics object")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError("metric names must be nonempty strings")
    for item in value.values():
        if (isinstance(item, bool) or not isinstance(item, (int, float)) or
                (isinstance(item, float) and not math.isfinite(item))):
            raise ValueError("metrics must contain finite numbers, excluding booleans")
    return value.copy()


class EvidencePackage:
    """A frozen protocol plus immutable success/failure receipts.

    ``create`` refuses an existing directory. Callers write raw observations,
    schedules and safe-format snapshots beneath this directory, then ``record``
    hashes those artifacts. A receipt cannot be replaced, including failed ones.
    New attempts need separately declared run IDs in a new protocol/package.
    """

    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, path, protocol, provenance):
        ids = list(_run_ids(protocol))
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("provenance must be a nonempty JSON object")
        # Validate everything before reserving the output directory.
        protocol_payload = _json_bytes(protocol)
        manifest = {
            "schema_version": 1,
            "protocol_sha256": sha256(protocol_payload).hexdigest(),
            "run_ids": ids,
            "provenance": provenance,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["manifest_sha256"] = sha256(_json_bytes(manifest)).hexdigest()
        _json_bytes(manifest)
        package = cls(path)
        package.path.mkdir(parents=True, exist_ok=False)
        (package.path / "runs").mkdir()
        _write_exclusive(package.path / "protocol.json", protocol)
        _write_exclusive(package.path / "manifest.json", manifest)
        return cls.open(package.path)

    @classmethod
    def open(cls, path):
        package = cls(path)
        package._verified()
        return package

    def _artifact(self, relative_path):
        if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
            raise ValueError("artifact path must be a relative POSIX path")
        relative = PurePosixPath(relative_path)
        if (not relative.parts or relative.is_absolute() or
                any(part in (".", "..") for part in relative.parts)):
            raise ValueError("artifact path must stay inside the package")
        if relative.as_posix() != relative_path:
            raise ValueError("artifact paths must use their normalized relative spelling")
        if relative.parts[0] in ("runs", "protocol.json", "manifest.json"):
            raise ValueError("raw artifacts must be separate from package metadata")
        candidate = self.path
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("artifact paths cannot contain symlinks")
        if not candidate.is_file():
            raise ValueError(f"artifact does not exist: {relative_path}")
        digest = sha256()
        size = 0
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return {"path": relative_path, "sha256": digest.hexdigest(), "size_bytes": size}

    def _receipt(self, receipt, expected_id, protocol_hash):
        if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1 or
                receipt.get("run_id") != expected_id or
                receipt.get("protocol_sha256") != protocol_hash):
            raise ValueError("run receipt does not match its frozen protocol")
        content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != sha256(_json_bytes(content)).hexdigest():
            raise ValueError("run receipt integrity check failed")
        status = receipt.get("status")
        if status not in ("completed", "failed"):
            raise ValueError("run status must be completed or failed")
        _metrics(receipt.get("metrics"), required=status == "completed")
        error = receipt.get("error")
        if status == "failed" and (not isinstance(error, str) or not error.strip()):
            raise ValueError("failed runs need a nonempty error description")
        if status == "completed" and error is not None:
            raise ValueError("completed runs cannot contain an error")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("receipt artifacts must be an object")
        for name, metadata in artifacts.items():
            if not isinstance(name, str) or not name or not isinstance(metadata, dict):
                raise ValueError("artifact labels and metadata are invalid")
            if self._artifact(metadata.get("path")) != metadata:
                raise ValueError(f"artifact integrity check failed: {name}")
        return receipt

    def _verified(self):
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("evidence package must be an ordinary directory")
        protocol = _read_json(self.path / "protocol.json")
        ids = _run_ids(protocol)
        manifest = _read_json(self.path / "manifest.json")
        digest = sha256(_json_bytes(protocol)).hexdigest()
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
                manifest.get("protocol_sha256") != digest or manifest.get("run_ids") != ids or
                not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]):
            raise ValueError("manifest does not match the frozen protocol")
        manifest_content = {key: value for key, value in manifest.items()
                            if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != sha256(_json_bytes(manifest_content)).hexdigest():
            raise ValueError("manifest integrity check failed")
        runs = self.path / "runs"
        if runs.is_symlink() or not runs.is_dir():
            raise ValueError("run receipts must be an ordinary directory")
        receipts = {}
        for entry in runs.iterdir():
            if entry.name.startswith(".receipt-") and not entry.is_symlink():
                continue  # An interrupted atomic write is not a recorded run.
            if entry.suffix != ".json" or entry.stem not in ids:
                raise ValueError(f"unplanned run receipt: {entry.name}")
            receipts[entry.stem] = self._receipt(_read_json(entry), entry.stem, digest)
        return protocol, manifest, receipts

    def record(self, run_id, *, status, metrics=None, artifacts=None, error=None):
        """Record a terminal result and hash existing artifacts, once only."""
        protocol, manifest, receipts = self._verified()
        if run_id not in protocol["run_ids"]:
            raise ValueError("run_id is not in the frozen protocol")
        if run_id in receipts:
            raise FileExistsError(f"run already recorded: {run_id}")
        if status not in ("completed", "failed"):
            raise ValueError("run status must be completed or failed")
        if artifacts is None:
            artifacts = {}
        if not isinstance(artifacts, dict):
            raise ValueError("artifacts must map labels to relative file paths")
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "protocol_sha256": manifest["protocol_sha256"],
            "status": status,
            "metrics": _metrics(metrics, required=status == "completed"),
            "artifacts": {label: self._artifact(path) for label, path in artifacts.items()},
            "error": error,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["receipt_sha256"] = sha256(_json_bytes(receipt)).hexdigest()
        self._receipt(receipt, run_id, manifest["protocol_sha256"])
        _write_exclusive(self.path / "runs" / f"{run_id}.json", receipt)
        return receipt

    def summary(self):
        """List every planned outcome; completion is not evidence of learning."""
        protocol, manifest, receipts = self._verified()
        grouped = {"pending": [], "completed": [], "failed": []}
        for run_id in protocol["run_ids"]:
            status = receipts[run_id]["status"] if run_id in receipts else "pending"
            grouped[status].append(run_id)
        return {
            "protocol_sha256": manifest["protocol_sha256"],
            "planned_runs": len(protocol["run_ids"]),
            **grouped,
            "execution_complete": not grouped["pending"],
        }
