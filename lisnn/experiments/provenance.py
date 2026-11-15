"""Capture source and environment identities without serializing secrets."""

from hashlib import sha256
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from lisnn.experiments.protocol import canonical_json
from lisnn.synapses import get_topology_validation


def capture_provenance(repository_root):
    """Hash tracked and nonignored source, including uncommitted edits.

    No environment dump, credentials, Git configuration or pickle is read.
    The package output should be under ignored logs/ so it cannot hash itself.
    """
    root = Path(repository_root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    revision = git("rev-parse", "HEAD").decode().strip()
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    files = []
    for name in sorted(set(names) - {""}):
        path = root / name
        if path.is_symlink():
            payload = os.readlink(path).encode()
            kind = "symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = "file"
        else:
            files.append({"path": name, "kind": "deleted"})
            continue
        files.append({"path": name, "kind": kind, "size_bytes": len(payload),
                      "sha256": sha256(payload).hexdigest()})
    return {
        "git_revision": revision,
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=normal").strip()),
        "source_sha256": sha256(canonical_json(files).encode()).hexdigest(),
        "source_files": files,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "platform": platform.platform(),
        "backend": "numpy",
        "topology_validation": get_topology_validation(),
    }
