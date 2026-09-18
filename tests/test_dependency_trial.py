import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_dependency_trial import verify_sources


def test_trial_requires_exact_pinned_source_and_rejects_path_escape(tmp_path):
    relative = "src/core/query_loop/law_search/nodes.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("# test source\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    topology = {"application_source_sha256": {relative: digest}}
    verify_sources(tmp_path, topology, digest)
    with pytest.raises(ValueError, match="must pin"):
        verify_sources(tmp_path, topology, "0" * 64)
    topology["application_source_sha256"]["../outside"] = "0" * 64
    with pytest.raises(ValueError, match="outside application"):
        verify_sources(tmp_path, topology, digest)
    del topology["application_source_sha256"]["../outside"]
    path.write_text("# changed source\n")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_sources(tmp_path, topology, digest)
