import json

import torch

from sceneshifter.parity import trace


def test_trace_filters_and_hashes_bfloat16(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENESHIFTER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SCENESHIFTER_TRACE_STEPS", "2")
    monkeypatch.setenv("SCENESHIFTER_TRACE_LAYERS", "7")
    monkeypatch.setattr(trace, "_TRACER", None)
    monkeypatch.setattr(trace, "_TRACER_DIR", None)

    value = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 2, 2)
    trace.trace_tensor("ignored", value, step=1, layer=7)
    trace.trace_tensor("kept", value, step=2, layer=7, pass_type="cond")

    rows = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "kept"
    assert rows[0]["shape"] == [1, 2, 2, 2]
    assert rows[0]["dtype"] == "torch.bfloat16"
    assert len(rows[0]["sha256"]) == 64
