"""Direct contracts for the CodeEdit feature."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.code_edit.feature import CodeEditFeature


@pytest.fixture
def feature(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    agent = SimpleNamespace(features={})
    feat = CodeEditFeature(agent=agent, code_root=str(root))
    return feat, root


def test_resolve_path_rejects_escape(feature):
    feat, _ = feature

    with pytest.raises(ValueError, match="escapes code root"):
        feat._resolve_path("../outside.py")


@pytest.mark.asyncio
async def test_code_read_returns_file_contents(feature):
    feat, root = feature
    target = root / "sample.py"
    target.write_text("line1\nline2\n", encoding="utf-8")

    result = await feat.code_read("sample.py")

    assert result["success"] is True
    assert result["path"] == "sample.py"
    assert result["content"] == "line1\nline2\n"
    assert result["total_lines"] == 3


@pytest.mark.asyncio
async def test_code_search_limits_and_reports_matches(feature):
    feat, root = feature
    (root / "a.py").write_text("alpha\nbeta alpha\n", encoding="utf-8")
    (root / "b.py").write_text("gamma\n", encoding="utf-8")

    result = await feat.code_search("alpha")

    assert result["success"] is True
    assert result["total_matches"] == 2
    assert result["matches"][0]["file"] == "a.py"


@pytest.mark.asyncio
async def test_code_edit_requires_unique_match_before_approval(feature):
    feat, root = feature
    target = root / "sample.py"
    target.write_text("dup\ndup\n", encoding="utf-8")

    result = await feat.code_edit("sample.py", "dup", "new")

    assert result["success"] is False
    assert "must be unique" in result["error"]


@pytest.mark.asyncio
async def test_code_edit_applies_change_after_approval(feature):
    feat, root = feature
    target = root / "sample.py"
    target.write_text("old\n", encoding="utf-8")
    feat._request_approval = lambda *args, **kwargs: __import__("asyncio").sleep(0, result=True)

    result = await feat.code_edit("sample.py", "old", "new", description="replace text")

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert result["description"] == "replace text"
