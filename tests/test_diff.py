"""GET /documents/{id}/diff 修订差异审计测试。

前半部分为 diff 算法单元测试（不依赖 FastAPI），
后半部分为端到端集成测试（每个用例使用独立临时 SQLite 文件）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.diff import diff_counts, diff_documents
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, content=None, doc_id="cfg", author=None):
    body = {"id": doc_id}
    if content is not None:
        body["content"] = content
    if author is not None:
        body["author"] = author
    r = client.post("/documents", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _patch(client, doc_id, base_revision, request_id, patch, author=None):
    body = {"base_revision": base_revision, "request_id": request_id, "patch": patch}
    if author is not None:
        body["author"] = author
    return client.patch(f"/documents/{doc_id}", json=body)


def _diff(client, doc_id, from_revision, to_revision):
    return client.get(
        f"/documents/{doc_id}/diff",
        params={"from_revision": from_revision, "to_revision": to_revision},
    )


# ---------------- diff_documents 算法单元测试 ----------------


def test_diff_empty_for_equal_documents():
    assert diff_documents({"a": 1, "b": {"x": [1, 2]}}, {"a": 1, "b": {"x": [1, 2]}}) == []
    assert diff_documents(None, None) == []
    assert diff_documents([1, 2], [1, 2]) == []


def test_diff_replace_scalar_at_root():
    assert diff_documents({"a": 1}, {"a": 2}) == [
        {"op": "replace", "path": "/a", "before": 1, "after": 2, "missing": []}
    ]
    # 根整体替换（类型变化）
    assert diff_documents({"a": 1}, [1, 2]) == [
        {"op": "replace", "path": "", "before": {"a": 1}, "after": [1, 2], "missing": []}
    ]


def test_diff_add_and_remove_keys():
    changes = diff_documents({"a": 1, "gone": {"n": 1}}, {"a": 1, "new": 3})
    assert changes == [
        {"op": "remove", "path": "/gone", "before": {"n": 1}, "after": None,
         "missing": ["after"]},
        {"op": "add", "path": "/new", "before": None, "after": 3,
         "missing": ["before"]},
    ]


def test_diff_recurses_nested_objects():
    changes = diff_documents(
        {"c": {"x": 1, "y": {"z": 1}}},
        {"c": {"x": 1, "y": {"z": 2, "w": True}}},
    )
    assert changes == [
        {"op": "add", "path": "/c/y/w", "before": None, "after": True,
         "missing": ["before"]},
        {"op": "replace", "path": "/c/y/z", "before": 1, "after": 2, "missing": []},
    ]


def test_diff_real_null_vs_missing():
    # 键真实存为 null → replace，missing 为空
    assert diff_documents({"a": None}, {"a": 1}) == [
        {"op": "replace", "path": "/a", "before": None, "after": 1, "missing": []}
    ]
    # 新增键的值恰好是真实 null：before 缺失（missing 标注），after 是真实 null
    assert diff_documents({}, {"a": None}) == [
        {"op": "add", "path": "/a", "before": None, "after": None,
         "missing": ["before"]}
    ]
    # 删除值恰好是真实 null 的键：before 是真实 null，after 缺失
    assert diff_documents({"a": None}, {}) == [
        {"op": "remove", "path": "/a", "before": None, "after": None,
         "missing": ["after"]}
    ]


def test_diff_arrays_compared_as_whole():
    # 数组不按下标展开：整体一次 replace
    assert diff_documents({"l": [1, 2]}, {"l": [1, 2, 3]}) == [
        {"op": "replace", "path": "/l", "before": [1, 2], "after": [1, 2, 3],
         "missing": []}
    ]
    # 对象 → 数组（类型变化）同样整体 replace
    assert diff_documents({"l": {"0": 1}}, {"l": [1]}) == [
        {"op": "replace", "path": "/l", "before": {"0": 1}, "after": [1],
         "missing": []}
    ]


def test_diff_paths_sorted_by_key_sequence():
    changes = diff_documents(
        {"b": 1, "a": {"z": 1, "y": 1}, "c": 1},
        {"b": 2, "a": {"z": 2, "y": 2}, "c": 2},
    )
    assert [c["path"] for c in changes] == ["/a/y", "/a/z", "/b", "/c"]


def test_diff_pointer_escapes_special_keys():
    changes = diff_documents({"a/b": 1, "t~x": 1}, {"a/b": 2, "t~x": 2})
    assert [c["path"] for c in changes] == ["/a~1b", "/t~0x"]


def test_diff_does_not_mutate_inputs():
    old = {"a": {"b": 1}}
    new = {"a": {"b": 2, "c": 3}}
    diff_documents(old, new)
    assert old == {"a": {"b": 1}}
    assert new == {"a": {"b": 2, "c": 3}}


def test_diff_counts_by_operation():
    changes = diff_documents(
        {"add_me_parent": {}, "del": 1, "same": 1, "chg": 1},
        {"add_me_parent": {"x": 1}, "same": 1, "chg": 2},
    )
    assert diff_counts(changes) == {"add": 1, "remove": 1, "replace": 1, "total": 3}
    assert diff_counts([]) == {"add": 0, "remove": 0, "replace": 0, "total": 0}


# ---------------- API 集成测试 ----------------


def test_diff_between_two_revisions(client):
    _create(client, {"a": 1, "b": 2, "c": {"x": 1}})
    assert _patch(client, "cfg", 1, "r2", {"a": 100}).status_code == 200
    assert _patch(client, "cfg", 2, "r3", {"c": {"y": 2}, "b": None}).status_code == 200

    r = _diff(client, "cfg", 1, 3)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_id"] == "cfg"
    assert body["from_revision"] == 1
    assert body["to_revision"] == 3
    assert body["changes"] == [
        {"op": "replace", "path": "/a", "before": 1, "after": 100, "missing": []},
        {"op": "remove", "path": "/b", "before": 2, "after": None,
         "missing": ["after"]},
        {"op": "add", "path": "/c/y", "before": None, "after": 2,
         "missing": ["before"]},
    ]
    assert body["counts"] == {"add": 1, "remove": 1, "replace": 1, "total": 3}


def test_diff_reverse_direction(client):
    _create(client, {"a": 1})
    assert _patch(client, "cfg", 1, "r2", {"a": 2, "n": 5}).status_code == 200
    body = _diff(client, "cfg", 2, 1).json()
    assert body["changes"] == [
        {"op": "replace", "path": "/a", "before": 2, "after": 1, "missing": []},
        {"op": "remove", "path": "/n", "before": 5, "after": None,
         "missing": ["after"]},
    ]
    assert body["counts"] == {"add": 0, "remove": 1, "replace": 1, "total": 2}


def test_diff_same_revision_returns_empty_changes(client):
    _create(client, {"a": 1, "l": [1, 2]})
    r = _diff(client, "cfg", 1, 1)
    assert r.status_code == 200
    body = r.json()
    assert body["changes"] == []
    assert body["counts"] == {"add": 0, "remove": 0, "replace": 0, "total": 0}


def test_diff_array_replaced_as_whole(client):
    _create(client, {"l": [1, 2], "keep": True})
    assert _patch(client, "cfg", 1, "r2", {"l": [1, 2, 3]}).status_code == 200
    body = _diff(client, "cfg", 1, 2).json()
    assert body["changes"] == [
        {"op": "replace", "path": "/l", "before": [1, 2], "after": [1, 2, 3],
         "missing": []}
    ]
    assert body["counts"]["replace"] == 1


def test_diff_distinguishes_real_null_from_missing(client):
    # 创建快照可原样存储真实 null；Merge Patch 中的 null 一律表示删除，
    # 因此反向比较（rev2 → rev1）才能看到“新增一个真实 null 键”。
    _create(client, {"a": None, "b": 1, "n": None})
    # rev2：a 改成 2、删除 b 与 n
    assert _patch(client, "cfg", 1, "r2", {"a": 2, "b": None, "n": None}).status_code == 200
    body = _diff(client, "cfg", 2, 1).json()
    by_path = {c["path"]: c for c in body["changes"]}
    # a：2 → 真实 null，是 replace，两侧都不缺失
    assert by_path["/a"] == {"op": "replace", "path": "/a", "before": 2,
                             "after": None, "missing": []}
    # b：新增普通值，before 的 null 是“缺失”
    assert by_path["/b"] == {"op": "add", "path": "/b", "before": None,
                             "after": 1, "missing": ["before"]}
    # n：新增且值恰为真实 null——after 是真实 null，before 的 null 是“缺失”
    assert by_path["/n"] == {"op": "add", "path": "/n", "before": None,
                             "after": None, "missing": ["before"]}
    # 正向（rev1 → rev2）：n 被删除，before 是真实 null，after 是“缺失”
    forward = {c["path"]: c for c in _diff(client, "cfg", 1, 2).json()["changes"]}
    assert forward["/n"] == {"op": "remove", "path": "/n", "before": None,
                             "after": None, "missing": ["after"]}


def test_diff_non_object_snapshots(client):
    _create(client, {"a": 1})
    assert _patch(client, "cfg", 1, "r2", [1, 2, 3]).status_code == 200
    body = _diff(client, "cfg", 1, 2).json()
    assert body["changes"] == [
        {"op": "replace", "path": "", "before": {"a": 1}, "after": [1, 2, 3],
         "missing": []}
    ]


def test_diff_missing_document_returns_404(client):
    r = _diff(client, "ghost", 1, 2)
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "not_found"


def test_diff_missing_revision_returns_404_and_names_it(client):
    _create(client, {"a": 1})
    assert _patch(client, "cfg", 1, "r2", {"a": 2}).status_code == 200

    # 仅终点缺失
    r = _diff(client, "cfg", 1, 99)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "revision_not_found"
    assert detail["document_id"] == "cfg"
    assert detail["from_revision"] == 1
    assert detail["to_revision"] == 99
    assert detail["missing_revisions"] == [99]

    # 仅起点缺失
    r = _diff(client, "cfg", 98, 2)
    assert r.status_code == 404
    assert r.json()["detail"]["missing_revisions"] == [98]

    # 两侧都缺失
    r = _diff(client, "cfg", 98, 99)
    assert r.status_code == 404
    assert r.json()["detail"]["missing_revisions"] == [98, 99]


def test_diff_params_required_and_validated(client):
    _create(client, {"a": 1})
    # 缺参数 → 422
    assert client.get("/documents/cfg/diff").status_code == 422
    assert client.get("/documents/cfg/diff?from_revision=1").status_code == 422
    # 修订号必须 >= 1
    assert _diff(client, "cfg", 0, 1).status_code == 422
    assert _diff(client, "cfg", 1, -1).status_code == 422


def test_diff_is_read_only(client):
    _create(client, {"a": 1})
    assert _patch(client, "cfg", 1, "r2", {"a": 2}).status_code == 200
    assert _diff(client, "cfg", 1, 2).status_code == 200
    # 差异查询不产生任何修订
    doc = client.get("/documents/cfg").json()
    assert doc["revision"] == 2
    hist = client.get("/documents/cfg/revisions").json()["items"]
    assert [h["revision"] for h in hist] == [1, 2]
