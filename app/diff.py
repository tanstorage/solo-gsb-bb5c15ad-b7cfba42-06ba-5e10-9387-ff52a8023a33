"""两个 JSON 快照之间的递归结构差异（add / remove / replace）。

规则概述
--------
* 两侧都是对象时按键递归：
  - 键只存在于旧侧 → ``remove``（``before`` 为旧值，``after`` 给 ``null``）；
  - 键只存在于新侧 → ``add``（``before`` 给 ``null``，``after`` 为新值）；
  - 两侧都有 → 递归比较子值；
* 两侧值相等（含数组逐元素相等、真实 ``null``）→ 无变更；
* 其余情况（标量不同、类型变化、数组不同）→ 在该路径整体 ``replace``；
  数组作为整体值比较，不按下标展开——与 RFC 7386 的整体替换语义一致；
* 键缺失用 ``missing`` 字段标注缺失侧（``"before"`` / ``"after"``），
  对应侧的值为 ``null``，以区别于真实存储的 JSON ``null``
  （与三方合并冲突响应的 ``missing`` 约定相同）。

变更项按 RFC 6901 路径（键序列）稳定排序：父路径先于子路径，同级按键名
字典序；路径渲染复用 :func:`app.merge.json_pointer`，根路径为 ``""``，
键中的 ``~`` / ``/`` 转义为 ``~0`` / ``~1``。

算法本身不依赖 FastAPI / SQLite，可独立单测。
"""
from __future__ import annotations

from typing import Any

from .merge import json_pointer

#: 变更操作类型
OP_ADD = "add"
OP_REMOVE = "remove"
OP_REPLACE = "replace"


def diff_documents(old: Any, new: Any) -> list[dict[str, Any]]:
    """递归比较 *old* 与 *new*，返回按 RFC 6901 路径稳定排序的变更列表。

    每项形如::

        {"op": "add" | "remove" | "replace",
         "path": "/a/b",          # RFC 6901 JSON Pointer，根为 ""
         "before": <旧值>,         # 键缺失时为 None
         "after":  <新值>,         # 键缺失时为 None
         "missing": ["before"]}   # 标注哪一侧是“键缺失”而非真实 null

    两侧完全相等时返回空列表。纯函数，不修改入参。
    """
    # 元素为 (键序列, 变更项)；键序列用于排序，响应里只保留变更项
    changes: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def emit(op: str, path: list[str], before: Any, after: Any, missing: list[str]) -> None:
        changes.append(
            (
                tuple(path),
                {
                    "op": op,
                    "path": json_pointer(path),
                    "before": before,
                    "after": after,
                    "missing": missing,
                },
            )
        )

    def node(before: Any, after: Any, path: list[str]) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in set(before) | set(after):
                if key not in after:
                    # 键被删除：after 给 null 并在 missing 中标注
                    emit(OP_REMOVE, path + [key], before[key], None, ["after"])
                elif key not in before:
                    # 新增键：before 给 null 并在 missing 中标注
                    emit(OP_ADD, path + [key], None, after[key], ["before"])
                else:
                    node(before[key], after[key], path + [key])
            return
        # 标量 / 数组 / null：整体比较（数组不按下标展开）
        if before == after:
            return
        emit(OP_REPLACE, path, before, after, [])

    node(old, new, [])
    # 稳定排序：按路径键序列（父先于子，同级按键名字典序），结果确定唯一
    changes.sort(key=lambda item: item[0])
    return [change for _, change in changes]


def diff_counts(changes: list[dict[str, Any]]) -> dict[str, int]:
    """按操作类型统计变更数，附 ``total``。"""
    counts = {OP_ADD: 0, OP_REMOVE: 0, OP_REPLACE: 0}
    for change in changes:
        counts[change["op"]] += 1
    counts["total"] = len(changes)
    return counts
