"""Cycle-safe, monotonic search-tree and transposition handling."""
from __future__ import annotations

import re
from typing import Any


def install_search_closure() -> None:
    from . import search_engine as search

    def next_node_id(tree: dict[str, Any]) -> str:
        seq = tree.get("next_node_seq")
        if not isinstance(seq, int) or seq < 1:
            existing: list[int] = []
            for node_id in tree.get("nodes", {}):
                match = re.fullmatch(r"n(\d+)", str(node_id))
                if match:
                    existing.append(int(match.group(1)))
            seq = max(existing, default=0) + 1
        tree["next_node_seq"] = seq + 1
        return f"n{seq}"

    def reachable(nodes: dict[str, Any], start: str, target: str) -> bool:
        stack = [start]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited or current not in nodes:
                continue
            visited.add(current)
            stack.extend(
                child for child in nodes[current].get("children", [])
                if isinstance(child, str)
            )
        return False

    def new_node(tree: dict[str, Any], plan_text: str, parent: str | None,
                 depth: int, note: str | None, rollout: dict[str, Any]) -> str:
        nodes = tree.setdefault("nodes", {})
        transposition = tree.setdefault("transposition", {})
        digest = search._hash(plan_text)
        existing = transposition.get(digest)
        if existing in nodes:
            nodes[existing]["visits"] = int(nodes[existing].get("visits", 0)) + 1
            if (
                parent
                and parent in nodes
                and existing != parent
                and not reachable(nodes, existing, parent)
                and existing not in nodes[parent].setdefault("children", [])
            ):
                nodes[parent]["children"].append(existing)
            return existing

        node_id = next_node_id(tree)
        nodes[node_id] = {
            "id": node_id,
            "parent": parent,
            "depth": depth,
            "note": note,
            "plan_text": plan_text,
            "score": rollout["score"],
            "value": rollout["value"],
            "q": rollout["value"],
            "visits": 1,
            "verify_ok": rollout["verify_ok"],
            "sim_ok": rollout["sim_ok"],
            "critiques": rollout["critiques"],
            "children": [],
        }
        transposition[digest] = node_id
        if parent:
            if parent not in nodes:
                raise KeyError(f"parent node {parent!r} does not exist")
            nodes[parent].setdefault("children", []).append(node_id)
        return node_id

    def select(tree: dict[str, Any], exploration: float, cost_penalty: float) -> str:
        nodes = tree.get("nodes", {})
        current = tree.get("root")
        if current not in nodes:
            raise RuntimeError("search tree has no valid root")
        visited: set[str] = set()
        while True:
            if current in visited:
                raise RuntimeError("search tree cycle detected during selection")
            visited.add(current)
            children = [
                child for child in nodes[current].get("children", [])
                if child in nodes
            ]
            if not children:
                return current
            current = max(
                children,
                key=lambda child: search._ucb(
                    tree, nodes[child], exploration, cost_penalty
                ),
            )

    def backprop(tree: dict[str, Any], node_id: str, value: float) -> None:
        nodes = tree.get("nodes", {})
        visited: set[str] = set()
        current: str | None = node_id
        while current:
            if current in visited:
                raise RuntimeError("search tree parent cycle detected during backpropagation")
            if current not in nodes:
                raise RuntimeError(f"search tree references missing parent node {current!r}")
            visited.add(current)
            node = nodes[current]
            visits = int(node.get("visits", 0)) + 1
            node["visits"] = visits
            node["q"] = (float(node.get("q", 0.0)) * (visits - 1) + value) / visits
            current = node.get("parent")
        if nodes:
            best = max(nodes.values(), key=lambda item: (item["value"], item["q"]))
            tree["best_node"] = best["id"]
            tree["best_value"] = best["value"]

    def prune(tree: dict[str, Any], margin: float) -> None:
        nodes = tree.get("nodes", {})
        if not nodes:
            return
        best_q = max(float(node.get("q", 0.0)) for node in nodes.values())
        removed: list[str] = []
        for node_id, node in list(nodes.items()):
            if (
                node_id != tree.get("root")
                and int(node.get("visits", 0)) > 1
                and not node.get("children")
                and float(node.get("q", 0.0)) < best_q - margin
            ):
                removed.append(node_id)
                del nodes[node_id]
        if not removed:
            return
        removed_set = set(removed)
        tree.setdefault("pruned", []).extend(removed)
        for node in nodes.values():
            node["children"] = [
                child for child in node.get("children", [])
                if child not in removed_set and child in nodes
            ]
        transposition = tree.setdefault("transposition", {})
        for digest, node_id in list(transposition.items()):
            if node_id in removed_set or node_id not in nodes:
                del transposition[digest]
        if tree.get("best_node") not in nodes and nodes:
            best = max(nodes.values(), key=lambda item: (item["value"], item["q"]))
            tree["best_node"] = best["id"]
            tree["best_value"] = best["value"]

    search._new_node = new_node
    search._select = select
    search._backprop = backprop
    search._prune = prune
