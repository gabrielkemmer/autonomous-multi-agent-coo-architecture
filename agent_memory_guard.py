"""
Reference implementation of Persistent Learned Memory and Pre-Flight Tool
Verification Loop for Autonomous Operational Agents.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional


class LearnedMemoryStore:
    """
    Thread-safe, atomic JSON-backed persistent memory for operational rules
    and verified entity mappings learned from executive corrections.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._state: Dict[str, Any] = {"rules": [], "contacts": {}, "audit_log": []}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._state = json.load(f)

    def _save(self) -> None:
        tmp_path = f"{self.file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.file_path)

    def save_rule(self, rule_text: str) -> None:
        clean = rule_text.strip()
        if clean and clean not in self._state["rules"]:
            self._state["rules"].append(clean)
            self._save()

    def save_verified_contact(self, alias: str, address: str, role: str) -> None:
        self._state["contacts"][alias.lower().strip()] = {
            "address": address.strip(),
            "role": role.strip(),
            "updated_at": int(time.time()),
        }
        self._save()

    def record_side_effect(self, tool_name: str, target: str, status: str) -> None:
        self._state["audit_log"].append({
            "tool": tool_name,
            "target": target,
            "status": status,
            "timestamp": int(time.time()),
        })
        self._save()


class PreFlightToolGuard:
    """
    Enforces that a target identifier passes a read-only verification check
    before any mutating or outbound communication tool is permitted to execute.
    """

    def __init__(self, verifier_fn: Callable[[str], bool], memory: LearnedMemoryStore):
        self.verifier_fn = verifier_fn
        self.memory = memory
        self._verified_targets: set[str] = set()

    def verify_target(self, target_id: str) -> bool:
        ok = bool(self.verifier_fn(target_id))
        if ok:
            self._verified_targets.add(target_id)
        return ok

    def execute_guarded_action(self, target_id: str, action_fn: Callable[[str], Any]) -> Any:
        if target_id not in self._verified_targets:
            if not self.verify_target(target_id):
                self.memory.record_side_effect("guarded_action", target_id, "blocked_unverified")
                raise PermissionError(f"Pre-flight verification failed for target: {target_id}")
        result = action_fn(target_id)
        self.memory.record_side_effect("guarded_action", target_id, "executed")
        return result
