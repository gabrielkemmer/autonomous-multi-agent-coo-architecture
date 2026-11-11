"""
Deterministic Multi-Step Agent Execution Loop (Reference Architecture)
======================================================================
Demonstrates how our production Autonomous COO & Operations Engine executes
multi-step reasoning with strict tool-call parsing, reply-context threading,
and pre-flight verification guards—without heavyweight framework bloat.
"""

import re
from typing import Any, Callable, Dict, List, Optional
from agent_memory_guard import LearnedMemoryStore, PreFlightToolGuard

TOOL_PATTERN = re.compile(r"\[\[TOOL:([A-Z_]+)\|([^\]]+)\]\]")


class DeterministicAgentRuntime:
    """
    Executes a bounded (max_steps=4) reasoning and tool-execution loop.
    Forces read-only verification before any outbound side-effect tool can run.
    """

    def __init__(
        self,
        llm_generate_fn: Callable[[str], str],
        memory: LearnedMemoryStore,
        guard: PreFlightToolGuard,
        outbound_sender_fn: Callable[[str, str], Dict[str, Any]],
        max_steps: int = 4,
    ):
        self.llm_generate = llm_generate_fn
        self.memory = memory
        self.guard = guard
        self.outbound_sender = outbound_sender_fn
        self.max_steps = max_steps

    def build_system_context(
        self,
        operator_message: str,
        reply_to_text: Optional[str] = None,
        reply_to_sender: Optional[str] = None,
    ) -> str:
        state = self.memory._state
        rules_block = "\n".join(f"- {r}" for r in state.get("rules", [])) or "None"
        contacts_block = "\n".join(
            f"- {alias}: {info['address']} ({info['role']})"
            for alias, info in state.get("contacts", {}).items()
        ) or "None"

        reply_context = ""
        if reply_to_text:
            sender_label = reply_to_sender or "Previous Message"
            reply_context = (
                f"\n[REPLYING TO PREVIOUS MESSAGE FROM {sender_label}]:\n"
                f"\"{reply_to_text.strip()}\"\n"
            )

        return (
            "SYSTEM DIRECTIVE: Deterministic Operations & COO Agent\n"
            "TEMPERATURE: 0.2 | FAIL-CLOSED SIDE EFFECTS: ENABLED\n\n"
            f"ACTIVE OPERATIONAL RULES:\n{rules_block}\n\n"
            f"VERIFIED CONTACTS:\n{contacts_block}\n"
            f"{reply_context}\n"
            f"OPERATOR DIRECTIVE:\n{operator_message}\n"
        )

    def run_turn(
        self,
        operator_message: str,
        reply_to_text: Optional[str] = None,
        reply_to_sender: Optional[str] = None,
    ) -> str:
        prompt = self.build_system_context(
            operator_message=operator_message,
            reply_to_text=reply_to_text,
            reply_to_sender=reply_to_sender,
        )

        for _step in range(self.max_steps):
            response = self.llm_generate(prompt)
            matches = TOOL_PATTERN.findall(response)
            if not matches:
                return response.strip()

            observations: List[str] = []
            for tool_name, raw_args in matches:
                args = [a.strip() for a in raw_args.split("|")]
                obs = self._dispatch_tool(tool_name, args)
                observations.append(f"[TOOL_RESULT:{tool_name}] -> {obs}")

            prompt += "\n" + response + "\n" + "\n".join(observations) + "\n"

        return "Max reasoning steps reached; halting for operator confirmation."

    def _dispatch_tool(self, tool_name: str, args: List[str]) -> str:
        if tool_name == "VERIFY_ENDPOINT" and len(args) >= 1:
            target = args[0]
            ok = self.guard.verify_target(target)
            return f"VERIFIED={ok} for target={target}"

        if tool_name == "SAVE_CONTACT" and len(args) >= 3:
            alias, address, role = args[0], args[1], args[2]
            self.memory.save_verified_contact(alias, address, role)
            return f"Contact '{alias}' persisted atomically."

        if tool_name == "SAVE_RULE" and len(args) >= 1:
            self.memory.save_rule(args[0])
            return "Operational rule persisted atomically."

        if tool_name == "SEND_MESSAGE" and len(args) >= 2:
            target, body = args[0], args[1]
            res = self.guard.execute_guarded_action(
                target, lambda t: self.outbound_sender(t, body)
            )
            return f"Dispatched successfully: {res}"

        return f"Rejected unknown or malformed tool call: {tool_name}"
