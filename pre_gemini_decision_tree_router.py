"""
Pre-LLM Decision Tree & Google Decision Model Router (Before Calling Gemini)
============================================================================
Production pattern used in KhronX, Venda Fácil CRM, and VOX:
Never let a raw user message hit a frontier generative LLM (Gemini 2.5 Pro/Flash)
without first passing through a deterministic Decision Tree / Gate Model.

Why we run a Decision Model BEFORE Gemini:
1. State & Guardrail Enforcement: Checks human-override flags ('somente vendedor',
   'ai_paused'), 24h window cutoffs, and open human tasks before spending a token.
2. Deterministic Fast-Path (0ms, $0): Handles known intents, opt-outs, button
   payloads, and regex-verifiable entities without invoking generative inference.
3. Constrained Branch Selection: When Gemini IS invoked, it does not "freely chat"—
   it is constrained to classify and fill parameters for a single valid node of
   the approved commercial Decision Tree (`responseMimeType="application/json"`).
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DecisionOutcome:
    branch_id: str
    used_gemini: bool
    allowed_to_act: bool
    next_crm_stage: str
    next_column_id: str
    response_text: str
    audit_reason: str


# Canonical Stage & Pipeline Mapping per Decision Branch
DECISION_TREE_BRANCH_MAP = {
    "ACCEPT_SLOT": ("Meeting Scheduled", "meeting_scheduled"),
    "REQUEST_ALT_SLOT": ("Qualification", "initial_contact"),
    "ASK_METHODOLOGY": ("Qualification", "initial_contact"),
    "ASK_PRICING": ("Qualification", "initial_contact"),
    "RAISE_OBJECTION": ("Qualification", "initial_contact"),
    "NEEDS_PARTNER_APPROVAL": ("Qualification", "initial_contact"),
    "HARD_OPT_OUT": ("Closed Lost", "lost_no_interest"),
    "ESCALATE_TO_HUMAN": ("Human Intervention", "human_handoff"),
}


class PreGeminiDecisionRouter:
    """
    Executes a 3-layer Decision Tree BEFORE calling Vertex AI Gemini:
      Layer 1: Operational State & Human-Handoff Gate (Deterministic Tree)
      Layer 2: Fast-Path Intent & Entity Classifier (Zero-Token Tree)
      Layer 3: Constrained Gemini Decision-Branch Evaluator (JSON Schema Locked)
    """

    def __init__(self, gemini_json_caller=None):
        self.gemini_json_caller = gemini_json_caller

    # -------------------------------------------------------------------------
    # LAYER 1: Operational State & Human-Handoff Decision Tree (Pre-LLM)
    # -------------------------------------------------------------------------
    def evaluate_operational_gates(self, contact: Dict[str, Any]) -> Optional[DecisionOutcome]:
        cadence = contact.get("cadence_state") or {}

        # Gate 1.1: Explicit Do-Not-Contact or Terminal Cadence State
        if contact.get("do_not_contact") or cadence.get("status") in (
            "lost_no_interest",
            "meeting_scheduled",
            "human_handoff",
        ):
            return DecisionOutcome(
                branch_id="BLOCKED_TERMINAL_STATE",
                used_gemini=False,
                allowed_to_act=False,
                next_crm_stage=contact.get("stage", ""),
                next_column_id=contact.get("column_id", ""),
                response_text="",
                audit_reason=f"Blocked by terminal state: {cadence.get('status')}",
            )

        # Gate 1.2: Chronological Interaction Command Audit ('somente vendedor' vs 'volte ia')
        paused_by_seller = False
        for inter in contact.get("interactions") or []:
            details = str(inter.get("details") or "").lower()
            if "somente vendedor" in details:
                paused_by_seller = True
            elif "volte ia" in details:
                paused_by_seller = False

        if paused_by_seller or contact.get("ai_paused"):
            return DecisionOutcome(
                branch_id="BLOCKED_HUMAN_OVERRIDE",
                used_gemini=False,
                allowed_to_act=False,
                next_crm_stage=contact.get("stage", ""),
                next_column_id=contact.get("column_id", ""),
                response_text="",
                audit_reason="Human operator lock active; Gemini invocation skipped.",
            )

        # Gate 1.3: Pending Human Task Check (AI only acts if task contains 'ia fazer')
        for task in contact.get("tasks") or []:
            if task.get("status") in ("todo", "pending"):
                blob = f"{task.get('title', '')} {task.get('description', '')}".lower()
                if "ia fazer" not in blob:
                    return DecisionOutcome(
                        branch_id="BLOCKED_PENDING_HUMAN_TASK",
                        used_gemini=False,
                        allowed_to_act=False,
                        next_crm_stage=contact.get("stage", ""),
                        next_column_id=contact.get("column_id", ""),
                        response_text="",
                        audit_reason="Open human task on card; waiting for human completion.",
                    )

        return None

    # -------------------------------------------------------------------------
    # LAYER 2: Fast-Path Deterministic Classifier (Pre-LLM)
    # -------------------------------------------------------------------------
    def evaluate_fast_path_tree(
        self, message: str, first_name: str, company_name: str, slot_a: str, slot_b: str
    ) -> Optional[DecisionOutcome]:
        clean = re.sub(r"[!?.]+$", "", (message or "").strip().lower()).strip()
        if not clean:
            return None

        # Fast-Path Branch A: Explicit Opt-Out / Stop
        if any(w in clean for w in ("nao tenho interesse", "não tenho interesse", "parar", "remover", "opt out")):
            stage, col = DECISION_TREE_BRANCH_MAP["HARD_OPT_OUT"]
            return DecisionOutcome(
                branch_id="HARD_OPT_OUT",
                used_gemini=False,
                allowed_to_act=True,
                next_crm_stage=stage,
                next_column_id=col,
                response_text=(
                    f"Understood, {first_name}. Thank you for letting me know—I won't follow up further. "
                    f"Wishing continued success to you and {company_name}."
                ),
                audit_reason="Matched deterministic opt-out rule (0 tokens spent).",
            )

        # Fast-Path Branch B: Direct Slot Confirmation ("sim", "combinado", "fechado", "ok")
        if clean in {"sim", "pode ser", "combinado", "fechado", "confirmo", "ok", "perfeito", "de acordo"}:
            stage, col = DECISION_TREE_BRANCH_MAP["ACCEPT_SLOT"]
            return DecisionOutcome(
                branch_id="ACCEPT_SLOT",
                used_gemini=False,
                allowed_to_act=True,
                next_crm_stage=stage,
                next_column_id=col,
                response_text=(
                    f"Perfect, {first_name}! I have locked in our session for {slot_a}. "
                    f"I will send the meeting link here shortly before we start to walk through "
                    f"the technical diagnostic for {company_name}."
                ),
                audit_reason="Matched deterministic slot confirmation (0 tokens spent).",
            )

        return None

    # -------------------------------------------------------------------------
    # LAYER 3: Constrained Gemini Decision-Branch Evaluator
    # -------------------------------------------------------------------------
    def route_and_execute(
        self,
        contact: Dict[str, Any],
        incoming_message: str,
        slot_a: str = "Thursday at 10:00",
        slot_b: str = "Friday at 15:30",
    ) -> DecisionOutcome:
        # Step 1: Run Operational Gate Tree
        gate_block = self.evaluate_operational_gates(contact)
        if gate_block is not None:
            return gate_block

        first_name = str(contact.get("name") or "there").split()[0].capitalize()
        company_name = str(contact.get("company_name") or contact.get("name") or "your company")

        # Step 2: Run Deterministic Fast-Path Tree
        fast_outcome = self.evaluate_fast_path_tree(
            incoming_message, first_name, company_name, slot_a, slot_b
        )
        if fast_outcome is not None:
            return fast_outcome

        # Step 3: Only invoke Gemini when contextual disambiguation is required,
        # and lock Gemini strictly to the Decision Tree branches via JSON schema.
        if self.gemini_json_caller:
            raw_json = self.gemini_json_caller(
                message=incoming_message,
                allowed_branches=list(DECISION_TREE_BRANCH_MAP.keys()),
                slot_a=slot_a,
                slot_b=slot_b,
            )
            branch = raw_json.get("branch_id", "ESCALATE_TO_HUMAN")
            if branch not in DECISION_TREE_BRANCH_MAP:
                branch = "ESCALATE_TO_HUMAN"
            stage, col = DECISION_TREE_BRANCH_MAP[branch]
            return DecisionOutcome(
                branch_id=branch,
                used_gemini=True,
                allowed_to_act=True,
                next_crm_stage=stage,
                next_column_id=col,
                response_text=raw_json.get("response_text", ""),
                audit_reason=f"Classified by Gemini within constrained Decision Tree branch: {branch}",
            )

        # Fail-closed fallback: escalate to human if Gemini is unreachable
        stage, col = DECISION_TREE_BRANCH_MAP["ESCALATE_TO_HUMAN"]
        return DecisionOutcome(
            branch_id="ESCALATE_TO_HUMAN",
            used_gemini=False,
            allowed_to_act=True,
            next_crm_stage=stage,
            next_column_id=col,
            response_text=(
                f"Got it, {first_name}! I am forwarding your question to a specialist on our team "
                f"so they can reply directly with the exact context."
            ),
            audit_reason="Fail-closed fallback to human specialist.",
        )
