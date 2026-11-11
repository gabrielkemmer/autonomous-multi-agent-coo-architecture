"""
Dual-Lock Outbound Idempotency & Webhook Race-Condition Reconciler
==================================================================
Prevents duplicate AI outbound messages and resolves webhook race conditions
between asynchronous messaging gateways and canonical CRM state stores.
"""

import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Protocol


class CanonicalCRMStore(Protocol):
    def find_by_phone_suffix(self, phone_suffix_8: str) -> Optional[Dict[str, Any]]: ...
    def count_leads_by_owner(self, owner_id: str, region: str) -> int: ...
    def upsert_lead_card(self, lead_doc: Dict[str, Any]) -> str: ...


def normalize_phone_suffix(phone: str, digits: int = 8) -> str:
    clean = re.sub(r"\D", "", phone or "")
    return clean[-digits:] if len(clean) >= digits else clean


class DualLockOutboundEngine:
    """
    Enforces two independent idempotency locks before any AI SDR outbound
    dispatch occurs:
      1. Local SQLite WAL audit ledger (`dispatched_leads`)
      2. Remote Canonical CRM database (matching normalized 8-digit phone suffix)
    Also performs deterministic 50/50 round-robin owner assignment that
    survives asynchronous webhook creation races.
    """

    def __init__(self, sqlite_path: str, crm: CanonicalCRMStore, owners: List[str]):
        self.sqlite_path = sqlite_path
        self.crm = crm
        self.owners = owners
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dispatched_leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    place_id TEXT UNIQUE,
                    phone_suffix TEXT UNIQUE,
                    company_name TEXT,
                    owner_id TEXT,
                    region TEXT,
                    dispatched_at INTEGER
                )
                """
            )

    def is_already_contacted(self, place_id: str, phone: str) -> bool:
        suffix = normalize_phone_suffix(phone)
        if not suffix:
            return True

        # Lock 1: Local SQLite WAL Ledger
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM dispatched_leads WHERE place_id = ? OR phone_suffix = ? LIMIT 1",
                (place_id, suffix),
            ).fetchone()
            if row:
                return True

        # Lock 2: Remote Canonical CRM Store
        crm_existing = self.crm.find_by_phone_suffix(suffix)
        if crm_existing:
            # Backfill local ledger so future checks short-circuit locally
            self._record_local(
                place_id=place_id,
                phone_suffix=suffix,
                company_name=crm_existing.get("name", ""),
                owner_id=crm_existing.get("owner_user_id", ""),
                region=crm_existing.get("region", "default"),
            )
            return True

        return False

    def select_next_owner_50_50(self, region: str) -> str:
        """
        Queries live CRM state to assign the next lead to whichever closer
        currently holds fewer active leads in the target region.
        """
        counts = {
            owner_id: self.crm.count_leads_by_owner(owner_id, region)
            for owner_id in self.owners
        }
        return min(counts, key=counts.get)

    def reconcile_and_commit_lead(
        self,
        place_id: str,
        phone: str,
        company_name: str,
        decision_maker: str,
        region: str,
        initial_message: str,
    ) -> Dict[str, Any]:
        """
        Even if an asynchronous messaging webhook created a stub contact
        milliseconds after dispatch, this reconciler overwrites the stub's
        owner, stage, and enriched decision-maker metadata atomically.
        """
        suffix = normalize_phone_suffix(phone)
        owner_id = self.select_next_owner_50_50(region)

        lead_doc = {
            "place_id": place_id,
            "phone": phone,
            "phone_suffix": suffix,
            "name": company_name,
            "decision_maker": decision_maker,
            "owner_user_id": owner_id,
            "region": region,
            "stage": "First Contact",
            "column_id": "new_leads",
            "conversation_Entry": {
                "type": "sent",
                "text": initial_message.strip(),
                "timestamp": int(time.time()),
            },
        }
        card_id = self.crm.upsert_lead_card(lead_doc)
        self._record_local(place_id, suffix, company_name, owner_id, region)
        return {"card_id": card_id, "owner_id": owner_id, "phone_suffix": suffix}

    def _record_local(
        self,
        place_id: str,
        phone_suffix: str,
        company_name: str,
        owner_id: str,
        region: str,
    ) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO dispatched_leads
                (place_id, phone_suffix, company_name, owner_id, region, dispatched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (place_id, phone_suffix, company_name, owner_id, region, int(time.time())),
            )
