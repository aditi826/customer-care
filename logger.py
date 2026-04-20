"""
logger.py - Centralized structured logging for the Support Agent system.
Logs every agent step, decision, API call, and outcome to file + memory.
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Optional


LOG_FILE = "agent_run_log.jsonl"
_memory_logs: list[dict] = []

# Configure Python's standard logger too
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
_std_logger = logging.getLogger("support_agent")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(record: dict) -> None:
    """Write log record to JSONL file and in-memory list."""
    _memory_logs.append(record)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_ticket_received(ticket: dict) -> None:
    record = {
        "event": "TICKET_RECEIVED",
        "timestamp": _now(),
        "ticket_id": ticket.get("ticket_id"),
        "subject": ticket.get("subject"),
        "customer_id": ticket.get("customer_id"),
        "priority": ticket.get("priority"),
    }
    _write(record)
    _std_logger.info(f"[TICKET_RECEIVED] {ticket.get('ticket_id')} — {ticket.get('subject')}")


def log_step(ticket_id: str, step: str, detail: Any, status: str = "ok") -> None:
    record = {
        "event": "AGENT_STEP",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "step": step,
        "status": status,
        "detail": detail,
    }
    _write(record)
    _std_logger.info(f"[STEP:{step}] {ticket_id} — {status} — {str(detail)[:120]}")


def log_llm_call(ticket_id: str, call_type: str, prompt_tokens: int, response_tokens: int, model: str) -> None:
    record = {
        "event": "LLM_CALL",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "call_type": call_type,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }
    _write(record)
    _std_logger.info(f"[LLM_CALL] {call_type} — tokens in:{prompt_tokens} out:{response_tokens}")


def log_classification(ticket_id: str, category: str, sub_category: str, confidence: float) -> None:
    record = {
        "event": "CLASSIFICATION",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "category": category,
        "sub_category": sub_category,
        "confidence": confidence,
    }
    _write(record)
    _std_logger.info(f"[CLASSIFY] {ticket_id} → {category}/{sub_category} (confidence={confidence:.2f})")


def log_knowledge_base_search(ticket_id: str, query: str, matched_policies: list[str]) -> None:
    record = {
        "event": "KB_SEARCH",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "query": query,
        "matched_policies": matched_policies,
    }
    _write(record)
    _std_logger.info(f"[KB_SEARCH] {ticket_id} — matched {len(matched_policies)} policies")


def log_decision(ticket_id: str, action: str, confidence_score: float, reasoning: str) -> None:
    record = {
        "event": "DECISION",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "action": action,
        "confidence_score": confidence_score,
        "reasoning": reasoning[:500],
    }
    _write(record)
    _std_logger.info(f"[DECISION] {ticket_id} → {action} (confidence={confidence_score:.2f})")


def log_email_sent(ticket_id: str, customer_email: str, subject: str, status: str, composio_response: Any) -> None:
    record = {
        "event": "EMAIL_SENT",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "to": customer_email,
        "subject": subject,
        "status": status,
        "composio_response": str(composio_response)[:300],
    }
    _write(record)
    _std_logger.info(f"[EMAIL] {ticket_id} → {customer_email} [{status}]")


def log_escalation(ticket_id: str, reason: str, assigned_to: str = "human_queue") -> None:
    record = {
        "event": "ESCALATION",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "reason": reason,
        "assigned_to": assigned_to,
    }
    _write(record)
    _std_logger.warning(f"[ESCALATE] {ticket_id} → {assigned_to} | reason: {reason}")


def log_resolution(ticket_id: str, status: str, summary: str, confidence_score: float) -> None:
    record = {
        "event": "RESOLUTION",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "status": status,
        "confidence_score": confidence_score,
        "summary": summary[:500],
    }
    _write(record)
    _std_logger.info(f"[RESOLVE] {ticket_id} → {status} (confidence={confidence_score:.2f})")


def log_error(ticket_id: str, error_msg: str, context: Optional[str] = None) -> None:
    record = {
        "event": "ERROR",
        "timestamp": _now(),
        "ticket_id": ticket_id,
        "error": error_msg,
        "context": context,
    }
    _write(record)
    _std_logger.error(f"[ERROR] {ticket_id} — {error_msg}")


def get_logs_for_ticket(ticket_id: str) -> list[dict]:
    """Retrieve all logs for a specific ticket from memory."""
    return [log for log in _memory_logs if log.get("ticket_id") == ticket_id]


def get_all_logs() -> list[dict]:
    """Get all in-memory logs."""
    return list(_memory_logs)


def get_recent_logs(n: int = 50) -> list[dict]:
    """Get last N log entries."""
    return _memory_logs[-n:]
