"""
agent.py - Main AgentOrchestrator.
Executes the full multi-step pipeline:
  Ticket → Classify → Gather Context → Search KB →
  Reason → Validate Confidence → Decide → Email → Log → Summary
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import logger as log
import tools
import llm_calls

CONFIDENCE_THRESHOLD = 0.60   # ShopWave policy: escalate if confidence below 0.6


class AgentOrchestrator:
    """
    Orchestrates the full ticket-handling pipeline with multi-step LLM reasoning.
    """

    def __init__(self, composio_api_key: Optional[str] = None, asi1_api_key: Optional[str] = None):
        self.composio_api_key = composio_api_key or os.getenv("COMPOSIO_API_KEY", "")
        if asi1_api_key:
            os.environ["ASI1_API_KEY"] = asi1_api_key

    # ─── Public entry point ────────────────────────────────────────────────

    def process_ticket(self, ticket_id: str) -> dict:
        """
        Full pipeline. Returns a result dict with all steps and final outcome.
        """
        start_time = time.time()
        pipeline_trace = []

        def add_trace(step: str, data: dict, status: str = "success"):
            pipeline_trace.append({
                "step": step,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            })

        # ── Step 0: Load ticket ───────────────────────────────────────────
        ticket = tools.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "error": f"Ticket {ticket_id} not found"}

        log.log_ticket_received(ticket)
        add_trace("ticket_loaded", {
            "ticket_id": ticket_id,
            "subject": ticket.get("subject", "No Subject"),
            "priority": ticket.get("priority", "medium"),
            "channel": ticket.get("channel", "unknown"),
        })

        # ── Step 1: Classify ticket ───────────────────────────────────────
        log.log_step(ticket_id, "CLASSIFY", "Running classification LLM")
        classification = llm_calls.classify_ticket(ticket, ticket_id)
        add_trace("classification", classification)

        # ── Step 2: Gather structured context ─────────────────────────────
        log.log_step(ticket_id, "GATHER_CONTEXT", "Fetching order, customer, products")

        order = tools.get_order(ticket.get("order_id", ""), ticket_id)
        if not order:
            add_trace("get_order", {"found": False, "order_id": ticket.get("order_id")}, "warn")
            order = {"order_id": "N/A", "status": "unknown", "items": [],
                     "total_amount": 0, "payment_method": "unknown", "refund_status": None}
        else:
            add_trace("get_order", {
                "order_id": order["order_id"],
                "status": order["status"],
                "total": order["total_amount"],
                "items": len(order["items"]),
            })

        customer = tools.get_customer(ticket.get("customer_id", ""), ticket_id)
        if not customer:
            add_trace("get_customer", {"found": False}, "warn")
            customer = {"customer_id": "unknown", "name": "Customer", "email": "unknown",
                        "tier": "bronze", "total_orders": 0, "return_count": 0, "complaint_count": 0}
        else:
            add_trace("get_customer", {
                "name": customer["name"],
                "tier": customer["tier"],
                "email": customer["email"],
            })

        products = tools.get_products_for_order(order, ticket_id)
        add_trace("get_products", {
            "products": [p["name"] for p in products]
        })

        # ── Step 3: Search knowledge base ─────────────────────────────────
        log.log_step(ticket_id, "SEARCH_KB", "Searching knowledge base")
        kb_query = f"{ticket.get('subject', '')} {ticket.get('description', ticket.get('body', ''))} {classification.get('category', '')} {customer.get('tier', '')}"
        policies = tools.search_knowledge_base(kb_query, ticket_id=ticket_id)
        # Also get tier-specific and escalation policies
        tier_policies = tools.search_knowledge_base(
            f"customer tier {customer.get('tier', 'bronze')} benefits", ticket_id=ticket_id
        )
        escalation_policies = tools.search_knowledge_base("escalate human", ticket_id=ticket_id)

        # Deduplicate
        all_policy_ids = {p["policy_id"] for p in policies}
        for p in tier_policies + escalation_policies:
            if p["policy_id"] not in all_policy_ids:
                policies.append(p)
                all_policy_ids.add(p["policy_id"])

        add_trace("knowledge_base_search", {
            "policies_found": [p["policy_id"] for p in policies],
            "categories": list({p["category"] for p in policies}),
        })

        # ── Step 4: Multi-step LLM Reasoning ─────────────────────────────
        log.log_step(ticket_id, "REASON", "Running multi-step reasoning LLM")
        reasoning = llm_calls.reason_about_ticket(
            ticket=ticket,
            customer=customer,
            order=order,
            products=products,
            policies=policies,
            classification=classification,
            ticket_id=ticket_id,
        )
        add_trace("reasoning", {
            "recommended_action": reasoning.get("recommended_action"),
            "reasoning_steps": reasoning.get("reasoning_steps", []),
            "policies_applied": reasoning.get("policy_ids_applied", []),
            "rule_violations": reasoning.get("rule_violations_found", []),
            "confidence_score": reasoning.get("confidence_score"),
            "escalate": reasoning.get("escalate"),
        })

        # ── Step 5: Validate Confidence ───────────────────────────────────
        log.log_step(ticket_id, "VALIDATE_CONFIDENCE", "Running confidence validation")
        validation = llm_calls.validate_confidence(reasoning, classification, ticket_id)
        final_confidence = validation.get("validated_confidence", reasoning.get("confidence_score", 0.5))
        safe_to_resolve = validation.get("safe_to_auto_resolve", False)

        add_trace("confidence_validation", {
            "original_confidence": reasoning.get("confidence_score"),
            "validated_confidence": final_confidence,
            "adjustment": validation.get("confidence_adjustment"),
            "flags": validation.get("flags", []),
            "safe_to_auto_resolve": safe_to_resolve,
            "notes": validation.get("validation_notes"),
        })

        # ── Step 6: Decision Gate ─────────────────────────────────────────
        log.log_step(ticket_id, "DECISION_GATE", f"confidence={final_confidence:.2f} threshold={CONFIDENCE_THRESHOLD}")

        # Force escalation if customer has too many complaints
        force_escalate = (
            customer.get("complaint_count", 0) >= 3
            or reasoning.get("escalate", False)
            or not safe_to_resolve
            or final_confidence < CONFIDENCE_THRESHOLD
        )

        # ── Step 7: Generate and send email (if resolving) ────────────────
        email_result = {"sent": False}
        email_content = {}

        if not force_escalate:
            log.log_step(ticket_id, "GENERATE_EMAIL", "Generating customer email")
            email_content = llm_calls.generate_customer_email(
                ticket=ticket,
                customer=customer,
                reasoning=reasoning,
                classification=classification,
                ticket_id=ticket_id,
            )
            add_trace("email_generated", {
                "subject": email_content.get("subject"),
                "body_preview": email_content.get("body", "")[:200] + "...",
            })

            # Send email via Composio
            log.log_step(ticket_id, "SEND_EMAIL", f"Sending to {customer.get('email')}")
            send_result = tools.send_email_via_composio(
                to_email=customer.get("email", ""),
                customer_name=customer.get("name", "Customer"),
                subject=email_content.get("subject", f"Re: {ticket['subject']}"),
                body=email_content.get("body", ""),
                ticket_id=ticket_id,
                composio_api_key=self.composio_api_key,
            )
            email_result = {"sent": True, **send_result}
            add_trace("email_sent", send_result)

            # Update ticket status
            tools.update_ticket_status(ticket_id, "resolved")
            final_status = "resolved"

        else:
            # Escalation path
            escalation_reason = reasoning.get("escalation_reason") or (
                f"Confidence score {final_confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
                if not reasoning.get("escalate")
                else reasoning.get("escalation_reason", "Policy escalation triggered")
            )
            log.log_escalation(ticket_id, escalation_reason)
            tools.update_ticket_status(ticket_id, "escalated")
            add_trace("escalation", {
                "reason": escalation_reason,
                "assigned_to": "human_queue",
                "flags": validation.get("flags", []),
                "confidence": final_confidence,
            })
            final_status = "escalated"

        # ── Step 8: Final Summary ─────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        human_summary = reasoning.get("human_summary", "No summary generated.")

        log.log_resolution(
            ticket_id=ticket_id,
            status=final_status,
            summary=human_summary,
            confidence_score=final_confidence,
        )

        result = {
            "success": True,
            "ticket_id": ticket_id,
            "final_status": final_status,
            "confidence_score": round(final_confidence, 3),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "classification": classification,
            "recommended_action": reasoning.get("recommended_action"),
            "action_details": reasoning.get("action_details", {}),
            "policies_applied": reasoning.get("policy_ids_applied", []),
            "rule_violations": reasoning.get("rule_violations_found", []),
            "customer": {
                "name": customer.get("name"),
                "email": customer.get("email"),
                "tier": customer.get("tier"),
            },
            "order_status": order.get("status"),
            "escalated": force_escalate,
            "escalation_reason": reasoning.get("escalation_reason") if force_escalate else None,
            "email_sent": email_result.get("sent", False),
            "email_subject": email_content.get("subject"),
            "email_body": email_content.get("body"),
            "human_summary": human_summary,
            "pipeline_trace": pipeline_trace,
            "processing_time_seconds": elapsed,
            "logs": log.get_logs_for_ticket(ticket_id),
        }

        log.log_step(ticket_id, "PIPELINE_COMPLETE", {
            "status": final_status,
            "elapsed_sec": elapsed,
            "confidence": final_confidence,
        })

        return result
