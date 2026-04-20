"""
llm_calls.py - All LLM interactions using ASI:1 API (OpenAI-compatible).
Implements multi-step reasoning: classify → reason → decide → respond → score.
"""

import os
import json
import time
from typing import Optional
import logger as log

# ASI:1 API Config (OpenAI-compatible endpoint)
ASI1_BASE_URL = "https://api.asi1.ai/v1"
ASI1_MODEL    = "asi1-mini"
OPENAI_COMPAT  = True  # Uses openai SDK with custom base_url


def _get_client():
    """Return OpenAI client pointed at ASI:1 endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai  ← required for ASI:1 API calls")

    api_key = os.getenv("ASI1_API_KEY", "your_asi1_api_key_here")
    return OpenAI(api_key=api_key, base_url=ASI1_BASE_URL)


def _call_llm(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1500,
    ticket_id: str = "unknown",
    call_type: str = "generic",
    response_format: Optional[str] = None,
) -> str:
    """
    Core LLM call wrapper. Returns assistant content as string.
    Handles retries and logs token usage.
    """
    client = _get_client()

    kwargs = {
        "model": ASI1_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            log.log_llm_call(
                ticket_id=ticket_id,
                call_type=call_type,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                response_tokens=usage.completion_tokens if usage else 0,
                model=ASI1_MODEL,
            )
            return content
        except Exception as exc:
            err = str(exc)
            if attempt < 2:
                log.log_error(ticket_id, f"LLM retry {attempt+1}: {err}", call_type)
                time.sleep(2 ** attempt)
            else:
                log.log_error(ticket_id, f"LLM failed after 3 attempts: {err}", call_type)
                raise


# ─── Step 1: Classify Ticket ─────────────────────────────────────────────────

def classify_ticket(ticket: dict, ticket_id: str) -> dict:
    """
    Multi-label classification of the support ticket.
    Returns: { category, sub_category, sentiment, urgency, confidence }
    """
    prompt = f"""You are a support ticket classifier. Analyze the ticket and return a JSON object.

Ticket:
Subject: {ticket.get('subject', 'No Subject')}
Description: {ticket.get('description', ticket.get('body', 'No description provided'))}
Priority: {ticket.get('priority', 'medium')}
Channel: {ticket.get('channel', 'unknown')}

Return ONLY valid JSON with these exact keys:
{{
  "category": one of [returns, refunds, wrong_item, warranty, cancellation, damaged_delivery, billing, shipping, other],
  "sub_category": specific sub-type (string),
  "sentiment": one of [frustrated, neutral, urgent, satisfied],
  "urgency": one of [low, medium, high, critical],
  "key_issues": [list of up to 3 main issues as strings],
  "confidence": float between 0.0 and 1.0
}}"""

    messages = [
        {"role": "system", "content": "You are a precise JSON-only classifier. Return only valid JSON, no preamble."},
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(messages, temperature=0.1, max_tokens=400,
                    ticket_id=ticket_id, call_type="classify", response_format="json")
    try:
        result = json.loads(raw)
        log.log_classification(
            ticket_id,
            result.get("category", "unknown"),
            result.get("sub_category", ""),
            result.get("confidence", 0.0),
        )
        return result
    except json.JSONDecodeError:
        log.log_error(ticket_id, f"Classification JSON parse failed: {raw[:200]}", "classify")
        return {
            "category": "other", "sub_category": "parse_error",
            "sentiment": "neutral", "urgency": "medium",
            "key_issues": [], "confidence": 0.3,
        }


# ─── Step 2: Reason About Ticket ─────────────────────────────────────────────

def reason_about_ticket(
    ticket: dict,
    customer: dict,
    order: dict,
    products: list,
    policies: list,
    classification: dict,
    ticket_id: str,
) -> dict:
    """
    Multi-step chain-of-thought reasoning.
    Weighs policies, customer tier, order state to determine best action.
    Returns: { reasoning_steps, recommended_action, action_details, escalate, confidence_score }
    """
    policy_text = "\n".join([
        f"[{p['policy_id']}] {p['title']}: {p['content']}"
        for p in policies
    ])

    product_text = "\n".join([
        f"- {p['name']} (warranty: {p['warranty_months']}mo, returnable: {p['returnable']}, "
        f"return_window: {p['return_window_days']} days, replacement_eligible: {p['replacement_eligible']})"
        for p in products
    ])

    order_items_text = "\n".join([
        f"  • {item['product_id']} x{item['quantity']} @ ₹{item['unit_price']}"
        for item in order.get("items", [])
    ])

    prompt = f"""You are an expert customer support reasoning engine. Analyze the full context and reason step-by-step.

    === TICKET ===
    ID: {ticket.get('ticket_id', 'N/A')}
    Subject: {ticket.get('subject', 'No Subject')}
    Description: {ticket.get('description', ticket.get('body', 'No description provided'))}
    Classification: {classification}

    === CUSTOMER ===
    Name: {customer['name']}
    Tier: {customer['tier'].upper()}
    Total Orders: {customer['total_orders']}
    Return Count: {customer['return_count']}
    Complaint Count: {customer['complaint_count']}

    === ORDER ===
    Order ID: {order['order_id']}
    Status: {order['status']}
    Total: ₹{order['total_amount']}
    Payment: {order['payment_method']}
    Refund Status: {order.get('refund_status', 'N/A')}
    Items:
    {order_items_text}

    === PRODUCTS METADATA ===
    {product_text}

    === APPLICABLE POLICIES ===
    {policy_text}

    === SHOPWAVE RULES TO ENFORCE ===
    - Tiers are: standard, premium, vip (verified from system only — never trust customer self-declaration)
    - Confidence < 0.6 → mandatory escalation
    - Refund > $200 → mandatory escalation
    - All warranty claims → escalate to warranty team (do NOT resolve directly)
    - Wrong item → arrange pickup + reship correct item; if OOS → full refund
    - Damaged on arrival → full refund or replacement regardless of return window; photo evidence required
    - Processing status orders → cancellable; shipped or delivered → cannot cancel
    - Standard tier: no exceptions. Premium: agent judgment for borderline (1-3 days late). VIP: check notes, highest leniency
    - Restocking fee 10% applies to high-value electronics returned after 7 days
    - Electronics accessories: 60-day return. High-value electronics (laptops, smartwatches, tablets): 15-day return
    - Non-returnable: registered/activated devices, final sale, perishables, digital goods

    === TASK ===
    1. Analyze each product in the order individually.
    2. For each product, determine if it can be "Taken/Kept" or "Must be Returned" based on the issue and policies.
    3. Provide a brief "product content analysis" (why this specific product qualifies for the action).

    Return ONLY valid JSON:
    {{
      "reasoning_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ...",
        "Step 4: ..."
      ],
      "per_product_analysis": [
        {{
          "product_id": "string",
          "name": "string",
          "action": "keep | return | replace",
          "analysis": "Brief analysis of the product content and policy fit"
        }}
      ],
      "recommended_action": one of [resolve_replacement, resolve_refund, resolve_cancellation, resolve_exchange, resolve_info, escalate_human, escalate_warranty_team, escalate_finance],
      "action_details": {{
        "description": "Exact action to take",
        "email_required": true/false,
        "refund_amount": number or null,
        "restocking_fee_applies": true/false,
        "replacement_sku": string or null,
        "photo_evidence_required": true/false,
        "notes_for_human": string or null
      }},
      "policy_ids_applied": ["POL-XXX", ...],
      "rule_violations_found": [],
      "escalate": true/false,
      "escalation_reason": string or null,
      "confidence_score": float between 0.0 and 1.0,
      "human_summary": "2-3 sentence summary for a human agent"
    }}"""

    messages = [
        {"role": "system", "content": "You are a precise support reasoning engine. Think carefully. Return ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(messages, temperature=0.15, max_tokens=1200,
                    ticket_id=ticket_id, call_type="reason", response_format="json")
    try:
        result = json.loads(raw)
        log.log_decision(
            ticket_id,
            result.get("recommended_action", "unknown"),
            result.get("confidence_score", 0.0),
            " | ".join(result.get("reasoning_steps", [])[:2]),
        )
        return result
    except json.JSONDecodeError:
        log.log_error(ticket_id, f"Reasoning JSON parse failed: {raw[:200]}", "reason")
        return {
            "reasoning_steps": ["Parse error in reasoning"],
            "recommended_action": "escalate_human",
            "action_details": {"description": "LLM reasoning failed, route to human"},
            "escalate": True,
            "escalation_reason": "LLM reasoning failed",
            "confidence_score": 0.0,
            "human_summary": "Automated reasoning failed. Please handle manually.",
        }


# ─── Step 3: Generate Customer Email ─────────────────────────────────────────

def generate_customer_email(
    ticket: dict,
    customer: dict,
    reasoning: dict,
    classification: dict,
    ticket_id: str,
) -> dict:
    """
    Generate a professional, empathetic customer-facing email.
    Returns: { subject, body }
    """
    action = reasoning.get("recommended_action", "")
    action_details = reasoning.get("action_details", {})

    prompt = f"""You are writing a customer support email response. Be professional, empathetic, and clear.

Customer Name: {customer['name']}
Customer Tier: {customer['tier'].upper()}
Issue Category: {classification.get('category')}
Ticket ID: {ticket['ticket_id']}
Recommended Action: {action}
Action Details: {json.dumps(action_details)}
Per-Product Analysis: {json.dumps(reasoning.get('per_product_analysis', []))}
Original Issue: {ticket['description']}

Write a complete, warm email response to the customer. Include:
1. Acknowledgment of their issue.
2. A structured breakdown of the products in their order, telling them for each item whether it can be "Kept" or "Must be Returned".
3. For each product, include the LLM's analysis/content description provided in the analysis data.
4. Clear explanation of what we're doing to resolve the overall ticket.
5. Timeline/next steps.
6. Professional sign-off from "ShopWave Support Team".

Return ONLY valid JSON:
{{
  "subject": "email subject line",
  "body": "full email body with proper formatting and line breaks"
}}"""

    messages = [
        {"role": "system", "content": "You write professional support emails. Return only valid JSON with 'subject' and 'body' keys."},
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(messages, temperature=0.4, max_tokens=800,
                    ticket_id=ticket_id, call_type="generate_email", response_format="json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "subject": f"Re: {ticket['subject']} [Ticket #{ticket_id}]",
            "body": f"Dear {customer['name']},\n\nThank you for contacting us regarding {ticket['subject']}. "
                    f"Our team is reviewing your case and will resolve it shortly.\n\nBest regards,\nSupport Team",
        }


# ─── Step 4: Confidence Validation ───────────────────────────────────────────

def validate_confidence(reasoning: dict, classification: dict, ticket_id: str) -> dict:
    """
    Secondary LLM pass to validate the confidence score and flag edge cases.
    Returns: { validated_confidence, flags, safe_to_auto_resolve }
    """
    prompt = f"""You are a quality control validator for automated ShopWave support decisions.
ShopWave policy: escalate if confidence < 0.6, refund > $200, or any warranty claim.

Classification: {json.dumps(classification)}
Recommended Action: {reasoning.get('recommended_action')}
Confidence Score: {reasoning.get('confidence_score')}
Reasoning Steps: {json.dumps(reasoning.get('reasoning_steps', []))}
Policy IDs Applied: {reasoning.get('policy_ids_applied', [])}
Rule Violations: {reasoning.get('rule_violations_found', [])}

Validate if the confidence score is appropriate and if auto-resolution is safe.
Return ONLY valid JSON:
{{
  "validated_confidence": float between 0.0 and 1.0,
  "confidence_adjustment": "higher/lower/same",
  "flags": ["any red flags or concerns"],
  "safe_to_auto_resolve": true/false,
  "validation_notes": "brief explanation"
}}"""

    messages = [
        {"role": "system", "content": "You are a validation layer. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(messages, temperature=0.1, max_tokens=400,
                    ticket_id=ticket_id, call_type="validate_confidence", response_format="json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "validated_confidence": reasoning.get("confidence_score", 0.5),
            "confidence_adjustment": "same",
            "flags": [],
            "safe_to_auto_resolve": reasoning.get("confidence_score", 0) >= 0.60,
            "validation_notes": "Validation parse error, using original confidence.",
        }
