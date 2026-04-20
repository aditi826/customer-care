"""
main.py - FastAPI server for the ShopWave Support Agent system.
"""

import os
import json
import traceback
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import logger as log
import tools
import llm_calls as llm
from agent import AgentOrchestrator

load_dotenv()

app = FastAPI(title="ShopWave Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY", "")
ASI1_API_KEY     = os.getenv("ASI1_API_KEY",     "")


# ─── Request models ───────────────────────────────────────────────────────────

class ProcessTicketRequest(BaseModel):
    ticket_id: str
    composio_api_key: Optional[str] = None
    asi1_api_key: Optional[str] = None

class CreateTicketRequest(BaseModel):
    subject: str
    description: str
    customer_id: str
    order_id: str
    priority: str = "medium"
    channel: str = "portal"

class SendEmailRequest(BaseModel):
    ticket_id: str
    to_email: str
    customer_name: str
    subject: str
    body: str
    composio_api_key: Optional[str] = None

class GenerateEmailRequest(BaseModel):
    ticket_id: str
    asi1_api_key: Optional[str] = None


# ─── UI ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ─── Tickets ──────────────────────────────────────────────────────────────────

@app.get("/api/tickets")
async def list_tickets():
    all_tickets = tools.list_tickets()
    return {"tickets": all_tickets, "total": len(all_tickets)}

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    t = tools.get_ticket(ticket_id)
    if not t:
        raise HTTPException(404, f"Ticket {ticket_id} not found")
    return t

@app.post("/api/tickets")
async def create_ticket(req: CreateTicketRequest):
    import random, string
    from datetime import datetime, timezone
    ticket_id = "TKT-" + "".join(random.choices(string.digits, k=4))
    new_ticket = {
        "ticket_id": ticket_id, "subject": req.subject,
        "description": req.description, "customer_id": req.customer_id,
        "order_id": req.order_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": req.priority, "status": "open", "channel": req.channel,
    }
    tools._tickets().append(new_ticket)
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "tickets.json"), "w", encoding="utf-8") as f:
        json.dump(tools._tickets(), f, indent=2)
    log.log_step(ticket_id, "TICKET_CREATED", new_ticket)
    return new_ticket


# ─── Agent pipeline ───────────────────────────────────────────────────────────

@app.post("/api/process/{ticket_id}")
async def process_ticket(ticket_id: str, request: Optional[ProcessTicketRequest] = None):
    composio_key = (request.composio_api_key if request else None) or COMPOSIO_API_KEY
    asi1_key     = (request.asi1_api_key     if request else None) or ASI1_API_KEY

    if not tools.get_ticket(ticket_id):
        raise HTTPException(404, f"Ticket {ticket_id} not found")

    agent = AgentOrchestrator(composio_api_key=composio_key, asi1_api_key=asi1_key)
    try:
        result = agent.process_ticket(ticket_id)
        return JSONResponse(content=result)
    except Exception as exc:
        tb = traceback.format_exc()
        log.log_error(ticket_id, str(exc), "process_ticket_endpoint")
        raise HTTPException(500, detail={"error": str(exc), "traceback": tb[-2000:]})


# ─── Email: generate draft ────────────────────────────────────────────────────

@app.post("/api/generate-email")
async def generate_email_draft(req: GenerateEmailRequest):
    """
    Generate an AI-written email draft for a ticket without running the full pipeline.
    Useful so the agent can compose email and let the human review before sending.
    """
    ticket_id = req.ticket_id
    if req.asi1_api_key:
        os.environ["ASI1_API_KEY"] = req.asi1_api_key

    ticket = tools.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(404, f"Ticket {ticket_id} not found")

    customer = tools.get_customer(ticket.get("customer_id", ""), ticket_id)
    if not customer:
        raise HTTPException(404, f"Customer for ticket {ticket_id} not found")

    order = tools.get_order(ticket.get("order_id", ""), ticket_id) or {}

    # Light classification for email context
    classification = llm.classify_ticket(ticket, ticket_id)

    # Lightweight reasoning stub (just enough for email generation)
    reasoning = {
        "recommended_action": "resolve_info",
        "action_details": {
            "description": f"Respond to customer about their {classification.get('category','issue')}.",
            "email_required": True,
        },
        "confidence_score": 0.85,
    }

    email_content = llm.generate_customer_email(
        ticket=ticket, customer=customer,
        reasoning=reasoning, classification=classification,
        ticket_id=ticket_id,
    )

    return {
        "ticket_id": ticket_id,
        "to_email": customer.get("email", ""),
        "customer_name": customer.get("name", ""),
        "customer_tier": customer.get("tier", "standard"),
        "subject": email_content.get("subject", f"Re: {ticket['subject']}"),
        "body": email_content.get("body", ""),
        "classification": classification,
    }


# ─── Email: send ─────────────────────────────────────────────────────────────

@app.post("/api/send-email")
async def send_email(req: SendEmailRequest):
    """
    Send an email to the customer via Composio Gmail.
    Subject and body can be the AI-generated draft or manually edited.
    """
    composio_key = req.composio_api_key or COMPOSIO_API_KEY

    result = tools.send_email_via_composio(
        to_email=req.to_email,
        customer_name=req.customer_name,
        subject=req.subject,
        body=req.body,
        ticket_id=req.ticket_id,
        composio_api_key=composio_key,
    )

    # Append email record to ticket in log
    log.log_email_sent(
        ticket_id=req.ticket_id,
        customer_email=req.to_email,
        subject=req.subject,
        status=result.get("status", "unknown"),
        composio_response=result,
    )

    return {
        "success": result.get("success", False),
        "status": result.get("status", "unknown"),
        "to": req.to_email,
        "subject": req.subject,
        "message_id": result.get("message_id"),
        "note": result.get("note"),
        "error": result.get("error"),
    }


# ─── Other endpoints ──────────────────────────────────────────────────────────

@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    o = tools.get_order(order_id)
    if not o:
        raise HTTPException(404, "Order not found")
    return o

@app.get("/api/customers/{customer_id}")
async def get_customer(customer_id: str):
    c = tools.get_customer(customer_id)
    if not c:
        raise HTTPException(404, "Customer not found")
    return c

@app.get("/api/knowledge-base/search")
async def search_kb(q: str, category: Optional[str] = None):
    results = tools.search_knowledge_base(q, category=category)
    return {"query": q, "results": results, "count": len(results)}

@app.get("/api/logs")
async def get_logs(ticket_id: Optional[str] = None, n: int = 100):
    logs = log.get_logs_for_ticket(ticket_id) if ticket_id else log.get_recent_logs(n)
    return {"logs": logs, "count": len(logs)}

@app.get("/api/stats")
async def get_stats():
    all_logs = log.get_all_logs()
    resolutions = [l for l in all_logs if l["event"] == "RESOLUTION"]
    escalations = [l for l in all_logs if l["event"] == "ESCALATION"]
    emails_sent = [l for l in all_logs if l["event"] == "EMAIL_SENT"]
    llm_calls_  = [l for l in all_logs if l["event"] == "LLM_CALL"]
    errors      = [l for l in all_logs if l["event"] == "ERROR"]
    avg_conf    = (sum(r.get("confidence_score",0) for r in resolutions)/len(resolutions)) if resolutions else 0
    return {
        "total_processed": len(resolutions),
        "escalated": len(escalations),
        "resolved": len([r for r in resolutions if r.get("status") == "resolved"]),
        "emails_sent": len(emails_sent),
        "llm_calls": len(llm_calls_),
        "errors": len(errors),
        "avg_confidence": round(avg_conf, 3),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  ShopWave Support Agent")
    print("=" * 60)
    print(f"  ASI:1 Key   : {'SET [OK]' if ASI1_API_KEY else 'NOT SET — demo mode'}")
    print(f"  Composio Key: {'SET [OK]' if COMPOSIO_API_KEY else 'NOT SET — email simulated'}")
    print("  UI          : http://localhost:8000")
    print("  API Docs    : http://localhost:8000/docs")
    print("=" * 60)
    # Using app object instead of string for better stability on Windows
    uvicorn.run(app, host="0.0.0.0", port=8000)
