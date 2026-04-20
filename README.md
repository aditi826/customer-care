# NEXUS — Support Intelligence Engine

Multi-source intelligent support agent with multi-step LLM reasoning, policy enforcement, confidence scoring, and Composio email delivery.

---

## Architecture

```
Ticket In
  │
  ├─ [Step 1] Classify Ticket          ← ASI:1 LLM
  ├─ [Step 2] Fetch Order              ← orders.json
  ├─ [Step 3] Fetch Customer           ← customers.json
  ├─ [Step 4] Fetch Products           ← products.json
  ├─ [Step 5] Search Knowledge Base    ← knowledge_base.json (keyword scoring)
  ├─ [Step 6] Multi-Step Reasoning     ← ASI:1 LLM (chain-of-thought)
  ├─ [Step 7] Validate Confidence      ← ASI:1 LLM (secondary validation)
  ├─ [Step 8] Decision Gate            ← confidence ≥ 70% → resolve | else → escalate
  ├─ [Step 9] Generate Email           ← ASI:1 LLM
  ├─ [Step 10] Send Email              ← Composio Gmail
  └─ [Step 11] Log & Summary           ← logger.py (JSONL + memory)
```

## File Structure

```
support_agent/
├── main.py              # FastAPI server + API endpoints
├── agent.py             # AgentOrchestrator (pipeline orchestrator)
├── llm_calls.py         # ASI:1 LLM calls (classify / reason / email / validate)
├── tools.py             # Data tools + Composio email
├── logger.py            # Structured JSONL logger
├── index.html           # Web UI
├── tickets.json         # Sample tickets
├── orders.json          # Sample orders
├── customers.json       # Sample customers
├── products.json        # Product catalog
├── knowledge_base.json  # Policy rules
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   ASI1_API_KEY    → https://asi1.ai
#   COMPOSIO_API_KEY → https://composio.ai

# 3. Run server
python main.py

# 4. Open browser
open http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/tickets` | List all tickets |
| GET | `/api/tickets/{id}` | Get single ticket |
| POST | `/api/process/{id}` | Run agent pipeline |
| GET | `/api/orders/{id}` | Get order |
| GET | `/api/customers/{id}` | Get customer |
| GET | `/api/knowledge-base/search?q=...` | Search KB |
| GET | `/api/logs` | All logs |
| GET | `/api/stats` | Summary stats |

## LLM Pipeline (ASI:1 API)

Uses `asi1-mini` via the OpenAI-compatible endpoint `https://api.asi1.ai/v1`.

4 LLM calls per ticket:
1. **Classify** — category, sub-category, sentiment, urgency
2. **Reason** — chain-of-thought with policies, customer tier, order state
3. **Validate** — confidence score verification (secondary check)
4. **Generate Email** — customer-facing response

## Confidence Scoring

- Score ≥ 0.70 → Auto-resolve + send email
- Score < 0.70 → Escalate to human queue
- Validation step can adjust score up/down

## Composio Email

Set `COMPOSIO_API_KEY` and connect your Gmail account at composio.ai.
Without key: emails are simulated (logged but not sent).

## Customer Tiers & Policy Rules

| Tier | Return Window | Refund Speed | Goodwill Credit |
|------|--------------|--------------|-----------------|
| Bronze | 30 days | Standard | — |
| Silver | 30 days | Standard | 5% on delays |
| Gold | 30 days | Expedited | 10% on delays |
| Platinum | 30 days | Expedited | 15% on delays + extended warranty |

Escalation triggers: confidence < 70%, 3+ complaints, refund > ₹50,000, legal threats.
