"""
tools.py - Data access layer for tickets, orders, customers, products, knowledge base.
Mock Composio email sending. In-memory ticket persistence.
"""

import json
import os
from typing import List, Dict, Any, Optional

# ─── JSON Data Files ─────────────────────────────────────────────────────────
TICKETS_FILE = "tickets.json"
CUSTOMERS_FILE = "customers.json"
ORDERS_FILE = "orders.json"
PRODUCTS_FILE = "products.json"
KB_FILE = "knowledge_base.json"

def load_json(filename: str) -> List[Dict]:
    """Load JSON file with error handling."""
    try:
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return []

_tickets_cache = None

def _tickets() -> List[Dict]:
    """Global in-memory tickets list."""
    global _tickets_cache
    if _tickets_cache is None:
        _tickets_cache = load_json(TICKETS_FILE)
    return _tickets_cache

def save_tickets():
    """Persist tickets to file."""
    path = os.path.join(os.path.dirname(__file__), TICKETS_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_tickets_cache, f, indent=2)

# ─── Tickets ────────────────────────────────────────────────────────────────

def list_tickets() -> List[Dict]:
    """List all tickets."""
    return _tickets()

def get_ticket(ticket_id: str) -> Optional[Dict]:
    """Get ticket by ID."""
    for t in _tickets():
        if t.get("ticket_id") == ticket_id:
            return t
    return None

def update_ticket_status(ticket_id: str, status: str):
    """Update ticket status."""
    for ticket in _tickets():
        if ticket.get("ticket_id") == ticket_id:
            ticket["status"] = status
            save_tickets()
            return
    raise ValueError(f"Ticket {ticket_id} not found")

# ─── Orders ─────────────────────────────────────────────────────────────────

def get_order(order_id: str, ticket_id: str = None) -> Optional[Dict]:
    """Get order by ID."""
    orders = load_json(ORDERS_FILE)
    for order in orders:
        if order.get("order_id") == order_id:
            return order
    if ticket_id:
        print(f"[TOOLS] Order {order_id} not found for ticket {ticket_id}")
    return None

# ─── Customers ──────────────────────────────────────────────────────────────

def get_customer(customer_id: str, ticket_id: str = None) -> Optional[Dict]:
    """Get customer by ID."""
    customers = load_json(CUSTOMERS_FILE)
    for cust in customers:
        if cust.get("customer_id") == customer_id:
            return cust
    if ticket_id:
        print(f"[TOOLS] Customer {customer_id} not found for ticket {ticket_id}")
    return None

# ─── Products ───────────────────────────────────────────────────────────────

def get_products_for_order(order: Dict, ticket_id: str = None) -> List[Dict]:
    """Get product details for order items."""
    products = load_json(PRODUCTS_FILE)
    product_map = {p["product_id"]: p for p in products}
    
    order_products = []
    for item in order.get("items", []):
        prod = product_map.get(item.get("product_id"))
        if prod:
            order_products.append(prod)
    
    if ticket_id and len(order_products) == 0:
        print(f"[TOOLS] No products found for order {order.get('order_id')} in ticket {ticket_id}")
    
    return order_products

# ─── Knowledge Base ────────────────────────────────────────────────────────

def search_knowledge_base(query: str, category: Optional[str] = None, ticket_id: Optional[str] = None) -> List[Dict]:
    """Simple keyword search in KB."""
    kb = load_json(KB_FILE)
    
    query_words = query.lower().split()
    matches = []
    
    for policy in kb:
        score = 0
        content = (policy.get("title", "") + " " + policy.get("content", "")).lower()
        for word in query_words:
            if word in content:
                score += 1
        if category and category.lower() not in content:
            score = 0
        
        if score > 0:
            policy["match_score"] = score
            matches.append(policy)
    
    matches.sort(key=lambda p: p["match_score"], reverse=True)
    return matches[:5]  # Top 5

# ─── Composio Mock ─────────────────────────────────────────────────────────

def send_email_via_composio(
    to_email: str, 
    customer_name: str, 
    subject: str, 
    body: str, 
    ticket_id: str,
    composio_api_key: str = ""
) -> Dict:
    """Mock Composio email send (logs only)."""
    print(f"[MOCK EMAIL] To: {to_email} | Subj: {subject[:50]}... | Ticket: {ticket_id}")
    return {
        "status": "sent (mock)",
        "message_id": f"MOCK-{ticket_id}",
        "provider": "composio-gmail-mock"
    }

