"""
LangGraph Fraud Investigator Agent
Layer 2 of Upgrade Roadmap: Agentic AI

UPGRADED: reasoning_node now calls real OpenAI GPT-4o-mini via LangChain.
Fallback to rule-based logic if OPENAI_API_KEY is missing (safe for dev/testing).

Environment variable required:
    OPENAI_API_KEY=sk-...   ← set in your .env file
"""

import os
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("fraud_investigator")

# --------------------------------------------------------------------------
# 0. LLM INITIALIZATION
#    Uses GPT-4o-mini — 10x cheaper than GPT-4, same quality for reasoning.
#    Falls back gracefully if no API key is set (safe for local dev).
# --------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o-mini",         # cheapest GPT-4 class model — perfect for demos
        temperature=0,                # deterministic decisions (no randomness)
        max_tokens=300,               # enough for a clear decision + reasoning
        api_key=OPENAI_API_KEY,
        timeout=10,                   # 10s max — don't block the request path
    )
    logger.info("✅ LangGraph Agent: Real OpenAI GPT-4o-mini loaded")
else:
    llm = None
    logger.warning("⚠️ LangGraph Agent: OPENAI_API_KEY not set — using rule-based fallback")


# --------------------------------------------------------------------------
# 1. DEFINE STATE
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    order_id:  str
    scores:    Dict[str, float]   # ensemble score + gnn score
    metadata:  Dict[str, Any]     # user history, IP, device, account age
    evidence:  List[str]          # findings from research_node
    decision:  str                # FINAL_APPROVE / FINAL_BLOCK / HUMAN_REVIEW
    reasoning: str                # human-readable explanation (from LLM or fallback)


# --------------------------------------------------------------------------
# 2. NODE 1 — EVALUATOR
#    Fast path: if both models strongly agree, skip the LLM entirely (saves cost).
# --------------------------------------------------------------------------
def evaluator_node(state: AgentState) -> AgentState:
    """
    Quick decision gate. Only borderline cases reach the LLM.
    - Both > 0.9  → instant BLOCK  (clear fraud, no LLM needed)
    - Both < 0.2  → instant APPROVE (clear safe, no LLM needed)
    - Otherwise   → INVESTIGATE (send to researcher → reasoner)
    """
    ensemble_score = state["scores"].get("ensemble", 0)
    gnn_score      = state["scores"].get("gnn", 0)

    if ensemble_score > 0.9 and gnn_score > 0.9:
        return {
            **state,
            "decision":  "FINAL_BLOCK",
            "reasoning": (
                f"Both models report critical risk "
                f"(Ensemble: {ensemble_score:.2f}, GNN: {gnn_score:.2f}). "
                f"Immediate block — no investigation needed."
            ),
        }

    if ensemble_score < 0.2 and gnn_score < 0.2:
        return {
            **state,
            "decision":  "FINAL_APPROVE",
            "reasoning": (
                f"Both models confirm very low risk "
                f"(Ensemble: {ensemble_score:.2f}, GNN: {gnn_score:.2f}). "
                f"Auto-approved."
            ),
        }

    # Borderline — needs full investigation
    return {**state, "decision": "INVESTIGATE"}


# --------------------------------------------------------------------------
# 3. NODE 2 — RESEARCHER
#    Simulates tool use: queries account DB, IP reputation, device history.
#    In production replace each block with real DB/API calls.
# --------------------------------------------------------------------------
def research_node(state: AgentState) -> AgentState:
    """
    Collects evidence signals for the LLM to reason over.
    Each finding is a plain-English sentence so the LLM can use it directly.
    """
    metadata = state["metadata"]
    evidence = []

    # --- Account age check ---
    account_age = metadata.get("account_age_days", 30)
    if account_age < 7:
        evidence.append(
            f"WARNING: Account is only {account_age} day(s) old — very new account."
        )
    elif account_age < 30:
        evidence.append(
            f"SIGNAL: Relatively new account ({account_age} days). Moderate risk signal."
        )
    else:
        evidence.append(
            f"OK: Established account ({account_age} days) with prior order history."
        )

    # --- IP / Proxy check ---
    if metadata.get("is_proxy", False):
        evidence.append(
            "WARNING: Transaction originated from a known high-risk proxy or VPN provider."
        )
    else:
        evidence.append("OK: IP address is not associated with proxy or VPN services.")

    # --- Device type check ---
    device = metadata.get("device", "unknown")
    if device == "unknown":
        evidence.append("SIGNAL: Device type is unrecognized or spoofed.")
    else:
        evidence.append(f"OK: Known device type '{device}' used for this transaction.")

    # --- Order count velocity check ---
    order_velocity = metadata.get("orders_last_hour", 0)
    if order_velocity > 10:
        evidence.append(
            f"WARNING: User placed {order_velocity} orders in the last hour — "
            f"extremely high velocity, possible bot activity."
        )
    elif order_velocity > 5:
        evidence.append(
            f"SIGNAL: User placed {order_velocity} orders in the last hour — elevated velocity."
        )
    else:
        evidence.append(
            f"OK: Normal order velocity ({order_velocity} orders/hour)."
        )

    # --- Location mismatch check ---
    if metadata.get("location_mismatch", False):
        evidence.append(
            "WARNING: Billing address location does not match the IP geolocation — "
            "possible stolen card or account takeover."
        )
    else:
        evidence.append("OK: Billing location matches IP geolocation.")

    logger.info(
        f"[Agent Research] Order {state['order_id']}: {len(evidence)} evidence signals collected."
    )

    return {**state, "evidence": evidence}


# --------------------------------------------------------------------------
# 4. NODE 3 — REASONER (THE REAL LLM CALL)
#    Sends a structured prompt to GPT-4o-mini with all scores + evidence.
#    Falls back to rule-based logic if OpenAI key is unavailable.
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior fraud investigator at Safe-Shop, a large e-commerce platform.
Your job is to analyze borderline fraud cases and make a final decision.

Rules:
- Be concise and precise. Your reasoning must be under 3 sentences.
- Your DECISION must be exactly one of: FINAL_BLOCK, FINAL_APPROVE, or HUMAN_REVIEW
- Use HUMAN_REVIEW only when evidence is genuinely contradictory and you cannot decide.
- Always state which evidence signals most influenced your decision.

Response format (strictly follow this):
DECISION: <FINAL_BLOCK|FINAL_APPROVE|HUMAN_REVIEW>
REASONING: <your concise explanation>"""


def reasoning_node(state: AgentState) -> AgentState:
    """
    The brain of the agent. Calls GPT-4o-mini with a structured prompt
    containing fraud scores + all collected evidence. Parses the response
    to extract a clean DECISION and REASONING.

    Falls back to deterministic rule-based logic if no API key is set.
    """
    scores   = state["scores"]
    evidence = state["evidence"]
    order_id = state["order_id"]

    # ── REAL LLM PATH ────────────────────────────────────────────────────────
    if llm is not None:
        evidence_block = "\n".join(f"  • {e}" for e in evidence)

        user_message = f"""Analyze this borderline fraud case:

Order ID: {order_id}
Ensemble Model Score: {scores.get('ensemble', 0):.4f}  (0=safe, 1=fraud)
GNN Ring Detection Score: {scores.get('gnn', 0):.4f}   (0=safe, 1=fraud)

Evidence collected:
{evidence_block}

Make your final decision."""

        try:
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])

            raw_text = response.content.strip()
            logger.info(f"[LLM Response] Order {order_id}:\n{raw_text}")

            # Parse the structured response
            decision  = "HUMAN_REVIEW"
            reasoning = raw_text  # fallback: use full response as reasoning

            for line in raw_text.splitlines():
                line = line.strip()
                if line.startswith("DECISION:"):
                    parsed = line.replace("DECISION:", "").strip().upper()
                    if parsed in ("FINAL_BLOCK", "FINAL_APPROVE", "HUMAN_REVIEW"):
                        decision = parsed
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            return {**state, "decision": decision, "reasoning": reasoning}

        except Exception as e:
            # If OpenAI call fails (timeout, quota, etc.) → fall through to rule-based
            logger.error(
                f"[LLM Error] Order {order_id}: OpenAI call failed ({e}). "
                f"Using rule-based fallback."
            )

    # ── RULE-BASED FALLBACK (no API key or LLM error) ────────────────────────
    has_warning  = any("WARNING" in e for e in evidence)
    has_ok       = any(e.startswith("OK") for e in evidence)
    ensemble_sc  = scores.get("ensemble", 0)
    gnn_sc       = scores.get("gnn", 0)

    if has_warning and (ensemble_sc > 0.5 or gnn_sc > 0.5):
        decision  = "FINAL_BLOCK"
        reasoning = (
            "Rule-based fallback: Multiple WARNING signals combined with elevated "
            f"model scores (Ensemble: {ensemble_sc:.2f}, GNN: {gnn_sc:.2f}). Blocking."
        )
    elif has_ok and ensemble_sc < 0.6 and gnn_sc < 0.6:
        decision  = "FINAL_APPROVE"
        reasoning = (
            "Rule-based fallback: Established account signals outweigh model flags. "
            f"Scores below threshold (Ensemble: {ensemble_sc:.2f}, GNN: {gnn_sc:.2f})."
        )
    else:
        decision  = "HUMAN_REVIEW"
        reasoning = (
            "Rule-based fallback: Evidence is contradictory. "
            "Sending to human review queue."
        )

    return {**state, "decision": decision, "reasoning": reasoning}


# --------------------------------------------------------------------------
# 5. CONSTRUCT THE LANGGRAPH STATE MACHINE
# --------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("evaluator", evaluator_node)
workflow.add_node("researcher", research_node)
workflow.add_node("reasoner",  reasoning_node)

workflow.set_entry_point("evaluator")

def route_after_evaluation(state: AgentState):
    """Conditional edge: fast-path exits skip the LLM entirely."""
    if state["decision"] == "INVESTIGATE":
        return "researcher"
    return END

workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {"researcher": "researcher", END: END}
)

workflow.add_edge("researcher", "reasoner")
workflow.add_edge("reasoner",   END)

# Compiled graph — imported by app.py as `fraud_investigator`
fraud_investigator = workflow.compile()


# --------------------------------------------------------------------------
# 6. DEMO RUNNER  (python fraud_investigator.py)
# --------------------------------------------------------------------------
def run_demo():
    print("\n" + "="*60)
    print("  Safe-Shop Fraud Investigator — LangGraph Agent Demo")
    print("="*60)

    test_cases = [
        {
            "label": "Case 1: Clear Fraud (fast-path block)",
            "input": {
                "order_id": "ORD-001",
                "scores":   {"ensemble": 0.95, "gnn": 0.92},
                "metadata": {"account_age_days": 1, "is_proxy": True,
                             "device": "unknown", "orders_last_hour": 15,
                             "location_mismatch": True},
                "evidence": [],
                "decision":  "",
                "reasoning": "",
            }
        },
        {
            "label": "Case 2: Borderline — LLM Investigation Triggered",
            "input": {
                "order_id": "ORD-552",
                "scores":   {"ensemble": 0.65, "gnn": 0.30},
                "metadata": {"account_age_days": 2, "is_proxy": True,
                             "device": "mobile", "orders_last_hour": 3,
                             "location_mismatch": False},
                "evidence": [],
                "decision":  "",
                "reasoning": "",
            }
        },
        {
            "label": "Case 3: Clear Safe (fast-path approve)",
            "input": {
                "order_id": "ORD-999",
                "scores":   {"ensemble": 0.05, "gnn": 0.08},
                "metadata": {"account_age_days": 365, "is_proxy": False,
                             "device": "desktop", "orders_last_hour": 1,
                             "location_mismatch": False},
                "evidence": [],
                "decision":  "",
                "reasoning": "",
            }
        },
    ]

    for case in test_cases:
        print(f"\n{'─'*60}")
        print(f"  {case['label']}")
        print(f"{'─'*60}")
        result = fraud_investigator.invoke(case["input"])
        print(f"  DECISION  : {result['decision']}")
        print(f"  REASONING : {result['reasoning']}")
        if result.get("evidence"):
            print(f"  EVIDENCE  :")
            for e in result["evidence"]:
                print(f"    {e}")

    print("\n" + "="*60)
    llm_status = "✅ Real OpenAI GPT-4o-mini" if llm else "⚠️  Rule-based fallback (set OPENAI_API_KEY)"
    print(f"  LLM Status: {llm_status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_demo()