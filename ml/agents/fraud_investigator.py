"""
LangGraph Fraud Investigator Agent
Layer 2 of Upgrade Roadmap: Agentic AI

This agent handles 'borderline' fraud cases by performing multi-step 
investigations and providing AI-generated reasoning for its decisions.
"""

import os
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# --------------------------------------------------------------------------
# 1. DEFINE STATE
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    order_id: str
    scores: Dict[str, float]  # Ensemble & GNN scores
    metadata: Dict[str, Any]  # User history, IP details, etc.
    evidence: List[str]       # Research findings
    decision: str              # FINAL_APPROVE / FINAL_BLOCK / HUMAN_REVIEW
    reasoning: str            # AI generated explanation

# --------------------------------------------------------------------------
# 2. NODES (The processing steps)
# --------------------------------------------------------------------------

def evaluator_node(state: AgentState):
    """Initial check: Does this order even need a deep investigation?"""
    scores = state["scores"]
    ensemble_score = scores.get("ensemble", 0)
    gnn_score      = scores.get("gnn", 0)
    
    # If both models strongly agree on fraud, block immediately (save costs)
    if ensemble_score > 0.9 and gnn_score > 0.9:
        return {
            "decision": "FINAL_BLOCK", 
            "reasoning": "Both Ensemble and GNN models report critical risk (>0.9). Immediate block triggered."
        }
    
    # If both models agree it's safe, approve immediately
    if ensemble_score < 0.2 and gnn_score < 0.2:
        return {
            "decision": "FINAL_APPROVE", 
            "reasoning": "Both models confirm very low risk pattern."
        }
    
    # Otherwise, it's a borderline case: INVESTIGATE
    return {"decision": "INVESTIGATE"}


def research_node(state: AgentState):
    """Simulates the 'Tool Use' phase: Querying DBs for more evidence."""
    metadata = state["metadata"]
    user_id  = metadata.get("user_id")
    
    # Logic: Simulate checking user history
    # In production, these would be real Tool/API calls
    evidence = []
    
    # Simulated check on account age
    if metadata.get("account_age_days", 0) < 7:
        evidence.append("WARNING: New account created less than 7 days ago.")
    else:
        evidence.append("SIGNAL: Established account with prior successful orders.")
        
    # Simulated check on IP/Location
    if metadata.get("is_proxy", False):
        evidence.append("WARNING: Transaction originated from a high-risk proxy/VPN.")
    
    return {"evidence": evidence}


def reasoning_node(state: AgentState):
    """The 'Brain' node: Uses an LLM to synthesize scores + evidence."""
    # We'll use a mocked LLM response if API key is missing, 
    # but the logic follows a real LLM prompt.
    
    scores   = state["scores"]
    evidence = "\n".join(state["evidence"])
    
    prompt = f"""
    You are a Fraud Investigator at an E-commerce company.
    Analyze the following order:
    - Order ID: {state['order_id']}
    - Ensemble Model Score: {scores.get('ensemble')}
    - GNN Ring Detection Score: {scores.get('gnn')}
    
    Additional Evidence gathered:
    {evidence}
    
    Provide your final decision (APPROVE, BLOCK, or REVIEW) and a concise summary of your reasoning.
    """
    
    # Demo logic: Synthesis of evidence
    has_warning = any("WARNING" in e for e in state["evidence"])
    
    if has_warning and (scores.get("ensemble", 0) > 0.5 or scores.get("gnn", 0) > 0.5):
        decision = "FINAL_BLOCK"
        reasoning = "Investigation confirmed multiple warning signals (high-risk proxy/new account) alongside elevated model scores. Blocking to prevent loss."
    else:
        decision = "FINAL_APPROVE"
        reasoning = "Initial model flags were overturned. Established account history provides sufficient weight to bypass the suspicious metadata flags."

    return {"decision": decision, "reasoning": reasoning}

# --------------------------------------------------------------------------
# 3. CONSTRUCT THE GRAPH
# --------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("researcher", research_node)
workflow.add_node("reasoner", reasoning_node)

# Set entry point
workflow.set_entry_point("evaluator")

# Define conditional edges
def route_after_evaluation(state: AgentState):
    if state["decision"] == "INVESTIGATE":
        return "researcher"
    return END

workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "researcher": "researcher",
        END: END
    }
)

workflow.add_edge("researcher", "reasoner")
workflow.add_edge("reasoner", END)

# Compile the graph
fraud_investigator = workflow.compile()

# --------------------------------------------------------------------------
# 4. DEMO RUNNER
# --------------------------------------------------------------------------
def run_demo():
    print("--- Running Fraud Investigator Agent (LangGraph) ---")
    
    # Case: A borderline order that needs investigation
    test_input = {
        "order_id": "ORD-552",
        "scores": {"ensemble": 0.65, "gnn": 0.3},
        "metadata": {
            "user_id": "user_22",
            "account_age_days": 2,
            "is_proxy": True
        },
        "evidence": []
    }
    
    results = fraud_investigator.invoke(test_input)
    
    print(f"\n[DECISION]: {results['decision']}")
    print(f"[REASONING]: {results['reasoning']}")
    if results['evidence']:
        print(f"[EVIDENCE COLLECTED]: {results['evidence']}")

if __name__ == "__main__":
    run_demo()
