import React, { useState } from "react";

export default function AgentsView({ orders }) {
  const [selectedAgent, setSelectedAgent] = useState("TriageBot-Alpha");
  const [isRunning, setIsRunning] = useState(false);
  const [targetId, setTargetId] = useState(orders[0]?.order_id || "ORD-94812");

  const [traceLogs, setTraceLogs] = useState([
    { ts: "19:54:12", agent: "TriageBot-Alpha", step: "INGEST", message: "Intercepted telemetry for ORD-1039 (Risk: 0.842). Extracting node graph features..." },
    { ts: "19:54:13", agent: "GraphSentinel-04", step: "GRAPH_REASON", message: "Detected 4 card tokens linked to fingerprint SHA-90fa21 in last 120s. Collusion weight: 0.91." },
    { ts: "19:54:14", agent: "TriageBot-Alpha", step: "RAG_RETRIEVE", message: "ChromaDB query matches 'Distributed BIN attack vector IN-MH' with 0.94 similarity." },
    { ts: "19:54:15", agent: "WalletSentry-v2", step: "SYNTHESIS", message: "Autonomously synthesized temporary rate limit rule RL-7729 (Freeze device fingerprint for 30m)." },
    { ts: "19:54:16", agent: "GhostFirewall-L9", step: "ENFORCE", message: "Rule RL-7729 deployed to Redis edge cache in 14ms. Order ORD-1039 blocked." },
  ]);

  const runSimulation = () => {
    setIsRunning(true);
    setTimeout(() => {
      const now = new Date().toLocaleTimeString();
      setTraceLogs((prev) => [
        { ts: now, agent: selectedAgent, step: "EVAL", message: `Evaluated ${targetId}: Autonomous policy check verified. No customer friction detected.` },
        ...prev,
      ]);
      setIsRunning(false);
    }, 1200);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Telemetry Header */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Total Agent Runs (24h)</p>
            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: "rgba(59,130,246,0.15)", color: "#60a5fa", fontWeight: 700, fontFamily: "var(--font-mono)" }}>
              3K+
            </span>
          </div>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>3,182</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>LangGraph autonomous executions</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Decision Precision</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", margin: "4px 0" }}>94.7%</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Grounded across ChromaDB RAG</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Mean Triage Latency</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#fbbf24", margin: "4px 0" }}>412ms</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>P50 reasoning & tool invocation</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Auto Mitigations</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#c084fc", margin: "4px 0" }}>2,190</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Rules synthesized & enforced</p>
        </div>
      </div>

      {/* Agent Fleet Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.8fr", gap: 14 }}>
        {/* Agent Roster */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[
            {
              name: "TriageBot-Alpha",
              role: "Autonomous Incident Investigator",
              tools: ["ChromaDB RAG", "Redis Feature Query", "Rule Synthesizer"],
              status: "ACTIVE",
              invocations: "1,420",
              accuracy: "96.2%",
            },
            {
              name: "GraphSentinel-04",
              role: "Collusion Ring & Network Analyst",
              tools: ["GNN Subgraph Sampler", "Device Fingerprinter", "Card Token Linker"],
              status: "ACTIVE",
              invocations: "1,118",
              accuracy: "93.8%",
            },
            {
              name: "WalletSentry-v2",
              role: "Cross-Merchant Defense & Velocity Limiter",
              tools: ["Redis Token Bucket", "Ghost Firewall L9", "Risk Scorer"],
              status: "ACTIVE",
              invocations: "644",
              accuracy: "94.1%",
            },
          ].map((agent) => {
            const isSel = selectedAgent === agent.name;
            return (
              <div
                key={agent.name}
                onClick={() => setSelectedAgent(agent.name)}
                style={{
                  background: isSel ? "rgba(255,255,255,0.06)" : "var(--bg-card)",
                  border: `1px solid ${isSel ? "rgba(255,255,255,0.2)" : "var(--border)"}`,
                  borderRadius: "var(--radius-md)",
                  padding: "14px 16px",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <i className="ti ti-robot" style={{ color: isSel ? "#ffffff" : "var(--text-muted)", fontSize: 16 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
                      {agent.name}
                    </span>
                  </div>
                  <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)", fontFamily: "var(--font-mono)" }}>
                    {agent.status}
                  </span>
                </div>
                <p style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 8 }}>{agent.role}</p>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 10 }}>
                  {agent.tools.map((t) => (
                    <span key={t} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(255,255,255,0.04)", color: "var(--text-muted)", border: "1px solid rgba(255,255,255,0.08)" }}>
                      {t}
                    </span>
                  ))}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 8 }}>
                  <span>Runs: {agent.invocations}</span>
                  <span>Accuracy: {agent.accuracy}</span>
                </div>
              </div>
            );
          })}

          {/* Interactive Trigger Sandbox */}
          <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: "#ffffff", marginBottom: 8 }}>Manual Agent Trigger Sandbox</p>
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <input
                type="text"
                value={targetId}
                onChange={(e) => setTargetId(e.target.value)}
                placeholder="Target Order ID..."
                style={{ flex: 1, fontSize: 11 }}
              />
              <button
                onClick={runSimulation}
                disabled={isRunning}
                style={{
                  background: "#ffffff",
                  color: "#08090b",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {isRunning ? "Simulating..." : "Dispatch"}
              </button>
            </div>
            <p style={{ fontSize: 9, color: "var(--text-muted)" }}>
              Executes LangGraph decision chain against the targeted order record.
            </p>
          </div>
        </div>

        {/* Live Execution Trace Feed */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            padding: 16,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border)", paddingBottom: 10, marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 6px #4ade80" }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
                AGENT DECISION GRAPH STREAM
              </span>
            </div>
            <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>LangGraph Engine v2.4</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, overflowY: "auto", flex: 1 }}>
            {traceLogs.map((log, i) => (
              <div
                key={i}
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                  borderRadius: 6,
                  padding: "10px 12px",
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "rgba(255,255,255,0.08)", color: "#ffffff", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                      {log.step}
                    </span>
                    <span style={{ fontSize: 11, color: "#a1a1aa", fontFamily: "var(--font-mono)" }}>{log.agent}</span>
                  </div>
                  <span style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{log.ts}</span>
                </div>
                <p style={{ fontSize: 11, color: "#e4e4e7", lineHeight: 1.5, margin: 0 }}>{log.message}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
