import React, { useState } from "react";

export default function ThreatsView({ orders, onBlockOrder }) {
  const [filter, setFilter] = useState("ALL");
  const [blockedSet, setBlockedSet] = useState(new Set());
  const [dismissedSet, setDismissedSet] = useState(new Set());

  const threats = orders.filter((o) => {
    if (dismissedSet.has(o.order_id)) return false;
    if (filter === "CRITICAL") return o.risk === "CRITICAL";
    if (filter === "HIGH") return o.risk === "HIGH";
    if (filter === "BLOCKED") return o.blocked || blockedSet.has(o.order_id);
    return o.risk === "CRITICAL" || o.risk === "HIGH" || o.blocked;
  });

  const totalAtRisk = threats.reduce((acc, o) => acc + o.amount, 0);

  const handleBlock = (orderId) => {
    setBlockedSet((prev) => new Set([...prev, orderId]));
    if (onBlockOrder) onBlockOrder(orderId);
  };

  const handleDismiss = (orderId) => {
    setDismissedSet((prev) => new Set([...prev, orderId]));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Header Threat KPI Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "#f87171", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>Active Critical Alerts</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#f87171", margin: "4px 0" }}>
            {orders.filter((o) => o.risk === "CRITICAL" && !dismissedSet.has(o.order_id)).length}
          </p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Requires analyst intervention</p>
        </div>

        <div style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "#fbbf24", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>High Risk In-Flight</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#fbbf24", margin: "4px 0" }}>
            {orders.filter((o) => o.risk === "HIGH" && !dismissedSet.has(o.order_id)).length}
          </p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Queued for LangGraph verification</p>
        </div>

        <div style={{ background: "rgba(168,85,247,0.06)", border: "1px solid rgba(168,85,247,0.2)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "#c084fc", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>L9 Automated Blocks</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#c084fc", margin: "4px 0" }}>
            {orders.filter((o) => o.blocked).length + blockedSet.size}
          </p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Ghost Firewall Interceptions</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>Total GMV at Risk</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>
            ₹{new Intl.NumberFormat().format(totalAtRisk.toFixed(0))}
          </p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Across current active threat queue</p>
        </div>
      </div>

      {/* Threat Filter Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "10px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <i className="ti ti-shield-x" style={{ color: "#ef4444", fontSize: 16 }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: "#ffffff" }}>Incident Response Console</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {["ALL", "CRITICAL", "HIGH", "BLOCKED"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                fontSize: 10,
                padding: "4px 10px",
                borderRadius: 4,
                background: filter === f ? (f === "CRITICAL" ? "rgba(239,68,68,0.25)" : "rgba(255,255,255,0.12)") : "transparent",
                border: `1px solid ${filter === f ? "rgba(255,255,255,0.2)" : "var(--border)"}`,
                color: filter === f ? "#ffffff" : "var(--text-muted)",
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Threats Queue */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {threats.length === 0 ? (
          <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "40px 20px", textAlign: "center" }}>
            <i className="ti ti-shield-check" style={{ fontSize: 32, color: "#4ade80", marginBottom: 8, display: "inline-block" }} />
            <h3 style={{ fontSize: 14, color: "#ffffff", marginBottom: 4 }}>All Active Threats Cleared</h3>
            <p style={{ fontSize: 11, color: "var(--text-muted)" }}>No pending unmitigated alerts match this filter.</p>
          </div>
        ) : (
          threats.map((o) => {
            const isBlocked = o.blocked || blockedSet.has(o.order_id);
            return (
              <div
                key={o.order_id}
                style={{
                  background: "var(--bg-card)",
                  border: `1px solid ${o.risk === "CRITICAL" ? "rgba(239,68,68,0.25)" : "var(--border)"}`,
                  borderRadius: "var(--radius-md)",
                  padding: "14px 18px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 16,
                }}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: 14, flex: 1 }}>
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 6,
                      background: o.risk === "CRITICAL" ? "rgba(239,68,68,0.15)" : "rgba(245,158,11,0.15)",
                      border: `1px solid ${o.risk === "CRITICAL" ? "rgba(239,68,68,0.3)" : "rgba(245,158,11,0.3)"}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                    }}
                  >
                    <i className={`ti ${isBlocked ? "ti-ban" : "ti-alert-octagon"}`} style={{ color: o.risk === "CRITICAL" ? "#f87171" : "#fbbf24", fontSize: 16 }} />
                  </div>

                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff" }}>
                        {o.order_id}
                      </span>
                      <span
                        style={{
                          fontSize: 9,
                          fontWeight: 600,
                          padding: "2px 6px",
                          borderRadius: 3,
                          fontFamily: "var(--font-mono)",
                          background: o.risk === "CRITICAL" ? "rgba(239,68,68,0.15)" : "rgba(245,158,11,0.15)",
                          color: o.risk === "CRITICAL" ? "#f87171" : "#fbbf24",
                        }}
                      >
                        {o.risk}
                      </span>
                      {isBlocked && (
                        <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(168,85,247,0.15)", color: "#c084fc", fontFamily: "var(--font-mono)" }}>
                          AUTO-BLOCKED L9
                        </span>
                      )}
                      <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                        {new Date(o.ts).toLocaleTimeString()}
                      </span>
                    </div>

                    <p style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 8 }}>
                      Amount: <strong style={{ color: "#ffffff" }}>₹{new Intl.NumberFormat().format(o.amount.toFixed(0))}</strong> &middot; Category: {o.category} &middot; Region: {o.region} &middot; Fraud Prob: <strong style={{ color: "#f87171" }}>{(o.fraud_score * 100).toFixed(1)}%</strong>
                    </p>

                    <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                      {o.shap_top?.map((factor, idx) => (
                        <span key={idx} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(255,255,255,0.04)", color: "#a1a1aa", border: "1px solid rgba(255,255,255,0.08)", fontFamily: "var(--font-mono)" }}>
                          {factor}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
                  {!isBlocked && (
                    <button
                      onClick={() => handleBlock(o.order_id)}
                      style={{
                        fontSize: 11,
                        fontWeight: 600,
                        padding: "6px 12px",
                        borderRadius: 5,
                        background: "rgba(239, 68, 68, 0.15)",
                        border: "1px solid rgba(239, 68, 68, 0.35)",
                        color: "#f87171",
                        cursor: "pointer",
                      }}
                    >
                      <i className="ti ti-ban" style={{ marginRight: 4 }} />
                      Block Actor
                    </button>
                  )}
                  <button
                    onClick={() => alert(`LangGraph autonomous agent investigation dispatched for ${o.order_id}`)}
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      padding: "6px 12px",
                      borderRadius: 5,
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid rgba(255, 255, 255, 0.12)",
                      color: "#ffffff",
                      cursor: "pointer",
                    }}
                  >
                    <i className="ti ti-robot" style={{ marginRight: 4 }} />
                    Investigate
                  </button>
                  <button
                    onClick={() => handleDismiss(o.order_id)}
                    style={{
                      fontSize: 11,
                      padding: "6px 10px",
                      borderRadius: 5,
                      background: "transparent",
                      border: "1px solid var(--border)",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                    }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
