import React, { useState } from "react";

export default function LiveFeedView({ orders, onBlockOrder }) {
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [selectedOrder, setSelectedOrder] = useState(orders[0] || null);
  const [isPaused, setIsPaused] = useState(false);

  const filtered = orders.filter((o) => {
    if (filter === "CRITICAL" && o.risk !== "CRITICAL") return false;
    if (filter === "HIGH" && o.risk !== "HIGH") return false;
    if (filter === "MEDIUM" && o.risk !== "MEDIUM") return false;
    if (filter === "LOW" && o.risk !== "LOW") return false;
    if (filter === "BLOCKED" && !o.blocked) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      return (
        o.order_id.toLowerCase().includes(q) ||
        o.category.toLowerCase().includes(q) ||
        o.region.toLowerCase().includes(q) ||
        (o.payment_method && o.payment_method.toLowerCase().includes(q))
      );
    }
    return true;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, height: "100%" }}>
      {/* Stream Controls Bar */}
      <div
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "12px 16px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: isPaused ? "#f59e0b" : "#22c55e",
                boxShadow: isPaused ? "0 0 8px #f59e0b" : "0 0 8px #22c55e",
              }}
            />
            <span style={{ fontSize: 12, fontWeight: 600, fontFamily: "var(--font-mono)", color: "#f4f4f5" }}>
              {isPaused ? "STREAM PAUSED" : "LIVE TRANSACTION INGESTION"}
            </span>
          </div>
          <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            ({orders.length} cached · {filtered.length} matched)
          </span>
          <button
            onClick={() => setIsPaused(!isPaused)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              fontSize: 11,
              padding: "4px 10px",
              background: isPaused ? "rgba(34, 197, 94, 0.15)" : "rgba(245, 158, 11, 0.15)",
              color: isPaused ? "#4ade80" : "#fbbf24",
              border: `1px solid ${isPaused ? "rgba(34, 197, 94, 0.3)" : "rgba(245, 158, 11, 0.3)"}`,
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            <i className={`ti ${isPaused ? "ti-player-play" : "ti-player-pause"}`} />
            {isPaused ? "Resume Stream" : "Pause Stream"}
          </button>
        </div>

        {/* Filters and Search */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ position: "relative" }}>
            <i
              className="ti ti-search"
              style={{
                position: "absolute",
                left: 8,
                top: "50%",
                transform: "translateY(-50%)",
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            />
            <input
              type="text"
              placeholder="Search ID, category, region..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ paddingLeft: 26, fontSize: 11, width: 200, height: 28 }}
            />
          </div>

          <div style={{ display: "flex", gap: 3 }}>
            {["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "BLOCKED"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  fontSize: 10,
                  padding: "4px 8px",
                  borderRadius: 4,
                  background:
                    filter === f
                      ? f === "CRITICAL"
                        ? "rgba(239,68,68,0.25)"
                        : f === "HIGH"
                        ? "rgba(245,158,11,0.25)"
                        : "rgba(255,255,255,0.12)"
                      : "transparent",
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
      </div>

      {/* Main Grid: Stream Table + Detailed Transaction Inspector */}
      <div style={{ display: "grid", gridTemplateColumns: selectedOrder ? "1.6fr 1fr" : "1fr", gap: 12, flex: 1, minHeight: 0 }}>
        {/* Transaction Table */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ overflowY: "auto", flex: 1 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "rgba(0,0,0,0.3)", borderBottom: "1px solid var(--border)", color: "var(--text-muted)", textAlign: "left" }}>
                  <th style={{ padding: "8px 12px" }}>TIMESTAMP</th>
                  <th style={{ padding: "8px 12px" }}>ORDER ID</th>
                  <th style={{ padding: "8px 12px" }}>AMOUNT</th>
                  <th style={{ padding: "8px 12px" }}>CATEGORY</th>
                  <th style={{ padding: "8px 12px" }}>REGION</th>
                  <th style={{ padding: "8px 12px" }}>RISK LEVEL</th>
                  <th style={{ padding: "8px 12px" }}>FRAUD SCORE</th>
                  <th style={{ padding: "8px 12px" }}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((o) => {
                  const isSel = selectedOrder?.order_id === o.order_id;
                  const isCrit = o.risk === "CRITICAL";
                  return (
                    <tr
                      key={o.order_id}
                      onClick={() => setSelectedOrder(o)}
                      style={{
                        background: isSel ? "rgba(255,255,255,0.06)" : isCrit ? "rgba(239,68,68,0.04)" : "transparent",
                        borderBottom: "1px solid rgba(255,255,255,0.04)",
                        cursor: "pointer",
                        transition: "background 0.15s ease",
                      }}
                      onMouseEnter={(e) => {
                        if (!isSel) e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isSel) e.currentTarget.style.background = isCrit ? "rgba(239,68,68,0.04)" : "transparent";
                      }}
                    >
                      <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                        {new Date(o.ts).toLocaleTimeString()}
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#ffffff", fontWeight: 600 }}>
                        {o.order_id}
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", fontWeight: 600, color: "#f4f4f5" }}>
                        ₹{new Intl.NumberFormat().format(o.amount.toFixed(0))}
                      </td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{o.category}</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{o.region}</td>
                      <td style={{ padding: "8px 12px" }}>
                        <span
                          style={{
                            fontSize: 10,
                            padding: "2px 6px",
                            borderRadius: 3,
                            fontFamily: "var(--font-mono)",
                            fontWeight: 600,
                            background:
                              o.risk === "CRITICAL"
                                ? "rgba(239,68,68,0.15)"
                                : o.risk === "HIGH"
                                ? "rgba(245,158,11,0.15)"
                                : "rgba(34,197,94,0.1)",
                            color:
                              o.risk === "CRITICAL"
                                ? "#f87171"
                                : o.risk === "HIGH"
                                ? "#fbbf24"
                                : "#4ade80",
                            border: `1px solid ${
                              o.risk === "CRITICAL"
                                ? "rgba(239,68,68,0.3)"
                                : o.risk === "HIGH"
                                ? "rgba(245,158,11,0.3)"
                                : "rgba(34,197,94,0.2)"
                            }`,
                          }}
                        >
                          {o.risk}
                        </span>
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", fontWeight: 700, color: o.fraud_score > 0.75 ? "#f87171" : "#a1a1aa" }}>
                        {o.fraud_score.toFixed(4)}
                      </td>
                      <td style={{ padding: "8px 12px" }}>
                        {o.blocked ? (
                          <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(168,85,247,0.15)", color: "#c084fc", border: "1px solid rgba(168,85,247,0.3)", fontFamily: "var(--font-mono)" }}>
                            BLOCKED
                          </span>
                        ) : (
                          <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)", fontFamily: "var(--font-mono)" }}>
                            CLEARED
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detailed Transaction Inspector Drawer */}
        {selectedOrder && (
          <div
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 14,
              overflowY: "auto",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
              <div>
                <p style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
                  TRANSACTION TELEMETRY DOSSIER
                </p>
                <h3 style={{ fontSize: 16, fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", margin: "4px 0" }}>
                  {selectedOrder.order_id}
                </h3>
                <p style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                  Inference Latency: <span style={{ color: "#4ade80", fontFamily: "var(--font-mono)" }}>{selectedOrder.latency_ms}ms</span>
                </p>
              </div>
              <button
                onClick={() => setSelectedOrder(null)}
                style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 14 }}
              >
                <i className="ti ti-x" />
              </button>
            </div>

            {/* Score Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
                <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 3 }}>Ensemble Score</p>
                <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: selectedOrder.fraud_score > 0.75 ? "#f87171" : "#fbbf24" }}>
                  {selectedOrder.fraud_score.toFixed(4)}
                </p>
              </div>
              <div style={{ background: "rgba(255,255,255,0.03)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
                <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 3 }}>GNN Collusion Score</p>
                <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: selectedOrder.gnn_score > 0.75 ? "#f87171" : "#a1a1aa" }}>
                  {selectedOrder.gnn_score?.toFixed(4) || "0.1420"}
                </p>
              </div>
            </div>

            {/* Top SHAP Drivers */}
            <div>
              <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                Primary Risk Factor Drivers (SHAP)
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {selectedOrder.shap_top?.map((factor, idx) => (
                  <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(255,255,255,0.02)", padding: "6px 8px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.05)" }}>
                    <span style={{ fontSize: 11, color: "#e4e4e7", fontFamily: "var(--font-mono)" }}>{factor}</span>
                    <span style={{ fontSize: 10, color: "#f87171", fontFamily: "var(--font-mono)" }}>
                      +{((4 - idx) * 0.21).toFixed(2)} impact
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, marginTop: "auto", paddingTop: 12, borderTop: "1px solid var(--border)" }}>
              <button
                onClick={() => {
                  if (onBlockOrder) onBlockOrder(selectedOrder.order_id);
                  selectedOrder.blocked = true;
                }}
                style={{
                  flex: 1,
                  background: "rgba(239, 68, 68, 0.15)",
                  border: "1px solid rgba(239, 68, 68, 0.35)",
                  color: "#f87171",
                  padding: "8px 12px",
                  borderRadius: 6,
                  fontWeight: 600,
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                <i className="ti ti-ban" style={{ marginRight: 5 }} />
                Instant Block (L9)
              </button>
              <button
                onClick={() => alert(`Agentic investigation dispatched for ${selectedOrder.order_id}`)}
                style={{
                  flex: 1,
                  background: "rgba(255, 255, 255, 0.06)",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  color: "#ffffff",
                  padding: "8px 12px",
                  borderRadius: 6,
                  fontWeight: 600,
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                <i className="ti ti-robot" style={{ marginRight: 5 }} />
                Run AI Triage
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
