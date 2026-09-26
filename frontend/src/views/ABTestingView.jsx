import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function ABTestingView({ orders }) {
  const [split, setSplit] = useState(10); // Challenger percentage
  const [promoted, setPromoted] = useState(false);

  const categories = ["Electronics", "Fashion", "Home", "Sports", "Beauty", "Books", "Gaming"];
  const barData = categories.map((cat, i) => ({
    name: cat,
    champion: +(0.82 + (i % 3) * 0.04).toFixed(3),
    challenger: +(0.86 + (i % 2) * 0.05).toFixed(3),
  }));

  const handlePromote = () => {
    if (window.confirm("Are you sure you want to promote Challenger (GNN-Transformer v5.0) to Champion production status? Traffic will shift to 100%.")) {
      setPromoted(true);
      setSplit(100);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Experiment Header */}
      <div
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "16px 20px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 14,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 14, fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
              SHADOW EXPERIMENT: EXP-2026-GNN-TRANSFORMER
            </span>
            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: promoted ? "rgba(34,197,94,0.1)" : "rgba(245,158,11,0.1)", color: promoted ? "#4ade80" : "#fbbf24", border: `1px solid ${promoted ? "rgba(34,197,94,0.2)" : "rgba(245,158,11,0.2)"}`, fontFamily: "var(--font-mono)" }}>
              {promoted ? "PROMOTED TO PROD" : "ACTIVE SHADOW EVALUATION"}
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-secondary)", margin: 0 }}>
            Comparing Production Ensemble v4.2 vs Next-Gen Graph Transformer v5.0 on live stream shadow traffic.
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Traffic Allocation:</span>
            <button
              onClick={() => setSplit(10)}
              style={{ fontSize: 10, padding: "4px 8px", borderRadius: 4, background: split === 10 ? "rgba(255,255,255,0.15)" : "transparent", color: split === 10 ? "#ffffff" : "var(--text-muted)", border: "1px solid var(--border)" }}
            >
              90 / 10
            </button>
            <button
              onClick={() => setSplit(20)}
              style={{ fontSize: 10, padding: "4px 8px", borderRadius: 4, background: split === 20 ? "rgba(255,255,255,0.15)" : "transparent", color: split === 20 ? "#ffffff" : "var(--text-muted)", border: "1px solid var(--border)" }}
            >
              80 / 20
            </button>
            <button
              onClick={() => setSplit(50)}
              style={{ fontSize: 10, padding: "4px 8px", borderRadius: 4, background: split === 50 ? "rgba(255,255,255,0.15)" : "transparent", color: split === 50 ? "#ffffff" : "var(--text-muted)", border: "1px solid var(--border)" }}
            >
              50 / 50
            </button>
          </div>

          <button
            onClick={handlePromote}
            disabled={promoted}
            style={{
              background: promoted ? "rgba(34,197,94,0.2)" : "#ffffff",
              color: promoted ? "#4ade80" : "#08090b",
              border: promoted ? "1px solid rgba(34,197,94,0.4)" : "none",
              padding: "8px 16px",
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 600,
              cursor: promoted ? "default" : "pointer",
            }}
          >
            {promoted ? "✓ Challenger Championed" : "Promote Challenger to 100%"}
          </button>
        </div>
      </div>

      {/* Head-to-Head Comparison Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        {/* Champion Card */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
              CHAMPION: Ensemble v4.2
            </h3>
            <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              Traffic Share: {100 - split}%
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
            <div style={{ background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>Fraud Recall</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", marginTop: 2 }}>91.3%</p>
            </div>
            <div style={{ background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>False Positive Rate</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#fbbf24", marginTop: 2 }}>1.18%</p>
            </div>
            <div style={{ background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>P99 Latency</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", marginTop: 2 }}>52ms</p>
            </div>
            <div style={{ background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 6, border: "1px solid var(--border)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>Monthly Prevented</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", marginTop: 2 }}>₹3.8M</p>
            </div>
          </div>
        </div>

        {/* Challenger Card */}
        <div style={{ background: "var(--bg-card)", border: "1px solid rgba(74,222,128,0.25)", borderRadius: "var(--radius-md)", padding: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: "#4ade80", fontFamily: "var(--font-mono)" }}>
              CHALLENGER: GNN-Transformer v5.0
            </h3>
            <span style={{ fontSize: 10, color: "#4ade80", fontFamily: "var(--font-mono)" }}>
              Traffic Share: {split}%
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
            <div style={{ background: "rgba(74,222,128,0.04)", padding: 10, borderRadius: 6, border: "1px solid rgba(74,222,128,0.15)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>Fraud Recall</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", marginTop: 2 }}>
                95.8% <span style={{ fontSize: 11, fontWeight: 500 }}>(+4.5%)</span>
              </p>
            </div>
            <div style={{ background: "rgba(74,222,128,0.04)", padding: 10, borderRadius: 6, border: "1px solid rgba(74,222,128,0.15)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>False Positive Rate</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", marginTop: 2 }}>
                0.58% <span style={{ fontSize: 11, fontWeight: 500 }}>(-51%)</span>
              </p>
            </div>
            <div style={{ background: "rgba(74,222,128,0.04)", padding: 10, borderRadius: 6, border: "1px solid rgba(74,222,128,0.15)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>P99 Latency</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", marginTop: 2 }}>
                38ms <span style={{ fontSize: 11, fontWeight: 500 }}>(-14ms)</span>
              </p>
            </div>
            <div style={{ background: "rgba(74,222,128,0.04)", padding: 10, borderRadius: 6, border: "1px solid rgba(74,222,128,0.15)" }}>
              <p style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase" }}>Est. Extra Savings</p>
              <p style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", marginTop: 2 }}>+₹420K/mo</p>
            </div>
          </div>
        </div>
      </div>

      {/* Head-to-Head Performance by Category */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
        <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 12 }}>
          Head-to-Head Precision by Product Category (Shadow Comparison)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#71717a" }} />
            <YAxis domain={[0.7, 1.0]} tick={{ fontSize: 9, fill: "#71717a" }} />
            <Tooltip
              contentStyle={{ background: "#121318", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 11 }}
            />
            <Bar dataKey="champion" fill="#71717a" name="Champion (v4.2)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="challenger" fill="#ffffff" name="Challenger (v5.0)" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
