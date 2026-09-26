import React, { useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

export default function ModelMetricsView({ metrics }) {
  const [retraining, setRetraining] = useState(false);
  const [retrainStep, setRetrainStep] = useState(null);

  const shapImportance = [
    { feature: "velocity_10m", score: 0.28 },
    { feature: "amount_zscore", score: 0.22 },
    { feature: "gnn_collusion_prob", score: 0.18 },
    { feature: "ip_distance_km", score: 0.14 },
    { feature: "device_fingerprint_entropy", score: 0.11 },
    { feature: "payment_failure_streak", score: 0.07 },
  ];

  const layerLatencies = [
    { stage: "L1: XGBoost Ensemble", p50: 18, p95: 36, p99: 52 },
    { stage: "L1: GNN Subgraph", p50: 31, p95: 64, p99: 89 },
    { stage: "L2: LangGraph Agent", p50: 412, p95: 780, p99: 980 },
    { stage: "L3: Redis Feature Store", p50: 1.2, p95: 2.1, p99: 3.4 },
    { stage: "L4: TFT Volume Forecast", p50: 140, p95: 220, p99: 310 },
  ];

  const handleRetrain = () => {
    setRetraining(true);
    setRetrainStep("Sampling 2.4M transactions from Data Lake...");
    setTimeout(() => setRetrainStep("Fitting XGBoost & LightGBM with BayesOpt..."), 1500);
    setTimeout(() => setRetrainStep("Evaluating out-of-time test split... PR-AUC: 0.894 (+0.002)"), 3200);
    setTimeout(() => {
      setRetrainStep("Model package validated. Ready for shadow A/B deployment.");
      setRetraining(false);
    }, 4800);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Top Champion Model Status Card */}
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
              CHAMPION: Ensemble v4.2.1-prod
            </span>
            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)", fontFamily: "var(--font-mono)" }}>
              PRODUCTION ACTIVE
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-secondary)", margin: 0 }}>
            Architecture: Calibrated Stacking (XGBoost + LightGBM + Isolation Forest + PyG GNN)
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {retrainStep && (
            <span style={{ fontSize: 11, color: "#fbbf24", fontFamily: "var(--font-mono)" }}>
              {retrainStep}
            </span>
          )}
          <button
            onClick={handleRetrain}
            disabled={retraining}
            style={{
              background: "#ffffff",
              color: "#08090b",
              border: "none",
              padding: "8px 16px",
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <i className={`ti ${retraining ? "ti-loader" : "ti-refresh"}`} />
            {retraining ? "Training Pipeline Active..." : "Trigger Model Retrain"}
          </button>
        </div>
      </div>

      {/* Primary KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {[
          { label: "PR-AUC Score", val: "0.892", sub: "+0.014 vs baseline", color: "#4ade80" },
          { label: "F1-Score", val: "0.864", sub: "Balanced threshold 0.55", color: "#4ade80" },
          { label: "Precision @ Recall 80%", val: "91.3%", sub: "< 0.8% false positive rate", color: "#ffffff" },
          { label: "Population Stability Index", val: "0.042", sub: "Status: No drift detected", color: "#60a5fa" },
        ].map((kpi, idx) => (
          <div key={idx} style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
            <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{kpi.label}</p>
            <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: kpi.color, margin: "4px 0" }}>{kpi.val}</p>
            <p style={{ fontSize: 10, color: "var(--text-muted)" }}>{kpi.sub}</p>
          </div>
        ))}
      </div>

      {/* Historical Performance Chart + Feature Importance Bar Chart */}
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
        {/* 30-Day Metrics Trend */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
          <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 12 }}>
            30-Day Production Model Stability (PR-AUC, F1, Precision, Recall)
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metrics} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="day" tick={{ fontSize: 9, fill: "#71717a" }} interval={4} />
              <YAxis domain={[0.7, 1.0]} tick={{ fontSize: 9, fill: "#71717a" }} />
              <Tooltip
                contentStyle={{ background: "#121318", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 11 }}
              />
              <Line type="monotone" dataKey="pr_auc" stroke="#ffffff" strokeWidth={1.8} dot={false} name="PR-AUC" />
              <Line type="monotone" dataKey="precision" stroke="#4ade80" strokeWidth={1.4} dot={false} name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#fbbf24" strokeWidth={1.4} dot={false} name="Recall" />
              <Line type="monotone" dataKey="f1" stroke="#a1a1aa" strokeWidth={1.2} strokeDasharray="3 3" dot={false} name="F1-Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Global SHAP Feature Importance */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
          <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 12 }}>
            Global Feature Importance (Mean |SHAP|)
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={shapImportance} layout="vertical" margin={{ top: 5, right: 15, left: 30, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis type="number" domain={[0, 0.35]} tick={{ fontSize: 9, fill: "#71717a" }} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize: 9, fill: "#a1a1aa", fontFamily: "var(--font-mono)" }} />
              <Tooltip
                contentStyle={{ background: "#121318", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 11 }}
              />
              <Bar dataKey="score" fill="#ffffff" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Latency Percentiles Across Pipeline */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
        <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 10 }}>
          Inference Latency Profile by Pipeline Stage
        </h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px" }}>INFERENCE STAGE</th>
              <th style={{ padding: "8px 12px" }}>P50 LATENCY</th>
              <th style={{ padding: "8px 12px" }}>P95 LATENCY</th>
              <th style={{ padding: "8px 12px" }}>P99 LATENCY</th>
              <th style={{ padding: "8px 12px" }}>SLA STATUS</th>
            </tr>
          </thead>
          <tbody>
            {layerLatencies.map((row, i) => (
              <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#ffffff" }}>{row.stage}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#4ade80" }}>{row.p50}ms</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#fbbf24" }}>{row.p95}ms</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: row.p99 > 500 ? "#f87171" : "#e4e4e7" }}>{row.p99}ms</td>
                <td style={{ padding: "8px 12px" }}>
                  <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)", fontFamily: "var(--font-mono)" }}>
                    HEALTHY
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
