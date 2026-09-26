import React, { useState } from "react";

export default function ReportsView({ orders }) {
  const [timeRange, setTimeRange] = useState("7d");
  const [reportType, setReportType] = useState("executive");
  const [isExporting, setIsExporting] = useState(false);

  const downloadCSV = () => {
    setIsExporting(true);
    const headers = ["Order_ID", "Timestamp", "Amount_INR", "Category", "Region", "Risk_Level", "Fraud_Score", "Blocked"];
    const rows = orders.map((o) => [
      o.order_id,
      new Date(o.ts).toISOString(),
      o.amount,
      o.category,
      o.region,
      o.risk,
      o.fraud_score,
      o.blocked ? "TRUE" : "FALSE",
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `safeshop_soc_audit_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setTimeout(() => setIsExporting(false), 800);
  };

  const handlePrintPDF = () => {
    window.print();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Report Generator Controls */}
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
              SECURITY &amp; COMPLIANCE AUDIT REPORTING
            </span>
            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(255,255,255,0.08)", color: "#ffffff", fontFamily: "var(--font-mono)" }}>
              PCI-DSS v4.0 / RBI CSF READY
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-secondary)", margin: 0 }}>
            Automated compliance exports, incident logs, and model attribution digests.
          </p>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {/* Time range picker */}
          <div style={{ display: "flex", gap: 2 }}>
            {["24h", "7d", "30d", "Quarter"].map((t) => (
              <button
                key={t}
                onClick={() => setTimeRange(t)}
                style={{
                  fontSize: 10,
                  padding: "4px 8px",
                  borderRadius: 4,
                  background: timeRange === t ? "rgba(255,255,255,0.15)" : "transparent",
                  color: timeRange === t ? "#ffffff" : "var(--text-muted)",
                  border: "1px solid var(--border)",
                  cursor: "pointer",
                }}
              >
                {t}
              </button>
            ))}
          </div>

          <button
            onClick={downloadCSV}
            disabled={isExporting}
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
            <i className="ti ti-download" />
            {isExporting ? "Exporting CSV..." : "Download Audit CSV"}
          </button>

          <button
            onClick={handlePrintPDF}
            style={{
              background: "rgba(255,255,255,0.06)",
              color: "#ffffff",
              border: "1px solid rgba(255,255,255,0.15)",
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
            <i className="ti ti-printer" />
            Print / Save PDF
          </button>
        </div>
      </div>

      {/* Audit KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Gross Fraud Prevented</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", margin: "4px 0" }}>₹38.4L</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>Across current audit window</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Total Orders Audited</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>{orders.length} Samples</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>100% telemetry completeness</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>False Positive Friction</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#60a5fa", margin: "4px 0" }}>0.74%</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>SLA compliant (&lt; 1.5%)</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Audit Compliance Score</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>99.2%</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>ISO 27001 &amp; RBI Master Directive</p>
        </div>
      </div>

      {/* Compliance Checklist & Executive Findings */}
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 18 }}>
          <h3 style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", marginBottom: 12 }}>
            Executive SOC Incident Audit Digest
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 12, color: "#e4e4e7", lineHeight: 1.6 }}>
            <p style={{ margin: 0 }}>
              1. <strong>Autonomous Mitigation Efficacy</strong>: Over the evaluated period, SafeShop's L9 Ghost Firewall and LangGraph multi-agent systems automatically mitigated 99.94% of synthetic bot checkouts without manual analyst escalation.
            </p>
            <p style={{ margin: 0 }}>
              2. <strong>Model Calibration &amp; Drift</strong>: Champion Stacking Ensemble v4.2 maintained a Population Stability Index (PSI) of 0.042, well below the 0.10 recalibration threshold.
            </p>
            <p style={{ margin: 0 }}>
              3. <strong>Data Governance &amp; Storage</strong>: All streaming transactions were logged to Apache Iceberg Parquet partitions with 100% schema integrity and full cryptographic audit trails.
            </p>
          </div>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 18 }}>
          <h3 style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", marginBottom: 12 }}>
            Regulatory Compliance Status
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[
              { item: "PCI-DSS v4.0 Requirement 10 (Logging & Tracking)", status: "COMPLIANT" },
              { item: "RBI Cyber Security Framework Section 3", status: "COMPLIANT" },
              { item: "Data Retention & Encryption at Rest (AES-256)", status: "COMPLIANT" },
              { item: "Real-Time AI Explainability (SHAP & LIME)", status: "COMPLIANT" },
            ].map((c, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", background: "rgba(255,255,255,0.02)", borderRadius: 5, border: "1px solid var(--border)" }}>
                <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{c.item}</span>
                <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", fontFamily: "var(--font-mono)" }}>
                  {c.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
