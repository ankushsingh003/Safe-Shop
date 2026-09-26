import React, { useState } from "react";

export default function SettingsModal({ isOpen, onClose }) {
  const [criticalThreshold, setCriticalThreshold] = useState(0.78);
  const [highThreshold, setHighThreshold] = useState(0.55);
  const [streamSpeed, setStreamSpeed] = useState("1500");
  const [autoBlockL9, setAutoBlockL9] = useState(true);
  const [ragAutoTriage, setRagAutoTriage] = useState(true);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
      }}
    >
      <div
        style={{
          background: "#101217",
          border: "1px solid rgba(255, 255, 255, 0.15)",
          borderRadius: 8,
          width: 480,
          padding: 22,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          boxShadow: "0 20px 50px rgba(0, 0, 0, 0.8)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255, 255, 255, 0.08)", paddingBottom: 10 }}>
          <div>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", margin: 0 }}>
              SOC ENGINE &amp; PIPELINE SETTINGS
            </h3>
            <p style={{ fontSize: 10, color: "var(--text-muted)", margin: "2px 0 0" }}>
              Configure threat thresholds, streaming interval, and firewall policies.
            </p>
          </div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 16 }}>
            <i className="ti ti-x" />
          </button>
        </div>

        {/* Sliders */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: "#ffffff" }}>CRITICAL Risk Threshold</span>
              <span style={{ color: "#f87171", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{criticalThreshold}</span>
            </div>
            <input
              type="range"
              min="0.60"
              max="0.95"
              step="0.01"
              value={criticalThreshold}
              onChange={(e) => setCriticalThreshold(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "#f87171" }}
            />
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: "#ffffff" }}>HIGH Risk Threshold</span>
              <span style={{ color: "#fbbf24", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{highThreshold}</span>
            </div>
            <input
              type="range"
              min="0.40"
              max="0.75"
              step="0.01"
              value={highThreshold}
              onChange={(e) => setHighThreshold(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "#fbbf24" }}
            />
          </div>

          <div>
            <label style={{ fontSize: 11, color: "#ffffff", display: "block", marginBottom: 6 }}>Live Stream Refresh Frequency</label>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { label: "1.0s (High load)", val: "1000" },
                { label: "1.5s (Standard)", val: "1500" },
                { label: "3.0s (Low CPU)", val: "3000" },
              ].map((s) => (
                <button
                  key={s.val}
                  onClick={() => setStreamSpeed(s.val)}
                  style={{
                    flex: 1,
                    padding: "6px 8px",
                    borderRadius: 4,
                    fontSize: 10,
                    background: streamSpeed === s.val ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.03)",
                    border: `1px solid ${streamSpeed === s.val ? "rgba(255,255,255,0.3)" : "var(--border)"}`,
                    color: streamSpeed === s.val ? "#ffffff" : "var(--text-muted)",
                    cursor: "pointer",
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 10 }}>
            <div>
              <p style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", margin: 0 }}>Ghost Firewall L9 Auto-Block</p>
              <p style={{ fontSize: 10, color: "var(--text-muted)", margin: 0 }}>Automatically drop packets exceeding critical threshold.</p>
            </div>
            <input
              type="checkbox"
              checked={autoBlockL9}
              onChange={(e) => setAutoBlockL9(e.target.checked)}
              style={{ width: 16, height: 16, cursor: "pointer", accentColor: "#ffffff" }}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <p style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", margin: 0 }}>LangGraph Auto-Triage</p>
              <p style={{ fontSize: 10, color: "var(--text-muted)", margin: 0 }}>Dispatch agent fleet on suspicious wallet velocity.</p>
            </div>
            <input
              type="checkbox"
              checked={ragAutoTriage}
              onChange={(e) => setRagAutoTriage(e.target.checked)}
              style={{ width: 16, height: 16, cursor: "pointer", accentColor: "#ffffff" }}
            />
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 12 }}>
          <button
            onClick={onClose}
            style={{ background: "#ffffff", color: "#08090b", border: "none", padding: "8px 16px", borderRadius: 4, fontWeight: 600, fontSize: 11, cursor: "pointer" }}
          >
            Save Preferences
          </button>
        </div>
      </div>
    </div>
  );
}
