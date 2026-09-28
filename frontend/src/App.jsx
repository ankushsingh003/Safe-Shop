import { useState, useRef, useEffect } from "react";
import SafeShopDashboard from "./SafeShopDashboard";
import SafeShopLogo from "./Logo";

const BOOT_LOG = [
  { text: "Windows PowerShell", color: "#ffffff", bold: true },
  { text: "Copyright (C) SafeShop SOC Platform. All rights reserved.", color: "#a1a1aa" },
  { text: "", color: "" },
  { text: "PS C:\\SafeShop\\admin> Initializing SOC console...", color: "#4ade80" },
  { text: "[OK] Connected to API: http://localhost:8000", color: "#4ade80" },
  { text: "[OK] Redis Feature Store: ONLINE", color: "#4ade80" },
  { text: "[OK] Kafka Broker: transactions.live (lag: 12ms)", color: "#4ade80" },
  { text: "[OK] GNN Model: L1_Ensemble v5.0 loaded", color: "#4ade80" },
  { text: "[OK] LangGraph Agents: 4/4 ACTIVE", color: "#4ade80" },
  { text: "[WARN] Ghost Firewall L9: 3 active rate-limits in effect", color: "#fbbf24" },
  { text: "", color: "" },
  { text: "Type 'help' for available commands.", color: "#60a5fa" },
  { text: "", color: "" },
];

const COMMANDS = {
  help: () => [
    { text: "Available commands:", color: "#60a5fa", bold: true },
    { text: "  status          - Show system health summary", color: "#e4e4e7" },
    { text: "  threats         - List active threat alerts", color: "#e4e4e7" },
    { text: "  agents          - Show AI agent fleet status", color: "#e4e4e7" },
    { text: "  block <id>      - Block an order ID", color: "#e4e4e7" },
    { text: "  clear           - Clear terminal", color: "#e4e4e7" },
    { text: "  restart         - Restart SOC engine", color: "#e4e4e7" },
    { text: "  exit            - Close console", color: "#e4e4e7" },
  ],
  status: () => [
    { text: "── SYSTEM STATUS ─────────────────────────────", color: "#71717a" },
    { text: "  API Server       : ONLINE  (latency: 14ms)", color: "#4ade80" },
    { text: "  Kafka Broker     : ONLINE  (lag: 12ms)", color: "#4ade80" },
    { text: "  Redis Store      : ONLINE  (hit-rate: 97.2%)", color: "#4ade80" },
    { text: "  ChromaDB         : ONLINE  (47 cases indexed)", color: "#4ade80" },
    { text: "  Ghost Firewall   : ACTIVE  (3 rules enforced)", color: "#fbbf24" },
    { text: "  GNN Ensemble     : LOADED  (AUC: 0.991)", color: "#4ade80" },
    { text: "  TFT Forecast     : READY   (horizon: 24h)", color: "#4ade80" },
    { text: "──────────────────────────────────────────────", color: "#71717a" },
  ],
  threats: () => [
    { text: "── ACTIVE THREATS ────────────────────────────", color: "#71717a" },
    { text: "  [CRITICAL] ORD-9847 - BIN stuffing detected (score: 0.97)", color: "#f87171" },
    { text: "  [CRITICAL] ORD-9012 - Card velocity breach x6 (score: 0.93)", color: "#f87171" },
    { text: "  [HIGH]     ORD-8391 - Device fingerprint anomaly (score: 0.81)", color: "#fbbf24" },
    { text: "  [HIGH]     ORD-7204 - Geo-IP mismatch IN→RU (score: 0.76)", color: "#fbbf24" },
    { text: "──────────────────────────────────────────────", color: "#71717a" },
    { text: `  Total: 2 CRITICAL, 2 HIGH — last updated ${new Date().toLocaleTimeString()}`, color: "#a1a1aa" },
  ],
  agents: () => [
    { text: "── AGENT FLEET STATUS ────────────────────────", color: "#71717a" },
    { text: "  TriageBot-Alpha     : ACTIVE  (34 cases/min)", color: "#4ade80" },
    { text: "  GraphSentinel-04   : ACTIVE  (analyzing subgraphs)", color: "#4ade80" },
    { text: "  WalletSentry-v2    : ACTIVE  (monitoring velocity)", color: "#4ade80" },
    { text: "  GhostFirewall-L9   : ACTIVE  (enforcing 3 rules)", color: "#c084fc" },
    { text: "──────────────────────────────────────────────", color: "#71717a" },
  ],
  restart: () => [
    { text: "[INFO] Sending SIGTERM to SOC engine processes...", color: "#60a5fa" },
    { text: "[INFO] Draining Kafka consumer group...", color: "#60a5fa" },
    { text: "[INFO] Flushing Redis write-behind buffer...", color: "#60a5fa" },
    { text: "[OK]   SOC engine restarted successfully in 1.2s", color: "#4ade80" },
  ],
};

function TerminalModal({ onClose }) {
  const [lines, setLines] = useState(BOOT_LOG);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);
  const [histIdx, setHistIdx] = useState(-1);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [lines]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleCmd = (raw) => {
    const cmd = raw.trim().toLowerCase();
    const parts = cmd.split(" ");
    const base = parts[0];
    const arg = parts.slice(1).join(" ");

    const echo = { text: `PS C:\\SafeShop\\admin> ${raw}`, color: "#e4e4e7" };

    if (cmd === "clear") { setLines([echo, { text: "", color: "" }]); setInput(""); return; }
    if (cmd === "exit") { onClose(); return; }

    let response;
    if (base === "block" && arg) {
      response = [
        { text: `[INFO] Submitting block order for ${arg.toUpperCase()}...`, color: "#60a5fa" },
        { text: `[OK]   ${arg.toUpperCase()} flagged and queued for Ghost Firewall L9 enforcement.`, color: "#4ade80" },
        { text: `[LOG]  Audit trail written to /var/log/safeshop/blocks.json`, color: "#71717a" },
      ];
    } else if (COMMANDS[base]) {
      response = COMMANDS[base]();
    } else {
      response = [{ text: `'${cmd}' is not recognized. Type 'help' for commands.`, color: "#f87171" }];
    }

    setLines((prev) => [...prev, echo, ...response, { text: "", color: "" }]);
    setHistory((prev) => [raw, ...prev]);
    setHistIdx(-1);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") { if (input.trim()) handleCmd(input); }
    if (e.key === "ArrowUp") { const i = Math.min(histIdx + 1, history.length - 1); setHistIdx(i); setInput(history[i] ?? ""); }
    if (e.key === "ArrowDown") { const i = Math.max(histIdx - 1, -1); setHistIdx(i); setInput(i === -1 ? "" : history[i]); }
    if (e.key === "Escape") onClose();
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", backdropFilter: "blur(6px)", zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={{ width: "min(820px, 95vw)", height: "min(540px, 85vh)", background: "#0c0c0c", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, display: "flex", flexDirection: "column", boxShadow: "0 30px 80px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.05)", overflow: "hidden" }}>
        {/* Title bar */}
        <div style={{ background: "#1a1a1a", padding: "8px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.08)", flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ display: "flex", gap: 6 }}>
              <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#ff5f56", display: "block" }} />
              <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#ffbd2e", display: "block" }} />
              <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#27c93f", display: "block" }} />
            </div>
            <span style={{ fontSize: 12, color: "#a1a1aa", fontFamily: "var(--font-mono)", marginLeft: 8 }}>Windows PowerShell — SafeShop SOC Admin Console</span>
          </div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", color: "#71717a", cursor: "pointer", fontSize: 16, lineHeight: 1 }}>✕</button>
        </div>

        {/* Terminal output */}
        <div onClick={() => inputRef.current?.focus()} style={{ flex: 1, overflowY: "auto", padding: "12px 16px", fontFamily: "var(--font-mono)", fontSize: 12, lineHeight: 1.65, cursor: "text" }}>
          {lines.map((l, i) => (
            <div key={i} style={{ color: l.color || "#e4e4e7", fontWeight: l.bold ? 700 : 400, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{l.text}</div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input row */}
        <div style={{ display: "flex", alignItems: "center", borderTop: "1px solid rgba(255,255,255,0.06)", padding: "8px 16px", gap: 8, background: "#0f0f0f", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "#4ade80", whiteSpace: "nowrap" }}>PS C:\SafeShop\admin&gt;</span>
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#e4e4e7", fontFamily: "var(--font-mono)", fontSize: 12, caretColor: "#4ade80" }}
            placeholder="type a command..."
            spellCheck={false}
            autoComplete="off"
          />
          <button onClick={() => { if (input.trim()) handleCmd(input); }} style={{ background: "rgba(74,222,128,0.1)", border: "1px solid rgba(74,222,128,0.25)", color: "#4ade80", fontFamily: "var(--font-mono)", fontSize: 11, padding: "3px 10px", borderRadius: 4, cursor: "pointer" }}>Run ↵</button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState("home"); // "home" or "monitor"
  const [consoleOpen, setConsoleOpen] = useState(false);

  if (view === "monitor") {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#09090b" }}>
        <div 
          style={{ 
            background: "#0d0e12", 
            padding: "8px 20px", 
            borderBottom: "1px solid rgba(255, 255, 255, 0.08)", 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center" 
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <button 
              onClick={() => setView("home")}
              style={{ 
                display: "inline-flex", 
                alignItems: "center", 
                gap: "8px", 
                background: "rgba(255, 255, 255, 0.04)", 
                border: "1px solid rgba(255, 255, 255, 0.12)", 
                color: "#e4e4e7", 
                padding: "6px 14px", 
                borderRadius: "6px", 
                cursor: "pointer", 
                fontSize: "12px",
                fontWeight: 500,
                transition: "all 0.15s ease"
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.08)";
                e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.2)";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.12)";
              }}
            >
              <i className="ti ti-arrow-left" style={{ fontSize: 13 }}></i>
              Return to Engineering Portal
            </button>
            <div style={{ height: 16, width: 1, background: "rgba(255, 255, 255, 0.1)" }} />
            <SafeShopLogo size={22} showText={true} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ 
              display: "inline-flex", 
              alignItems: "center", 
              gap: 6, 
              fontSize: "11px", 
              color: "#a1a1aa", 
              fontFamily: "var(--font-mono)", 
              background: "rgba(255, 255, 255, 0.03)", 
              padding: "4px 10px", 
              borderRadius: "4px",
              border: "1px solid rgba(255, 255, 255, 0.06)"
            }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 8px #22c55e" }}></span>
              SOC TELEMETRY LIVE
            </span>
          </div>
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          <SafeShopDashboard />
        </div>
      </div>
    );
  }

  return (
    <div 
      style={{ 
        minHeight: "100vh", 
        background: "#08090b", 
        color: "#f4f4f5", 
        fontFamily: "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif", 
        display: "flex", 
        flexDirection: "column",
        position: "relative",
        overflowX: "hidden"
      }}
    >
      {/* Subtle Ambient Background Glow (Monochrome/Dark Graphite) */}
      <div 
        style={{
          position: "absolute",
          top: 0,
          left: "50%",
          transform: "translateX(-50%)",
          width: "900px",
          height: "450px",
          background: "radial-gradient(circle at 50% 10%, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 40%, transparent 70%)",
          pointerEvents: "none",
          zIndex: 0
        }}
      />

      {/* Internal Security Operations Header */}
      <header 
        style={{ 
          padding: "18px 40px", 
          display: "flex", 
          justifyContent: "space-between", 
          alignItems: "center", 
          borderBottom: "1px solid rgba(255, 255, 255, 0.07)",
          background: "rgba(10, 11, 14, 0.8)",
          backdropFilter: "blur(12px)",
          position: "relative",
          zIndex: 10
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <SafeShopLogo size={32} showText={true} textVariant="full" />
          <div style={{ height: 24, width: 1, background: "rgba(255, 255, 255, 0.1)", margin: "0 6px" }} />
          <span 
            style={{ 
              fontSize: "11px", 
              fontWeight: 600, 
              color: "#71717a", 
              letterSpacing: "0.08em", 
              textTransform: "uppercase",
              fontFamily: "var(--font-mono)"
            }}
          >
            Internal Engineering Hub
          </span>
        </div>

        {/* Right Header: Internal Telemetry & Direct Console Access */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div 
            style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: 8, 
              padding: "5px 12px", 
              background: "rgba(255, 255, 255, 0.03)", 
              border: "1px solid rgba(255, 255, 255, 0.08)", 
              borderRadius: "6px",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
              color: "#a1a1aa"
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 6px #22c55e" }} />
            <span>NODE: PROD-US-EAST // CLUSTER 01</span>
          </div>

          <button
                      onClick={() => setConsoleOpen(true)}
            style={{
              background: "#ffffff",
              color: "#09090b",
              border: "none",
              padding: "8px 16px",
              borderRadius: "6px",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.15s ease",
              boxShadow: "0 2px 10px rgba(255, 255, 255, 0.1)"
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "translateY(-1px)";
              e.currentTarget.style.boxShadow = "0 4px 15px rgba(255, 255, 255, 0.2)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 2px 10px rgba(255, 255, 255, 0.1)";
            }}
          >
            <i className="ti ti-terminal" style={{ fontSize: 13 }} />
            <span>Open Console</span>
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <main 
        style={{ 
          flex: 1, 
          display: "flex", 
          flexDirection: "column", 
          alignItems: "center", 
          justifyContent: "center", 
          padding: "70px 24px 60px", 
          textAlign: "center",
          position: "relative",
          zIndex: 1
        }}
      >
        {/* Subtle Tech Badge */}
        <div 
          style={{ 
            padding: "6px 14px", 
            background: "rgba(255, 255, 255, 0.04)", 
            border: "1px solid rgba(255, 255, 255, 0.12)", 
            borderRadius: "100px", 
            color: "#e4e4e7", 
            fontSize: "12px", 
            fontWeight: 500, 
            letterSpacing: "0.04em",
            marginBottom: "28px", 
            display: "inline-flex", 
            alignItems: "center", 
            gap: "8px",
            boxShadow: "inset 0 1px 0 rgba(255, 255, 255, 0.1)"
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#a1a1aa", display: "inline-block" }}></span>
          <span>SAFE-SHOP CYBER DEFENSE // AUTONOMOUS ML INFERENCE ENGINE</span>
        </div>
        
        {/* Hero Title with Metallic Monochrome Gradient */}
        <h1 
          style={{ 
            fontSize: "52px", 
            fontWeight: 800, 
            lineHeight: 1.12, 
            letterSpacing: "-0.04em", 
            maxWidth: "860px", 
            marginBottom: "20px",
            color: "#ffffff"
          }}
        >
          Autonomous Fraud Detection &amp;{" "}
          <span 
            style={{ 
              background: "linear-gradient(180deg, #FFFFFF 0%, #A1A1AA 60%, #71717A 100%)", 
              WebkitBackgroundClip: "text", 
              WebkitTextFillColor: "transparent" 
            }}
          >
            Threat Operations.
          </span>
        </h1>
        
        <p 
          style={{ 
            fontSize: "15px", 
            color: "#a1a1aa", 
            maxWidth: "640px", 
            lineHeight: 1.65, 
            marginBottom: "40px",
            fontWeight: 400
          }}
        >
          Internal security engineering console for SafeShop e-commerce infrastructure. Real-time inference across transaction graph networks, automated bot mitigation, and agentic SOC alert triage.
        </p>

        {/* CTA Buttons in Sleek White & Graphite */}
        <div style={{ display: "flex", gap: "14px", alignItems: "center" }}>
          <button 
            onClick={() => setView("monitor")}
            style={{ 
              background: "#ffffff", 
              color: "#08090b", 
              border: "none", 
              padding: "14px 30px", 
              borderRadius: "8px", 
              fontSize: "14px", 
              fontWeight: 600, 
              cursor: "pointer", 
              boxShadow: "0 0 25px rgba(255, 255, 255, 0.15)", 
              display: "flex", 
              alignItems: "center", 
              gap: "8px", 
              transition: "all 0.18s ease" 
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 8px 30px rgba(255, 255, 255, 0.25)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 0 25px rgba(255, 255, 255, 0.15)";
            }}
          >
            <i className="ti ti-activity-heartbeat" style={{ fontSize: 16 }}></i>
            Launch SOC Monitoring Console
          </button>

          <button 
            onClick={() => setView("monitor")}
            style={{ 
              background: "rgba(255, 255, 255, 0.04)", 
              color: "#e4e4e7", 
              border: "1px solid rgba(255, 255, 255, 0.12)", 
              padding: "14px 26px", 
              borderRadius: "8px", 
              fontSize: "14px", 
              fontWeight: 500, 
              cursor: "pointer", 
              display: "flex",
              alignItems: "center",
              gap: "8px",
              transition: "all 0.18s ease" 
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.08)";
              e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.2)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
              e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.12)";
            }}
          >
            <i className="ti ti-cpu" style={{ fontSize: 16, color: "#a1a1aa" }}></i>
            Live Model Telemetry
          </button>
        </div>

        {/* Telemetry Metrics in Monochrome / Dark Slate Cards */}
        <div 
          style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(4, 1fr)", 
            gap: "16px", 
            marginTop: "64px", 
            maxWidth: "960px", 
            width: "100%" 
          }}
        >
          {[
            { label: "GMV Protected / Month", value: "$4.2B+", desc: "Across 14 Global Regions" },
            { label: "P99 Inference Latency", value: "< 18ms", desc: "Edge Evaluated In-Stream" },
            { label: "Threat Mitigation Rate", value: "99.94%", desc: "Autonomous ML Intercepts" },
            { label: "Throughput Capacity", value: "4.8M ops/s", desc: "Kafka + Spark Cluster" }
          ].map((stat, i) => (
            <div 
              key={i}
              style={{
                background: "rgba(18, 19, 24, 0.6)",
                border: "1px solid rgba(255, 255, 255, 0.07)",
                borderRadius: "10px",
                padding: "20px 22px",
                textAlign: "left",
                transition: "all 0.2s ease"
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.15)";
                e.currentTarget.style.background = "rgba(24, 25, 32, 0.8)";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.07)";
                e.currentTarget.style.background = "rgba(18, 19, 24, 0.6)";
              }}
            >
              <div 
                style={{ 
                  fontSize: "26px", 
                  fontWeight: 700, 
                  color: "#ffffff", 
                  fontFamily: "var(--font-mono)", 
                  letterSpacing: "-0.03em",
                  marginBottom: "6px" 
                }}
              >
                {stat.value}
              </div>
              <div style={{ color: "#e4e4e7", fontSize: "12px", fontWeight: 600, marginBottom: "4px" }}>
                {stat.label}
              </div>
              <div style={{ color: "#71717a", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
                {stat.desc}
              </div>
            </div>
          ))}
        </div>

        {/* Security Engine Architecture Summary */}
        <div 
          style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(3, 1fr)", 
            gap: "16px", 
            marginTop: "20px", 
            maxWidth: "960px", 
            width: "100%" 
          }}
        >
          <div 
            style={{ 
              background: "rgba(14, 15, 19, 0.4)", 
              border: "1px solid rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px", 
              padding: "16px 20px", 
              textAlign: "left" 
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, color: "#e4e4e7", fontSize: 13, fontWeight: 600 }}>
              <i className="ti ti-network" style={{ color: "#a1a1aa", fontSize: 16 }}></i>
              Graph Neural Collusion Network
            </div>
            <p style={{ fontSize: 12, color: "#71717a", lineHeight: 1.5, margin: 0 }}>
              Identifies synthetic identity rings and synchronized checkout bot attacks across credit token networks.
            </p>
          </div>

          <div 
            style={{ 
              background: "rgba(14, 15, 19, 0.4)", 
              border: "1px solid rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px", 
              padding: "16px 20px", 
              textAlign: "left" 
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, color: "#e4e4e7", fontSize: 13, fontWeight: 600 }}>
              <i className="ti ti-robot" style={{ color: "#a1a1aa", fontSize: 16 }}></i>
              LangGraph Autonomous Triage
            </div>
            <p style={{ fontSize: 12, color: "#71717a", lineHeight: 1.5, margin: 0 }}>
              Agentic investigator bots dynamically synthesize rules, verify vector embeddings, and freeze suspicious wallets.
            </p>
          </div>

          <div 
            style={{ 
              background: "rgba(14, 15, 19, 0.4)", 
              border: "1px solid rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px", 
              padding: "16px 20px", 
              textAlign: "left" 
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, color: "#e4e4e7", fontSize: 13, fontWeight: 600 }}>
              <i className="ti ti-server-bolt" style={{ color: "#a1a1aa", fontSize: 16 }}></i>
              Zero-Trust Transaction Scoring
            </div>
            <p style={{ fontSize: 12, color: "#71717a", lineHeight: 1.5, margin: 0 }}>
              Ensemble of XGBoost, Isolation Forests, and TFT forecasting with continuous stream drift evaluation.
            </p>
          </div>
        </div>
      </main>

      {/* Internal Footer */}
      <footer 
        style={{ 
          padding: "16px 40px", 
          borderTop: "1px solid rgba(255, 255, 255, 0.05)", 
          display: "flex", 
          justifyContent: "space-between", 
          alignItems: "center",
          fontSize: "11px",
          color: "#52525b",
          fontFamily: "var(--font-mono)"
        }}
      >
        <div>
          <span>SAFESHOP SOC v5.2 // INTERNAL ACCESS PROTOCOL // LEVEL-4 AUTHORIZED</span>
        </div>
        <div>
          <span>CONFIDENTIAL · SEC-ENG ML RED TEAM USE ONLY</span>
        </div>
      </footer>
      {consoleOpen && <TerminalModal onClose={() => setConsoleOpen(false)} />}
    </div>
  );
}
