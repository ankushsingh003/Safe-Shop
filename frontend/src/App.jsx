import { useState } from "react";
import SafeShopDashboard from "./SafeShopDashboard";
import SafeShopLogo from "./Logo";

export default function App() {
  const [view, setView] = useState("home"); // "home" or "monitor"

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
            onClick={() => setView("monitor")}
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
            <span>Open Console</span>
            <i className="ti ti-arrow-right" style={{ fontSize: 12 }}></i>
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
    </div>
  );
}
