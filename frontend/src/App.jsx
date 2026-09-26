import { useState } from "react";
import SafeShopDashboard from "./SafeShopDashboard";

export default function App() {
  const [view, setView] = useState("home"); // "home" or "monitor"

  if (view === "monitor") {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <div style={{ background: "var(--bg-surface)", padding: "10px 20px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <button 
            onClick={() => setView("home")}
            style={{ display: "flex", alignItems: "center", gap: "8px", background: "transparent", border: "1px solid rgba(255,255,255,0.1)", color: "var(--text-primary)", padding: "6px 12px", borderRadius: "var(--radius-md)", cursor: "pointer", fontSize: "12px" }}
          >
            <i className="ti ti-arrow-left"></i> Back to SafeShop Inc.
          </button>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>SOC ENVIRONMENT LIVE</span>
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          <SafeShopDashboard />
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-base)", color: "var(--text-primary)", fontFamily: "Inter, sans-serif", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header style={{ padding: "24px 48px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 20px rgba(59,130,246,0.4)" }}>
            <i className="ti ti-shield-bolt" style={{ fontSize: 20, color: "#fff" }} />
          </div>
          <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>SafeShop<span style={{ color: "#3b82f6" }}>.ai</span></span>
        </div>
        <nav style={{ display: "flex", gap: "24px" }}>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none", fontSize: "14px", fontWeight: 500 }}>Platform</a>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none", fontSize: "14px", fontWeight: 500 }}>Solutions</a>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none", fontSize: "14px", fontWeight: 500 }}>Company</a>
        </nav>
      </header>

      {/* Hero Section */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 24px", textAlign: "center" }}>
        <div style={{ padding: "8px 16px", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: "20px", color: "#60a5fa", fontSize: "13px", fontWeight: 600, marginBottom: "32px", display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#60a5fa", display: "inline-block", boxShadow: "0 0 10px #60a5fa" }}></span>
          Introducing Next-Gen Fraud Intelligence
        </div>
        
        <h1 style={{ fontSize: "64px", fontWeight: 800, lineHeight: 1.1, letterSpacing: "-0.03em", maxWidth: "800px", marginBottom: "24px" }}>
          Securing Global E-Commerce with <span style={{ background: "linear-gradient(to right, #3b82f6, #8b5cf6, #ec4899)", WebkitBackgroundClip: "text", color: "transparent" }}>Autonomous AI</span>.
        </h1>
        
        <p style={{ fontSize: "18px", color: "var(--text-secondary)", maxWidth: "600px", lineHeight: 1.6, marginBottom: "48px" }}>
          SafeShop is a leading e-commerce platform processing millions of transactions daily. Our Security Engineering team builds state-of-the-art ML pipelines, Graph Neural Networks, and Agentic AI to detect and neutralize fraud before it happens.
        </p>

        <div style={{ display: "flex", gap: "16px" }}>
          <button 
            onClick={() => setView("monitor")}
            style={{ background: "linear-gradient(135deg, #3b82f6, #2563eb)", color: "#fff", border: "none", padding: "16px 32px", borderRadius: "8px", fontSize: "16px", fontWeight: 600, cursor: "pointer", boxShadow: "0 10px 25px -5px rgba(59, 130, 246, 0.4)", display: "flex", alignItems: "center", gap: "8px", transition: "transform 0.2s" }}
            onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
            onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
          >
            Access SOC Monitoring <i className="ti ti-arrow-right"></i>
          </button>
          <button 
            style={{ background: "rgba(255,255,255,0.05)", color: "var(--text-primary)", border: "1px solid rgba(255,255,255,0.1)", padding: "16px 32px", borderRadius: "8px", fontSize: "16px", fontWeight: 600, cursor: "pointer", transition: "background 0.2s" }}
            onMouseOver={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.1)"}
            onMouseOut={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
          >
            Read Architecture Whitepaper
          </button>
        </div>

        {/* Stats Section */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "48px", marginTop: "80px", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "60px", maxWidth: "900px", width: "100%" }}>
          <div>
            <div style={{ fontSize: "36px", fontWeight: 800, color: "#fff", fontFamily: "var(--font-mono)", marginBottom: "8px" }}>$4.2B+</div>
            <div style={{ color: "var(--text-muted)", fontSize: "14px" }}>GMV Protected Monthly</div>
          </div>
          <div>
            <div style={{ fontSize: "36px", fontWeight: 800, color: "#fff", fontFamily: "var(--font-mono)", marginBottom: "8px" }}>&lt;18ms</div>
            <div style={{ color: "var(--text-muted)", fontSize: "14px" }}>P99 Inference Latency</div>
          </div>
          <div>
            <div style={{ fontSize: "36px", fontWeight: 800, color: "#fff", fontFamily: "var(--font-mono)", marginBottom: "8px" }}>99.9%</div>
            <div style={{ color: "var(--text-muted)", fontSize: "14px" }}>Fraud Prevention Rate</div>
          </div>
        </div>
      </main>
    </div>
  );
}
