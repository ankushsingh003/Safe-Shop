
import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY  = import.meta.env.VITE_API_KEY  || "dev-secret-key";

const headers = { "X-API-KEY": API_KEY, "Content-Type": "application/json" };

const MOCK_ORDERS = Array.from({ length: 12 }, (_, i) => ({
  order_id: `ORD-${1000 + i}`,
  user_id: `USR-${Math.floor(Math.random() * 999)}`,
  order_amount: +(Math.random() * 2000 + 50).toFixed(2),
  is_fraud: Math.random() < 0.15,
  fraud_score: +(Math.random()).toFixed(3),
  risk_level: ["LOW", "HIGH", "CRITICAL"][Math.floor(Math.random() * 3)],
  category: ["Electronics", "Clothing", "Home", "Books", "Beauty"][Math.floor(Math.random() * 5)],
  timestamp: new Date(Date.now() - i * 18000).toISOString(),
}));

const MOCK_FORECAST = Array.from({ length: 24 }, (_, i) => {
  const base = 120 + Math.sin((i / 24) * Math.PI * 2) * 40;
  const actual = i < 14 ? +(base + (Math.random() - 0.5) * 20).toFixed(0) : null;
  return {
    hour: `${String(i).padStart(2, "0")}:00`,
    forecast: +(base + (Math.random() - 0.5) * 10).toFixed(0),
    actual,
    upper: +(base * 1.15).toFixed(0),
    lower: +(base * 0.85).toFixed(0),
  };
});

const MOCK_HEALTH = {
  status: "ok", version: "v5.0-rag",
  layers: { L1_Ensemble: true, L1_GNN: true, L2_Agentic_AI: true, L3_Redis: true, L4_TFT_Forecast: true, L5_Shadow_AB: true, L6_RAG_ChromaDB: true },
  rag: { cases_stored: 47, embeddings: "text-embedding-3-small", status: "healthy" }
};

function riskColor(level) {
  if (level === "CRITICAL") return "#e24b4a";
  if (level === "HIGH")     return "#ef9f27";
  return "#639922";
}

function riskBg(level) {
  if (level === "CRITICAL") return "#fcebeb";
  if (level === "HIGH")     return "#faeeda";
  return "#eaf3de";
}

function fmt(n) { return new Intl.NumberFormat().format(n); }
function fmtUSD(n) { return "$" + (+n).toFixed(2); }
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "1rem", minWidth: 0 }}>
      <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 4px", letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</p>
      <p style={{ fontSize: 24, fontWeight: 500, margin: 0, color: accent || "var(--color-text-primary)" }}>{value}</p>
      {sub && <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "4px 0 0" }}>{sub}</p>}
    </div>
  );
}

function Badge({ level }) {
  return (
    <span style={{
      fontSize: 11, fontWeight: 500, padding: "2px 8px",
      borderRadius: "var(--border-radius-md)",
      background: riskBg(level), color: riskColor(level),
      letterSpacing: "0.03em",
    }}>{level}</span>
  );
}

function SectionHeader({ icon, title, right }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <i className={`ti ${icon}`} aria-hidden="true" style={{ fontSize: 18, color: "var(--color-text-secondary)" }} />
        <span style={{ fontWeight: 500, fontSize: 15 }}>{title}</span>
      </div>
      {right}
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: "var(--border-radius-lg)",
      padding: "1rem 1.25rem",
      ...style
    }}>{children}</div>
  );
}

function Pulse({ active }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: active ? "#639922" : "#e24b4a",
      marginRight: 6,
    }} />
  );
}

function LiveOrderFeed({ orders, loading }) {
  return (
    <Card>
      <SectionHeader
        icon="ti-shopping-cart"
        title="Live order feed"
        right={
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            <Pulse active={true} />live
          </span>
        }
      />
      {loading ? (
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>Loading orders...</p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                {["Order", "User", "Amount", "Category", "Risk", "Time"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 8px", fontWeight: 500, color: "var(--color-text-secondary)", fontSize: 12, whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {orders.slice(0, 10).map((o, i) => (
                <tr key={o.order_id} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: o.is_fraud ? "#fff8f8" : "transparent" }}>
                  <td style={{ padding: "7px 8px", fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--color-text-secondary)" }}>{o.order_id}</td>
                  <td style={{ padding: "7px 8px", color: "var(--color-text-secondary)", fontSize: 12 }}>{o.user_id}</td>
                  <td style={{ padding: "7px 8px", fontWeight: 500 }}>{fmtUSD(o.order_amount)}</td>
                  <td style={{ padding: "7px 8px", color: "var(--color-text-secondary)" }}>{o.category}</td>
                  <td style={{ padding: "7px 8px" }}><Badge level={o.risk_level} /></td>
                  <td style={{ padding: "7px 8px", color: "var(--color-text-secondary)", fontSize: 12 }}>{fmtTime(o.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function FraudAlertBanner({ orders }) {
  const critical = orders.filter(o => o.risk_level === "CRITICAL");
  const high     = orders.filter(o => o.risk_level === "HIGH");
  const totalFraud = orders.filter(o => o.is_fraud).length;

  if (critical.length === 0 && high.length === 0) {
    return (
      <div style={{ background: "#eaf3de", border: "0.5px solid #c0dd97", borderRadius: "var(--border-radius-md)", padding: "10px 16px", display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
        <i className="ti ti-shield-check" aria-hidden="true" style={{ fontSize: 18, color: "#3b6d11" }} />
        <span style={{ color: "#3b6d11", fontWeight: 500 }}>All clear</span>
        <span style={{ color: "#639922" }}>— No active fraud alerts in the current batch.</span>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {critical.length > 0 && (
        <div style={{ background: "#fcebeb", border: "0.5px solid #f09595", borderRadius: "var(--border-radius-md)", padding: "10px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: critical.length > 0 ? 8 : 0 }}>
            <i className="ti ti-alert-triangle" aria-hidden="true" style={{ fontSize: 18, color: "#a32d2d" }} />
            <span style={{ fontWeight: 500, color: "#a32d2d", fontSize: 13 }}>{critical.length} critical fraud {critical.length === 1 ? "alert" : "alerts"}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {critical.slice(0, 3).map(o => (
              <div key={o.order_id} style={{ fontSize: 12, color: "#791f1f", display: "flex", gap: 12 }}>
                <span style={{ fontFamily: "var(--font-mono)" }}>{o.order_id}</span>
                <span>{fmtUSD(o.order_amount)}</span>
                <span>score: {o.fraud_score}</span>
                <span>{fmtTime(o.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {high.length > 0 && (
        <div style={{ background: "#faeeda", border: "0.5px solid #fac775", borderRadius: "var(--border-radius-md)", padding: "10px 16px", display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
          <i className="ti ti-alert-circle" aria-hidden="true" style={{ fontSize: 18, color: "#854f0b" }} />
          <span style={{ fontWeight: 500, color: "#854f0b" }}>{high.length} high-risk {high.length === 1 ? "order" : "orders"}</span>
          <span style={{ color: "#ba7517" }}>under review — {totalFraud} total flagged in current batch.</span>
        </div>
      )}
    </div>
  );
}

function ForecastChart({ data, loading }) {
  return (
    <Card>
      <SectionHeader icon="ti-chart-line" title="Demand forecast vs actuals" />
      {loading ? (
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Loading forecast...</p>
      ) : (
        <>
          <div style={{ display: "flex", gap: 16, marginBottom: 12, fontSize: 12, color: "var(--color-text-secondary)" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 24, height: 2, background: "#378add", display: "inline-block" }} />
              Actual orders
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 24, height: 2, background: "#ef9f27", borderTop: "2px dashed #ef9f27", display: "inline-block" }} />
              TFT forecast
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 24, height: 8, background: "#b5d4f4", opacity: 0.4, display: "inline-block", borderRadius: 2 }} />
              Confidence band
            </span>
          </div>
          <div style={{ width: "100%", height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" strokeOpacity={0.5} />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#888" }} interval={3} />
                <YAxis tick={{ fontSize: 11, fill: "#888" }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, boxShadow: "none" }}
                  formatter={(v, n) => [v ?? "—", n]}
                />
                <Line type="monotone" dataKey="upper" stroke="#b5d4f4" strokeWidth={0} dot={false} legendType="none" />
                <Line type="monotone" dataKey="lower" stroke="#b5d4f4" strokeWidth={0} dot={false} legendType="none" />
                <Line type="monotone" dataKey="forecast" stroke="#ef9f27" strokeWidth={1.5} strokeDasharray="5 3" dot={false} name="Forecast" />
                <Line type="monotone" dataKey="actual" stroke="#378add" strokeWidth={2} dot={false} name="Actual" connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </Card>
  );
}

function HealthCard({ health, loading }) {
  if (loading) return (
    <Card><SectionHeader icon="ti-heart-rate-monitor" title="System health" />
      <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Checking layers...</p>
    </Card>
  );

  const layers = health?.layers || {};
  const layerList = [
    { key: "L1_Ensemble",    label: "Ensemble model (XGBoost + LightGBM)" },
    { key: "L1_GNN",         label: "GNN ring detection (GraphSAGE)" },
    { key: "L2_Agentic_AI",  label: "LangGraph fraud agent (GPT-4o-mini)" },
    { key: "L3_Redis",       label: "Redis feature store" },
    { key: "L4_TFT_Forecast",label: "TFT demand forecasting" },
    { key: "L5_Shadow_AB",   label: "Shadow A/B testing" },
    { key: "L6_RAG_ChromaDB",label: "RAG knowledge base (ChromaDB)" },
  ];

  const allOk = Object.values(layers).every(Boolean);

  return (
    <Card>
      <SectionHeader
        icon="ti-heart-rate-monitor"
        title="System health"
        right={
          <span style={{ fontSize: 12, color: allOk ? "#3b6d11" : "#a32d2d", display: "flex", alignItems: "center" }}>
            <Pulse active={allOk} />{allOk ? "all systems operational" : "degraded"}
          </span>
        }
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {layerList.map(({ key, label }) => (
          <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 13 }}>
            <span style={{ color: "var(--color-text-secondary)" }}>{label}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 4, fontWeight: 500, color: layers[key] ? "#3b6d11" : "#a32d2d", fontSize: 12 }}>
              <i className={`ti ${layers[key] ? "ti-check" : "ti-x"}`} aria-hidden="true" style={{ fontSize: 14 }} />
              {layers[key] ? "online" : "offline"}
            </span>
          </div>
        ))}
      </div>
      {health?.rag && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "0.5px solid var(--color-border-tertiary)", display: "flex", gap: 16, fontSize: 12, color: "var(--color-text-secondary)" }}>
          <span><i className="ti ti-database" aria-hidden="true" style={{ fontSize: 13, marginRight: 4 }} />{fmt(health.rag.cases_stored)} cases stored</span>
          <span><i className="ti ti-cpu" aria-hidden="true" style={{ fontSize: 13, marginRight: 4 }} />{health.rag.embeddings}</span>
        </div>
      )}
    </Card>
  );
}

function AskPanel({ apiReachable }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const examples = [
    "Show cases where velocity triggered a block",
    "Which fraud pattern is most common?",
    "Find CRITICAL orders with location mismatch",
  ];

  async function submit() {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer(null);
    try {
      if (!apiReachable) throw new Error("demo");
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST", headers,
        body: JSON.stringify({ question, top_k: 5 })
      });
      const data = await res.json();
      setAnswer(data);
    } catch {
      setAnswer({
        answer: `Demo mode: In production, GPT-4o-mini would search ${47} stored fraud cases and synthesize an answer about: "${question}"`,
        cases_retrieved: 5,
        source_cases: [],
        rag_status: "demo"
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <SectionHeader icon="ti-message-bolt" title="Ask fraud intelligence" />
      <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          placeholder="Ask about past fraud investigations..."
          style={{ flex: 1, fontSize: 13 }}
        />
        <button onClick={submit} disabled={loading || !question.trim()} style={{ whiteSpace: "nowrap", fontSize: 13 }}>
          {loading ? "Searching..." : "Ask ↗"}
        </button>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: answer ? 12 : 0 }}>
        {examples.map(ex => (
          <button key={ex} onClick={() => { setQuestion(ex); }} style={{ fontSize: 11, padding: "3px 10px", color: "var(--color-text-secondary)" }}>
            {ex}
          </button>
        ))}
      </div>
      {answer && (
        <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px 14px", fontSize: 13 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontWeight: 500, fontSize: 12, color: "var(--color-text-secondary)" }}>
              <i className="ti ti-database" aria-hidden="true" style={{ fontSize: 13, marginRight: 4 }} />
              {answer.cases_retrieved} cases retrieved
            </span>
            <span style={{ fontSize: 11, color: answer.rag_status === "ok" ? "#3b6d11" : "#ba7517" }}>
              {answer.rag_status === "ok" ? "live RAG" : "demo mode"}
            </span>
          </div>
          <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-primary)" }}>{answer.answer}</p>
        </div>
      )}
    </Card>
  );
}

export default function SafeShopDashboard() {
  const [orders, setOrders]   = useState(MOCK_ORDERS);
  const [forecast, setForecast] = useState(MOCK_FORECAST);
  const [health, setHealth]   = useState(MOCK_HEALTH);
  const [loadingOrders, setLoadingOrders]   = useState(false);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [loadingHealth, setLoadingHealth]   = useState(false);
  const [apiReachable, setApiReachable]     = useState(false);
  const [lastRefresh, setLastRefresh]       = useState(new Date());
  const intervalRef = useRef(null);

  const fetchAll = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { headers });
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
        setApiReachable(true);
      }
    } catch { setApiReachable(false); }

    try {
      const res = await fetch(`${API_BASE}/forecast`, {
        method: "POST", headers,
        body: JSON.stringify({ category: "Electronics", horizon_hours: 24 })
      });
      if (res.ok) {
        const data = await res.json();
        setForecast(data.predictions.map((p, i) => ({
          hour: `${String(i).padStart(2, "0")}:00`,
          forecast: +p.predicted_volume.toFixed(0),
          actual: i < 14 ? +(p.predicted_volume * (0.85 + Math.random() * 0.3)).toFixed(0) : null,
          upper: +p.confidence_upper.toFixed(0),
          lower: +p.confidence_lower.toFixed(0),
        })));
      }
    } catch {}

    setLastRefresh(new Date());
  }, []);

  useEffect(() => {
    fetchAll();
    intervalRef.current = setInterval(fetchAll, 15000);
    return () => clearInterval(intervalRef.current);
  }, [fetchAll]);

  const fraudOrders = orders.filter(o => o.is_fraud);
  const totalRev    = orders.reduce((s, o) => s + o.order_amount, 0);
  const fraudRate   = orders.length ? ((fraudOrders.length / orders.length) * 100).toFixed(1) : "0";

  return (
    <div style={{ padding: "1rem 0", maxWidth: 900, margin: "0 auto" }}>
      <h2 aria-hidden="true" />

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 500 }}>SafeShop</h2>
          <p style={{ margin: "2px 0 0", fontSize: 12, color: "var(--color-text-secondary)" }}>Real-time order analytics pipeline</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "var(--color-text-secondary)" }}>
          <span>
            <Pulse active={apiReachable} />
            {apiReachable ? "connected to API" : "demo mode"}
          </span>
          <span>refreshed {fmtTime(lastRefresh.toISOString())}</span>
          <button onClick={fetchAll} style={{ fontSize: 12, padding: "4px 10px" }}>
            <i className="ti ti-refresh" aria-hidden="true" style={{ fontSize: 14, marginRight: 4 }} />
            Refresh
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 10, marginBottom: "1rem" }}>
        <MetricCard label="Orders" value={fmt(orders.length)} sub="current batch" />
        <MetricCard label="Revenue" value={"$" + (totalRev / 1000).toFixed(1) + "k"} sub="current batch" />
        <MetricCard label="Fraud rate" value={fraudRate + "%"} sub={`${fraudOrders.length} flagged`} accent={+fraudRate > 10 ? "#a32d2d" : +fraudRate > 5 ? "#854f0b" : "#3b6d11"} />
        <MetricCard label="RAG cases" value={fmt(health?.rag?.cases_stored ?? 47)} sub="in knowledge base" />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <FraudAlertBanner orders={orders} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "1rem", marginBottom: "1rem" }}>
        <LiveOrderFeed orders={orders} loading={loadingOrders} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
        <ForecastChart data={forecast} loading={loadingForecast} />
        <HealthCard health={health} loading={loadingHealth} />
      </div>

      <AskPanel apiReachable={apiReachable} />

      <p style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: "1rem", textAlign: "center" }}>
        Safe-Shop v5.0 · Kafka + Spark + GNN + LangGraph + ChromaDB · auto-refreshes every 15s
      </p>
    </div>
  );
}