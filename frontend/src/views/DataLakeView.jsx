import React from "react";

export default function DataLakeView() {
  const topics = [
    { topic: "transactions.live", partitions: 16, ingress: "4.8M msg/s", lag: "12 ms", status: "HEALTHY" },
    { topic: "gnn.subgraph.edges", partitions: 8, ingress: "840K msg/s", lag: "24 ms", status: "HEALTHY" },
    { topic: "agent.actions.audit", partitions: 4, ingress: "12K msg/s", lag: "4 ms", status: "HEALTHY" },
    { topic: "firewall.blocks.sink", partitions: 4, ingress: "48K msg/s", lag: "2 ms", status: "HEALTHY" },
  ];

  const features = [
    { name: "user_txn_count_10m", type: "INT32", ttl: "600s", p99_ms: 0.8, freshness: "< 1s" },
    { name: "card_token_velocity_1h", type: "INT32", ttl: "3600s", p99_ms: 0.9, freshness: "< 1s" },
    { name: "device_fingerprint_entropy", type: "FLOAT32", ttl: "86400s", p99_ms: 1.1, freshness: "< 5s" },
    { name: "billing_shipping_distance_km", type: "FLOAT32", ttl: "PERSIST", p99_ms: 1.2, freshness: "STATIC" },
    { name: "gnn_collusion_cluster_id", type: "INT64", ttl: "1800s", p99_ms: 1.4, freshness: "< 2s" },
    { name: "ip_asn_reputation_score", type: "FLOAT32", ttl: "7200s", p99_ms: 0.7, freshness: "< 3s" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Infrastructure Telemetry KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Kafka Stream Rate</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>4.8M msg/s</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>16 active broker partitions</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Redis Cache Hit Rate</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#4ade80", margin: "4px 0" }}>99.98%</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>1.2ms P99 key lookup</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Spark Microbatches</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#60a5fa", margin: "4px 0" }}>100ms</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>8 worker cluster executor</p>
        </div>

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "14px 18px" }}>
          <p style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Historical Warehouse</p>
          <p style={{ fontSize: 24, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#ffffff", margin: "4px 0" }}>48M rows</p>
          <p style={{ fontSize: 10, color: "var(--text-muted)" }}>2.4 TB Parquet on Iceberg</p>
        </div>
      </div>

      {/* Kafka Topics Status */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
        <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 12 }}>
          Kafka Event Streaming Topics
        </h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px" }}>TOPIC NAME</th>
              <th style={{ padding: "8px 12px" }}>PARTITIONS</th>
              <th style={{ padding: "8px 12px" }}>INGRESS THROUGHPUT</th>
              <th style={{ padding: "8px 12px" }}>CONSUMER LAG</th>
              <th style={{ padding: "8px 12px" }}>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {topics.map((t, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#ffffff" }}>{t.topic}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{t.partitions}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#ffffff", fontWeight: 600 }}>{t.ingress}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#4ade80" }}>{t.lag}</td>
                <td style={{ padding: "8px 12px" }}>
                  <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(34,197,94,0.1)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)", fontFamily: "var(--font-mono)" }}>
                    {t.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Online Feature Store Schema */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: 16 }}>
        <h3 style={{ fontSize: 12, fontWeight: 600, color: "#ffffff", marginBottom: 12 }}>
          Redis Online Feature Store &mdash; Active Features (L3)
        </h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px" }}>FEATURE NAME</th>
              <th style={{ padding: "8px 12px" }}>DATATYPE</th>
              <th style={{ padding: "8px 12px" }}>CACHE TTL</th>
              <th style={{ padding: "8px 12px" }}>P99 LOOKUP</th>
              <th style={{ padding: "8px 12px" }}>FRESHNESS</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#ffffff" }}>{f.name}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{f.type}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#a1a1aa" }}>{f.ttl}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#4ade80" }}>{f.p99_ms}ms</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#60a5fa" }}>{f.freshness}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
