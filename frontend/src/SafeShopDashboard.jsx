import { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY  = import.meta.env.VITE_API_KEY  || "dev-secret-key";
const HEADERS  = { "X-API-KEY": API_KEY, "Content-Type": "application/json" };

const CATEGORIES   = ["Electronics","Fashion","Home & Garden","Sports","Beauty","Books","Gaming","Auto"];
const DEVICE_TYPES = ["mobile","desktop","tablet","smart_tv"];
const REGIONS      = ["IN-MH","IN-DL","IN-KA","IN-TN","IN-WB","IN-GJ","IN-RJ","IN-UP","IN-HR","IN-PB"];
const PAYMENT_MET  = ["UPI","Credit Card","Debit Card","Net Banking","Wallet","BNPL","COD"];

function sr(seed) { let x = Math.sin(seed+1)*10000; return x - Math.floor(x); }

function mkOrder(i, now = Date.now()) {
  const rng = (o=0) => sr(i*17+3+o);
  const score = +(rng(1)*0.98).toFixed(3);
  const risk = score>0.78?"CRITICAL":score>0.55?"HIGH":score>0.3?"MEDIUM":"LOW";
  return {
    order_id: `ORD-${2024000+i}`, user_id: `USR-${String(Math.floor(rng(2)*89999)+10000)}`,
    amount: +(rng(3)*49950+50).toFixed(2), category: CATEGORIES[Math.floor(rng(4)*CATEGORIES.length)],
    device: DEVICE_TYPES[Math.floor(rng(5)*DEVICE_TYPES.length)], region: REGIONS[Math.floor(rng(6)*REGIONS.length)],
    payment: PAYMENT_MET[Math.floor(rng(7)*PAYMENT_MET.length)], fraud_score: score, is_fraud: score>0.55, risk,
    velocity: Math.floor(rng(8)*12), location_mismatch: rng(9)>0.7,
    gnn_score: +(rng(10)*0.95).toFixed(3),
    shap_top: ["velocity_spike","location_mismatch","amount_zscore","device_fingerprint","ip_reputation"].slice(0,Math.floor(rng(11)*3)+1),
    ts: new Date(now - i*4200 - rng(12)*2000).toISOString(),
    latency_ms: Math.floor(rng(13)*85+12), blocked: score>0.78, agent_triggered: score>0.55 && score<0.78,
  };
}

function genOrders(n=40) { return Array.from({length:n},(_,i)=>mkOrder(i)); }

function genTimeline() {
  const now = Date.now();
  return Array.from({length:60},(_,i)=>{
    const t = new Date(now-(59-i)*60000);
    const base = 12+Math.sin((i/60)*Math.PI*2)*4;
    const fraud = Math.floor(base*(0.08+sr(i*7)*0.07));
    return { time:`${String(t.getHours()).padStart(2,"0")}:${String(t.getMinutes()).padStart(2,"0")}`, total:Math.floor(base+(sr(i*3)-0.5)*3), fraud, blocked:Math.floor(fraud*0.8) };
  });
}

function genForecast() {
  const now = Date.now();
  return Array.from({length:48},(_,i)=>{
    const h = new Date(now-(47-i)*1800000);
    const base = 800+Math.sin((i/48)*Math.PI*2)*250+Math.sin((i/12)*Math.PI)*120;
    return { label:`${String(h.getHours()).padStart(2,"0")}:${i%2===0?"00":"30"}`, forecast:+(base+(sr(i*5)-0.5)*30).toFixed(0), actual: i<36?+(base+(sr(i*11)-0.5)*80).toFixed(0):null, upper:+(base*1.12).toFixed(0), lower:+(base*0.88).toFixed(0) };
  });
}

function genMetrics() {
  return Array.from({length:30},(_,i)=>({ day:`D-${30-i}`, pr_auc:+(0.78+sr(i*3)*0.12).toFixed(3), f1:+(0.81+sr(i*7)*0.1).toFixed(3), precision:+(0.84+sr(i*11)*0.08).toFixed(3), recall:+(0.77+sr(i*13)*0.12).toFixed(3) }));
}

const LAYER_PERF = [
  { layer:"Ensemble", calls:24821, p50:18, p99:52, blocked:1823, precision:0.913 },
  { layer:"GNN", calls:12440, p50:31, p99:89, blocked:704, precision:0.881 },
  { layer:"Agent", calls:3182, p50:412, p99:980, blocked:2190, precision:0.947 },
  { layer:"Redis", calls:48202, p50:2, p99:8, blocked:0, precision:null },
  { layer:"TFT", calls:1440, p50:140, p99:310, blocked:0, precision:null },
  { layer:"Firewall", calls:9841, p50:1, p99:4, blocked:9841, precision:null },
];

const INIT_ORDERS = genOrders(40);
const INIT_TL = genTimeline();
const INIT_FORECAST = genForecast();
const INIT_METRICS = genMetrics();

const fmtNum = n => new Intl.NumberFormat().format(n);
const fmtPct = n => `${(n*100).toFixed(1)}%`;
const fmtMs = n => `${n}ms`;
const fmtTime = iso => { const d=new Date(iso); return `${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}:${String(d.getSeconds()).padStart(2,"0")}`; };

function riskStyle(r) {
  if(r==="CRITICAL") return {color:"#f87171",bg:"rgba(239,68,68,0.12)",border:"rgba(239,68,68,0.35)"};
  if(r==="HIGH") return {color:"#fbbf24",bg:"rgba(245,158,11,0.12)",border:"rgba(245,158,11,0.35)"};
  if(r==="MEDIUM") return {color:"#60a5fa",bg:"rgba(96,165,250,0.10)",border:"rgba(96,165,250,0.3)"};
  return {color:"#4ade80",bg:"rgba(74,222,128,0.08)",border:"rgba(74,222,128,0.25)"};
}

function RiskBadge({level,small}) {
  const s=riskStyle(level);
  return <span style={{fontSize:small?10:11,fontWeight:600,padding:small?"1px 6px":"2px 8px",borderRadius:3,background:s.bg,color:s.color,border:`1px solid ${s.border}`,letterSpacing:"0.05em",fontFamily:"var(--font-mono)"}}>{level}</span>;
}

function ScoreBar({value}) {
  const pct=Math.min(value*100,100);
  const color=value>0.78?"#ef4444":value>0.55?"#f59e0b":value>0.3?"#3b82f6":"#22c55e";
  return (
    <div style={{display:"flex",alignItems:"center",gap:6}}>
      <div style={{flex:1,height:4,background:"rgba(255,255,255,0.06)",borderRadius:2,overflow:"hidden"}}>
        <div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:2,transition:"width 0.4s ease"}}/>
      </div>
      <span style={{fontSize:10,fontFamily:"var(--font-mono)",color,minWidth:34,textAlign:"right"}}>{value.toFixed(3)}</span>
    </div>
  );
}

function LiveDot({active=true}) {
  return (
    <span style={{position:"relative",display:"inline-flex",marginRight:4}}>
      <span style={{width:7,height:7,borderRadius:"50%",background:active?"#22c55e":"#ef4444",display:"block"}}/>
      {active && <span style={{position:"absolute",inset:0,borderRadius:"50%",background:"#22c55e",animation:"threat-ping 1.5s ease-out infinite"}}/>}
    </span>
  );
}

const ChartTooltip = ({active,payload,label}) => {
  if(!active || !payload?.length) return null;
  return (
    <div style={{background:"#1c2030",border:"1px solid rgba(255,255,255,0.1)",borderRadius:6,padding:"8px 12px",fontSize:12,boxShadow:"0 8px 24px rgba(0,0,0,0.5)"}}>
      <p style={{color:"#7c8399",marginBottom:4,fontSize:11}}>{label}</p>
      {payload.map((p,i)=><p key={i} style={{color:p.color||p.fill,marginBottom:1}}><span style={{color:"#7c8399"}}>{p.name}: </span>{p.value ?? "—"}</p>)}
    </div>
  );
};

function Card({children,style,glow}) {
  return <div style={{background:"var(--bg-card)",border:"1px solid var(--border)",borderRadius:"var(--radius-lg)",overflow:"hidden",boxShadow:glow||"none",...style}}>{children}</div>;
}

function CardHeader({title,subtitle,right,icon,accent}) {
  return (
    <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 14px",borderBottom:"1px solid var(--border)",background:"rgba(255,255,255,0.015)"}}>
      <div style={{display:"flex",alignItems:"center",gap:8}}>
        {icon && <div style={{width:26,height:26,borderRadius:6,background:accent?`${accent}22`:"rgba(255,255,255,0.05)",display:"flex",alignItems:"center",justifyContent:"center",border:accent?`1px solid ${accent}40`:"1px solid var(--border)"}}>
          <i className={`ti ${icon}`} style={{fontSize:13,color:accent||"var(--text-secondary)"}}/>
        </div>}
        <div>
          <p style={{fontSize:12,fontWeight:600,color:"var(--text-primary)",letterSpacing:"0.01em"}}>{title}</p>
          {subtitle && <p style={{fontSize:10,color:"var(--text-muted)",marginTop:1}}>{subtitle}</p>}
        </div>
      </div>
      {right}
    </div>
  );
}

/* ── TOP KPI METRICS ────────────────────────────────────────────────────────── */
function TopMetrics({orders}) {
  const crit=orders.filter(o=>o.risk==="CRITICAL").length;
  const fraud=orders.filter(o=>o.is_fraud).length;
  const blocked=orders.filter(o=>o.blocked).length;
  const gmv=orders.reduce((s,o)=>s+o.amount,0);
  const fraudRate=orders.length?fraud/orders.length:0;
  const avgLat=orders.length?Math.round(orders.reduce((s,o)=>s+o.latency_ms,0)/orders.length):0;
  const agentCalls=orders.filter(o=>o.agent_triggered).length;

  const metrics=[
    {label:"Total GMV",value:`₹${(gmv/100000).toFixed(1)}L`,color:"#e8eaf0",sub:`${fmtNum(orders.length)} orders`,icon:"ti-shopping-bag",accent:"#3b82f6"},
    {label:"Fraud Rate",value:fmtPct(fraudRate),color:fraudRate>0.12?"#f87171":fraudRate>0.07?"#fbbf24":"#4ade80",sub:`${fraud} flagged`,icon:"ti-shield-x",accent:"#ef4444",glow:fraudRate>0.1},
    {label:"Critical Threats",value:crit,color:crit>3?"#f87171":"#fbbf24",sub:"need review",icon:"ti-alert-triangle",accent:"#f59e0b",glow:crit>3},
    {label:"Auto-Blocked",value:blocked,color:"#c084fc",sub:"Firewall L9",icon:"ti-ban",accent:"#8b5cf6"},
    {label:"Agent Calls",value:agentCalls,color:"#67e8f9",sub:"L2 LangGraph",icon:"ti-robot",accent:"#06b6d4"},
    {label:"Avg Latency",value:fmtMs(avgLat),color:avgLat>60?"#fbbf24":"#4ade80",sub:"P50 inference",icon:"ti-clock-bolt",accent:"#10b981"},
  ];

  return (
    <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:10}}>
      {metrics.map((m,i)=>(
        <div key={i} style={{background:"var(--bg-card)",border:`1px solid ${m.glow?"rgba(239,68,68,0.3)":"var(--border)"}`,borderRadius:"var(--radius-md)",padding:"12px 14px",position:"relative",overflow:"hidden",boxShadow:m.glow?"0 0 16px rgba(239,68,68,0.15)":"none",animation:`fade-up 0.3s ease ${i*0.05}s both`}}>
          <div style={{position:"absolute",top:0,left:0,right:0,height:2,background:`linear-gradient(90deg,${m.accent}00,${m.accent}99,${m.accent}00)`}}/>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}>
            <span style={{fontSize:9,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:"0.08em",lineHeight:1.3}}>{m.label}</span>
            <div style={{width:22,height:22,borderRadius:4,background:`${m.accent}18`,display:"flex",alignItems:"center",justifyContent:"center"}}>
              <i className={`ti ${m.icon}`} style={{fontSize:11,color:m.accent}}/>
            </div>
          </div>
          <p style={{fontSize:20,fontWeight:700,color:m.color,fontFamily:"var(--font-mono)",letterSpacing:"-0.02em",marginBottom:2}}>{m.value}</p>
          <p style={{fontSize:9,color:"var(--text-muted)"}}>{m.sub}</p>
        </div>
      ))}
    </div>
  );
}

/* ── LIVE ORDER TABLE ───────────────────────────────────────────────────────── */
function LiveOrderFeed({orders}) {
  const [selected,setSelected] = useState(null);
  const [filter,setFilter] = useState("ALL");
  const filtered = filter==="ALL"?orders:orders.filter(o=>o.risk===filter);
  const displayed = filtered.slice(0,20);

  return (
    <Card style={{display:"flex",flexDirection:"column"}}>
      <CardHeader title="Live Transaction Stream" subtitle={`${orders.length} orders · streaming`} icon="ti-activity" accent="#3b82f6"
        right={<div style={{display:"flex",gap:5,alignItems:"center"}}>
          <LiveDot/><span style={{fontSize:10,color:"var(--text-secondary)",marginRight:8}}>LIVE</span>
          {["ALL","CRITICAL","HIGH","MEDIUM","LOW"].map(f=>(
            <button key={f} onClick={()=>setFilter(f)} style={{fontSize:9,padding:"2px 7px",background:filter===f?(f==="CRITICAL"?"rgba(239,68,68,0.2)":f==="HIGH"?"rgba(245,158,11,0.2)":"rgba(59,130,246,0.2)"):"transparent",border:`1px solid ${filter===f?"rgba(255,255,255,0.18)":"var(--border)"}`,color:filter===f?"var(--text-primary)":"var(--text-muted)",letterSpacing:"0.04em"}}>{f}</button>
          ))}
        </div>}
      />
      <div style={{overflowY:"auto",maxHeight:340}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
          <thead style={{position:"sticky",top:0,background:"var(--bg-card)",zIndex:1}}>
            <tr>
              {["Order ID","User","Amount","Category","Device","Region","Payment","Score","Risk","Agent","Latency","Status"].map(h=>(
                <th key={h} style={{textAlign:"left",padding:"7px 8px",fontSize:9,color:"var(--text-muted)",fontWeight:600,letterSpacing:"0.07em",textTransform:"uppercase",borderBottom:"1px solid var(--border)",whiteSpace:"nowrap"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayed.map((o,i)=>{
              const isSel = selected?.order_id===o.order_id;
              return (
                <tr key={o.order_id} onClick={()=>setSelected(isSel?null:o)}
                  style={{background:isSel?"rgba(59,130,246,0.08)":o.risk==="CRITICAL"?"rgba(239,68,68,0.04)":"transparent",borderBottom:"1px solid rgba(255,255,255,0.035)",cursor:"pointer",transition:"background 0.1s",animation:i<3?`slide-in-right 0.3s ease ${i*0.04}s both`:"none"}}
                  onMouseEnter={e=>{e.currentTarget.style.background="rgba(255,255,255,0.03)";}}
                  onMouseLeave={e=>{e.currentTarget.style.background=isSel?"rgba(59,130,246,0.08)":o.risk==="CRITICAL"?"rgba(239,68,68,0.04)":"transparent";}}
                >
                  <td style={{padding:"6px 8px",fontFamily:"var(--font-mono)",color:"#60a5fa",fontSize:10,whiteSpace:"nowrap"}}>
                    {o.risk==="CRITICAL" && <span style={{marginRight:3,color:"#f87171",animation:"blink-critical 1s infinite"}}>&#x2B24;</span>}
                    {o.order_id}
                  </td>
                  <td style={{padding:"6px 8px",color:"var(--text-secondary)",fontFamily:"var(--font-mono)",fontSize:10}}>{o.user_id}</td>
                  <td style={{padding:"6px 8px",fontWeight:600,color:o.amount>20000?"#fbbf24":"var(--text-primary)",whiteSpace:"nowrap"}}>&#8377;{fmtNum(o.amount.toFixed(0))}</td>
                  <td style={{padding:"6px 8px",color:"var(--text-secondary)"}}>{o.category}</td>
                  <td style={{padding:"6px 8px",color:"var(--text-muted)",fontSize:10}}>{o.device}</td>
                  <td style={{padding:"6px 8px",color:"var(--text-muted)",fontFamily:"var(--font-mono)",fontSize:10}}>{o.region}</td>
                  <td style={{padding:"6px 8px",color:"var(--text-muted)"}}>{o.payment}</td>
                  <td style={{padding:"6px 8px",minWidth:90}}><ScoreBar value={o.fraud_score}/></td>
                  <td style={{padding:"6px 8px"}}><RiskBadge level={o.risk} small/></td>
                  <td style={{padding:"6px 8px",textAlign:"center"}}>
                    {o.agent_triggered ? <i className="ti ti-robot" style={{color:"#67e8f9",fontSize:11}}/> : <span style={{color:"var(--text-muted)",fontSize:9}}>—</span>}
                  </td>
                  <td style={{padding:"6px 8px",fontFamily:"var(--font-mono)",color:o.latency_ms>60?"#fbbf24":"var(--text-muted)",fontSize:10}}>{o.latency_ms}ms</td>
                  <td style={{padding:"6px 8px"}}>
                    {o.blocked ? <span style={{fontSize:9,color:"#f87171",background:"rgba(239,68,68,0.12)",padding:"1px 6px",borderRadius:3,border:"1px solid rgba(239,68,68,0.25)",fontFamily:"var(--font-mono)"}}>BLOCKED</span>
                    : o.is_fraud ? <span style={{fontSize:9,color:"#fbbf24",background:"rgba(245,158,11,0.1)",padding:"1px 6px",borderRadius:3,border:"1px solid rgba(245,158,11,0.25)",fontFamily:"var(--font-mono)"}}>REVIEW</span>
                    : <span style={{fontSize:9,color:"#4ade80",background:"rgba(74,222,128,0.08)",padding:"1px 6px",borderRadius:3,border:"1px solid rgba(74,222,128,0.2)",fontFamily:"var(--font-mono)"}}>PASS</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {selected && (
        <div style={{borderTop:"1px solid var(--border)",padding:"12px 14px",background:"rgba(59,130,246,0.05)",animation:"fade-up 0.2s ease"}}>
          <p style={{fontSize:9,color:"var(--text-muted)",marginBottom:6,textTransform:"uppercase",letterSpacing:"0.07em"}}>Order Detail — {selected.order_id}</p>
          <div style={{display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-start"}}>
            {[["Ensemble",selected.fraud_score.toFixed(4),riskStyle(selected.risk).color],["GNN",selected.gnn_score.toFixed(4),"#c084fc"],["Velocity",`${selected.velocity}x/min`,selected.velocity>8?"#f87171":"var(--text-secondary)"],["Loc Mismatch",selected.location_mismatch?"YES":"NO",selected.location_mismatch?"#f87171":"#4ade80"],["Latency",`${selected.latency_ms}ms`,"var(--text-muted)"],["Time",fmtTime(selected.ts),"var(--text-muted)"]].map(([l,v,c],j)=>(
              <div key={j}><p style={{fontSize:9,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:"0.07em",marginBottom:2}}>{l}</p><p style={{fontSize:13,fontWeight:600,color:c,fontFamily:"var(--font-mono)"}}>{v}</p></div>
            ))}
            <div>
              <p style={{fontSize:9,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:"0.07em",marginBottom:4}}>SHAP Drivers (L10)</p>
              <div style={{display:"flex",gap:5}}>
                {selected.shap_top.map((s,k)=><span key={k} style={{fontSize:9,padding:"2px 7px",borderRadius:3,background:"rgba(139,92,246,0.15)",color:"#c084fc",border:"1px solid rgba(139,92,246,0.3)",fontFamily:"var(--font-mono)"}}>{s}</span>)}
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

/* ── THREAT TIMELINE CHART ──────────────────────────────────────────────────── */
function FraudTimeline({data}) {
  return (
    <Card>
      <CardHeader title="Threat Volume — 60min" subtitle="orders / fraud / blocked per min" icon="ti-chart-area-line" accent="#ef4444" right={<span style={{fontSize:10,color:"var(--text-muted)"}}>Rolling window</span>}/>
      <div style={{padding:"12px 6px 6px"}}>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:-16}}>
            <defs>
              <linearGradient id="gT" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/><stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/></linearGradient>
              <linearGradient id="gF" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/><stop offset="95%" stopColor="#ef4444" stopOpacity={0}/></linearGradient>
              <linearGradient id="gB" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.25}/><stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/></linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
            <XAxis dataKey="time" tick={{fontSize:9,fill:"#4a5168"}} interval={9}/>
            <YAxis tick={{fontSize:9,fill:"#4a5168"}}/>
            <Tooltip content={<ChartTooltip/>}/>
            <Area type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={1.5} fill="url(#gT)" name="Total" dot={false}/>
            <Area type="monotone" dataKey="fraud" stroke="#ef4444" strokeWidth={1.5} fill="url(#gF)" name="Fraud" dot={false}/>
            <Area type="monotone" dataKey="blocked" stroke="#8b5cf6" strokeWidth={1.5} fill="url(#gB)" name="Blocked" dot={false}/>
          </AreaChart>
        </ResponsiveContainer>
        <div style={{display:"flex",gap:14,justifyContent:"center",marginTop:6}}>
          {[["#3b82f6","Total"],["#ef4444","Fraud"],["#8b5cf6","Blocked"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:4}}><div style={{width:16,height:2,background:c,borderRadius:1}}/><span style={{fontSize:9,color:"var(--text-muted)"}}>{l}</span></div>
          ))}
        </div>
      </div>
    </Card>
  );
}

/* ── MODEL PERFORMANCE CHART ────────────────────────────────────────────────── */
function ModelPerformance({data}) {
  const latest = data[data.length-1];
  return (
    <Card>
      <CardHeader title="Model Metrics — 30d" subtitle="Champion ensemble" icon="ti-chart-dots" accent="#8b5cf6"
        right={<div style={{display:"flex",gap:10,fontSize:10}}>
          <span style={{color:"#c084fc"}}>PR-AUC <strong style={{fontFamily:"var(--font-mono)",color:"#e8eaf0"}}>{latest?.pr_auc}</strong></span>
          <span style={{color:"#67e8f9"}}>F1 <strong style={{fontFamily:"var(--font-mono)",color:"#e8eaf0"}}>{latest?.f1}</strong></span>
        </div>}
      />
      <div style={{padding:"12px 6px 6px"}}>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={data} margin={{top:4,right:6,bottom:0,left:-16}}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
            <XAxis dataKey="day" tick={{fontSize:9,fill:"#4a5168"}} interval={6}/>
            <YAxis domain={[0.7,1.0]} tick={{fontSize:9,fill:"#4a5168"}}/>
            <Tooltip content={<ChartTooltip/>}/>
            <ReferenceLine y={0.85} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 3"/>
            <Line type="monotone" dataKey="pr_auc" stroke="#8b5cf6" strokeWidth={2} dot={false} name="PR-AUC"/>
            <Line type="monotone" dataKey="f1" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="F1"/>
            <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={1} dot={false} name="Precision" strokeDasharray="4 2"/>
            <Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={1} dot={false} name="Recall" strokeDasharray="4 2"/>
          </LineChart>
        </ResponsiveContainer>
        <div style={{display:"flex",gap:10,justifyContent:"center",marginTop:6}}>
          {[["#8b5cf6","PR-AUC"],["#06b6d4","F1"],["#10b981","Prec"],["#f59e0b","Recall"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:3}}><div style={{width:14,height:2,background:c}}/><span style={{fontSize:9,color:"var(--text-muted)"}}>{l}</span></div>
          ))}
        </div>
      </div>
    </Card>
  );
}

/* ── DEMAND FORECAST CHART ──────────────────────────────────────────────────── */
function DemandForecast({data}) {
  return (
    <Card>
      <CardHeader title="Demand Forecast — TFT L4" subtitle="48h window · 90% CI" icon="ti-trending-up" accent="#10b981"
        right={<span style={{fontSize:10,color:"#4ade80"}}><i className="ti ti-cpu" style={{marginRight:3,fontSize:10}}/>Active</span>}
      />
      <div style={{padding:"12px 6px 6px"}}>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:-16}}>
            <defs>
              <linearGradient id="gBand" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.08}/><stop offset="95%" stopColor="#10b981" stopOpacity={0.01}/></linearGradient>
              <linearGradient id="gAct" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15}/><stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/></linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
            <XAxis dataKey="label" tick={{fontSize:9,fill:"#4a5168"}} interval={7}/>
            <YAxis tick={{fontSize:9,fill:"#4a5168"}}/>
            <Tooltip content={<ChartTooltip/>}/>
            <Area type="monotone" dataKey="upper" stroke="none" fill="url(#gBand)" name="Upper"/>
            <Area type="monotone" dataKey="lower" stroke="none" fill="var(--bg-card)" name="Lower"/>
            <Area type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} fill="url(#gAct)" name="Actual" dot={false} connectNulls={false}/>
            <Line type="monotone" dataKey="forecast" stroke="#10b981" strokeWidth={1.5} strokeDasharray="5 3" dot={false} name="Forecast"/>
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/* ── INTELLIGENCE SHIELD LAYER STATUS ───────────────────────────────────────── */
function LayerStatus({health,layerPerf}) {
  const layers = health?.layers || {};
  const config = [
    {key:"L1_Ensemble",label:"L1  Ensemble",sub:"XGBoost + LightGBM + IF",icon:"ti-stack-2",accent:"#8b5cf6",lk:"Ensemble"},
    {key:"L1_GNN",label:"L1  GNN Ring",sub:"GraphSAGE · Ring Detect",icon:"ti-network",accent:"#8b5cf6",lk:"GNN"},
    {key:"L2_Agentic_AI",label:"L2  Agent",sub:"LangGraph · GPT-4o-mini",icon:"ti-robot",accent:"#06b6d4",lk:"Agent"},
    {key:"L3_Redis",label:"L3  Feature Store",sub:"Redis · <3ms lookup",icon:"ti-database",accent:"#f59e0b",lk:"Redis"},
    {key:"L4_TFT_Forecast",label:"L4  TFT Forecast",sub:"Temporal Fusion Transformer",icon:"ti-trending-up",accent:"#10b981",lk:"TFT"},
    {key:"L5_Shadow_AB",label:"L5  Shadow A/B",sub:"Champion vs Challenger",icon:"ti-test-pipe",accent:"#f59e0b",lk:null},
    {key:"L6_RAG_ChromaDB",label:"L6  RAG KB",sub:"ChromaDB · OpenAI Embed",icon:"ti-brain",accent:"#8b5cf6",lk:null},
  ];
  return (
    <Card>
      <CardHeader title="Intelligence Shield" subtitle="10-Layer Fraud Defense" icon="ti-shield-half" accent="#8b5cf6"
        right={<div style={{display:"flex",alignItems:"center",gap:5}}><LiveDot/><span style={{fontSize:10,color:"#4ade80"}}>All Nominal</span></div>}
      />
      <div style={{padding:"6px 0"}}>
        {config.map(({key,label,sub,icon,accent,lk})=>{
          const ok = layers[key]!==false;
          const perf = lk ? layerPerf.find(p=>p.layer===lk) : null;
          return (
            <div key={key} style={{display:"flex",alignItems:"center",gap:8,padding:"7px 14px",borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
              <div style={{width:26,height:26,borderRadius:5,background:ok?`${accent}15`:"rgba(239,68,68,0.1)",border:`1px solid ${ok?accent+"30":"rgba(239,68,68,0.3)"}`,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
                <i className={`ti ${icon}`} style={{fontSize:12,color:ok?accent:"#ef4444"}}/>
              </div>
              <div style={{flex:1,minWidth:0}}>
                <p style={{fontSize:11,fontWeight:600,color:"var(--text-primary)",fontFamily:"var(--font-mono)"}}>{label}</p>
                <p style={{fontSize:9,color:"var(--text-muted)"}}>{sub}</p>
              </div>
              <div style={{display:"flex",gap:12,alignItems:"center"}}>
                {perf && <>
                  <div style={{textAlign:"right"}}><p style={{fontSize:9,color:"var(--text-muted)"}}>P50</p><p style={{fontSize:10,fontFamily:"var(--font-mono)",color:perf.p50>200?"#fbbf24":"#4ade80"}}>{perf.p50}ms</p></div>
                  <div style={{textAlign:"right"}}><p style={{fontSize:9,color:"var(--text-muted)"}}>P99</p><p style={{fontSize:10,fontFamily:"var(--font-mono)",color:perf.p99>500?"#f87171":"#fbbf24"}}>{perf.p99}ms</p></div>
                  {perf.blocked>0 && <div style={{textAlign:"right"}}><p style={{fontSize:9,color:"var(--text-muted)"}}>Blocked</p><p style={{fontSize:10,fontFamily:"var(--font-mono)",color:"#f87171"}}>{fmtNum(perf.blocked)}</p></div>}
                  {perf.precision && <div style={{textAlign:"right"}}><p style={{fontSize:9,color:"var(--text-muted)"}}>Prec</p><p style={{fontSize:10,fontFamily:"var(--font-mono)",color:"#c084fc"}}>{fmtPct(perf.precision)}</p></div>}
                </>}
                <span style={{fontSize:9,padding:"2px 7px",borderRadius:3,fontFamily:"var(--font-mono)",background:ok?"rgba(34,197,94,0.1)":"rgba(239,68,68,0.1)",color:ok?"#4ade80":"#f87171",border:`1px solid ${ok?"rgba(34,197,94,0.2)":"rgba(239,68,68,0.25)"}`}}>{ok?"ONLINE":"OFFLINE"}</span>
              </div>
            </div>
          );
        })}
      </div>
      {health?.rag && <div style={{padding:"8px 14px",borderTop:"1px solid var(--border)",display:"flex",gap:16,background:"rgba(139,92,246,0.05)"}}>
        {[["RAG Cases",fmtNum(health.rag.cases_stored),"#c084fc"],["Vector DB","ChromaDB","var(--text-secondary)"],["Embeddings",health.rag.embeddings,"var(--text-secondary)"]].map(([l,v,c],i)=>(
          <div key={i}><p style={{fontSize:9,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:"0.07em",marginBottom:2}}>{l}</p><p style={{fontSize:12,fontWeight:600,color:c,fontFamily:"var(--font-mono)"}}>{v}</p></div>
        ))}
      </div>}
    </Card>
  );
}

/* ── SHADOW A/B TESTING PANEL ───────────────────────────────────────────────── */
function ShadowABPanel({orders}) {
  const total = orders.length||1;
  const champWins=Math.floor(total*0.61);
  const challWins=Math.floor(total*0.27);
  const barData = CATEGORIES.map((c,i)=>({ name:c.split(" ")[0], champion:+(0.80+sr(i*3)*0.12).toFixed(3), challenger:+(0.75+sr(i*7)*0.15).toFixed(3) }));
  return (
    <Card>
      <CardHeader title="Shadow A/B — L5" subtitle="Champion vs Challenger" icon="ti-test-pipe" accent="#f59e0b"
        right={<span style={{fontSize:10,color:"#fbbf24"}}>Champion +{((champWins/total-challWins/total)*100).toFixed(1)}pp</span>}
      />
      <div style={{padding:12}}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:12}}>
          {[{l:"Champion Wins",v:champWins,c:"#4ade80",ic:"ti-trophy"},{l:"Challenger Wins",v:challWins,c:"#f87171",ic:"ti-sword"},{l:"Draws",v:total-champWins-challWins,c:"#7c8399",ic:"ti-minus"},{l:"KS Drift",v:"0.023",c:"#4ade80",ic:"ti-chart-scatter"}].map((m,i)=>(
            <div key={i} style={{background:"rgba(255,255,255,0.025)",borderRadius:5,padding:"8px 10px",border:"1px solid var(--border)",display:"flex",alignItems:"center",gap:8}}>
              <i className={`ti ${m.ic}`} style={{fontSize:14,color:m.c}}/>
              <div><p style={{fontSize:9,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:"0.06em"}}>{m.l}</p><p style={{fontSize:14,fontWeight:700,color:m.c,fontFamily:"var(--font-mono)"}}>{m.v}</p></div>
            </div>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={barData} margin={{top:0,right:2,bottom:0,left:-22}} barGap={1}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
            <XAxis dataKey="name" tick={{fontSize:8,fill:"#4a5168"}}/>
            <YAxis domain={[0.6,1.0]} tick={{fontSize:8,fill:"#4a5168"}}/>
            <Tooltip content={<ChartTooltip/>}/>
            <Bar dataKey="champion" fill="#10b981" name="Champion" radius={[2,2,0,0]}/>
            <Bar dataKey="challenger" fill="#f59e0b" name="Challenger" radius={[2,2,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/* ── ACTIVE THREAT QUEUE ────────────────────────────────────────────────────── */
function ActiveAlerts({orders}) {
  const criticals = orders.filter(o=>o.risk==="CRITICAL").slice(0,6);
  const [dismissed,setDismissed] = useState(new Set());
  const active = criticals.filter(o=>!dismissed.has(o.order_id));
  return (
    <Card glow="0 0 20px rgba(239,68,68,0.08)">
      <CardHeader title="Active Threat Queue" subtitle="Requires action" icon="ti-bell-ringing" accent="#ef4444"
        right={<span style={{fontSize:10,color:"#f87171",animation:active.length>3?"blink-critical 1.5s infinite":"none"}}>{active.length} active</span>}
      />
      <div style={{padding:"6px 0",overflowY:"auto",maxHeight:260}}>
        {active.length===0 ? (
          <div style={{padding:"16px 14px",textAlign:"center"}}>
            <i className="ti ti-shield-check" style={{fontSize:22,color:"#4ade80",display:"block",marginBottom:4}}/>
            <p style={{fontSize:11,color:"#4ade80"}}>No active threats</p>
          </div>
        ) : active.map((o,i)=>(
          <div key={o.order_id} style={{display:"flex",alignItems:"flex-start",gap:8,padding:"8px 14px",borderBottom:"1px solid rgba(239,68,68,0.07)",animation:`fade-up 0.2s ease ${i*0.06}s both`}}>
            <div style={{width:6,height:6,borderRadius:"50%",marginTop:5,background:"#ef4444",flexShrink:0,boxShadow:"0 0 6px rgba(239,68,68,0.6)",animation:"pulse-dot 1s infinite"}}/>
            <div style={{flex:1,minWidth:0}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:2}}>
                <span style={{fontSize:11,fontFamily:"var(--font-mono)",color:"#60a5fa"}}>{o.order_id}</span>
                <span style={{fontSize:9,color:"var(--text-muted)"}}>{fmtTime(o.ts)}</span>
              </div>
              <p style={{fontSize:10,color:"var(--text-secondary)",marginBottom:3}}>
                &#8377;{fmtNum(o.amount.toFixed(0))} &middot; {o.category} &middot; {o.region} &middot; <span style={{color:"#f87171",fontFamily:"var(--font-mono)"}}>{o.fraud_score.toFixed(4)}</span>
              </p>
              <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>
                {o.shap_top.map((s,j)=><span key={j} style={{fontSize:8,padding:"1px 5px",borderRadius:2,background:"rgba(239,68,68,0.1)",color:"#fca5a5",border:"1px solid rgba(239,68,68,0.2)",fontFamily:"var(--font-mono)"}}>{s}</span>)}
              </div>
            </div>
            <div style={{display:"flex",gap:3,flexShrink:0}}>
              <button style={{fontSize:9,padding:"2px 7px",background:"rgba(239,68,68,0.15)",border:"1px solid rgba(239,68,68,0.3)",color:"#f87171"}}>Block</button>
              <button onClick={()=>setDismissed(d=>new Set([...d,o.order_id]))} style={{fontSize:9,padding:"2px 7px",color:"var(--text-muted)"}}>Dismiss</button>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ── RAG INTELLIGENCE PANEL ─────────────────────────────────────────────────── */
function RAGPanel({health}) {
  const [q,setQ] = useState("");
  const [ans,setAns] = useState(null);
  const [loading,setLoading] = useState(false);
  const examples = ["Show CRITICAL cases in Electronics","IPs with highest velocity","Top fraud patterns by category","Cases where GNN and ensemble disagreed"];

  async function submit() {
    if(!q.trim()) return;
    setLoading(true); setAns(null);
    try {
      const res = await fetch(`${API_BASE}/ask`,{method:"POST",headers:HEADERS,body:JSON.stringify({question:q,top_k:5})});
      if(res.ok) { setAns(await res.json()); } else throw new Error();
    } catch {
      setAns({answer:`[DEMO MODE] GPT-4o-mini would search ${health?.rag?.cases_stored??47} investigations for: "${q}". Uses ChromaDB + OpenAI embeddings for semantic retrieval and grounded analyst-grade answers.`,cases_retrieved:5,rag_status:"demo"});
    } finally { setLoading(false); }
  }

  return (
    <Card>
      <CardHeader title="Fraud Intelligence RAG — L6" subtitle="Natural language query over investigations" icon="ti-brain" accent="#8b5cf6"
        right={<span style={{fontSize:10,color:"#c084fc"}}><i className="ti ti-database" style={{marginRight:3,fontSize:10}}/>{fmtNum(health?.rag?.cases_stored??47)} cases</span>}
      />
      <div style={{padding:12}}>
        <div style={{display:"flex",gap:7,marginBottom:8}}>
          <div style={{flex:1,position:"relative"}}>
            <i className="ti ti-search" style={{position:"absolute",left:9,top:"50%",transform:"translateY(-50%)",fontSize:12,color:"var(--text-muted)",pointerEvents:"none"}}/>
            <input type="text" value={q} onChange={e=>setQ(e.target.value)} onKeyDown={e=>e.key==="Enter"&&submit()} placeholder="Ask about past fraud investigations..." style={{width:"100%",paddingLeft:28,fontSize:12}}/>
          </div>
          <button onClick={submit} disabled={loading||!q.trim()} style={{background:"rgba(139,92,246,0.15)",border:"1px solid rgba(139,92,246,0.35)",color:"#c084fc",padding:"5px 12px",fontWeight:600,fontSize:12}}>
            {loading ? "Searching..." : "Query \u2197"}
          </button>
        </div>
        <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom:ans?10:0}}>
          {examples.map(ex=><button key={ex} onClick={()=>setQ(ex)} style={{fontSize:10,padding:"2px 8px",background:"rgba(255,255,255,0.03)",color:"var(--text-muted)"}}>{ex}</button>)}
        </div>
        {ans && <div style={{background:"rgba(139,92,246,0.07)",borderRadius:7,padding:"10px 12px",border:"1px solid rgba(139,92,246,0.2)",animation:"fade-up 0.2s ease"}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
            <span style={{fontSize:10,color:"var(--text-muted)"}}>{ans.cases_retrieved} cases retrieved</span>
            <span style={{fontSize:9,color:ans.rag_status==="ok"?"#4ade80":"#fbbf24"}}>{ans.rag_status==="ok"?"\u25CF LIVE RAG":"\u25CB DEMO"}</span>
          </div>
          <p style={{fontSize:12,lineHeight:1.7,color:"var(--text-accent)",fontStyle:"italic"}}>{ans.answer}</p>
        </div>}
      </div>
    </Card>
  );
}

/* ── TOP BAR ────────────────────────────────────────────────────────────────── */
function TopBar({apiReachable,onRefresh}) {
  const [now,setNow] = useState(new Date());
  useEffect(()=>{const t=setInterval(()=>setNow(new Date()),1000);return()=>clearInterval(t);},[]);
  return (
    <div style={{height:50,background:"var(--bg-surface)",borderBottom:"1px solid var(--border)",display:"flex",alignItems:"center",padding:"0 18px",gap:14,flexShrink:0,position:"sticky",top:0,zIndex:100}}>
      <div style={{display:"flex",alignItems:"center",gap:9}}>
        <div style={{width:28,height:28,borderRadius:7,background:"linear-gradient(135deg,#3b82f6,#8b5cf6)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 0 10px rgba(59,130,246,0.35)"}}>
          <i className="ti ti-shield-bolt" style={{fontSize:15,color:"#fff"}}/>
        </div>
        <div>
          <p style={{fontSize:13,fontWeight:700,color:"var(--text-primary)",letterSpacing:"-0.01em"}}>SafeShop</p>
          <p style={{fontSize:9,color:"var(--text-muted)",letterSpacing:"0.06em"}}>ML FRAUD INTELLIGENCE PLATFORM</p>
        </div>
      </div>
      <div style={{width:1,height:26,background:"var(--border)",margin:"0 4px"}}/>
      {["Overview","Transactions","Models","Alerts","Settings"].map((t,i)=>(
        <button key={t} style={{fontSize:11,padding:"4px 10px",background:i===0?"rgba(59,130,246,0.15)":"transparent",border:i===0?"1px solid rgba(59,130,246,0.3)":"1px solid transparent",color:i===0?"#60a5fa":"var(--text-muted)"}}>{t}</button>
      ))}
      <div style={{flex:1}}/>
      <div style={{display:"flex",alignItems:"center",gap:12,fontSize:11}}>
        <div style={{display:"flex",alignItems:"center",gap:5}}>
          <LiveDot active={apiReachable}/>
          <span style={{color:apiReachable?"#4ade80":"#f87171",fontFamily:"var(--font-mono)",fontSize:10}}>{apiReachable?"API CONNECTED":"DEMO MODE"}</span>
        </div>
        <span style={{fontFamily:"var(--font-mono)",fontSize:10,color:"var(--text-muted)"}}>{now.toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false})} IST</span>
        <button onClick={onRefresh} style={{display:"flex",alignItems:"center",gap:4,fontSize:11,background:"rgba(59,130,246,0.1)",border:"1px solid rgba(59,130,246,0.22)",color:"#60a5fa",padding:"4px 10px"}}>
          <i className="ti ti-refresh" style={{fontSize:11}}/>Refresh
        </button>
        <div style={{width:26,height:26,borderRadius:"50%",background:"linear-gradient(135deg,#8b5cf6,#3b82f6)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:700,color:"#fff",cursor:"pointer"}}>AK</div>
      </div>
    </div>
  );
}

/* ── LEFT SIDEBAR ───────────────────────────────────────────────────────────── */
function Sidebar({orders}) {
  const crit = orders.filter(o=>o.risk==="CRITICAL").length;
  const high = orders.filter(o=>o.risk==="HIGH").length;
  const navItems = [
    {icon:"ti-layout-dashboard",label:"Dashboard",active:true},
    {icon:"ti-activity",label:"Live Feed",badge:orders.length},
    {icon:"ti-shield-x",label:"Threats",badge:crit||null,danger:true},
    {icon:"ti-robot",label:"AI Agents"},
    {icon:"ti-chart-dots",label:"Model Metrics"},
    {icon:"ti-test-pipe",label:"A/B Testing"},
    {icon:"ti-brain",label:"RAG KB"},
    {icon:"ti-database",label:"Data Lake"},
    {icon:"ti-report-analytics",label:"Reports"},
  ];
  const layerList = [
    {id:"L1",label:"Ensemble + GNN",c:"#4ade80"},{id:"L2",label:"Agent (LangGraph)",c:"#4ade80"},
    {id:"L3",label:"Redis Feature Store",c:"#4ade80"},{id:"L4",label:"TFT Forecast",c:"#4ade80"},
    {id:"L5",label:"Shadow A/B",c:"#4ade80"},{id:"L6",label:"RAG ChromaDB",c:"#4ade80"},
    {id:"L9",label:"Ghost Firewall",c:"#c084fc"},
  ];
  return (
    <div style={{width:195,flexShrink:0,background:"var(--bg-surface)",borderRight:"1px solid var(--border)",display:"flex",flexDirection:"column",overflowY:"auto"}}>
      <div style={{padding:"6px 8px",borderBottom:"1px solid var(--border)"}}>
        <p style={{fontSize:8,color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",padding:"7px 8px 4px"}}>Navigation</p>
        {navItems.map((item,i)=>(
          <div key={i} style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"6px 8px",borderRadius:5,cursor:"pointer",background:item.active?"rgba(59,130,246,0.12)":"transparent",border:item.active?"1px solid rgba(59,130,246,0.2)":"1px solid transparent",marginBottom:1}}>
            <div style={{display:"flex",alignItems:"center",gap:7}}>
              <i className={`ti ${item.icon}`} style={{fontSize:13,color:item.active?"#60a5fa":item.danger&&item.badge?"#f87171":"var(--text-muted)"}}/>
              <span style={{fontSize:11,color:item.active?"#60a5fa":"var(--text-secondary)"}}>{item.label}</span>
            </div>
            {item.badge && <span style={{fontSize:9,padding:"1px 4px",borderRadius:3,background:item.danger?"rgba(239,68,68,0.2)":"rgba(59,130,246,0.2)",color:item.danger?"#f87171":"#60a5fa",fontFamily:"var(--font-mono)",fontWeight:600}}>{item.badge}</span>}
          </div>
        ))}
      </div>
      <div style={{padding:"8px 14px",borderBottom:"1px solid var(--border)"}}>
        <p style={{fontSize:8,color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:5,marginTop:3}}>Shield Layers</p>
        {layerList.map(l=>(
          <div key={l.id} style={{display:"flex",alignItems:"center",gap:7,marginBottom:4}}>
            <span style={{width:5,height:5,borderRadius:"50%",background:l.c,flexShrink:0}}/>
            <span style={{fontSize:9,color:"var(--text-muted)",flex:1}}>{l.id} {l.label}</span>
          </div>
        ))}
      </div>
      <div style={{padding:"10px 14px"}}>
        <p style={{fontSize:8,color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:7}}>Threat Summary</p>
        {[{l:"CRITICAL",n:crit,c:"#f87171",bg:"rgba(239,68,68,0.08)"},{l:"HIGH",n:high,c:"#fbbf24",bg:"rgba(245,158,11,0.08)"},{l:"BLOCKED",n:orders.filter(o=>o.blocked).length,c:"#c084fc",bg:"rgba(139,92,246,0.08)"}].map(({l,n,c,bg})=>(
          <div key={l} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"4px 7px",borderRadius:4,background:bg,marginBottom:3}}>
            <span style={{fontSize:9,color:c,fontFamily:"var(--font-mono)"}}>{l}</span>
            <span style={{fontSize:12,fontWeight:700,color:c,fontFamily:"var(--font-mono)"}}>{n}</span>
          </div>
        ))}
      </div>
      <div style={{marginTop:"auto",padding:"10px 14px",borderTop:"1px solid var(--border)"}}>
        <p style={{fontSize:9,color:"var(--text-muted)"}}>SafeShop SOC v5.0</p>
        <p style={{fontSize:8,color:"var(--text-muted)",marginTop:1}}>Kafka &middot; Spark &middot; GNN &middot; LangGraph</p>
      </div>
    </div>
  );
}

/* ── MAIN DASHBOARD ─────────────────────────────────────────────────────────── */
export default function SafeShopDashboard() {
  const [orders,setOrders] = useState(INIT_ORDERS);
  const [timeline,setTimeline] = useState(INIT_TL);
  const [forecast,setForecast] = useState(INIT_FORECAST);
  const [metrics] = useState(INIT_METRICS);
  const [health,setHealth] = useState({
    status:"ok", version:"v5.0-rag",
    layers:{L1_Ensemble:true,L1_GNN:true,L2_Agentic_AI:true,L3_Redis:true,L4_TFT_Forecast:true,L5_Shadow_AB:true,L6_RAG_ChromaDB:true},
    rag:{cases_stored:47,embeddings:"text-embedding-3-small",status:"healthy"}
  });
  const [apiReachable,setApiReachable] = useState(false);
  const tickRef = useRef(0);

  const fetchAll = useCallback(async()=>{
    try { const res=await fetch(`${API_BASE}/health`,{headers:HEADERS}); if(res.ok){setHealth(await res.json());setApiReachable(true);} } catch { setApiReachable(false); }
    try {
      const res=await fetch(`${API_BASE}/forecast`,{method:"POST",headers:HEADERS,body:JSON.stringify({category:"Electronics",horizon_hours:24})});
      if(res.ok){ const d=await res.json(); setForecast(d.predictions.map((p,i)=>({label:`${String(i).padStart(2,"0")}:00`,forecast:+p.predicted_volume.toFixed(0),actual:i<18?+(p.predicted_volume*(0.85+Math.random()*0.3)).toFixed(0):null,upper:+p.confidence_upper.toFixed(0),lower:+p.confidence_lower.toFixed(0)}))); }
    } catch {}
  },[]);

  useEffect(()=>{
    fetchAll();
    const liveInterval = setInterval(()=>{
      tickRef.current += 1;
      setOrders(prev=>[mkOrder(1000+tickRef.current,Date.now()),...prev].slice(0,40));
      setTimeline(genTimeline());
    }, 1500);
    const apiInterval = setInterval(fetchAll, 20000);
    return ()=>{ clearInterval(liveInterval); clearInterval(apiInterval); };
  },[fetchAll]);

  return (
    <div style={{display:"flex",flexDirection:"column",height:"100vh",background:"var(--bg-base)"}}>
      <TopBar apiReachable={apiReachable} onRefresh={fetchAll}/>
      <div style={{display:"flex",flex:1,overflow:"hidden"}}>
        <Sidebar orders={orders}/>
        <div style={{flex:1,overflowY:"auto",padding:"14px 16px",display:"flex",flexDirection:"column",gap:12}}>
          <TopMetrics orders={orders}/>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
            <FraudTimeline data={timeline}/>
            <ModelPerformance data={metrics}/>
            <DemandForecast data={forecast}/>
          </div>
          <LiveOrderFeed orders={orders}/>
          <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr 1fr",gap:12}}>
            <LayerStatus health={health} layerPerf={LAYER_PERF}/>
            <ShadowABPanel orders={orders}/>
            <ActiveAlerts orders={orders}/>
          </div>
          <RAGPanel health={health}/>
          <p style={{fontSize:9,color:"var(--text-muted)",textAlign:"center",padding:"6px 0 2px"}}>SafeShop SOC v5.0 &middot; Kafka + Spark + GNN + LangGraph + ChromaDB + TFT &middot; Live stream every 1.5s</p>
        </div>
      </div>
    </div>
  );
}