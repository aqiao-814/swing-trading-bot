"""Render the five-year backtest findings into a self-contained HTML page.

Reads ``artifacts/backtest5y/findings.json`` (written by ``backtest_5y.py``) and
emits ``artifacts/backtest5y/findings.html`` -- a single file with inline CSS,
an embedded JSON data island, and vanilla-JS SVG charts (no external requests,
so it works as a GitHub Pages page or a claude.ai Artifact unchanged).

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/build_findings_page.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "artifacts" / "backtest5y" / "findings.json"
OUT = ROOT / "artifacts" / "backtest5y" / "findings.html"

PAGE = r"""<style>
:root{
  color-scheme: light;
  --plane:#eef0f3; --surface:#ffffff; --surface-2:#f7f8fa;
  --ink:#0f1216; --ink-2:#464c55; --muted:#878d96;
  --hair:#e3e6ea; --hair-2:#eef0f3; --baseline:#c8ccd3;
  --accent:#2a78d6;
  --s-port:#2a78d6; --s-qqq:#008300; --s-spy:#d05389; --s-ew:#c98500;
  --pos:#0a7d3c; --neg:#c4322f; --warn:#b9860b;
  --shadow:0 1px 2px rgba(15,18,22,.05),0 8px 24px -12px rgba(15,18,22,.14);
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --plane:#0a0c0f; --surface:#14171b; --surface-2:#1a1e23;
    --ink:#f2f4f7; --ink-2:#aab1bb; --muted:#727a84;
    --hair:#242a31; --hair-2:#1d2228; --baseline:#333a42;
    --accent:#3987e5;
    --s-port:#3987e5; --s-qqq:#2aa32a; --s-spy:#d55181; --s-ew:#d9a441;
    --pos:#38c46a; --neg:#f0645f; --warn:#e0a92a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px -14px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --plane:#0a0c0f; --surface:#14171b; --surface-2:#1a1e23;
  --ink:#f2f4f7; --ink-2:#aab1bb; --muted:#727a84;
  --hair:#242a31; --hair-2:#1d2228; --baseline:#333a42;
  --accent:#3987e5;
  --s-port:#3987e5; --s-qqq:#2aa32a; --s-spy:#d55181; --s-ew:#d9a441;
  --pos:#38c46a; --neg:#f0645f; --warn:#e0a92a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px -14px rgba(0,0,0,.6);
}

*{box-sizing:border-box;}
.wrap{
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  background:var(--plane); color:var(--ink); font-family:var(--sans);
  font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased;
  padding:clamp(18px,4vw,56px) clamp(16px,4vw,40px) 72px; min-height:100%;
}
.page{max-width:1080px; margin:0 auto;}
.mono{font-family:var(--mono);}
.tnum{font-variant-numeric:tabular-nums;}

/* masthead */
.eyebrow{font-family:var(--mono); font-size:.72rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--accent); font-weight:600;}
h1{font-size:clamp(1.9rem,4.6vw,3.1rem); line-height:1.04; margin:.32em 0 .18em;
  letter-spacing:-.02em; text-wrap:balance; font-weight:680;}
.dek{font-size:clamp(1.02rem,2.2vw,1.2rem); color:var(--ink-2); max-width:64ch;
  margin:0; text-wrap:pretty;}
.meta{display:flex; flex-wrap:wrap; gap:6px 20px; margin-top:20px;
  font-family:var(--mono); font-size:.78rem; color:var(--muted);}
.meta b{color:var(--ink-2); font-weight:600;}

/* callout */
.callout{margin:28px 0 8px; border:1px solid var(--hair); border-left:3px solid var(--warn);
  background:var(--surface); border-radius:12px; padding:16px 20px; box-shadow:var(--shadow);}
.callout h3{margin:0 0 4px; font-size:.82rem; letter-spacing:.1em; text-transform:uppercase;
  font-family:var(--mono); color:var(--warn);}
.callout p{margin:0; color:var(--ink-2); font-size:.95rem;}

/* kpi strip */
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:26px 0 8px;}
.kpi{background:var(--surface); border:1px solid var(--hair); border-radius:14px;
  padding:16px 18px; box-shadow:var(--shadow);}
.kpi .k-label{font-family:var(--mono); font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted);}
.kpi .k-val{font-size:clamp(1.5rem,3vw,1.95rem); font-weight:680; letter-spacing:-.02em;
  margin-top:6px; line-height:1;}
.kpi .k-sub{font-family:var(--mono); font-size:.74rem; color:var(--ink-2); margin-top:6px;}
.pos{color:var(--pos);} .neg{color:var(--neg);}

/* sections */
section{margin-top:44px;}
.sec-head{display:flex; align-items:baseline; justify-content:space-between; gap:16px;
  margin-bottom:14px; flex-wrap:wrap;}
h2{font-size:1.35rem; letter-spacing:-.01em; margin:0; font-weight:660;}
.sec-note{color:var(--muted); font-size:.85rem; font-family:var(--mono);}

.card{background:var(--surface); border:1px solid var(--hair); border-radius:16px;
  padding:20px 22px; box-shadow:var(--shadow);}

/* legend */
.legend{display:flex; flex-wrap:wrap; gap:8px 18px; margin:2px 0 14px;
  font-family:var(--mono); font-size:.78rem; color:var(--ink-2);}
.legend span{display:inline-flex; align-items:center; gap:7px;}
.swatch{width:14px; height:3px; border-radius:2px; display:inline-block;}
.swatch.dash{background:repeating-linear-gradient(90deg,currentColor 0 5px,transparent 5px 9px);}

/* chart */
.chart{position:relative;}
.chart svg{display:block; width:100%; height:auto; overflow:visible;}
.grid-line{stroke:var(--hair-2);} .axis-line{stroke:var(--baseline);}
.ax-label{fill:var(--muted); font-family:var(--mono); font-size:11px;}
.end-label{font-family:var(--mono); font-size:11px; font-weight:600;}
.tip{position:fixed; pointer-events:none; opacity:0; transform:translate(-50%,-100%);
  background:var(--ink); color:var(--surface); font-family:var(--mono); font-size:11.5px;
  padding:8px 10px; border-radius:8px; white-space:nowrap; z-index:5; transition:opacity .08s;
  box-shadow:0 6px 20px -6px rgba(0,0,0,.5); line-height:1.5;}
:root[data-theme="dark"] .tip, :root:where(:not([data-theme="light"])) .tip{--x:0;}
.tip b{font-weight:600;} .tip .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle;}
.crosshair{stroke:var(--baseline); stroke-dasharray:3 3;}

/* small multiples */
.grid-cards{display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px;}
.ycard{background:var(--surface); border:1px solid var(--hair); border-radius:14px;
  padding:16px 18px 12px; box-shadow:var(--shadow);}
.ycard-top{display:flex; align-items:baseline; justify-content:space-between; margin-bottom:2px;}
.ycard-top .yr{font-family:var(--mono); font-weight:680; font-size:1.15rem; letter-spacing:.02em;}
.badge{font-family:var(--mono); font-size:.78rem; font-weight:600; padding:2px 8px;
  border-radius:999px; border:1px solid var(--hair);}
.badge.up{color:var(--pos); background:color-mix(in srgb,var(--pos) 12%,transparent);}
.badge.down{color:var(--neg); background:color-mix(in srgb,var(--neg) 12%,transparent);}
.ycard-stats{display:flex; gap:16px; margin-top:8px; font-family:var(--mono);
  font-size:.76rem; color:var(--ink-2);}
.ycard-stats b{color:var(--muted); font-weight:500;}

/* table */
.tbl-wrap{overflow-x:auto; border:1px solid var(--hair); border-radius:14px; background:var(--surface); box-shadow:var(--shadow);}
table{border-collapse:collapse; width:100%; font-size:.86rem; min-width:640px;}
th,td{padding:11px 14px; text-align:right; font-variant-numeric:tabular-nums;
  border-bottom:1px solid var(--hair-2); white-space:nowrap;}
th{font-family:var(--mono); font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--surface);}
th:first-child,td:first-child{text-align:left; font-family:var(--mono);}
tbody tr:last-child td{border-bottom:none;}
tbody tr.total td{font-weight:680; background:var(--surface-2); border-top:2px solid var(--hair);}
td.mono{font-family:var(--mono);}

/* caveats */
.caveats{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px;}
.caveat{background:var(--surface); border:1px solid var(--hair); border-radius:14px; padding:16px 18px;}
.caveat h4{margin:0 0 5px; font-size:.9rem; font-weight:660;}
.caveat p{margin:0; color:var(--ink-2); font-size:.88rem;}
.caveat .tag{font-family:var(--mono); font-size:.66rem; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted);}

/* deploy footer */
.deploy{margin-top:44px; background:var(--surface); border:1px solid var(--hair); border-radius:16px;
  padding:24px 26px; box-shadow:var(--shadow);}
.deploy h2{margin-bottom:6px;} .deploy p{color:var(--ink-2); margin:0 0 14px; max-width:68ch;}
.flow{display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-family:var(--mono); font-size:.8rem;}
.flow .node{border:1px solid var(--hair); background:var(--surface-2); border-radius:10px; padding:8px 12px;}
.flow .node.hot{border-color:var(--accent); color:var(--accent);}
.flow .arrow{color:var(--muted);}
a.link{color:var(--accent); text-decoration:none; border-bottom:1px solid color-mix(in srgb,var(--accent) 40%,transparent);}
a.link:hover{border-bottom-color:var(--accent);}
footer.fine{margin-top:34px; color:var(--muted); font-family:var(--mono); font-size:.74rem;
  border-top:1px solid var(--hair); padding-top:16px; display:flex; gap:8px 20px; flex-wrap:wrap;}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:4px;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>

<div class="wrap"><div class="page">

  <header>
    <div class="eyebrow">Swingbot Research &middot; Walk-Forward Backtest</div>
    <h1 id="headline">Five years, one policy, learning as it goes</h1>
    <p class="dek" id="dek"></p>
    <div class="meta" id="meta"></div>
  </header>

  <div class="callout">
    <h3>Read this before the number impresses you</h3>
    <p id="honesty"></p>
  </div>

  <div class="kpis" id="kpis"></div>

  <section id="overall-sec">
    <div class="sec-head">
      <h2>Growth of $100,000</h2>
      <span class="sec-note">rebased to 100 at inception &middot; simulated capital</span>
    </div>
    <div class="card">
      <div class="legend" id="overall-legend"></div>
      <div class="chart" id="overall-chart"></div>
    </div>
  </section>

  <section id="years-sec">
    <div class="sec-head">
      <h2>Year by year</h2>
      <span class="sec-note">portfolio vs QQQ &middot; each panel rebased to 100 at the year's open</span>
    </div>
    <div class="grid-cards" id="year-cards"></div>
  </section>

  <section id="table-sec">
    <div class="sec-head"><h2>The full ledger, by year</h2></div>
    <div class="tbl-wrap"><table id="metrics-table"></table></div>
  </section>

  <section id="health-sec">
    <div class="sec-head">
      <h2>Is the model actually deciding?</h2>
      <span class="sec-note">conviction spread across the universe, per bar</span>
    </div>
    <div class="card">
      <p style="margin:0 0 14px;color:var(--ink-2);font-size:.92rem" id="health-note"></p>
      <div class="chart" id="health-chart"></div>
    </div>
  </section>

  <section id="caveat-sec">
    <div class="sec-head"><h2>How to read this</h2></div>
    <div class="caveats" id="caveats"></div>
  </section>

  <div class="deploy">
    <h2>What happens to this model now</h2>
    <p>The policy that walked through these five years isn't thrown away. Its trained
      weights become the <b>seed for the live 30-minute paper bot</b>: at inception the
      engine loads this checkpoint, clears the per-symbol recurrence (a daily-bar trace
      shouldn't leak into a half-hour loop), then refines it on recent 30-minute history
      before trading forward. That is what &ldquo;learn from the backtest&rdquo; means here &mdash;
      the live book starts from five years of lived experience, not a cold policy.</p>
    <div class="flow">
      <span class="node">5-yr daily backtest</span>
      <span class="arrow">&rarr;</span>
      <span class="node hot">rrl_latest.bin</span>
      <span class="arrow">&rarr;</span>
      <span class="node">seed 30m inception</span>
      <span class="arrow">&rarr;</span>
      <span class="node">refine on 30m history</span>
      <span class="arrow">&rarr;</span>
      <span class="node">live, every 30 min from the open</span>
    </div>
    <p style="margin-top:14px;margin-bottom:0">Live dashboard:
      <a class="link" href="https://aqiao-814.github.io/swingbot-live/">aqiao-814.github.io/swingbot-live</a></p>
  </div>

  <footer class="fine" id="footer"></footer>

</div></div>

<div class="tip" id="tip"></div>

<script id="findings" type="application/json">__FINDINGS_JSON__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById("findings").textContent);
  const tip = document.getElementById("tip");
  const pct = (x,dp=1)=> (x==null?"—":(x>=0?"+":"")+ (x*100).toFixed(dp)+"%");
  const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim()
                    || getComputedStyle(document.querySelector(".wrap")).getPropertyValue(n).trim();
  const arrow = x => x==null?"":(x>=0?"↑":"↓");

  // ---------- copy ----------
  const o=D.overall, m=D.meta, h=D.health||{};
  document.getElementById("dek").innerHTML =
    `A single recurrent-reinforcement policy trading the ${m.n_symbols}-name ${m.universe.toUpperCase()} `+
    `on daily bars, ${m.inception} → ${m.end}. It learns online from every realized bar, so this run `+
    `both <b>measures</b> the strategy and <b>trains</b> the model the live bot inherits.`;
  document.getElementById("meta").innerHTML = [
    ["universe", m.universe.toUpperCase()],["model","RRL · shared weights"],
    ["capital","$"+m.starting_capital.toLocaleString()],["bars","daily"],
    ["kill switches","off (measurement)"]
  ].map(([k,v])=>`<span><b>${v}</b> ${k}</span>`).join("");
  const excQQQ = o.excess_vs_qqq;
  document.getElementById("honesty").innerHTML =
    `The headline &mdash; <b>${pct(o.total_return)}</b> vs QQQ&rsquo;s ${pct(o.benchmarks.QQQ)} &mdash; is `+
    `flattered by two things this backtest can&rsquo;t escape: the universe is <b>today&rsquo;s</b> `+
    `${m.universe.toUpperCase()} (survivors only, so every name already &ldquo;made it&rdquo;), and the policy runs `+
    `<b>long-only in a five-year bull market</b>. Its Sharpe is ${o.sharpe.toFixed(2)} &mdash; below 1 &mdash; and it `+
    `drew down ${pct(o.max_drawdown)}, deeper than the index. Treat the excess as <i>beta plus survivorship</i> `+
    `until an out-of-sample forward record says otherwise.`;

  // ---------- KPIs ----------
  const cls = x => x>=0?"pos":"neg";
  const kpis = [
    {l:"Total return", v:pct(o.total_return), c:cls(o.total_return), s:`${o.n_days} trading days`},
    {l:"CAGR", v:pct(o.cagr), c:cls(o.cagr), s:"5-yr compound"},
    {l:"Sharpe", v:o.sharpe.toFixed(2), c:o.sharpe>=1?"pos":"", s:`vol ${pct(o.ann_vol)} ann.`},
    {l:"Max drawdown", v:pct(o.max_drawdown), c:"neg", s:"peak to trough"},
    {l:"Excess vs QQQ", v:pct(excQQQ), c:cls(excQQQ), s:`QQQ ${pct(o.benchmarks.QQQ)}`},
    {l:"Trades", v:o.n_trades.toLocaleString(), c:"", s:`~${Math.round(o.n_trades/5)}/yr · low turnover`},
  ];
  document.getElementById("kpis").innerHTML = kpis.map(k=>
    `<div class="kpi"><div class="k-label">${k.l}</div>`+
    `<div class="k-val ${k.c} tnum">${k.v}</div><div class="k-sub">${k.s}</div></div>`).join("");

  // ---------- generic line chart ----------
  function lineChart(mount, cfg){
    // cfg: {ts:[], series:[{key,name,color,fill?}], H, showY, showX, endLabels, fmt, yfmt, hline, hlabel}
    const fmt = cfg.fmt || (v=>pct(v/100-1));            // tooltip value formatter
    const yfmt = cfg.yfmt || (t=>String(t));             // y-axis tick formatter
    const W=760, H=cfg.H||300, PL=cfg.showY?46:10, PR=cfg.endLabels?52:12, PT=14, PB=cfg.showX?26:12;
    const iw=W-PL-PR, ih=H-PT-PB;
    const ts=cfg.ts, n=ts.length;
    let lo=Infinity, hi=-Infinity;
    cfg.series.forEach(s=> s.vals.forEach(v=>{ if(v<lo)lo=v; if(v>hi)hi=v; }));
    const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
    const X=i=> PL + (n<2?iw/2:iw*i/(n-1));
    const Y=v=> PT + ih*(1-(v-lo)/(hi-lo));

    // gridlines at rounded index levels (100 = start)
    const ticks=[]; const span=hi-lo; let step= span>140?50: span>70?25: span>28?10:5;
    let g=Math.ceil(lo/step)*step;
    for(; g<=hi; g+=step) ticks.push(g);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="line chart" preserveAspectRatio="none">`;
    ticks.forEach(t=>{ const y=Y(t).toFixed(1);
      svg+=`<line class="grid-line" x1="${PL}" x2="${PL+iw}" y1="${y}" y2="${y}" stroke-width="1"/>`;
      if(cfg.showY) svg+=`<text class="ax-label" x="${PL-8}" y="${(+y+3.5)}" text-anchor="end">${yfmt(t)}</text>`;
    });
    // baseline at 100 (the rebase origin) for equity charts
    if(lo<100&&hi>100){ const y=Y(100).toFixed(1);
      svg+=`<line class="axis-line" x1="${PL}" x2="${PL+iw}" y1="${y}" y2="${y}" stroke-width="1"/>`; }
    // optional horizontal reference line (e.g. the kill-switch threshold)
    if(cfg.hline!=null && cfg.hline>lo && cfg.hline<hi){ const y=Y(cfg.hline).toFixed(1);
      svg+=`<line x1="${PL}" x2="${PL+iw}" y1="${y}" y2="${y}" stroke="var(--neg)" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>`;
      if(cfg.hlabel) svg+=`<text class="ax-label" x="${PL+iw}" y="${(+y-5)}" text-anchor="end" fill="var(--neg)">${cfg.hlabel}</text>`;
    }
    // x labels: first, mid, last
    if(cfg.showX){ [0,Math.floor(n/2),n-1].forEach(i=>{
      svg+=`<text class="ax-label" x="${X(i).toFixed(1)}" y="${H-8}" text-anchor="middle">${ts[i].slice(0,7)}</text>`;});}
    // series paths
    cfg.series.forEach(s=>{
      const col=cssv(s.color);
      const pts=s.vals.map((v,i)=>`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
      if(s.fill){
        svg+=`<path d="M${X(0).toFixed(1)},${Y(lo).toFixed(1)} L${pts.join(" L")} L${X(n-1).toFixed(1)},${Y(lo).toFixed(1)} Z" `+
             `fill="${col}" fill-opacity="0.08" stroke="none"/>`;
      }
      svg+=`<polyline points="${pts.join(" ")}" fill="none" stroke="${col}" `+
           `stroke-width="${s.w||2}" stroke-linejoin="round" stroke-linecap="round" `+
           `${s.dash?`stroke-dasharray="${s.dash}"`:""}/>`;
      if(cfg.endLabels){ const last=s.vals[n-1];
        svg+=`<text class="end-label" x="${(X(n-1)+7).toFixed(1)}" y="${(Y(last)+3.5).toFixed(1)}" fill="${col}">${s.name}</text>`;}
    });
    svg+=`<line class="crosshair" id="ch-${mount.id}" x1="0" x2="0" y1="${PT}" y2="${PT+ih}" stroke-width="1" style="opacity:0"/>`;
    svg+=`</svg>`;
    mount.innerHTML=svg;

    const svgEl=mount.querySelector("svg"), ch=mount.querySelector(".crosshair");
    const dots=cfg.series.map(s=>{ const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
      c.setAttribute("r","3.5"); c.setAttribute("fill",cssv(s.color)); c.setAttribute("stroke","var(--surface)");
      c.setAttribute("stroke-width","1.5"); c.style.opacity="0"; svgEl.appendChild(c); return c; });
    function move(ev){
      const r=svgEl.getBoundingClientRect();
      const px=(ev.touches?ev.touches[0].clientX:ev.clientX)-r.left;
      const fx=px/r.width*W; let i=Math.round((fx-PL)/(iw)*(n-1));
      i=Math.max(0,Math.min(n-1,i)); if(!isFinite(i))return;
      const xc=X(i);
      ch.setAttribute("x1",xc); ch.setAttribute("x2",xc); ch.style.opacity="1";
      let rows="";
      cfg.series.forEach((s,k)=>{ const v=s.vals[i];
        dots[k].setAttribute("cx",xc); dots[k].setAttribute("cy",Y(v)); dots[k].style.opacity="1";
        rows+=`<div><span class="dot" style="background:${cssv(s.color)}"></span>${s.name} <b>${fmt(v)}</b></div>`; });
      tip.innerHTML=`<div style="color:var(--muted);margin-bottom:3px">${ts[i]}</div>${rows}`;
      tip.style.opacity="1";
      const rc=xc/W*r.width + r.left, ry=r.top+window.scrollY+PT;
      tip.style.left=rc+"px"; tip.style.top=(ry)+"px";
    }
    function leave(){ ch.style.opacity="0"; tip.style.opacity="0"; dots.forEach(d=>d.style.opacity="0"); }
    svgEl.addEventListener("mousemove",move); svgEl.addEventListener("mouseleave",leave);
    svgEl.addEventListener("touchmove",move,{passive:true}); svgEl.addEventListener("touchend",leave);
    return ()=>lineChart(mount,cfg); // re-render fn for theme swap
  }

  const rerenders=[];

  // ---------- overall chart ----------
  const oc=document.getElementById("overall-chart");
  const OS=[
    {key:"port",name:"Portfolio",color:"--s-port",fill:true,w:2.4},
    {key:"qqq",name:"QQQ",color:"--s-qqq"},
    {key:"spy",name:"SPY",color:"--s-spy"},
    {key:"ew",name:"Equal-wt",color:"--s-ew"},
  ].filter(s=>o.series[s.key]).map(s=>({...s,vals:o.series[s.key]}));
  document.getElementById("overall-legend").innerHTML = OS.map(s=>
    `<span><i class="swatch" style="background:${cssv(s.color)}"></i>${s.name} `+
    `<b class="tnum">${pct((s.vals[s.vals.length-1])/100-1)}</b></span>`).join("");
  const drawOverall=()=>lineChart(oc,{ts:o.series.ts,series:OS,H:340,showY:true,showX:true,endLabels:false});
  rerenders.push(drawOverall);

  // ---------- year cards ----------
  const yc=document.getElementById("year-cards");
  yc.innerHTML = D.years.map(y=>`
    <div class="ycard">
      <div class="ycard-top">
        <span class="yr">${y.year}${y.year==2021||y.year==2026?'<span style="font-size:.7rem;color:var(--muted)"> partial</span>':''}</span>
        <span class="badge ${y.total_return>=0?'up':'down'}">${arrow(y.total_return)} ${pct(y.total_return)}</span>
      </div>
      <div class="chart" id="yc-${y.year}"></div>
      <div class="ycard-stats">
        <span><b>QQQ</b> ${pct(y.benchmarks.QQQ)}</span>
        <span><b>vs</b> <span class="${y.excess_vs_qqq>=0?'pos':'neg'}">${pct(y.excess_vs_qqq)}</span></span>
        <span><b>Sharpe</b> ${y.sharpe.toFixed(2)}</span>
        <span><b>maxDD</b> ${pct(y.max_drawdown)}</span>
      </div>
    </div>`).join("");
  const drawYears=()=> D.years.forEach(y=>{
    const mnt=document.getElementById("yc-"+y.year);
    lineChart(mnt,{ts:y.series.ts,H:150,showY:false,showX:false,endLabels:false,
      series:[{key:"port",name:"Portfolio",color:"--s-port",fill:true,w:2,vals:y.series.port},
              {key:"qqq",name:"QQQ",color:"--s-qqq",w:1.5,dash:"4 3",vals:y.series.qqq}]});
  });
  rerenders.push(drawYears);

  // ---------- table ----------
  const rows = D.years.map(y=>({
    label:y.year, ret:y.total_return, sh:y.sharpe, dd:y.max_drawdown, vol:y.ann_vol,
    q:y.benchmarks.QQQ, s:y.benchmarks.SPY, e:y.benchmarks.EW, ex:y.excess_vs_qqq, t:y.n_trades
  }));
  const totRow={label:"5-yr",ret:o.total_return,sh:o.sharpe,dd:o.max_drawdown,vol:o.ann_vol,
    q:o.benchmarks.QQQ,s:o.benchmarks.SPY,e:o.benchmarks.EW,ex:o.excess_vs_qqq,t:o.n_trades};
  const cell=(x,dp=1)=>`<td class="tnum ${x>=0?'pos':'neg'}">${pct(x,dp)}</td>`;
  const plain=(x)=>`<td class="tnum">${x==null?'—':x.toFixed(2)}</td>`;
  const head=`<thead><tr><th>Period</th><th>Return</th><th>Sharpe</th><th>Max DD</th>`+
    `<th>Ann vol</th><th>QQQ</th><th>SPY</th><th>Eq-wt</th><th>vs QQQ</th><th>Trades</th></tr></thead>`;
  const body = rows.map(r=>`<tr><td>${r.label}</td>${cell(r.ret)}${plain(r.sh)}${cell(r.dd)}`+
    `<td class="tnum">${pct(r.vol)}</td>${cell(r.q)}${cell(r.s)}${cell(r.e)}${cell(r.ex)}`+
    `<td class="tnum">${r.t.toLocaleString()}</td></tr>`).join("");
  const tot=`<tr class="total"><td>${totRow.label}</td>${cell(totRow.ret)}${plain(totRow.sh)}${cell(totRow.dd)}`+
    `<td class="tnum">${pct(totRow.vol)}</td>${cell(totRow.q)}${cell(totRow.s)}${cell(totRow.e)}${cell(totRow.ex)}`+
    `<td class="tnum">${totRow.t.toLocaleString()}</td></tr>`;
  document.getElementById("metrics-table").innerHTML = head+`<tbody>${body}${tot}</tbody>`;

  // ---------- health ----------
  if(h.ts && h.ts.length){
    document.getElementById("health-note").innerHTML =
      `Sizing here is &ldquo;conviction-weighted,&rdquo; so it only means something if convictions actually `+
      `<i>spread out</i>. Across the walk the cross-sectional spread averaged `+
      `<b>&sigma; = ${h.mean_conviction_std.toFixed(3)}</b> and sat above the 0.99 saturation line `+
      `<b>${(h.mean_frac_saturated*100).toFixed(0)}%</b> of the time &mdash; thin enough that the ranking often `+
      `collapses toward the sort&rsquo;s tiebreak. Read the returns as coming mostly from <i>being long the right kind `+
      `of names</i>, not from finely graded conviction. (${h.n_updates.toLocaleString()} online updates total.)`+
      `<br><br>The backtest also surfaced <i>why</i>: the recurrent weight drifted to <b>u &gt; 1</b>, which makes `+
      `<code>F_t = tanh(w&middot;x + u&middot;F_{t-1} + b)</code> explosive &mdash; convictions pin to &plusmn;1 within a few `+
      `bars regardless of features. The live 30-minute bot ships a fix for exactly this: it caps <b>|u| &le; 0.7</b> `+
      `so the recurrence stays a contraction, which restores a healthy conviction spread (&sigma; &asymp; 0.25 across `+
      `the full universe) that the model-health kill switch is happy with.`;
    const hc=document.getElementById("health-chart");
    // downsample health series to ~260 pts for a light payload path
    const stride=Math.max(1,Math.floor(h.ts.length/260));
    const hts=[], hv=[]; for(let i=0;i<h.ts.length;i+=stride){hts.push(h.ts[i]); hv.push(h.conviction_std[i]*1000);}
    const drawHealth=()=>{
      // values are sigma*1000 for a readable axis; format back to real sigma in labels/tooltip.
      lineChart(hc,{ts:hts,H:200,showY:true,showX:true,endLabels:false,
        fmt:v=>"σ "+(v/1000).toFixed(3), yfmt:t=>(t/1000).toFixed(3),
        hline:50, hlabel:"model-health halt (σ 0.05)",
        series:[{key:"cs",name:"conviction σ",color:"--warn",fill:true,w:1.8,vals:hv}]});
    };
    rerenders.push(drawHealth);
  } else { document.getElementById("health-sec").style.display="none"; }

  // ---------- caveats ----------
  const caveats=[
    {t:"Survivorship",h:"The universe is today's index",p:"nasdaq100 membership is a 2025 snapshot. Names that were delisted or dropped never enter the test, so every stock in it is a past winner. This inflates any long-biased backtest — there is no free delisting-inclusive dataset."},
    {t:"Signal",h:"Near-zero measured rank signal",p:"Prior walk-forward work put this policy's per-date RankIC at ~0 (mean +0.004, fails the 0.02 gate). Cross-sectional stock-picking skill is not established; the equity curve leans on broad long exposure."},
    {t:"Risk",h:"Drawdown deeper than the index",p:`The book drew down ${pct(o.max_drawdown)} against a shallower fall in QQQ, and concentrates in ~10 names at up to 20% each. Higher return came with higher risk, not free alpha.`},
    {t:"Statistics",h:"One seed, no deflation",p:"A single random seed and one configuration. Sharpe "+o.sharpe.toFixed(2)+" is below 1 and below the deflated-Sharpe credibility bar; treat it as a point estimate, not a distribution."},
    {t:"Transfer",h:"Daily bars, 30-minute deployment",p:"This measurement is on daily bars (the only source with 5 years of history). The live bot trades 30-minute bars, whose features live on a different time scale — the seed is refined on 30m history before it trades, but the transfer is a hypothesis, not a proven equivalence."},
    {t:"Setup",h:"Kill switches were off",p:"The production drawdown / model-health halts are disabled here so a safety stop wouldn't freeze a 5-year measurement. Live, they can flatten the book — realized live results will differ."},
  ];
  document.getElementById("caveats").innerHTML = caveats.map(c=>
    `<div class="caveat"><div class="tag">${c.t}</div><h4>${c.h}</h4><p>${c.p}</p></div>`).join("");

  // ---------- footer ----------
  document.getElementById("footer").innerHTML =
    `<span>Generated ${D.generated||""}</span><span>All capital simulated — no real orders</span>`+
    `<span>swingbot · ContinualRRL</span>`;

  // ---------- render + theme reactivity ----------
  function renderAll(){ rerenders.forEach(f=>f()); }
  renderAll();
  window.addEventListener("resize",()=>{ /* svg is fluid; tooltip recomputes on hover */ });
  const mo=new MutationObserver(renderAll);
  mo.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  if(window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",renderAll);
})();
</script>
"""


def main() -> None:
    findings = json.loads(SRC.read_text())
    findings["generated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = PAGE.replace("__FINDINGS_JSON__", json.dumps(findings, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
