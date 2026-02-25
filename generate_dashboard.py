"""
生成Bot展示页面的数据 + HTML

用法: python3 generate_dashboard.py
输出: profiles/dashboard.html
"""

import os
import sys
import json
import hashlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.backtest.engine import run_backtest, BacktestResult
from src.strategy.schema import BotConfig, WeightVector


def load_unique_bots(gen_dir: str):
    """加载并去重Bot。"""
    bots = []
    seen = set()
    for fname in sorted(os.listdir(gen_dir)):
        if not fname.startswith("bot_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(gen_dir, fname)) as f:
            data = json.load(f)
        bot = BotConfig.from_dict(data["bot"])
        fp = bot.weights.fingerprint()
        if fp not in seen:
            seen.add(fp)
            bots.append(bot)
    return bots


def run_and_collect(bots, df, regime):
    """回测所有Bot，收集equity curve和trades明细。"""
    all_data = []
    timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else list(range(len(df)))

    for bot in bots:
        print(f"  回测 {bot.bot_id} ({bot.name})...")
        result = run_backtest(df, bot, regime)

        eq = result.equity_curve.tolist()
        eq_timestamps = []
        step = max(1, len(timestamps) // len(eq)) if len(eq) > 0 else 1
        for i in range(len(eq)):
            idx = min(i * step, len(timestamps) - 1)
            ts = timestamps[idx]
            eq_timestamps.append(str(ts)[:16] if hasattr(ts, "strftime") else str(ts)[:16])

        trades_list = []
        for t in result.trades:
            entry_ts = str(timestamps[min(t.entry_idx, len(timestamps)-1)])[:16] if t.entry_idx < len(timestamps) else ""
            exit_ts = str(timestamps[min(t.exit_idx, len(timestamps)-1)])[:16] if t.exit_idx and t.exit_idx < len(timestamps) else ""
            trades_list.append({
                "entry_idx": t.entry_idx,
                "entry_price": round(t.entry_price, 2),
                "exit_idx": t.exit_idx,
                "exit_price": round(t.exit_price, 2) if t.exit_price else 0,
                "direction": "Long" if t.direction == 1 else "Short",
                "leverage": t.leverage,
                "margin": round(t.margin, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct * 100, 2),
                "exit_reason": t.exit_reason,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
            })

        tags = bot.tags or []
        source = "LLM" if "llm_evolved" in tags else "Elite" if "elite" in tags else "Genetic"

        peak = result.equity_curve.expanding().max()
        drawdown = ((result.equity_curve - peak) / peak).tolist()
        dd_timestamps = eq_timestamps[:len(drawdown)]

        bot_entry = {
            "bot_id": bot.bot_id,
            "name": bot.name,
            "source": source,
            "generation": bot.generation,
            "weights": bot.weights.to_dict(),
            "metrics": result.to_dict(),
            "equity": eq,
            "equity_timestamps": eq_timestamps,
            "drawdown": [round(d * 100, 2) for d in drawdown],
            "trades": trades_list,
        }
        all_data.append(bot_entry)
        print(f"    return={result.total_return:.2%}, trades={result.total_trades}, sharpe={result.sharpe_ratio:.2f}")

    return all_data


def _downsample(arr, max_points=500):
    """将过长数组降采样到max_points。"""
    if len(arr) <= max_points:
        return arr, list(range(len(arr)))
    step = len(arr) / max_points
    indices = [int(i * step) for i in range(max_points)]
    indices[-1] = len(arr) - 1
    return [arr[i] for i in indices], indices


def build_html(bots_data, price_data, regime_data, output_path):
    """生成自包含HTML展示页面。"""
    colors = ['#58a6ff','#56d364','#bc8cff','#f0883e','#f85149','#3fb950','#d2a8ff','#79c0ff','#ffa657']
    for i, b in enumerate(bots_data):
        eq, idx = _downsample(b["equity"])
        ts = [b["equity_timestamps"][j] if j < len(b["equity_timestamps"]) else "" for j in idx]
        dd = [b["drawdown"][j] if j < len(b["drawdown"]) else 0 for j in idx]
        b["equity_ds"] = eq
        b["timestamps_ds"] = ts
        b["drawdown_ds"] = dd
        b["color"] = colors[i % len(colors)]

    data_json = json.dumps(bots_data, ensure_ascii=False)
    price_json = json.dumps(price_data, ensure_ascii=False)
    regime_json = json.dumps(regime_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Bot - 策略进化结果</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }}
.header {{ background: linear-gradient(135deg, #161b22, #1c2333); padding: 28px 32px; border-bottom: 1px solid #30363d; }}
.header h1 {{ font-size: 24px; color: #f0f6fc; letter-spacing: -0.5px; }}
.header .sub {{ color: #8b949e; margin-top: 6px; font-size: 14px; display: flex; gap: 20px; flex-wrap: wrap; }}
.header .sub span {{ display: inline-flex; align-items: center; gap: 4px; }}
.tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.container {{ max-width: 1440px; margin: 0 auto; padding: 20px 24px; }}
.overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 12px; text-align: center; }}
.stat-card .value {{ font-size: 26px; font-weight: 700; color: #58a6ff; }}
.stat-card .label {{ font-size: 11px; color: #8b949e; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}

.compare-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin: 20px 0; }}
.compare-box h3 {{ font-size: 15px; color: #f0f6fc; margin-bottom: 12px; }}

.bot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; margin: 16px 0; }}
.bot-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; cursor: pointer; transition: all 0.2s; }}
.bot-card:hover {{ border-color: #58a6ff; transform: translateY(-1px); }}
.bot-card.active {{ border-color: #58a6ff; box-shadow: 0 0 0 1px #58a6ff, 0 4px 12px rgba(88,166,255,0.15); }}
.bot-card .bot-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
.bot-card .bot-id {{ font-size: 14px; font-weight: 600; color: #f0f6fc; }}
.bot-card .bot-sub {{ font-size: 11px; color: #8b949e; margin-bottom: 8px; }}
.source {{ font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }}
.source-LLM {{ background: #1f3a5f; color: #58a6ff; }}
.source-Elite {{ background: #2a1f3f; color: #bc8cff; }}
.source-Genetic {{ background: #1f3f2a; color: #56d364; }}
.bot-card .bot-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
.bot-card .bot-stats .s {{ text-align: center; }}
.bot-card .bot-stats .s .v {{ font-size: 15px; font-weight: 700; }}
.bot-card .bot-stats .s .l {{ font-size: 9px; color: #8b949e; text-transform: uppercase; }}
.positive {{ color: #56d364; }}
.negative {{ color: #f85149; }}
.neutral {{ color: #8b949e; }}

.detail-panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; margin: 20px 0; display: none; }}
.detail-panel.show {{ display: block; }}
.detail-panel h2 {{ font-size: 18px; color: #f0f6fc; margin-bottom: 4px; }}
.detail-panel .detail-sub {{ font-size: 13px; color: #8b949e; margin-bottom: 16px; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }}
.chart-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ font-size: 12px; color: #8b949e; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}

.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin: 12px 0; }}
.metric {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center; }}
.metric .mv {{ font-size: 18px; font-weight: 700; }}
.metric .ml {{ font-size: 10px; color: #8b949e; margin-top: 2px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ background: #0d1117; color: #8b949e; padding: 8px 10px; text-align: left; font-weight: 600; border-bottom: 1px solid #30363d; position: sticky; top: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; }}
tr:hover td {{ background: #1c2333; }}
.trades-wrap {{ max-height: 420px; overflow-y: auto; border: 1px solid #30363d; border-radius: 8px; }}

.weights-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-top: 12px; }}
.weight-group {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }}
.weight-group h4 {{ font-size: 11px; color: #58a6ff; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.weight-group .wi {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; border-bottom: 1px solid #21262d; }}
.weight-group .wi:last-child {{ border-bottom: none; }}
.weight-group .wi .wk {{ color: #8b949e; }}
.weight-group .wi .wv {{ color: #f0f6fc; font-weight: 500; font-family: 'SF Mono', 'Fira Code', monospace; }}

.section-title {{ font-size: 15px; color: #f0f6fc; margin: 24px 0 10px; padding-bottom: 8px; border-bottom: 1px solid #30363d; font-weight: 600; }}
.empty-msg {{ text-align: center; padding: 40px; color: #484f58; font-size: 14px; }}

@media (max-width: 768px) {{
  .chart-row {{ grid-template-columns: 1fr; }}
  .bot-grid {{ grid-template-columns: 1fr; }}
  .container {{ padding: 12px; }}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Trade Bot Dashboard</h1>
  <div class="sub">
    <span>BTC/USDT 1H</span>
    <span>148天回测</span>
    <span>118代LLM进化</span>
    <span>Claude Sonnet 4.6</span>
  </div>
</div>
<div class="container">
  <div class="overview" id="overview"></div>

  <div class="compare-box">
    <h3>全Bot权益对比 (对数坐标)</h3>
    <canvas id="compareChart" height="80"></canvas>
  </div>

  <div class="section-title">Bot 列表</div>
  <div class="bot-grid" id="botGrid"></div>
  <div class="detail-panel" id="detailPanel"></div>
</div>
<script>
const BOTS = {data_json};
const PRICE = {price_json};
const REGIME = {regime_json};

let currentChart1 = null, currentChart2 = null, currentChart3 = null;

function fmtVal(v) {{
  if (v >= 1e9) return (v/1e9).toFixed(1)+'B';
  if (v >= 1e6) return (v/1e6).toFixed(1)+'M';
  if (v >= 1e3) return (v/1e3).toFixed(1)+'K';
  return v.toFixed(0);
}}

function fmtRet(r) {{
  if (Math.abs(r) >= 100) return (r).toFixed(0) + '%';
  return r.toFixed(1) + '%';
}}

function init() {{
  renderOverview();
  renderCompareChart();
  renderBotGrid();
  if (BOTS.length > 0) selectBot(0);
}}

function renderOverview() {{
  const profitable = BOTS.filter(b => b.metrics.total_return > 0).length;
  const totalTrades = BOTS.reduce((s, b) => s + b.metrics.total_trades, 0);
  const avgSharpe = (BOTS.reduce((s, b) => s + b.metrics.sharpe_ratio, 0) / BOTS.length).toFixed(2);
  const bestReturn = Math.max(...BOTS.map(b => b.metrics.total_return));
  const avgMDD = (BOTS.reduce((s, b) => s + b.metrics.max_drawdown, 0) / BOTS.length * 100).toFixed(1);
  document.getElementById('overview').innerHTML = `
    <div class="stat-card"><div class="value">${{BOTS.length}}</div><div class="label">总Bot数</div></div>
    <div class="stat-card"><div class="value positive">${{profitable}}</div><div class="label">盈利Bot</div></div>
    <div class="stat-card"><div class="value">${{totalTrades}}</div><div class="label">总交易</div></div>
    <div class="stat-card"><div class="value positive">${{fmtRet(bestReturn*100)}}</div><div class="label">最佳收益</div></div>
    <div class="stat-card"><div class="value">${{avgSharpe}}</div><div class="label">平均Sharpe</div></div>
    <div class="stat-card"><div class="value negative">${{avgMDD}}%</div><div class="label">平均MDD</div></div>
  `;
}}

function renderCompareChart() {{
  const maxLen = Math.max(...BOTS.map(b => b.equity_ds.length));
  const labels = BOTS[0].timestamps_ds;
  const datasets = BOTS.map(b => ({{
    label: b.bot_id,
    data: b.equity_ds,
    borderColor: b.color,
    backgroundColor: 'transparent',
    pointRadius: 0,
    borderWidth: 1.5,
    tension: 0.1,
  }}));
  new Chart(document.getElementById('compareChart'), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ color: '#8b949e', usePointStyle: true, pointStyle: 'line', padding: 16, font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + fmtVal(ctx.parsed.y) }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#484f58', maxTicksLimit: 12, maxRotation: 0 }}, grid: {{ color: '#21262d' }} }},
        y: {{ type: 'logarithmic', ticks: {{ color: '#484f58', callback: v => '$'+fmtVal(v) }}, grid: {{ color: '#21262d' }}, min: 100 }}
      }}
    }}
  }});
}}

function renderBotGrid() {{
  let sorted = BOTS.map((b,i) => ({{...b, idx: i}})).sort((a,b) => b.metrics.total_return - a.metrics.total_return);
  let html = '';
  sorted.forEach(b => {{
    const ret = b.metrics.total_return;
    const cls = ret > 0 ? 'positive' : 'negative';
    const mdd = b.metrics.max_drawdown;
    html += `<div class="bot-card" id="card_${{b.idx}}" onclick="selectBot(${{b.idx}})">
      <div class="bot-header">
        <span class="bot-id" style="color:${{b.color}}">${{b.bot_id}}</span>
        <span class="source source-${{b.source}}">${{b.source}}</span>
      </div>
      <div class="bot-sub">${{b.weights.indicator_set}} / ${{b.weights.entry_condition}}</div>
      <div class="bot-stats">
        <div class="s"><div class="v ${{cls}}">${{fmtRet(ret*100)}}</div><div class="l">收益</div></div>
        <div class="s"><div class="v">${{b.metrics.sharpe_ratio.toFixed(2)}}</div><div class="l">Sharpe</div></div>
        <div class="s"><div class="v negative">${{(mdd*100).toFixed(0)}}%</div><div class="l">MDD</div></div>
        <div class="s"><div class="v">${{b.metrics.total_trades}}</div><div class="l">交易</div></div>
      </div>
    </div>`;
  }});
  document.getElementById('botGrid').innerHTML = html;
}}

function selectBot(idx) {{
  document.querySelectorAll('.bot-card').forEach(c => c.classList.remove('active'));
  document.getElementById('card_' + idx).classList.add('active');
  const b = BOTS[idx];
  const m = b.metrics;
  const panel = document.getElementById('detailPanel');
  panel.classList.add('show');

  const retCls = m.total_return > 0 ? 'positive' : 'negative';

  const dims = {{
    '体制感知 (Regime)': ['regime_focus','regime_detector_version','regime_switch_freq'],
    '时间框架 (Timeframe)': ['primary_timeframe','secondary_timeframe','decision_freq'],
    '技术指标 (Indicators)': ['num_indicators','indicator_set','ma_period_range','rsi_period','adx_period'],
    '杠杆仓位 (Leverage)': ['base_leverage','leverage_dynamic','position_sizing','max_position_per_trade','risk_per_trade'],
    '进出场 (Entry/Exit)': ['entry_condition','sl_type','tp_type','exit_on_regime_change','entry_confirmation'],
    '数据源权重 (Weights)': ['price_weight','volume_weight','funding_rate_weight','onchain_weight','meme_sentiment_weight'],
    '决策风格 (Decision)': ['reasoning_depth','temperature','creativity_bias','sub_agent_count','bias_towards_action'],
    '极端开关 (Extreme)': ['allow_blowup','reverse_logic_prob','max_dd_tolerance','yolo_mode','regime_override_prob'],
  }};

  let weightsHtml = '';
  for (const [group, keys] of Object.entries(dims)) {{
    let items = keys.map(k => {{
      let v = b.weights[k];
      if (v === undefined || v === null) return '';
      if (typeof v === 'number' && !Number.isInteger(v)) v = v.toFixed(4);
      return `<div class="wi"><span class="wk">${{k}}</span><span class="wv">${{v}}</span></div>`;
    }}).filter(Boolean).join('');
    weightsHtml += `<div class="weight-group"><h4>${{group}}</h4>${{items}}</div>`;
  }}

  let regimeHtml = '';
  if (m.regime_performance && Object.keys(m.regime_performance).length > 0) {{
    for (const [r, rp] of Object.entries(m.regime_performance)) {{
      const rpCls = rp.return > 0 ? 'positive' : 'negative';
      regimeHtml += `<div class="metric"><div class="mv ${{rpCls}}">${{fmtRet(rp.return*100)}}</div><div class="ml">${{r}} (${{rp.trades}}笔, 胜${{(rp.win_rate*100).toFixed(0)}}%)</div></div>`;
    }}
  }} else {{
    regimeHtml = '<div class="empty-msg">无交易数据</div>';
  }}

  let tradesHtml = '';
  if (b.trades.length > 0) {{
    tradesHtml = b.trades.map((t, i) => {{
      const pCls = t.pnl_pct > 0 ? 'positive' : 'negative';
      return `<tr>
        <td>${{i+1}}</td>
        <td>${{t.entry_time}}</td>
        <td>${{t.exit_time}}</td>
        <td style="font-weight:600">${{t.direction}}</td>
        <td>${{t.leverage}}x</td>
        <td>${{t.entry_price.toLocaleString()}}</td>
        <td>${{t.exit_price.toLocaleString()}}</td>
        <td class="${{pCls}}" style="font-weight:600">${{t.pnl_pct > 0 ? '+' : ''}}${{t.pnl_pct.toFixed(2)}}%</td>
        <td><span class="tag" style="background:#21262d;color:#8b949e">${{t.exit_reason}}</span></td>
      </tr>`;
    }}).join('');
  }}

  panel.innerHTML = `
    <h2 style="color:${{b.color}}">${{b.bot_id}}</h2>
    <div class="detail-sub">${{b.name}} · ${{b.source}} · Gen ${{b.generation}} · ${{b.weights.indicator_set}} / ${{b.weights.entry_condition}}</div>
    <div class="metrics-grid">
      <div class="metric"><div class="mv ${{retCls}}">${{fmtRet(m.total_return*100)}}</div><div class="ml">总收益</div></div>
      <div class="metric"><div class="mv">${{m.sharpe_ratio.toFixed(2)}}</div><div class="ml">Sharpe</div></div>
      <div class="metric"><div class="mv negative">${{(m.max_drawdown*100).toFixed(1)}}%</div><div class="ml">最大回撤</div></div>
      <div class="metric"><div class="mv">${{(m.win_rate*100).toFixed(0)}}%</div><div class="ml">胜率</div></div>
      <div class="metric"><div class="mv">${{m.profit_factor === Infinity ? '∞' : m.profit_factor.toFixed(2)}}</div><div class="ml">盈亏比</div></div>
      <div class="metric"><div class="mv">${{m.total_trades}}</div><div class="ml">交易</div></div>
      <div class="metric"><div class="mv positive">${{(m.avg_win*100).toFixed(1)}}%</div><div class="ml">均盈</div></div>
      <div class="metric"><div class="mv negative">${{(m.avg_loss*100).toFixed(1)}}%</div><div class="ml">均亏</div></div>
      <div class="metric"><div class="mv">${{m.max_consecutive_wins}} / ${{m.max_consecutive_losses}}</div><div class="ml">连胜/连亏</div></div>
    </div>
    ${{Object.keys(m.regime_performance || {{}}).length > 0 ? '<div class="section-title">Regime 表现</div><div class="metrics-grid">' + regimeHtml + '</div>' : ''}}
    <div class="section-title">权益曲线 & 回撤</div>
    <div class="chart-row">
      <div class="chart-box"><h3>权益曲线</h3><canvas id="equityChart"></canvas></div>
      <div class="chart-box"><h3>回撤曲线</h3><canvas id="ddChart"></canvas></div>
    </div>
    ${{b.trades.length > 0 ? `
      <div class="section-title">盈亏分布 (${{b.trades.length}} 笔)</div>
      <div class="chart-box" style="margin-bottom:16px"><canvas id="pnlChart" height="90"></canvas></div>
      <div class="section-title">交易明细</div>
      <div class="trades-wrap"><table>
        <thead><tr><th>#</th><th>开仓</th><th>平仓</th><th>方向</th><th>杠杆</th><th>开仓价</th><th>平仓价</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>${{tradesHtml}}</tbody>
      </table></div>
    ` : '<div class="empty-msg">该Bot在回测期间无交易</div>'}}
    <div class="section-title">42维权重向量</div>
    <div class="weights-grid">${{weightsHtml}}</div>
  `;

  renderCharts(b);
  panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
}}

function renderCharts(b) {{
  if (currentChart1) currentChart1.destroy();
  if (currentChart2) currentChart2.destroy();
  if (currentChart3) currentChart3.destroy();

  const labels = b.timestamps_ds;
  const chartOpts = {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#484f58', maxTicksLimit: 8, maxRotation: 0, font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
    }}
  }};

  const useLog = Math.max(...b.equity_ds) / Math.min(...b.equity_ds.filter(v => v > 0)) > 100;

  currentChart1 = new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [{{ data: b.equity_ds, borderColor: b.color, backgroundColor: b.color + '18', fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.1 }}]
    }},
    options: {{
      ...chartOpts,
      plugins: {{ ...chartOpts.plugins, tooltip: {{ callbacks: {{ label: ctx => '$' + fmtVal(ctx.parsed.y) }} }} }},
      scales: {{
        ...chartOpts.scales,
        y: useLog
          ? {{ type: 'logarithmic', ticks: {{ color: '#484f58', callback: v => '$'+fmtVal(v), font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
          : {{ ticks: {{ color: '#484f58', callback: v => '$'+fmtVal(v), font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
      }}
    }}
  }});

  currentChart2 = new Chart(document.getElementById('ddChart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [{{ data: b.drawdown_ds, borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.1)', fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.1 }}]
    }},
    options: {{
      ...chartOpts,
      scales: {{
        ...chartOpts.scales,
        y: {{ ticks: {{ color: '#484f58', callback: v => v.toFixed(0) + '%', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
      }}
    }}
  }});

  if (b.trades.length > 0) {{
    const pnls = b.trades.map(t => t.pnl_pct);
    const pnlColors = pnls.map(p => p > 0 ? '#56d364' : '#f85149');
    currentChart3 = new Chart(document.getElementById('pnlChart'), {{
      type: 'bar',
      data: {{
        labels: pnls.map((_, i) => '#' + (i+1)),
        datasets: [{{ data: pnls, backgroundColor: pnlColors, borderRadius: 2 }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toFixed(2) + '%' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#484f58', maxTicksLimit: 40, font: {{ size: 9 }} }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#484f58', callback: v => v.toFixed(0) + '%', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
        }}
      }}
    }});
  }}
}}

init();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def main():
    gen_dir = os.path.join(ROOT_DIR, "profiles", "gen_118")
    if not os.path.exists(gen_dir):
        dirs = sorted([d for d in os.listdir(os.path.join(ROOT_DIR, "profiles")) if d.startswith("gen_")])
        gen_dir = os.path.join(ROOT_DIR, "profiles", dirs[-1]) if dirs else os.path.join(ROOT_DIR, "profiles")

    print(f"加载Bot: {gen_dir}")
    bots = load_unique_bots(gen_dir)
    print(f"  去重后: {len(bots)} 个独特Bot")

    print(f"\n加载行情数据...")
    df = fetch_ohlcv("BTC/USDT", "1h", 148)
    regime = classify_regime(df, version="v1")
    print(f"  {len(df)} candles")

    print(f"\n回测中...")
    bots_data = run_and_collect(bots, df, regime)

    price_data = {
        "timestamps": [str(t)[:16] for t in df["timestamp"].tolist()],
        "close": df["close"].tolist(),
    }

    summary = regime_summary(df, regime)
    regime_data = {r: {"pct": round(s["pct"], 3), "count": s["count"]} for r, s in summary.items()}

    output_path = os.path.join(ROOT_DIR, "profiles", "dashboard.html")
    build_html(bots_data, price_data, regime_data, output_path)
    print(f"\n完成! 打开: {output_path}")


if __name__ == "__main__":
    main()
