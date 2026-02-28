"""
精选20 Bot 进化测试

从历史3批 110 个 bot 中多维择优 20 个极端代表，
用 15m K线 + 每周反思进化重跑，对比有/无反思效果。

维度: 赚得多 / 亏得多 / 历史峰值高 / 回撤最小 / Sharpe补齐
"""

import os, sys, json, time
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.strategy.decision import DecisionParams
from src.generator.llm_tuner import LLMTuner, format_market_context
from agent_backtest import run_agent_backtest, run_with_reflection

TIMEFRAME = "15m"
DAYS = 148
EXCHANGE = "okx"
REFLECTION_BARS = 672       # 7 days × 24h × 4
INITIAL_CAPITAL = 10000.0


def load_all_bots():
    """从 3 批结果中加载所有 bot，统一格式。"""
    all_bots = []

    with open(os.path.join(ROOT, "agent_batch_result/all_bots.json")) as f:
        for b in json.load(f):
            eq = b.get("equity", [])
            e0 = eq[0] if eq else 1
            all_bots.append({
                "name": b["name"], "batch": "batch_20",
                "prompt": b.get("prompt", ""),
                "personality": b.get("personality", ""),
                "params": b["params"],
                "result": b["result"],
                "equity": eq,
                "peak": (max(eq) / e0 - 1) * 100 if eq else 0,
            })

    with open(os.path.join(ROOT, "agent_80_result/all_bots.json")) as f:
        for b in json.load(f):
            eq = b.get("equity", [])
            e0 = eq[0] if eq else 1
            all_bots.append({
                "name": b["name"], "batch": "diverse_80",
                "prompt": b.get("prompt", ""),
                "personality": b.get("personality", ""),
                "params": b["params"],
                "result": b["result"],
                "equity": eq,
                "peak": (max(eq) / e0 - 1) * 100 if eq else 0,
            })

    with open(os.path.join(ROOT, "agent_20_reflect/all_bots.json")) as f:
        for b in json.load(f):
            eq = b.get("equity", [])
            e0 = eq[0] if eq else 1
            all_bots.append({
                "name": b["name"], "batch": "reflect_20",
                "prompt": b.get("prompt", ""),
                "personality": b.get("personality", ""),
                "params": b["params"],
                "result": b["result_evo"],
                "equity": eq,
                "peak": (max(eq) / e0 - 1) * 100 if eq else 0,
            })

    return [b for b in all_bots if b["result"]["total_trades"] > 0]


def select_top20(all_bots):
    """多维度择优：赚得多/亏得多/峰值高/回撤小/Sharpe补齐。"""
    for i, b in enumerate(all_bots):
        b["_key"] = f"{b['name']}_{b['batch']}_{i}"

    selected_keys = set()
    selected = []

    def add_from(pool, n):
        added = 0
        for b in pool:
            if b["_key"] not in selected_keys and added < n:
                selected_keys.add(b["_key"])
                selected.append(b)
                added += 1

    add_from(sorted(all_bots, key=lambda x: x["result"]["total_return"], reverse=True), 5)
    add_from(sorted(all_bots, key=lambda x: x["result"]["total_return"]), 5)
    add_from(sorted(all_bots, key=lambda x: x["peak"], reverse=True), 5)
    add_from(sorted(all_bots, key=lambda x: (
        x["result"]["max_drawdown"] if x["result"]["max_drawdown"] > 0 else 999
    )), 5)

    if len(selected) < 20:
        add_from(sorted(all_bots, key=lambda x: x["result"]["sharpe_ratio"], reverse=True),
                 20 - len(selected))

    return selected


def main():
    print("=" * 60)
    print("  精选20 Bot · 进化测试 (15m · 每周反思)")
    print("=" * 60)

    # ─── 1. 选 Bot ───
    print("\n[1] 从历史 110 个 bot 中多维择优...")
    all_bots = load_all_bots()
    top20 = select_top20(all_bots)
    print(f"  有效 bot: {len(all_bots)} | 精选: {len(top20)}")
    for i, b in enumerate(top20):
        r = b["result"]
        ret = r["total_return"] * 100
        blow = r.get("blowup_count", 0)
        print(f"  {i+1:2d}. {b['name']:10s} ({b['batch']:12s}) "
              f"ret={ret:+8.1f}% peak={b['peak']:+8.1f}% "
              f"DD={r['max_drawdown']*100:.1f}% blow={blow} "
              f"lev={b['params'].get('base_leverage',0):.0f}x")
    sys.stdout.flush()

    # ─── 2. 数据 ───
    print(f"\n[2] 加载 {TIMEFRAME} K线 ({DAYS}天) via {EXCHANGE}...")
    df = fetch_ohlcv("BTC/USDT", TIMEFRAME, DAYS, exchange_id=EXCHANGE)
    regime = classify_regime(df, version="v1", min_duration=48 * 4)
    print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
    summary = regime_summary(df, regime)
    for r, s in summary.items():
        print(f"  {r}: {s['pct']:.0%}")
    sys.stdout.flush()

    # ─── 3. 回测 (有反思 vs 无反思) ───
    tuner = LLMTuner()
    n_segs = max(1, len(df) // REFLECTION_BARS)
    print(f"\n[3] 回测 {len(top20)} 个 Bot | {n_segs} 周段 | "
          f"预计 {len(top20) * n_segs} 次反思调用")
    sys.stdout.flush()

    final = []
    for i, bot in enumerate(top20):
        tag = f"[{i+1}/{len(top20)}]"
        params = DecisionParams.from_dict(bot["params"])
        prompt = bot["prompt"]

        bt_base = run_agent_backtest(df, params, regime, initial_capital=INITIAL_CAPITAL)

        bt_evo, evo_log = run_with_reflection(
            df, params, regime, tuner,
            user_prompt=prompt,
            reflection_interval=REFLECTION_BARS,
            initial_capital=INITIAL_CAPITAL,
            verbose=True,
        )

        base_ret = bt_base.total_return
        evo_ret = bt_evo.total_return
        delta = evo_ret - base_ret

        print(f"  {tag} {bot['name']:10s} | "
              f"无反思: {base_ret*100:+8.1f}% (💥{bt_base.blowup_count}) | "
              f"有反思: {evo_ret*100:+8.1f}% (💥{bt_evo.blowup_count}) | "
              f"进化{len(evo_log)}轮 | Δ{delta*100:+.1f}%")
        sys.stdout.flush()

        final.append({
            "id": f"top_{i+1:02d}",
            "name": bot["name"],
            "personality": bot["personality"],
            "prompt": prompt,
            "original_batch": bot["batch"],
            "params": params.to_dict(),
            "result_base": bt_base.to_dict(),
            "result_evo": bt_evo.to_dict(),
            "equity_base": bt_base.equity_curve.tolist(),
            "equity": bt_evo.equity_curve.tolist(),
            "evolution_log": evo_log,
            "trades": [
                {
                    "entry_idx": t.entry_idx,
                    "exit_idx": t.exit_idx,
                    "direction": "LONG" if t.direction == 1 else "SHORT",
                    "entry_price": round(t.entry_price, 2),
                    "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                    "pnl_pct": round(t.pnl_pct * 100, 2),
                    "leverage": t.leverage,
                    "margin": round(t.margin, 2),
                    "exit_reason": t.exit_reason,
                }
                for t in bt_evo.trades
            ],
        })

    # ─── 4. 保存 ───
    out_dir = os.path.join(ROOT, "agent_top20_evolve")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_bots.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    btc_prices = df["close"].tolist()
    with open(os.path.join(out_dir, "btc_prices.json"), "w") as f:
        json.dump(btc_prices, f)
    print(f"\n  结果: {out_dir}/")

    # ─── 5. Dashboard ───
    print("\n[4] 生成对比看板...")
    dash_data = []
    for r in final:
        dash_data.append({
            "id": r["id"], "name": r["name"],
            "personality": r["personality"],
            "prompt": r["prompt"],
            "reasoning": "",
            "params": r["params"],
            "result": r["result_evo"],
            "equity": r["equity"],
            "trades": r["trades"],
        })
    from run_batch_agents import build_dashboard
    html = build_dashboard(dash_data, btc_prices)
    html_path = os.path.join(ROOT, "agent_top20_evolve_dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  看板: {html_path}")

    # ─── 6. 汇总 ───
    print(f"\n{'='*60}")
    print(f"  进化效果汇总 (精选20 · 15m · 每周反思)")
    print(f"{'='*60}")
    print(f"  {'Bot':10s} {'来源':12s} {'无反思':>10s} {'有反思':>10s} "
          f"{'Δ':>8s} {'Sharpe变化':>14s} {'爆仓变化':>10s}")
    print(f"  {'-'*72}")

    improved = 0
    sorted_final = sorted(final,
                          key=lambda x: x["result_evo"]["total_return"] - x["result_base"]["total_return"],
                          reverse=True)
    for r in sorted_final:
        rb = r["result_base"]
        re = r["result_evo"]
        d = re["total_return"] - rb["total_return"]
        if d > 0:
            improved += 1
        blow_change = f"{rb.get('blowup_count',0)}→{re.get('blowup_count',0)}"
        print(f"  {r['name']:10s} {r['original_batch']:12s} "
              f"{rb['total_return']*100:+9.1f}% "
              f"{re['total_return']*100:+9.1f}% "
              f"{d*100:+7.1f}% "
              f"{rb['sharpe_ratio']:+.2f}→{re['sharpe_ratio']:+.2f}  "
              f"{blow_change:>8s}")

    print(f"\n  反思改善率: {improved}/{len(final)} ({improved/max(1,len(final))*100:.0f}%)")
    print("  完成!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
