"""
20-Bot 进化测试 (15m K线 · 50x杠杆上限 · 每周反思)

用法:
  OPENROUTER_API_KEY=sk-or-... python3 -u run_20_reflect.py
"""

import os, sys, json, time
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.generator.prompt_expander import StrategyDesigner
from src.generator.llm_tuner import LLMTuner, format_market_context
from agent_backtest import run_agent_backtest, run_with_reflection

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  20 条精选描述 —— 覆盖激进度/方向/信号/出场/人格
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOT_DESCRIPTIONS = [
    # 激进度光谱 (5)
    "我是个胆小鬼，千万别让我亏钱",
    "中等风险，追求长期正收益",
    "激进一点没关系",
    "像凉兮一样100倍杠杆滚仓",
    "赌狗模式，暴富或归零",

    # 方向偏好 (4)
    "只做多，永远看多BTC",
    "只做空，BTC必须跌到2万",
    "涨了做多跌了做空，双向通吃",
    "只在暴跌时抄底",

    # 信号偏好 (3)
    "我只相信均线金叉死叉",
    "布林带收口后突破方向跟进",
    "EMA和Supertrend必须同向才开仓",

    # 出场风格 (3)
    "止损要紧，1%就跑",
    "让利润奔跑直到趋势反转",
    "浮盈加仓浮亏砍仓",

    # 人格/文化 (3)
    "我是日本武士，纪律至上",
    "我是对冲基金经理，只看Sharpe",
    "我是AI机器人，纯数学最优",

    # 策略思路 (2)
    "海龟交易法，突破加仓",
    "逆势交易，所有人恐惧时我贪婪",
]

LEVERAGE_CAP = 50.0
TIMEFRAME = "15m"
DAYS = 148
EXCHANGE = "okx"
REFLECTION_BARS = 672       # 7 days × 24h × 4 (15m bars)
INITIAL_CAPITAL = 10000.0


def clamp_leverage(params):
    """将杠杆限制在 LEVERAGE_CAP 以内。"""
    if params.base_leverage > LEVERAGE_CAP:
        params.base_leverage = LEVERAGE_CAP
    if params.max_leverage > LEVERAGE_CAP:
        params.max_leverage = LEVERAGE_CAP
    params.max_leverage = max(params.base_leverage, params.max_leverage)
    return params


def main():
    n = len(BOT_DESCRIPTIONS)
    print("=" * 60)
    print(f"  20-Bot 进化测试 | {TIMEFRAME} | ≤{LEVERAGE_CAP:.0f}x | 每周反思")
    print("=" * 60)

    # ─── 1. 数据 ───
    print(f"\n[1] 下载 {TIMEFRAME} K线 ({DAYS}天) via {EXCHANGE}...")
    df = fetch_ohlcv("BTC/USDT", TIMEFRAME, DAYS, exchange_id=EXCHANGE)
    regime = classify_regime(df, version="v1", min_duration=48 * 4)
    print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
    summary = regime_summary(df, regime)
    for r, s in summary.items():
        print(f"  {r}: {s['pct']:.0%}")
    sys.stdout.flush()

    warmup = min(720 * 4, len(df))
    ctx = format_market_context(df.iloc[:warmup], regime.iloc[:warmup])

    # ─── 2. AI 设计策略 ───
    print(f"\n[2] StrategyDesigner 为 {n} 个 Bot 生成策略...")
    sys.stdout.flush()
    designer = StrategyDesigner()
    designs = []

    for i, desc in enumerate(BOT_DESCRIPTIONS):
        tag = f"[{i+1}/{n}]"
        try:
            bot = designer.design(desc, market_context=ctx)
            p = clamp_leverage(bot["params"])
            dir_str = "做多" if p.long_bias > 0.7 else ("做空" if p.long_bias < 0.3 else "双向")
            print(f"  {tag} {bot['name']:8s} ← \"{desc[:25]}\" | "
                  f"{dir_str} {p.base_leverage:.0f}x | "
                  f"仓位{p.risk_per_trade*100:.0f}% | "
                  f"滚仓={'开' if p.rolling_enabled else '关'}")
            designs.append({
                "id": f"bot_{i+1:03d}",
                "name": bot["name"],
                "personality": bot["personality"],
                "user_input": desc,
                "reasoning": bot["reasoning"],
                "params": p,
            })
        except Exception as e:
            print(f"  {tag} ✗ \"{desc[:25]}\" 失败: {e}")
        sys.stdout.flush()
        time.sleep(0.5)

    print(f"\n  成功生成: {len(designs)}/{n}")

    # ─── 3. 回测 (有反思 vs 无反思) ───
    tuner = LLMTuner()
    n_segs = max(1, len(df) // REFLECTION_BARS)
    print(f"\n[3] 回测 {len(designs)} 个Bot | {n_segs} 周段 | "
          f"预计 {len(designs) * n_segs} 次LLM反思调用")
    sys.stdout.flush()

    final = []
    for i, d in enumerate(designs):
        tag = f"[{i+1}/{len(designs)}]"
        p = d["params"]

        # 无反思基线
        bt_base = run_agent_backtest(df, p, regime, initial_capital=INITIAL_CAPITAL)
        base_ret = bt_base.total_return

        # 有反思进化
        bt_evo, evo_log = run_with_reflection(
            df, p, regime, tuner,
            user_prompt=d["user_input"],
            reflection_interval=REFLECTION_BARS,
            initial_capital=INITIAL_CAPITAL,
            verbose=True,
        )
        evo_ret = bt_evo.total_return
        delta = evo_ret - base_ret

        blow_base = bt_base.blowup_count
        blow_evo = bt_evo.blowup_count

        print(f"  {tag} {d['name']:8s} | "
              f"无反思: {base_ret*100:+7.1f}% (💥{blow_base}) | "
              f"有反思: {evo_ret*100:+7.1f}% (💥{blow_evo}) | "
              f"进化{len(evo_log)}轮 | Δ{delta*100:+.1f}%")
        sys.stdout.flush()

        final.append({
            "id": d["id"],
            "name": d["name"],
            "personality": d["personality"],
            "prompt": d["user_input"],
            "reasoning": d["reasoning"],
            "params": p.to_dict(),
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
    out_dir = os.path.join(ROOT, "agent_20_reflect")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_bots.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    btc_prices = df["close"].tolist()
    with open(os.path.join(out_dir, "btc_prices.json"), "w") as f:
        json.dump(btc_prices, f)
    print(f"\n  结果已保存: {out_dir}/")

    # ─── 5. Dashboard ───
    print("\n[4] 生成对比看板...")
    # 构造 dashboard 兼容格式（用进化后的结果）
    dash_data = []
    for r in final:
        dash_data.append({
            "id": r["id"],
            "name": r["name"],
            "personality": r["personality"],
            "prompt": r["prompt"],
            "reasoning": r["reasoning"],
            "params": r["params"],
            "result": r["result_evo"],
            "equity": r["equity"],
            "trades": r["trades"],
        })
    from run_batch_agents import build_dashboard
    html = build_dashboard(dash_data, btc_prices)
    html_path = os.path.join(ROOT, "agent_20_reflect_dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  看板: {html_path}")

    # ─── 6. 进化效果汇总 ───
    print(f"\n{'='*60}")
    print(f"  进化效果汇总 ({TIMEFRAME} · ≤{LEVERAGE_CAP:.0f}x · 每周反思)")
    print(f"{'='*60}")
    print(f"  {'Bot':10s} {'无反思':>10s} {'有反思':>10s} {'Δ':>8s} {'Sharpe变化':>12s}")
    print(f"  {'-'*50}")

    improved = 0
    for r in sorted(final, key=lambda x: x["result_evo"]["total_return"] - x["result_base"]["total_return"], reverse=True):
        rb = r["result_base"]
        re = r["result_evo"]
        d = re["total_return"] - rb["total_return"]
        if d > 0:
            improved += 1
        print(f"  {r['name']:10s} "
              f"{rb['total_return']*100:+9.1f}% "
              f"{re['total_return']*100:+9.1f}% "
              f"{d*100:+7.1f}% "
              f"{rb['sharpe_ratio']:+.2f}→{re['sharpe_ratio']:+.2f}")

    print(f"\n  反思改善率: {improved}/{len(final)} ({improved/max(1,len(final))*100:.0f}%)")
    print("  完成!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
