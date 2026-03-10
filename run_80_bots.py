"""
80-Bot 多样性压力测试

用 StrategyDesigner 从80条用户自然语言直接生成策略参数，
不使用任何模板，千人千面。
"""

import os, sys, json, time
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.generator.prompt_expander import StrategyDesigner
from src.generator.llm_tuner import LLMTuner, format_market_context
from agent_backtest import run_agent_backtest, run_with_reflection

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  80 条用户描述 —— 覆盖风险/方向/信号/频率/人格/文化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER_DESCRIPTIONS = [
    # ─── 激进度光谱（10条）───
    "我是个胆小鬼，千万别让我亏钱",
    "稳健一点，年化15%就满足了",
    "中等风险，追求长期正收益",
    "我可以承受30%回撤",
    "激进一点没关系",
    "全仓梭哈干他",
    "像凉兮一样100倍杠杆滚仓",
    "赌狗模式，暴富或归零",
    "125倍杠杆做空，不成功便成仁",
    "用1%的钱博100倍的收益",

    # ─── 方向偏好（10条）───
    "只做多，永远看多BTC",
    "只做空，BTC必须跌到2万",
    "涨了做多跌了做空，双向通吃",
    "主做多偶尔做空保护一下",
    "7成时间做空3成做多",
    "完全中性，让数据决定方向",
    "牛市做多熊市做空",
    "震荡行情两边吃",
    "只在暴跌时抄底",
    "只在暴涨时做空",

    # ─── 信号偏好（10条）───
    "我只相信均线金叉死叉",
    "RSI超卖就买超买就卖",
    "MACD金叉放量时入场",
    "布林带收口后突破方向跟进",
    "什么指标都不看，随缘交易",
    "只看成交量，放量就进缩量就出",
    "ATR暴涨时入场赌大波动",
    "用5分钟级别的超短线逻辑",
    "看周线级别的大趋势",
    "EMA和Supertrend必须同向才开仓",

    # ─── 出场风格（10条）───
    "止损要紧，1%就跑",
    "宽止损，给行情呼吸空间",
    "不设止损，靠信号反转出场",
    "移动止损锁利润",
    "固定止盈2倍止损距离",
    "赚10%就跑绝不贪心",
    "让利润奔跑直到趋势反转",
    "用时间止损，持仓超过3天就平",
    "盈亏比至少1:3才值得做",
    "浮盈加仓浮亏砍仓",

    # ─── 人格/文化（15条）───
    "我是一个佛系玩家，随缘定投",
    "我是退休教师，用养老金小小交易",
    "我是程序员，相信数据和统计",
    "我是赌场荷官，懂概率但忍不住赌",
    "我是日本武士，纪律至上",
    "我是华尔街量化分析师",
    "我是越南散户，跟着大V操作",
    "我是矿工，BTC信仰者",
    "我是做市商，两头吃价差",
    "我是末日论者，做空一切",
    "我是大学生，拿生活费炒币",
    "我是对冲基金经理，只看Sharpe",
    "我是家庭主妇，儿子教我炒币",
    "我是AI机器人，纯数学最优",
    "我是巴菲特的信徒，价值投资",

    # ─── 具体策略思路（10条）───
    "海龟交易法，突破加仓",
    "马丁格尔策略，亏了加倍",
    "网格交易，每隔500刀挂一单",
    "动量追踪，涨最猛的时候追进去",
    "波段操作，吃一段就跑",
    "滚仓策略，浮盈全部加码",
    "只做趋势的中间段，不抄底不逃顶",
    "逆势交易，所有人恐惧时我贪婪",
    "配对交易思路做均值回归",
    "高频做市，一天交易100次",

    # ─── 极端/创意（5条）───
    "如果明天是世界末日你会怎么交易",
    "用掷硬币决定方向但用严格风控管理",
    "只在整数关口（70000/80000）交易",
    "当别人爆仓时我入场",
    "我要做那个在熊市里悄悄赚钱的人",
]


def main():
    n = len(USER_DESCRIPTIONS)
    print("=" * 60)
    print(f"  80-Bot 多样性测试 ({n} 条用户描述)")
    print("=" * 60)

    # ─── 1. 数据 ───
    print("\n[1] 加载行情数据...")
    df = fetch_ohlcv("BTC/USDT", "1h", 148)
    regime = classify_regime(df, version="v1", min_duration=48)
    print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
    summary = regime_summary(df, regime)
    for r, s in summary.items():
        print(f"  {r}: {s['pct']:.0%}")

    warmup = min(720, len(df))
    ctx = format_market_context(df.iloc[:warmup], regime.iloc[:warmup])

    # ─── 2. AI 设计策略 ───
    print(f"\n[2] StrategyDesigner 为 {n} 个 Bot 生成策略...")
    designer = StrategyDesigner()
    designs = []

    for i, desc in enumerate(USER_DESCRIPTIONS):
        tag = f"[{i+1}/{n}]"
        try:
            bot = designer.design(desc)
            p = bot["params"]
            dir_str = "做多" if p.long_bias > 0.7 else ("做空" if p.long_bias < 0.3 else "双向")
            print(f"  {tag} {bot['name']:6s} ← \"{desc[:20]}...\" | "
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
            print(f"  {tag} ✗ \"{desc[:20]}...\" 失败: {e}")
        time.sleep(0.5)

    # ─── 3. 回测 ───
    enable_reflect = os.environ.get("REFLECT", "0") == "1"
    reflect_interval = int(os.environ.get("REFLECT_INTERVAL", "24"))
    tuner = LLMTuner() if enable_reflect else None

    if enable_reflect:
        print(f"\n[3] 回测 {len(designs)} 个 Bot（反思进化 每{reflect_interval}h）...")
    else:
        print(f"\n[3] 回测 {len(designs)} 个 Bot...")

    final = []
    for i, d in enumerate(designs):
        tag = f"[{i+1}/{len(designs)}]"

        evo_log = []
        if enable_reflect and tuner:
            bt, evo_log = run_with_reflection(
                df, d["params"], regime, tuner,
                user_prompt=d["user_input"],
                reflection_interval=reflect_interval,
                verbose=True,
            )
        else:
            bt = run_agent_backtest(df, d["params"], regime)

        ret = bt.total_return
        sign = "+" if ret >= 0 else ""
        blow_tag = f" 💥{bt.blowup_count}" if bt.blowup_count > 0 else ""
        evo_tag = f" 进化{len(evo_log)}轮" if evo_log else ""

        print(f"  {tag} {d['name']:6s}: {sign}{ret*100:7.1f}% | "
              f"S={bt.sharpe_ratio:5.2f} | DD={bt.max_drawdown*100:5.1f}% | "
              f"{bt.total_trades:4d}笔{blow_tag}{evo_tag}")

        final.append({
            "id": d["id"],
            "name": d["name"],
            "personality": d["personality"],
            "prompt": d["user_input"],
            "reasoning": d["reasoning"],
            "params": d["params"].to_dict(),
            "result": bt.to_dict(),
            "equity": bt.equity_curve.tolist(),
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
                for t in bt.trades
            ],
        })

    # ─── 4. 保存 ───
    out_dir = os.path.join(ROOT, "agent_80_result")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_bots.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    btc_prices = df["close"].tolist()
    with open(os.path.join(out_dir, "btc_prices.json"), "w") as f:
        json.dump(btc_prices, f)
    print(f"\n  结果: {out_dir}/")

    # ─── 5. Dashboard ───
    print("\n[4] 生成对比看板...")
    from run_batch_agents import build_dashboard
    html = build_dashboard(final, btc_prices)
    html_path = os.path.join(ROOT, "agent_80_dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  看板: {html_path}")

    # ─── 6. 汇总 ───
    print(f"\n{'='*60}")
    print(f"  汇总排名 (Top 10 / Bottom 10)")
    print(f"{'='*60}")
    ranked = sorted(final, key=lambda x: x["result"]["sharpe_ratio"], reverse=True)
    print("  ── Top 10 ──")
    for i, r in enumerate(ranked[:10]):
        ret = r["result"]["total_return"] * 100
        blow = r["result"].get("blowup_count", 0)
        blow_tag = f" 💥{blow}" if blow > 0 else ""
        print(f"  #{i+1:2d} {r['name']:6s} | {'+' if ret>=0 else ''}{ret:7.1f}% | "
              f"S={r['result']['sharpe_ratio']:5.2f} | "
              f"DD={r['result']['max_drawdown']*100:5.1f}% | "
              f"{r['result']['total_trades']:4d}笔{blow_tag}")
        print(f"       ← \"{r['prompt'][:30]}\"")

    print("  ── Bottom 10 ──")
    for i, r in enumerate(ranked[-10:]):
        ret = r["result"]["total_return"] * 100
        blow = r["result"].get("blowup_count", 0)
        blow_tag = f" 💥{blow}" if blow > 0 else ""
        idx = len(ranked) - 10 + i + 1
        print(f"  #{idx:2d} {r['name']:6s} | {'+' if ret>=0 else ''}{ret:7.1f}% | "
              f"S={r['result']['sharpe_ratio']:5.2f} | "
              f"DD={r['result']['max_drawdown']*100:5.1f}% | "
              f"{r['result']['total_trades']:4d}笔{blow_tag}")
        print(f"       ← \"{r['prompt'][:30]}\"")

    # 多样性统计
    print(f"\n  ── 多样性统计 ──")
    levs = [r["params"]["base_leverage"] for r in final]
    biases = [r["params"]["long_bias"] for r in final]
    thresholds = [r["params"]["entry_threshold"] for r in final]
    rolls = sum(1 for r in final if r["params"]["rolling_enabled"])
    blowups = sum(1 for r in final if r["result"].get("blowup_count", 0) > 0)
    profitable = sum(1 for r in final if r["result"]["total_return"] > 0)

    print(f"  杠杆范围: {min(levs):.0f}x ~ {max(levs):.0f}x")
    print(f"  方向分布: 做多{sum(1 for b in biases if b>0.7)} | "
          f"做空{sum(1 for b in biases if b<0.3)} | "
          f"双向{sum(1 for b in biases if 0.3<=b<=0.7)}")
    print(f"  阈值范围: {min(thresholds):.2f} ~ {max(thresholds):.2f}")
    print(f"  滚仓开启: {rolls}/{len(final)}")
    print(f"  爆仓bot数: {blowups}/{len(final)}")
    print(f"  盈利bot数: {profitable}/{len(final)}")


if __name__ == "__main__":
    main()
