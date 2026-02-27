"""生成每个Bot的自然语言交易叙述，按天分组，不省略任何交易。"""
import sys, json, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime
from src.backtest.engine import run_backtest
from src.strategy.schema import BotConfig


def fmt_money(v):
    av = abs(v)
    if av >= 1e9: return f"{av/1e9:.2f}亿"
    if av >= 1e6: return f"{av/1e6:.2f}百万"
    if av >= 1e4: return f"{av/1e4:.2f}万"
    return f"{av:,.2f}"


def fmt_price(v):
    return f"{v:,.2f}"


def narrate_bot(bot, result, timestamps):
    trades = result.trades
    s = result.to_dict()

    lines = []
    lines.append(f"## {bot.bot_id}（{bot.name}）")
    lines.append(f"总成绩：收益 {s['total_return']*100:,.0f}% | "
                 f"{s['total_trades']}笔交易 | "
                 f"胜率 {s['win_rate']*100:.0f}% | "
                 f"最大回撤 {s['max_drawdown']*100:.0f}%")
    lines.append("")

    if not trades:
        lines.append("整个回测期间没有产生任何交易。")
        lines.append("原因：策略参数组合（Supertrend+ADX + MultiBar确认）存在逻辑矛盾，入场条件永远无法满足。")
        lines.append("\n---\n")
        return "\n".join(lines)

    reason_map = {
        "take_profit": "止盈", "stop_loss": "止损",
        "trailing_stop": "移动止损", "trailing_tp": "移动止盈",
        "regime_change": "行情切换", "regime_change_loss": "行情切换止损",
        "liquidation": "爆仓", "end_of_data": "结束平仓",
        "max_drawdown": "最大回撤止损",
    }

    trade_records = []
    for i, t in enumerate(trades):
        entry_ts = str(timestamps[min(t.entry_idx, len(timestamps)-1)])
        exit_ts = str(timestamps[min(t.exit_idx, len(timestamps)-1)]) if t.exit_idx is not None else ""
        entry_date = entry_ts[:10]
        exit_date = exit_ts[:10]
        entry_time = entry_ts[11:16]
        exit_time = exit_ts[11:16]
        trade_records.append({
            "idx": i,
            "entry_date": entry_date, "entry_time": entry_time,
            "exit_date": exit_date, "exit_time": exit_time,
            "direction": t.direction, "leverage": t.leverage,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "margin": t.margin, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "position_size": t.margin * t.leverage,
        })

    day_events = defaultdict(lambda: {"opens": [], "closes": []})
    for tr in trade_records:
        day_events[tr["entry_date"]]["opens"].append(tr)
        if tr["exit_date"]:
            day_events[tr["exit_date"]]["closes"].append(tr)

    capital = 10000.0
    trade_capital = {}
    for tr in trade_records:
        capital += tr["pnl"]
        trade_capital[tr["idx"]] = capital

    capital = 10000.0
    prev_capital = 10000.0
    for date in sorted(day_events.keys()):
        ev = day_events[date]
        opens = ev["opens"]
        closes = ev["closes"]

        if not opens and not closes:
            continue

        day_pnl = sum(tr["pnl"] for tr in closes)
        day_wins = sum(1 for tr in closes if tr["pnl"] > 0)
        day_losses = len(closes) - day_wins

        end_capital = prev_capital + day_pnl
        for tr in closes:
            pass

        last_trade_in_day = None
        all_day_trades = sorted(
            [(tr["idx"], tr) for tr in closes],
            key=lambda x: x[0]
        )
        if all_day_trades:
            last_trade_in_day = all_day_trades[-1][0]
            end_capital = trade_capital[last_trade_in_day]

        day_header = f"### {date}"
        if closes:
            pnl_sign = "+" if day_pnl >= 0 else ""
            day_header += f"  |  平仓{len(closes)}笔（{day_wins}盈{day_losses}亏）"
            day_header += f"  |  当日盈亏 {pnl_sign}${fmt_money(day_pnl)}"
            day_header += f"  |  账户 ${fmt_money(end_capital)}"
        if opens and not closes:
            day_header += f"  |  开仓{len(opens)}笔"

        lines.append(day_header)

        printed_idx = set()

        for tr in sorted(closes, key=lambda x: x["idx"]):
            i = tr["idx"]
            if i in printed_idx:
                continue
            printed_idx.add(i)

            dir_cn = "做多" if tr["direction"] == 1 else "做空"
            reason_cn = reason_map.get(tr["exit_reason"], tr["exit_reason"])
            pnl_sign = "+" if tr["pnl"] >= 0 else "-"
            result_icon = "+" if tr["pnl"] >= 0 else "-"

            line = (f"  [{result_icon}] 第{i+1}笔：{tr['entry_date']} {tr['entry_time']} {dir_cn} "
                    f"开仓 {fmt_price(tr['entry_price'])}，"
                    f"保证金 ${fmt_money(tr['margin'])}（{tr['leverage']}x，仓位 ${fmt_money(tr['position_size'])}）"
                    f" → {tr['exit_time']} {reason_cn} "
                    f"平仓 {fmt_price(tr['exit_price'])}，"
                    f"盈亏 {pnl_sign}${fmt_money(tr['pnl'])}（{tr['pnl_pct']*100:+.1f}%）")
            lines.append(line)

        for tr in opens:
            if tr["idx"] in printed_idx:
                continue
            if tr["exit_date"] != date:
                dir_cn = "做多" if tr["direction"] == 1 else "做空"
                line = (f"  [>] 第{tr['idx']+1}笔：{tr['entry_time']} {dir_cn} "
                        f"开仓 {fmt_price(tr['entry_price'])}，"
                        f"保证金 ${fmt_money(tr['margin'])}（{tr['leverage']}x，仓位 ${fmt_money(tr['position_size'])}）"
                        f" → 持仓中...")
                lines.append(line)

        lines.append("")
        prev_capital = end_capital

    win_count = sum(1 for t in trades if t.pnl > 0)
    loss_count = len(trades) - win_count
    total_profit = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = sum(t.pnl for t in trades if t.pnl < 0)
    final_capital = trade_capital[len(trades) - 1]

    lines.append(f"### 总结")
    lines.append(f"- {len(trades)}笔交易：{win_count}胜 {loss_count}负")
    lines.append(f"- 总盈利：+${fmt_money(total_profit)}")
    lines.append(f"- 总亏损：-${fmt_money(abs(total_loss))}")
    lines.append(f"- 起始资金：$10,000 → 最终：${fmt_money(final_capital)}")
    lines.append("\n---\n")
    return "\n".join(lines)


def main():
    df = fetch_ohlcv("BTC/USDT", "1h", 148)
    regime = classify_regime(df, version="v1")
    timestamps = df["timestamp"].tolist()

    gen_dir = "profiles/gen_118"
    bots = []
    seen = set()
    for fname in sorted(os.listdir(gen_dir)):
        if not fname.startswith("bot_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(gen_dir, fname)) as f:
            bot = BotConfig.from_dict(json.load(f)["bot"])
        fp = bot.weights.fingerprint()
        if fp not in seen:
            seen.add(fp)
            bots.append(bot)

    output = []
    for bot in bots:
        print(f"回测 {bot.bot_id}...")
        result = run_backtest(df, bot, regime)
        output.append(narrate_bot(bot, result, timestamps))

    text = "\n".join(output)
    out_path = "profiles/trade_narrative.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n输出: {out_path}")
    print(text)


if __name__ == "__main__":
    main()
