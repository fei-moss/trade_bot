"""单独跑利弗莫尔 bot，验证性格锁定效果。"""

import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.strategy.decision import DecisionParams
from src.generator.llm_tuner import LLMTuner
from agent_backtest import run_agent_backtest, run_with_reflection

TIMEFRAME = "15m"
DAYS = 148
EXCHANGE = "okx"
REFLECTION_BARS = 672
INITIAL_CAPITAL = 10000.0
LEVERAGE_CAP = 150.0

bot_file = os.path.join(ROOT, "agent_top20_evolve/top_01_利弗莫尔.json")
with open(bot_file) as f:
    bot = json.load(f)

params = DecisionParams.from_dict(bot["params"])
params.base_leverage = max(1.0, min(params.base_leverage, LEVERAGE_CAP))
params.max_leverage = max(1.0, min(params.max_leverage, LEVERAGE_CAP))
prompt = bot["prompt"]

print("=" * 60)
print("  利弗莫尔 · 性格锁定验证")
print("=" * 60)
print(f"  性格参数 (应全程不变):")
print(f"    long_bias={params.long_bias:.2f}  base_leverage={params.base_leverage:.0f}x")
print(f"    risk_per_trade={params.risk_per_trade:.0%}  max_position_pct={params.max_position_pct:.0%}")
print(f"    rolling: enabled={params.rolling_enabled} trigger={params.rolling_trigger_pct:.0%} "
      f"reinvest={params.rolling_reinvest_pct:.0%} max={params.rolling_max_times}")
print(f"    weights: trend={params.trend_weight:.2f} mom={params.momentum_weight:.2f} "
      f"mr={params.mean_revert_weight:.2f} vol={params.volume_weight:.2f} "
      f"vola={params.volatility_weight:.2f}")

print(f"\n[1] 加载 {TIMEFRAME} K线 ({DAYS}天) via {EXCHANGE}...")
df = fetch_ohlcv("BTC/USDT", TIMEFRAME, DAYS, exchange_id=EXCHANGE)
regime = classify_regime(df, version="v1", min_duration=48 * 4)
print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
summary = regime_summary(df, regime)
for r, s in summary.items():
    print(f"  {r}: {s['pct']:.0%}")

print(f"\n[2] 无反思回测...")
t0 = time.time()
bt_base = run_agent_backtest(df, params, regime, initial_capital=INITIAL_CAPITAL)
print(f"  收益: {bt_base.total_return*100:+.1f}% | Sharpe: {bt_base.sharpe_ratio:.2f} | "
      f"交易: {bt_base.total_trades} | 爆仓: {bt_base.blowup_count}")

print(f"\n[3] 有反思回测 (性格锁定)...")
tuner = LLMTuner()
bt_evo, evo_log = run_with_reflection(
    df, params, regime, tuner,
    user_prompt=prompt,
    reflection_interval=REFLECTION_BARS,
    initial_capital=INITIAL_CAPITAL,
    verbose=True,
)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print(f"  利弗莫尔 · 结果对比 ({elapsed/60:.1f}分钟)")
print(f"{'=' * 60}")
print(f"  {'指标':12s} {'无反思':>12s} {'有反思(锁定)':>14s} {'Δ':>10s}")
print(f"  {'-'*50}")

base_ret = bt_base.total_return * 100
evo_ret = bt_evo.total_return * 100
delta_ret = evo_ret - base_ret
print(f"  {'收益':12s} {base_ret:+11.1f}% {evo_ret:+13.1f}% {delta_ret:+9.1f}%")

print(f"  {'Sharpe':12s} {bt_base.sharpe_ratio:+12.2f} {bt_evo.sharpe_ratio:+14.2f} "
      f"{bt_evo.sharpe_ratio - bt_base.sharpe_ratio:+10.2f}")

print(f"  {'最大回撤':12s} {bt_base.max_drawdown*100:11.1f}% {bt_evo.max_drawdown*100:13.1f}% "
      f"{(bt_evo.max_drawdown - bt_base.max_drawdown)*100:+9.1f}%")

print(f"  {'交易数':12s} {bt_base.total_trades:12d} {bt_evo.total_trades:14d}")
print(f"  {'胜率':12s} {bt_base.win_rate*100:11.1f}% {bt_evo.win_rate*100:13.1f}%")
print(f"  {'爆仓':12s} {bt_base.blowup_count:12d} {bt_evo.blowup_count:14d}")

print(f"\n  进化 {len(evo_log)} 轮:")
for log in evo_log:
    changes = log.get("key_changes", {})
    change_str = " | ".join(f"{k}: {v}" for k, v in changes.items()) if changes else "无变化"
    print(f"    #{log['generation']:2d} 收益={log['segment_return']*100:+.1f}% "
          f"交易={log['trades_count']} 胜率={log['win_rate']*100:.0f}% | {change_str}")

out = {
    "name": "利弗莫尔",
    "params": params.to_dict(),
    "result_base": bt_base.to_dict(),
    "result_evo": bt_evo.to_dict(),
    "equity_base": bt_base.equity_curve.tolist(),
    "equity_evo": bt_evo.equity_curve.tolist(),
    "evolution_log": evo_log,
}
out_file = os.path.join(ROOT, "agent_top20_evolve/top_01_利弗莫尔_locked.json")
with open(out_file, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n  结果保存到: {out_file}")
