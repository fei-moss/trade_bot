# Trade Bot - LLM 驱动的策略发现工厂

## 项目定位

这个项目**不是交易执行系统**，而是一个 **LLM 驱动的策略发现与进化工厂**。

```
用户自然语言描述 → [StrategyDesigner/LLM] → DecisionParams → [快脑决策函数] → 回测 → [每周反思进化] → 最终策略
```

核心思路：让 LLM 从真实行情中"悟"出策略参数，通过周期性反思不断进化。

## 双脑架构

```
┌─────────────────────────────────────────────────┐
│  慢脑 (LLM)                                      │
│  ├─ StrategyDesigner: 自然语言 → DecisionParams  │
│  ├─ LLMTuner.tune: 初始参数生成                   │
│  └─ LLMTuner.reflect: 周期性反思进化              │
│       ├─ 输入: 本周期交易日志 + 市场环境            │
│       ├─ 输入: 累计收益/峰值/BTC总变化 (全局视野)   │
│       ├─ 输入: 初始参数快照 (性格漂移检测)          │
│       └─ 输出: 微调后的 DecisionParams             │
├─────────────────────────────────────────────────┤
│  快脑 (决策函数)                                   │
│  compute_signals(df, params, regime)              │
│  → 每根K线计算 composite signal → 开多/开空/不动   │
└─────────────────────────────────────────────────┘
```

## 核心参数: DecisionParams

~30 个连续参数，LLM 可精细调节：

| 类别 | 关键参数 |
|------|----------|
| 信号权重(5维) | trend / momentum / mean_revert / volume / volatility |
| 交易阈值 | entry_threshold (0.05~0.55), exit_threshold |
| 方向偏好 | long_bias (0=只空, 0.5=双向, 1=只多) |
| 杠杆仓位 | base_leverage (1~200x), risk_per_trade |
| 止损止盈 | sl_atr_mult, tp_rr_ratio, trailing |
| 滚仓 | rolling_enabled, trigger_pct, reinvest_pct |
| Regime | regime_sensitivity, exit_on_regime_change |

信号计算: `composite = Σ(weight_i × signal_i)`，超过 entry_threshold 则开仓。

## 项目结构

```
trade_bot/
├── CLAUDE.md                       # 本文档
├── config/settings.yaml            # 全局配置
├── requirements.txt
│
├── src/                            # 核心库
│   ├── strategy/
│   │   ├── decision.py             # ★ DecisionParams + compute_signals (快脑)
│   │   └── indicators.py           # 20+ 技术指标 (EMA/RSI/MACD/BB/ATR/ADX/Supertrend...)
│   ├── data/
│   │   ├── fetcher.py              # ccxt 数据下载 + csv 缓存 (支持 binance/okx 等)
│   │   └── regime.py               # 行情分类: BULL / BEAR / SIDEWAYS
│   ├── backtest/
│   │   └── engine.py               # 回测引擎 (杠杆/多空/滚仓/止盈止损/爆仓/复活)
│   └── generator/
│       ├── llm_tuner.py            # ★ 慢脑: tune(初始生成) + reflect(周期反思进化)
│       └── prompt_expander.py      # StrategyDesigner: 自然语言 → 参数 + 性格
│
├── agent_backtest.py               # 回测主逻辑 + run_with_reflection (进化回测)
├── run_batch_agents.py             # 批量运行 + build_dashboard (HTML看板)
├── run_80_bots.py                  # 80 bot 多样性测试
├── run_20_reflect.py               # 20 bot 反思进化测试
├── run_top20_evolve.py             # 精选 20 bot 进化 (多维择优)
│
├── data_cache/                     # K线缓存
│   ├── binance_BTC_USDT_1h_148d.csv
│   └── okx_BTC_USDT_15m_148d.csv
│
├── agent_batch_result/             # 第1批 20 bot (经典人设, 1h)
├── agent_80_result/                # 第2批 70 bot (千人千面, 1h)
├── agent_20_reflect/               # 第3批 20 bot (首次反思测试, 15m)
└── agent_top20_evolve/             # 第4批 精选20 bot (多维择优+进化, 15m)
```

## 关键机制

### 1. StrategyDesigner (prompt_expander.py)

用户输入随意的自然语言（如"像凉兮一样干"、"我是个胆小鬼"），LLM 从第一性原理出发，生成：
- `name`: 独特 bot 名
- `personality`: 2-3 句性格描述
- `reasoning`: 参数推导过程
- `DecisionParams`: 完整参数集

**不使用模板**，纯生成式。

### 2. 反思进化 (llm_tuner.reflect + run_with_reflection)

每 N 根 K 线暂停一次，LLM 根据实战结果微调参数：

```
每个周期 LLM 收到:
  ① 本周期交易日志 (胜/亏/止损原因)
  ② 本周期市场环境 (价格变化 + regime)
  ③ 全局累计摘要 (总收益/峰值/回撤/BTC总变化/近3周期趋势)
  ④ 初始参数快照 (用于性格漂移检测)

LLM 输出:
  微调后的 DecisionParams + 反思理由
```

**防漂移机制**:
- 自动检测 long_bias / leverage / threshold 是否偏离初始值超过阈值
- 偏离过大时在 prompt 中插入警告
- 单参数调整幅度限制 15%
- prompt 强调"先看大局再看细节"、"短期亏损不是改变方向的理由"

### 3. 爆仓复活 (agent_backtest.py)

资金跌至初始资金 1% 以下时触发复活：
- 重置资金到初始金额
- blowup_count +1
- 累计收益计入 -100%
- 继续交易（避免激进 bot 提前退出）

### 4. 回测引擎 (engine.py)

- 保证金模型（开仓扣保证金，平仓归还）
- 支持多空 + 杠杆 (1x~200x)
- 滚仓：浮盈触发 → 用浮盈开新仓 → 老仓止损移至成本价
- 止损：ATR 倍数 | 止盈：风险回报比 | 移动止损
- 爆仓检测 + 复活机制

### 5. Dashboard (run_batch_agents.py: build_dashboard)

HTML 看板功能：
- 对比排名表（收益/最高盈利/爆仓/Sharpe/回撤/胜率/盈亏比）可排序
- Chart.js 累计收益曲线 + BTC 叠加（可缩放/拖拽/全选/反选）
- 信号权重雷达图
- 每个 bot 的详细卡片（性格/参数/交易记录）

## 运行方式

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...

# 批量跑 20 个预设 bot (1h)
python3 -u run_batch_agents.py

# 80 bot 多样性测试 (1h)
python3 -u run_80_bots.py

# 20 bot 反思进化 (15m, 每周反思)
python3 -u run_20_reflect.py

# 精选 20 bot 进化测试 (15m, 多维择优)
python3 -u run_top20_evolve.py
```

## 历史实验结果

| 批次 | Bot数 | 数据 | 特点 | 结论 |
|------|-------|------|------|------|
| batch_20 | 20 | 1h/148d | 经典人设 (利弗莫尔/凉兮等) | 方向判断准的 bot 收益极高 |
| diverse_80 | 70 | 1h/148d | 自然语言千人千面 | 杠杆是双刃剑，高杠杆峰值高但爆仓多 |
| reflect_20 | 20 | 15m/148d | 首次反思测试 | 反思改善率 45%，激进型受益最大 |
| top20_evolve | 20 | 15m/148d | 多维择优+进化 | 进行中... |

**反思机制核心发现**:
- 爆仓型 bot 受益极大（赌神: -2730% → +42.7%，爆仓28次→0次）
- 原本方向正确的盈利型 bot 反而被"纠偏"变差（利弗莫尔: +55.7% → +9.0%）
- 原因: 短视偏差 — LLM 每周只看局部"横盘"，忽略了5个月的大趋势
- 改进: v2 reflect 加入累计上下文 + 性格漂移检测 + 惯性约束

## 编码规范

- Python 3.11+
- 使用 dataclass（轻量）
- 数据用 pandas DataFrame，缓存用 csv
- LLM 调用使用 openai SDK（通过 OpenRouter 兼容所有模型）
- 默认模型: anthropic/claude-sonnet-4.6

## 注意事项

- **不要修改 decision.py 中 DecisionParams 的字段名** — 这是 LLM 输出格式的接口契约
- **不要减少 indicators.py 中的指标类型** — compute_signals 会引用
- **不要修改 Regime 值 (BULL/BEAR/SIDEWAYS)** — 多模块依赖
- 回测引擎资金管理用保证金模型（capital 是可用保证金）
- fetcher.py 分页逻辑已适配 OKX（每次最多返回 300 根，自动翻页）
