# Trade Bot - 策略发现工厂 v2.0

## 项目定位

这个项目**不是交易执行系统**，而是一个**策略发现与生成工厂**。

核心理念变化（v1→v2）：从"LLM设计策略规则"进化为"42维权重向量定义Bot DNA"。

```
真实行情数据 → [42维权重生成] → 100个Bot(各自独特DNA) → 回测+筛选 → Bot Profile → 交给openclaw管理/执行
```

每个Bot的"性格"由42个参数横跨9大维度唯一确定。回测时权重被翻译为具体交易规则；实盘时权重作为LLM决策上下文。

## 核心目标

- **100个旗帜鲜明、差异极大的Bot**，通过42维权重向量随机采样+多样性筛选实现
- **任何行情**下都有1-2个表现特别好、1-2个表现特别差的Bot
- **可进化**：通过遗传算法(交叉+变异)产生下一代，Top Bot的权重作为种子
- 策略是**可解释的规则组合**（非黑箱），便于封装为skill

## 42维权重向量 — 9大维度

| 维度 | 参数数 | 关键参数 |
|------|--------|----------|
| 体制感知 | 3 | regime_focus, detector_version, switch_freq |
| 时间框架 | 3 | primary_timeframe, secondary_timeframe, decision_freq |
| 技术指标组合 | 5 | indicator_set(8种), num_indicators, ma_period_range, rsi_period, adx_period |
| 杠杆与仓位 | 5 | base_leverage(1-200x), leverage_dynamic, position_sizing(6种), risk_per_trade |
| 进出场规则 | 5 | entry_condition(5种), sl_type(6种), tp_type(5种), exit_on_regime_change |
| 进化与回测 | 6 | evolution_style, fitness_metric, exploration_rate |
| 数据源权重 | 5 | price/volume/funding/onchain/meme 权重(归一化) |
| 决策风格 | 5 | reasoning_depth, temperature, creativity_bias, bias_towards_action |
| 极端化开关 | 5 | yolo_mode, reverse_logic_prob, max_dd_tolerance, allow_blowup |

## 项目结构

```
trade_bot/
├── main.py                       # 主入口：生成模式 + --evolve进化模式
├── config/settings.yaml          # 全局配置（含LLM设置）
├── requirements.txt              # Python依赖
├── Dockerfile                    # Docker构建
├── docker-compose.yml            # Docker编排
├── src/
│   ├── strategy/
│   │   ├── schema.py             # ★ 核心：42维WeightVector + BotConfig + PARAM_SPACE
│   │   ├── indicators.py         # 20+技术指标计算（统一接口compute_indicator）
│   │   └── signals.py            # 权重→信号翻译器（8种指标集×5种入场条件）
│   ├── data/
│   │   ├── fetcher.py            # ccxt数据下载+csv缓存
│   │   ├── regime.py             # v1/v2/v3行情分类 → BULL/BEAR/SIDEWAYS
│   │   └── formatter.py          # K线数据LLM可读格式化
│   ├── backtest/
│   │   └── engine.py             # 回测引擎（6种SL/5种TP/6种仓位管理/极端开关）
│   ├── generator/
│   │   ├── weight_generator.py   # 随机生成权重向量 + 遗传交叉/变异
│   │   ├── llm_evolver.py        # ★ LLM驱动进化（OpenRouter + 混合进化 + 进化日志）
│   │   └── diversity.py          # 双重多样性筛选（权重余弦距离+收益相关性）
│   └── output/
│       └── profiler.py           # Bot Profile输出 + skill_summary生成
├── profiles/                     # 最终产物：每个Bot一个JSON + _index.json + evolution_log.json
├── data_cache/                   # 行情数据缓存（csv）
└── .claude/                      # Claude Code项目配置
```

## 关键设计

### 权重向量 Schema (src/strategy/schema.py)

`WeightVector` dataclass 包含42个字段，每个字段对应一个决策参数。
`PARAM_SPACE` 定义每个参数的类型(discrete/continuous)和有效范围。
`BotConfig` 包裹 WeightVector + 元数据（bot_id, generation, parent_ids等）。

### 信号生成 (src/strategy/signals.py)

8种指标集映射：
- MA+RSI: EMA交叉 + RSI过滤
- MACD+BB: MACD交叉 + 布林带回归
- Supertrend+ADX: 强趋势跟踪
- Ichimoku+KDJ: 云图 + 随机指标
- EMA+Volume: 趋势 + 成交量确认
- Stochastic+CCI: 超买超卖反转
- PurePriceAction: 支撑阻力 + 蜡烛形态
- Multi3: RSI + MACD + Bollinger 组合

5种入场条件：Strict(AND) / Loose(OR) / MomentumBreak / MeanReversion / RegimeConfirmed

### 回测引擎 (src/backtest/engine.py)

- 6种止损: None / Fixed / ATR / Trailing / RegimeAdaptive / MaxDD
- 5种止盈: RR2 / RR3 / RR5 / Trailing / None
- 6种仓位管理: Fixed / Kelly / Martingale / Anti-Martingale / AllIn / Volatility
- 动态杠杆: Fixed / RegimeScale / VolatilityScale
- 极端开关: YOLO倍率, 反向逻辑, 最大回撤容忍, 爆仓控制

### 进化机制

**遗传进化** (src/generator/weight_generator.py):
1. 按fitness_metric排序 → 保留Top 20% elite
2. Elite两两交叉(50%概率) → 变异(20%参数随机重置) → 归一化+一致性约束

**LLM驱动进化** (src/generator/llm_evolver.py):
1. 加载当前代profiles + 回测结果
2. 构建prompt: Top/Bottom Bot权重 + 行情环境 + 参数空间说明
3. LLM分析成功/失败模式，定向生成改进的权重向量
4. 校验LLM输出 → 回测 → 输出下一代

**混合进化** (`hybrid_evolve`): Elite保留(20%) + LLM后代(30-50%) + 遗传填充(剩余)

### 多样性保障 (src/generator/diversity.py)

双重多样性评分 = 0.4 × 权重余弦距离 + 0.6 × 收益相关性
确保选出的Bot不仅DNA不同，行为也不同。

### 数据流水线 (main.py)

**生成模式** (默认):
```
Step 1: fetch_ohlcv() 下载K线数据 (ccxt + Binance)
Step 2: classify_regime() 标记行情 BULL/BEAR/SIDEWAYS (支持v1/v2/v3)
Step 3: generate_batch() 随机生成150个42维权重向量
Step 4: batch_backtest() 回测所有Bot
Step 5: select_diverse_subset() 双重多样性筛选出100个
Step 6: generate_all_profiles() 输出Bot Profile JSON
```

**进化模式** (`--evolve`):
```
Step 1: load_profiles() 加载当前代Bot
Step 2: fetch_ohlcv() + classify_regime() 加载行情数据
Step 3: hybrid_evolve() 混合进化 (Elite + LLM + Genetic)
Step 4: batch_backtest() 回测新一代Bot
Step 5: generate_all_profiles() 输出到 profiles/gen_N/
```

## 运行方式

```bash
pip install -r requirements.txt

# 生成第0代
python main.py --symbols BTC/USDT --bots 10 --generate 25

# LLM进化（需要OPENROUTER_API_KEY）
export OPENROUTER_API_KEY=sk-or-...
python main.py --evolve --bots 10 --evolve-count 5 --model anthropic/claude-sonnet-4-20250514

# Docker运行
docker compose up --build
docker compose run trade-bot --evolve --bots 10 --evolve-count 5
```

## 当前状态

- [x] 42维权重向量Schema定义
- [x] 技术指标库（20+指标）
- [x] 8种指标集信号生成引擎
- [x] 数据采集模块（ccxt + 缓存）
- [x] 行情regime识别（v1/v2/v3）
- [x] 回测引擎（多种SL/TP/仓位管理/极端开关）
- [x] 随机权重生成器 + 唯一性校验
- [x] 遗传交叉 + 变异 + 进化机制
- [x] 双重多样性评分与筛选
- [x] Profile输出模块
- [ ] 端到端测试（尚未跑通完整流水线）
- [ ] 手续费模拟（config中已预留commission_rate）
- [ ] 多币种回测（当前只用第一个symbol回测）
- [ ] 滑点模拟
- [x] LLM驱动进化（OpenRouter + 混合进化 + 进化日志）
- [ ] Profile → openclaw skill 转换器

## 编码规范

- Python 3.11+
- 使用 dataclass 而非 pydantic（轻量）
- 数据用 pandas DataFrame，缓存用 parquet
- Bot配置序列化为 JSON
- 遗传操作使用 random 模块（可设seed复现）
- LLM调用使用 openai SDK（通过OpenRouter兼容所有模型）

## 注意事项

- **不要修改schema.py中PARAM_SPACE的key名**，权重向量格式是和外部系统的接口契约
- **不要减少indicators.py中的指标类型**，信号生成器会引用这些指标
- **不要修改Regime的值(BULL/BEAR/SIDEWAYS)**，多个模块依赖这些字符串
- 回测引擎中的资金管理用保证金模型（capital是可用保证金，开仓扣保证金，平仓归还）
- 归一化组(data_source权重)的总和必须=1.0
