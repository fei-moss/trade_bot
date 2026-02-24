# Trade Bot - 策略发现工厂

## 项目定位

这个项目**不是交易执行系统**，而是一个**策略发现与生成工厂**。

```
真实行情数据 → [策略发现引擎] → 100个标准化Bot策略Profile → 交给openclaw管理/执行
```

最终产出物是一组JSON策略配置文件(profiles/)，每个文件完整描述一个bot的"性格"和交易规则，可被openclaw等外部平台直接消费。后期这些策略会被封装进skills，让agent快速复制。

## 核心目标

- 让AI从近4个月的**真实行情**中"悟"出策略，而非手动设计
- 生成**100个旗帜鲜明、差异极大**的bot策略
- **任何行情**下都有1-2个表现特别好、1-2个表现特别差的bot
- 策略是**可解释的规则组合**（非黑箱），便于封装为skill

## 多样性维度（不只是保守→激进的光谱）

| 维度 | 变化范围 |
|------|----------|
| 原型(10种) | 趋势追踪、均值回归、突破猎手、刮头皮、波段、动量、逆势、波动率、定投、网格 |
| 激进度(5级) | 极保守(1x/1%) → 极激进(50x滚仓) |
| 方向(3种) | 只做多 / 只做空 / 双向 |
| 指标组合(20+) | EMA/RSI/MACD/布林/超级趋势/一目均衡/蜡烛形态/唐奇安/CCI/MFI... |
| 决策逻辑 | 单指标阈值 / 多指标投票 / 指标背离 / 序列条件 |
| 出场方式 | 固定止盈 / 移动止损 / 信号反转 / 时间退出 |
| 仓位管理 | 固定 / 凯利 / 马丁 / 反马丁 / 波动率缩放 |
| 滚仓 | 不滚 / 浮盈30%滚 / 浮盈50%滚（用浮盈当保证金开新仓） |

## 项目结构

```
trade_bot/
├── main.py                          # 主入口：6步流水线（数据→regime→LLM生成→回测→筛选→输出）
├── config/settings.yaml             # 全局配置
├── requirements.txt                 # Python依赖
├── src/
│   ├── strategy/
│   │   ├── schema.py                # ★ 核心：策略基因组Schema（BotStrategy dataclass）
│   │   ├── indicators.py            # 20+技术指标计算（统一接口compute_indicator）
│   │   └── signals.py               # 条件→买卖信号翻译器（evaluate_condition）
│   ├── data/
│   │   ├── fetcher.py               # ccxt数据下载+parquet缓存
│   │   └── regime.py                # 行情分类：trending_up/down, ranging, volatile
│   ├── backtest/
│   │   └── engine.py                # 回测引擎（支持杠杆/多空/滚仓/止盈止损/移动止损/爆仓）
│   ├── generator/
│   │   ├── llm_generator.py         # LLM策略生成（Anthropic API，带多样性矩阵和滚仓prompt）
│   │   └── diversity.py             # 收益相关性聚类 + 多样性筛选
│   └── output/
│       └── profiler.py              # Bot Profile输出 + skill_summary生成
├── profiles/                        # 最终产物：每个bot一个JSON文件 + _index.json
├── data_cache/                      # 行情数据缓存（parquet）
└── .claude/                         # Claude Code项目配置
```

## 关键设计

### 策略Schema (src/strategy/schema.py)

`BotStrategy` dataclass 是所有bot的统一描述格式：
- `archetype`: 策略原型（trend_follower, mean_reverter, breakout_hunter等10种）
- `aggressiveness`: 激进度（ultra_conservative → ultra_aggressive 5级）
- `position`: 仓位配置（杠杆/仓位/方向/滚仓参数）
- `entry_rules`: 入场规则（多组条件，组间OR，组内AND）
- `exit_rule`: 出场规则（止盈/止损/移动止损/时间退出/信号退出）
- `risk`: 风控（日最大亏损/最大回撤/冷却期）

### 滚仓机制 (src/backtest/engine.py)

`position.rolling = true` 时启用：
1. 持仓浮盈达到 `rolling_trigger_pct`（如30%）时触发
2. 取 `rolling_reinvest_pct`（如80%）的浮盈作为新仓保证金
3. 以当前价、相同方向和杠杆开新仓
4. `rolling_move_stop=true` 时老仓止损移到成本价（保本）
5. 最多滚 `rolling_max_times` 次
6. 效果：单边行情中指数级放大收益，回调时新仓爆仓但老仓保本

### 多样性保障 (src/generator/diversity.py)

1. 生成150个候选（多余50%用于筛选）
2. 计算所有策略收益曲线的相关性矩阵
3. 层次聚类分组
4. 每组选最好+最差的代表（最差的也要选，因为在某行情中最差 = 在另一行情中可能最好）
5. 剩余名额按多样性分数补齐

### 数据流水线 (main.py)

```
Step 1: fetch_ohlcv() 下载4个月K线 (ccxt + Binance)
Step 2: classify_regime() 标记行情阶段
Step 3: StrategyGenerator.generate_batch() LLM批量生成策略
Step 4: batch_backtest() 回测所有策略
Step 5: select_diverse_subset() 多样性筛选出100个
Step 6: generate_all_profiles() 输出Bot Profile JSON
```

## 运行方式

```bash
pip install -r requirements.txt
# 需要设置 ANTHROPIC_API_KEY 环境变量
python main.py --symbols BTC/USDT ETH/USDT --bots 100 --generate 150
```

## 当前状态

- [x] 策略Schema定义（含滚仓）
- [x] 技术指标库（20+指标）
- [x] 信号生成引擎
- [x] 数据采集模块（ccxt + 缓存）
- [x] 行情regime识别
- [x] 回测引擎（含滚仓+爆仓）
- [x] LLM策略生成器（含多样性矩阵）
- [x] 多样性评分与筛选
- [x] Profile输出模块
- [ ] 端到端测试（尚未跑通完整流水线）
- [ ] 手续费模拟（config中已预留commission_rate）
- [ ] 多币种回测（当前只用第一个symbol回测）
- [ ] 滑点模拟
- [ ] Profile → openclaw skill 转换器

## 编码规范

- Python 3.11+
- 使用 dataclass 而非 pydantic（轻量）
- 数据用 pandas DataFrame，缓存用 parquet
- 策略配置序列化为 JSON（可选YAML）
- LLM调用使用 anthropic SDK

## 注意事项

- **不要修改schema.py中的字段名**，profile格式是和外部系统的接口契约
- **不要减少indicators.py中的指标类型**，LLM生成的策略会引用这些指标
- 回测引擎中的资金管理改用了保证金模型（capital是可用保证金，开仓扣保证金，平仓归还）
- 滚仓的新仓保证金来源是浮盈，从capital中预扣
