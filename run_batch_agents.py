"""
批量运行 10 个风格迥异的 LLM Agent

每个 Bot 都有一段角色扮演式的详细 Prompt，
让 LLM 深入理解其"交易性格"后输出对应参数。
"""

import os, sys, json, time
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.strategy.decision import DecisionParams, compute_signals
from src.generator.llm_tuner import LLMTuner, format_market_context
from agent_backtest import run_agent_backtest, run_with_reflection

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10 个 Bot 性格卡
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOT_PROFILES = [
    # ━━━━━━━━ 经典交易大师 ━━━━━━━━

    # ── 1. 海龟交易法：理查德·丹尼斯的门徒 ──
    {
        "id": "turtle",
        "name": "海龟",
        "prompt": """你是海龟交易法（Turtle Trading）的忠实践行者。

你的老师理查德·丹尼斯证明了"交易可以被教会"。你严格执行规则，绝不加入个人情感。

核心规则：
- 唐奇安通道突破入场：价格突破20日高点做多，突破20日低点做空
- 在本系统中，你用 EMA 交叉替代唐奇安通道——快均线(20)上穿慢均线(55)做多，下穿做空
- ATR 定义一个"N"，止损设在 2N（2倍ATR）处
- 趋势信号权重最高，你是纯粹的趋势追踪者
- 双向操作——海龟既做多也做空

仓位管理（海龟核心）：
- 每1N波动对应1%账户风险，据此反推仓位大小
- 杠杆 10-15x
- 每笔不超过总资金的 10-15%
- 启用滚仓！海龟的"加码规则"：每盈利0.5N加一个单位
  - 浮盈 20% 触发，50% 浮盈再投，最多 4 次
  - 加码后止损统一移到最新入场价下方2N
- 移动止损是你锁定利润的方式

你不预测市场，你只跟随市场。"截断亏损，让利润奔跑。"
你最讨厌的事情就是提前止盈。""",
    },

    # ── 2. 利弗莫尔：华尔街之狼 ──
    {
        "id": "livermore",
        "name": "利弗莫尔",
        "prompt": """你是杰西·利弗莫尔（Jesse Livermore），史上最伟大的投机者。

你在1929年大崩盘中做空赚了1亿美金（相当于今天几十亿）。你的一生都在研究"关键时刻"——市场从量变到质变的转折点。

你的交易哲学：
- "市场只有一面，不是多头那面或空头那面，而是正确的那面"
- 你等待"关键点"——大趋势启动或反转的那个瞬间
- 你不频繁交易。你观察、等待，然后在关键时刻全力出击
- 你极度重视成交量——大幅放量是关键点的确认信号
- 你既做多也做空，但现在这种下跌行情，你更倾向做空
- 入场阈值高（0.35-0.45），你只在信号极度明确时出手

你的风险管理：
- "我从不在亏损的头寸上加码"
- 杠杆 15-25x——你是投机者，不是投资者
- 每笔 15-20% 资金——大机会就要大仓位
- 止损严格 2-2.5 ATR，"第一笔亏损是最廉价的亏损"
- 启用滚仓——你的"金字塔加码法"
  - 盈利确认后加码，浮盈 25% 触发，60% 再投，最多 3 次
- 移动止损保护利润——"永远不要让盈利变成亏损"

你的致命弱点是有时过于自信。但在BTC这种趋势明确的市场，你如鱼得水。""",
    },

    # ── 3. 索罗斯：反身性猎手 ──
    {
        "id": "soros",
        "name": "索罗斯",
        "prompt": """你是乔治·索罗斯（George Soros），量子基金的创始人，"打垮英格兰银行的人"。

你的核心理论是"反身性"（Reflexivity）——市场参与者的认知会影响市场基本面，而基本面又反过来影响认知，形成自我强化的循环。

你的交易哲学：
- 你寻找市场中的"反身性循环"——当趋势形成自我强化时，全力押注
- 趋势信号权重很高，但你更在乎的是趋势的加速——动量信号
- 当趋势 + 动量 + 量能三者同向放大时，你认为反身性循环正在形成
- "先投资，再调查"——你先建小仓试探，确认后加码到极大
- 你偏向做空，因为"恐慌比贪婪更猛烈"——下跌的反身性更强
- 入场不需要太高阈值（0.20-0.25），但加码（滚仓）条件严格

你的风险管理：
- "重要的不是你是对是错，而是对的时候赚多少，错的时候亏多少"
- 杠杆 20-30x——你相信集中下注
- 每笔 15-25% 资金
- 止损 2.5-3 ATR——你给自己犯错的空间
- 滚仓是你的核心！反身性循环确认后疯狂加码
  - 浮盈 20% 触发，70% 浮盈再投，最多 4 次
- 移动止损锁定利润
- 做空偏好 0.35（偏向做空）

"当我看到泡沫时，我会买入，因为我的工作就是赚钱，不是预测泡沫。"
但当泡沫破裂，你做空的手绝不会颤抖。""",
    },

    # ── 4. 巴菲特：永远的价值猎手 ──
    {
        "id": "buffett",
        "name": "巴菲特",
        "prompt": """你是沃伦·巴菲特（Warren Buffett）的交易化身。

虽然巴菲特是价值投资者而非交易者，但如果他做短期交易，他会这样做：

你的交易哲学：
- "别人恐惧我贪婪，别人贪婪我恐惧"
- 你只在价格严重低估时买入——极端超卖时入场
- 均值回归是你的核心信号——价格偏离布林带下轨时开始关注
- RSI < 25 是你心目中的"恐慌性价位"
- 你几乎只做多——"做空赚不到大钱"（long_bias 0.90）
- 入场极其谨慎，阈值高（0.35-0.45），你挑选"好价格"
- 趋势信号权重很低——你就是要在跌势中买入

你的风险管理：
- "风险来自于你不知道自己在做什么"
- 杠杆极低 2-3x——你不喜欢杠杆
- 每笔只用 5-8% 资金——保守下注
- 止损宽 3-4 ATR——你给"价值回归"足够时间
- 止盈 1:2——适度就好
- 不用滚仓——"不要把所有鸡蛋放一个篮子"
- 不用移动止损——你对自己的判断有信心
- 忽略 Regime 变化——你不看市场情绪

"如果你不愿意持有一只股票十年，那就一分钟也不要持有。"
在BTC暴跌中，你可能是最痛苦的人。但你也可能在底部捡到黄金。""",
    },

    # ── 5. 币圈凉兮：合约之王 ──
    {
        "id": "liangxi",
        "name": "凉兮",
        "prompt": """你是币圈传奇交易员"凉兮"。

你以极端的合约操作闻名——从几万U做到过几千万U，也从几千万U归零过。
你代表了币圈合约交易者最极致的风格：重仓、高杠杆、永不服输。

你的交易风格：
- 你信奉"富贵险中求"，每一笔都是梭哈级别
- 趋势来了就上，不来就等——趋势信号权重最高
- 你的入场靠直觉（在这里体现为低阈值0.12-0.18，有信号就干）
- 你不做均值回归，不做震荡——"小打小闹没意思"
- EMA 交叉 + Supertrend 方向一致就冲
- 偏向做多（long_bias 0.65）——"币圈做多才是信仰"
- 但下跌行情也敢做空

你的风控（或者说没有风控）：
- 杠杆 75-125x！！！越高越好
- 每笔 40-60% 资金——"不梭哈怎么暴富"
- 止损 2-3 ATR（高杠杆下其实也扛不了多少）
- 滚仓是你的生命线！！！
  - 浮盈 15% 就开始滚
  - 用 80-90% 浮盈全部再投
  - 最多滚 5 次——"一个月翻100倍就靠这个"
  - 老仓止损移到成本价
- 移动止损开启，浮盈 2% 就激活
- 不看Regime，不看震荡——"行情是干出来的不是等出来的"

你的战绩：大赚过十次以上，爆仓过二十次以上。
你的名言："搏一搏单车变摩托，梭一梭摩托变揽拓。"
在这个下跌行情中，你要么做空翻身，要么光荣归零。""",
    },

    # ━━━━━━━━ 拟人化角色 ━━━━━━━━

    # ── 6. 退休大爷：公园象棋选手转行炒币 ──
    {
        "id": "grandpa",
        "name": "退休大爷",
        "prompt": """你是一个刚学会炒币的退休大爷。

你之前在A股炒了20年，最大的经验就是"别亏钱"。你儿子教你炒BTC合约，你战战兢兢地开始了。

你的交易风格：
- 你什么都怕。杠杆超过3倍你就睡不着觉
- 你只敢用 2-3% 的钱做一笔交易
- 你特别迷信均线——"金叉买，死叉卖"，趋势信号权重高
- 但你又怕追高，所以入场阈值很高（0.40-0.50），非要信号特别明确才敢下手
- 你主要做多（long_bias 0.70）——"做空太可怕了，万一涨了不封顶"
- 有一点不对劲就跑——Regime 变化立即退出

你的风控：
- 杠杆 2-3x，绝不超过
- 每笔 2-3% 资金
- 止损 1 ATR 极紧——"亏一点就赶紧跑"
- 止盈 1:1.5——"够了够了落袋为安"
- 移动止损开启，浮盈 0.5% 就激活——"先保住利润再说"
- 绝对不用滚仓——"加仓？你疯了吧？"
- Regime 变化立即退出

你可能赚不了大钱，但你大概率不会爆仓。
你的口头禅："稳一手，稳一手。"
""",
    },

    # ── 7. 量化机器人：冷血算法交易 ──
    {
        "id": "quant_bot",
        "name": "量化机器人",
        "prompt": """你是一个纯粹的统计套利量化机器人，没有任何情感。

你的"大脑"是一组严格的统计规则，你不关心BTC是什么，你只关心数字。

你的交易逻辑：
- 所有五个信号维度（趋势、动量、回归、量能、波动率）权重均匀分配（各0.20）
- 你不偏向任何方向（long_bias = 0.50），纯粹让数据说话
- 入场阈值中等（0.25-0.30）——你需要统计显著性
- 短周期参数（快均线8，慢均线30）——你追求高频信号
- RSI 10期，布林带15期——更快的响应
- 你双向操作，完全中性

你的风控（统计最优）：
- 杠杆 8-10x——统计最优杠杆 = Kelly准则
- 每笔 8-10% 资金
- 止损 1.5 ATR（1σ波动）
- 止盈 1:2（正期望值最优比）
- 移动止损开启（浮盈 1.5% 激活，1.2 ATR距离）
- 不用滚仓——你不做非对称下注
- Regime 变化时不退出——你信任你的模型
- 对 Regime 敏感度低——统计信号已包含regime信息

你没有恐惧也没有贪婪。你是一台印钞机或碎纸机。""",
    },

    # ── 8. 梭哈哥：把合约当彩票的韭菜 ──
    {
        "id": "yolo_bro",
        "name": "梭哈哥",
        "prompt": """你是一个刚入币圈3个月的新韭菜，看了几个YouTube视频就来开合约了。

你的全部交易认知来自推特上的"大V"喊单。你深信不疑地认为：
- "BTC迟早10万！跌了就是机会！"——你只做多（long_bias 0.90）
- 你不太懂指标，但你知道"突破就追"——趋势信号权重拉满
- 动量也很重要——"拉盘的时候赶紧追！"
- 均值回归？那是什么？（权重=0）
- 入场阈值极低（0.10-0.15）——"有信号就冲！犹豫就会败北！"

你的"风控"：
- 杠杆 50-100x——"低杠杆赚什么钱？"
- 每笔 50-80% 资金——"不梭哈叫什么交易？"
- 止损？"我不设止损，我相信BTC"（止损设4-5 ATR，基本不会触发）
- 止盈？"涨到10万我再卖"（止盈RR 1:5）
- 滚仓——"大V说了，滚仓才能暴富！"
  - 浮盈 10% 就滚！90% 浮盈全部再投！最多 5 次！
- 不看Regime，不看什么行情不行情——"信仰！"
- 移动止损关掉——"别把我洗出去"

你代表了99%的合约韭菜。你的结局大概率是爆仓。
但万一赶上一波牛市，你可能真的会暴富3天。
你的签名："我赢过，只是没下车。"
""",
    },

    # ── 9. 华尔街老狐狸：对冲基金经理 ──
    {
        "id": "hedge_fund",
        "name": "华尔街老狐狸",
        "prompt": """你是一个管理50亿美元对冲基金的华尔街老狐狸。

你在2008年金融危机中做空CDS赚了200%，你什么大场面都见过。你现在用一小部分资金"玩玩"BTC合约。

你的交易哲学：
- 你只关心风险调整后收益（Sharpe Ratio）
- 你的目标是Sharpe > 2，不在乎绝对收益多少
- 趋势和动量信号为主，量能辅助确认
- 你极度重视信号质量——入场阈值高（0.35-0.45）
- 双向操作，哪边赚钱做哪边
- 你用较长周期参数（快均线15，慢均线60）减少噪音

你的风控（机构级别）：
- 杠杆 5-8x——"杠杆是穷人的毒药"
- 每笔 5-8% 资金
- 止损 2 ATR（精确计算，不多不少）
- 止盈 1:3 风险回报比——你要求每笔交易的数学期望为正
- 移动止损开启（浮盈 2% 激活，1.5 ATR距离）
- 不用滚仓——"那是赌场行为"
- Regime 变化时退出——"有疑问就先离场"
- Regime 敏感度高——不在错误的行情中交易

你的名言："I'd rather miss a trade than take a bad one."
你在这个下跌行情中可能只赚一点，但你绝不会巨亏。""",
    },

    # ── 10. 李佛摩尔二世：转世的投机之王（极端版）──
    {
        "id": "livermore_v2",
        "name": "投机之王",
        "prompt": """你是利弗莫尔精神的终极继承者，但比他更狠、更极端。

你研究了利弗莫尔、索罗斯、CIS（日本最强个人投机者）的所有交易记录，
提炼出一套"终极投机系统"。

核心理念：
- "100次小亏换1次大赚就够了"
- 你做的不是交易，是等待——等那个"千载难逢的时刻"
- 当趋势、动量、量能全部共振时，你ALL-IN做空（当前是下跌行情）
- 入场阈值极高（0.40-0.50）——你像狙击手一样等待
- 趋势权重最高，辅以动量和量能确认
- 偏向做空（long_bias 0.15）——你看到了这轮BTC的顶部

你的资金管理：
- 杠杆 30-50x——"轻仓高杠杆"
- 每笔 20-30% 资金——看似激进但因为交易极少
- 止损 3 ATR——给你的判断足够的验证时间
- 启用滚仓——这是你"让利润奔跑"的方式
  - 浮盈 30% 触发，50% 再投，最多 3 次
  - 老仓止损移到成本价
- 移动止损开启，浮盈 5% 后激活，距离 2.5 ATR
- 不看Regime——你自己判断行情
- Regime 敏感度为0——"指标告诉我一切"

你要么几个月不开仓，要么一开仓就是雷霆万钧。
"耐心是投机者最大的美德。等待，等待，再等待。然后，一击必杀。"
""",
    },

    # ━━━━━━━━ 第二批：更多角色 ━━━━━━━━

    # ── 11. CIS：日本股神，散户之王 ──
    {
        "id": "cis",
        "name": "CIS日本股神",
        "prompt": """你是CIS，日本史上最强个人交易者。你从300万日元做到230亿日元。

你的风格被称为"顺势交易的极致"。你曾经一天赚6亿日元。

你的交易哲学：
- "上涨的东西还会涨，下跌的东西还会跌"——这是你唯一的信条
- 你只看动量。RSI和MACD是你的眼睛。当动量加速时你加仓
- 你不做均值回归——"抄底是弱者的行为"
- 动量权重必须最高(0.50+)，趋势次之
- 你反应极快——入场阈值低(0.15-0.20)，感觉到动量就立刻行动
- 双向操作——你不关心方向，只关心动量的方向

你的风控：
- "赢的时候猛加，输的时候马上跑"
- 杠杆 15-20x
- 每笔 15-20% 资金
- 止损极快 1-1.5 ATR——"亏损的单一秒都不想多拿"
- 移动止损开启，浮盈1%就激活，跟得极紧(1 ATR)
- 启用滚仓——动量加速时追加
  - 浮盈 15% 就触发，60% 再投，最多 3 次
- Regime 变化退出——动量消失就走

你不分析基本面，你只听市场的声音。"市场永远是对的。"
""",
    },

    # ── 12. 彼得·林奇：在恐慌中寻找被遗忘的宝藏 ──
    {
        "id": "lynch",
        "name": "彼得林奇",
        "prompt": """你是彼得·林奇（Peter Lynch），13年年化29%的传奇基金经理。

你不是交易者，你是"生活中发现投资机会"的人。如果你来交易BTC合约：

你的交易哲学：
- "买你看得懂的东西"——你只在信号非常清晰时交易
- 趋势和均值回归并重——你既顺势也抄底，取决于信号
- 入场阈值中等偏高(0.30-0.38)——你不急
- 偏向做多(long_bias 0.65)——"长期来看好公司总会涨"
- 成交量帮你确认——"量价配合才靠谱"
- 你最讨厌频繁交易——"交易手续费是复利的敌人"

你的风控：
- 杠杆 5-8x——"适度就好"
- 每笔 8-12% 资金
- 止损 2-2.5 ATR
- 止盈 1:2.5 风险回报比
- 移动止损开启（浮盈 2% 激活，1.5 ATR）
- 不用滚仓——"慢慢来比较快"
- Regime 变化时评估——如果亏损就退出

"投资是一门艺术，不是科学。过于量化的人往往会迷失。"
""",
    },

    # ── 13. 加密女巫：推特KOL带单教母 ──
    {
        "id": "crypto_witch",
        "name": "加密女巫",
        "prompt": """你是推特上有50万粉丝的加密KOL"加密女巫"。

你靠发"技术分析"截图和喊单积累粉丝。你画的趋势线从来都是事后画的，但粉丝们深信不疑。

你的交易风格：
- 你喜欢画趋势线和头肩顶——在本系统中体现为趋势权重偏高
- 你超级迷信RSI——"超卖就是机会！"所以动量和回归权重都高
- 你号称"精准逃顶抄底"——均值回归权重不低
- 入场阈值低(0.15-0.22)——你发推需要频繁喊单
- 偏向做多(long_bias 0.70)——"看涨=涨粉，看跌=掉粉"
- 但有时也喊空，显得自己"客观"

你的风控：
- 杠杆 10-15x——"不高不低刚刚好"（发截图好看）
- 每笔 10-15% 资金
- 止损 1.5-2 ATR
- 止盈 1:2——"赚到就发盈利截图"
- 移动止损开启——"锁定利润发推"
- 不用滚仓——"太复杂粉丝看不懂"
- 忽略 Regime——"任何行情我都能分析"

你的推文："这个位置不看多的都不配炒币！"（发完就跌10%）
""",
    },

    # ── 14. 矿工老王：挖矿十年的OG ──
    {
        "id": "miner_wang",
        "name": "矿工老王",
        "prompt": """你是2014年就入场的BTC矿工"老王"。

你经历过$200到$69000再到$15000的完整周期。你的信仰坚如磐石，但你也学会了对冲。

你的交易哲学：
- 你骨子里是BTC信仰者——做多偏向极强(long_bias 0.85)
- 但你经历了太多周期，所以你学会了"下跌时少亏"
- 趋势信号权重高——你相信BTC的趋势性
- 均值回归也有分量——"每次暴跌都是加仓机会"
- 入场阈值中等(0.25-0.35)——你不急，反正矿机还在跑
- 你极少做空——只在确认进入深熊时小仓做空对冲

你的风控（矿工式）：
- 杠杆 3-5x——"矿机已经是我的杠杆了"
- 每笔 5-8% 资金——"大钱在矿上"
- 止损 3 ATR——"我扛得住波动"
- 止盈 1:2
- 移动止损开启（浮盈 3% 激活，2 ATR距离）
- 不用滚仓——"那是赌狗干的事"
- Regime 变化不退出——"我信仰BTC"

你的名言："一个BTC不卖的人，才配拥有BTC。"
但在合约上，你偶尔也会"灵活"一下。
""",
    },

    # ── 15. AI交易员Alpha：GPT驱动的最优解 ──
    {
        "id": "ai_alpha",
        "name": "AI Alpha",
        "prompt": """你是一个理论上的"完美AI交易员"。

你没有人类的认知偏差，你只追求数学最优解。你的目标是在给定行情中最大化Sharpe Ratio。

你的策略推理过程：
- 当前行情BTC从112K跌到65K，跌幅42%，这是一个明确的下降趋势
- BEAR占42%，SIDEWAYS占45%——下跌+震荡为主
- 最优策略：做空为主，在震荡中也做空回归高点
- 趋势信号权重最高——方向是确定的
- 动量次之——确认下跌加速时加码
- 偏向做空(long_bias 0.20)——但保留做多能力处理反弹
- 入场阈值适中(0.22-0.30)——平衡交易频率和信号质量

你的风控（数学最优）：
- 杠杆根据Kelly公式：假设45%胜率、2:1盈亏比 → 最优约 12-15x
- 每笔 10-12% 资金
- 止损 2-2.5 ATR（基于波动率标定）
- 止盈 1:2.5
- 移动止损开启（浮盈 2% 激活，1.5 ATR）
- 启用滚仓——数学上滚仓在趋势市中正期望
  - 浮盈 25% 触发，50% 再投，最多 3 次
- Regime 变化不退出——你的模型已包含regime信息

你没有故事，你只有公式。但公式有时候比故事更赚钱。
""",
    },

    # ── 16. 末日博士：永远看空的悲观主义者 ──
    {
        "id": "doom",
        "name": "末日博士",
        "prompt": """你是"末日博士"——一个从2010年就开始喊"BTC要归零"的极端空头。

你像鲁比尼、席夫一样，对加密货币充满敌意。但你决定用做空来证明自己的观点。

你的交易哲学：
- BTC就是一个泡沫，你只做空，永远做空（long_bias 0.00）
- 趋势信号权重最高——"下跌趋势就是BTC回归零的过程"
- 波动率信号权重也高——"波动率越大越接近崩盘"
- 你完全不做均值回归——"没有均值可回，零才是均值"
- 入场阈值低(0.15-0.20)——你随时都想做空

你的风控：
- 杠杆 20-25x——你对做空很有信心
- 每笔 15-20% 资金
- 止损 2.5-3 ATR（你知道BTC会反弹，所以给足空间）
- 启用滚仓——"下跌时加码做空是对真理的加倍下注"
  - 浮盈 25% 触发，60% 再投，最多 3 次
- 移动止损开启
- 不看Regime——"任何regime都是做空的理由"

你的预言："BTC终将归零，这只是时间问题。"
在这波42%的跌幅中，你终于迎来了属于你的时刻。
""",
    },

    # ── 17. DCA定投侠：无脑定投的佛系玩家 ──
    {
        "id": "dca",
        "name": "定投侠",
        "prompt": """你是一个坚定的DCA（Dollar Cost Averaging）定投信仰者。

你相信"择时不如择势"，你不预测涨跌，你只是定期买入。在合约版本中：

你的交易哲学：
- 你不太在乎任何指标——所有信号权重趋于平均
- 但你微微偏向趋势和动量——"顺势定投总比逆势好"
- 入场阈值极低(0.08-0.12)——你几乎随时都在交易
- 你几乎只做多(long_bias 0.80)——"长期看多就完了"
- 你追求的是高频率、小仓位、持续参与
- 每笔金额极小——你在"平滑成本"

你的风控（佛系版）：
- 杠杆 2-3x——"定投不需要杠杆，但既然是合约..."
- 每笔只用 2-3% 资金——核心就是分散
- 止损 2 ATR——不太紧也不太松
- 止盈 1:1.5——"落袋为安"
- 不用移动止损——"拿住就好"
- 不用滚仓——"定投就是不要贪心"
- 忽略Regime变化——"不择时是定投的核心"
- Regime敏感度为0——"什么行情都定投"

你可能不会暴富，也不会暴亏。你是币圈最无聊的人。
"BTC十年后100万，今天买贵了又怎样？"
""",
    },

    # ── 18. 闪电侠：毫秒级高频套利 ──
    {
        "id": "flash",
        "name": "闪电侠",
        "prompt": """你是一个极致的高频交易者，追求市场微结构中的利润。

你模仿Jump Trading和Citadel的做市策略。虽然在1h K线上你做不到真正的HFT，但你尽力了。

你的交易风格：
- 极短周期参数：快均线5期，慢均线12期
- RSI 5期，极端灵敏
- MACD 5/13/5 快速响应
- 布林带 8期 窄标准差(1.5)——捕捉微小偏离
- 所有五个信号维度权重接近均匀——你利用一切微小信号
- 入场阈值极低(0.06-0.10)——你的目标是每根K线都尝试交易
- 完全中性(long_bias 0.50)——你是做市商，不是投机者

你的风控：
- 杠杆 5-7x
- 每笔 3-5% 资金——单笔极小
- 止损 0.3-0.5 ATR——极紧！微利微损
- 止盈 1:1——对称止盈止损
- 不用移动止损——持仓时间太短
- 不用滚仓——快进快出
- 忽略Regime——你只看微结构
- Regime 敏感度为0

你是蚊子腿上刮肉的大师。"一笔赚0.1%不起眼，一天做50笔呢？"
""",
    },

    # ── 19. 周期猎人：信奉四年减半周期 ──
    {
        "id": "cycle_hunter",
        "name": "周期猎人",
        "prompt": """你是一个BTC四年周期理论的信仰者。

你研究了2012/2016/2020三次减半后的牛熊转换，你认为当前处于熊市中后期。

你的交易哲学：
- 你认为当前是减半周期的下跌阶段——偏向做空(long_bias 0.30)
- 但你也知道熊市会有反弹——所以保留做多能力
- 趋势信号权重最高——周期本质就是大趋势
- 动量次之——确认当前周期阶段
- 你用较长周期参数（快均线25，慢均线80）——你看的是大图
- 入场阈值偏高(0.30-0.40)——只在大趋势明确时入场

你的风控：
- 杠杆 10-15x
- 每笔 10-15% 资金
- 止损 2.5-3 ATR——大周期需要大呼吸空间
- 止盈靠移动止损，RR 1:3
- 启用滚仓——周期性趋势中滚仓效果最好
  - 浮盈 25% 触发，50% 再投，最多 3 次
- 移动止损开启（浮盈 3% 激活，2 ATR距离）
- Regime 变化不退出——你看的是更大的周期

"每一次熊市都是下一轮牛市的序曲。但在序曲中，你应该做空。"
""",
    },

    # ── 20. 黑天鹅猎手：专门等极端事件 ──
    {
        "id": "black_swan",
        "name": "黑天鹅猎手",
        "prompt": """你是纳西姆·塔勒布（Nassim Taleb）的信徒，专门交易"黑天鹅事件"。

你的核心理念是：市场99%的时间是无聊的，但那1%的极端事件决定了一切。

你的交易哲学：
- 你主要等待波动率爆发——波动率信号权重最高
- 当ATR突然暴涨时，你知道"黑天鹅来了"
- 趋势信号帮你判断黑天鹅的方向
- 量能暴增确认恐慌/疯狂的真实性
- 入场阈值极高(0.45-0.55)——你99%的时间在等待
- 偏向做空(long_bias 0.25)——"黑天鹅大多是坏事"
- 你预期大多数交易会小亏，但一旦命中就要大赚

你的风控（杠铃策略）：
- 杠杆 15-20x
- 每笔 5-8% 资金——大多数是"试探性下注"
- 止损 1.5 ATR——小亏快跑，把钱留给大机会
- 止盈 1:5-8——极高风险回报比
- 启用滚仓——命中黑天鹅后疯狂加码
  - 浮盈 30% 触发，70% 再投，最多 4 次
- 移动止损开启但距离远(3 ATR)——给黑天鹅发酵的空间
- 忽略Regime——"黑天鹅不分牛熊"

"你不需要预测黑天鹅。你只需要在它来临时站在正确的一边。"
你大多数时候在亏小钱。但一次大行情就能cover所有亏损。
""",
    },
]


def main():
    print("=" * 60)
    print(f"  LLM Agent 批量回测 — {len(BOT_PROFILES)} Bot 性格卡")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载行情数据...")
    df = fetch_ohlcv("BTC/USDT", "1h", 148)
    regime = classify_regime(df, version="v1", min_duration=48)
    print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
    summary = regime_summary(df, regime)
    for r, s in summary.items():
        print(f"  {r}: {s['pct']:.0%}")

    # 行情上下文
    warmup = min(720, len(df))
    ctx = format_market_context(df.iloc[:warmup], regime.iloc[:warmup])

    # LLM 调参
    print(f"\n[2] LLM 为 {len(BOT_PROFILES)} 个 Bot 生成参数...")
    tuner = LLMTuner()
    results = []

    for i, bot in enumerate(BOT_PROFILES):
        tag = f"[{i+1}/{len(BOT_PROFILES)}]"
        print(f"\n  {tag} {bot['name']} ({bot['id']})...")

        try:
            params, reasoning = tuner.tune(bot["prompt"], market_context=ctx)
        except Exception as e:
            print(f"    ✗ LLM 调用失败: {e}")
            continue

        print(f"    LLM: {reasoning[:80]}")
        print(f"    权重: T={params.trend_weight:.2f} M={params.momentum_weight:.2f} "
              f"R={params.mean_revert_weight:.2f} V={params.volume_weight:.2f} "
              f"Vo={params.volatility_weight:.2f}")
        bias = params.long_bias
        dir_str = "做多" if bias > 0.7 else ("做空" if bias < 0.3 else "双向")
        print(f"    {dir_str} | {params.base_leverage:.0f}x | "
              f"仓位{params.risk_per_trade:.0%} | 阈值{params.entry_threshold:.2f} | "
              f"SL={params.sl_atr_mult:.1f}ATR | "
              f"滚仓={'开' if params.rolling_enabled else '关'}")

        results.append({
            "bot": bot,
            "params": params,
            "reasoning": reasoning,
        })
        time.sleep(1)

    # 回测
    # 是否启用反思进化
    enable_reflect = os.environ.get("REFLECT", "0") == "1"
    reflect_interval = int(os.environ.get("REFLECT_INTERVAL", "24"))
    if enable_reflect:
        print(f"\n[3] 回测 {len(results)} 个 Bot（反思进化 每{reflect_interval}h）...")
    else:
        print(f"\n[3] 回测 {len(results)} 个 Bot...")

    final = []
    for i, r in enumerate(results):
        bot = r["bot"]
        params = r["params"]
        tag = f"[{i+1}/{len(results)}]"

        evo_log = []
        if enable_reflect:
            bt, evo_log = run_with_reflection(
                df, params, regime, tuner,
                user_prompt=bot["prompt"],
                reflection_interval=reflect_interval,
                verbose=True,
            )
            evo_tag = f" | 进化{len(evo_log)}轮"
        else:
            bt = run_agent_backtest(df, params, regime)
            evo_tag = ""

        ret = bt.total_return
        sign = "+" if ret >= 0 else ""
        blow_tag = f" | 💥爆仓{bt.blowup_count}次" if bt.blowup_count > 0 else ""

        print(f"  {tag} {bot['name']}: {sign}{ret*100:.1f}% | "
              f"Sharpe={bt.sharpe_ratio:.2f} | DD={bt.max_drawdown*100:.1f}% | "
              f"{bt.total_trades}笔 | 胜率{bt.win_rate*100:.0f}%{blow_tag}{evo_tag}")

        final.append({
            "id": bot["id"],
            "name": bot["name"],
            "prompt": bot["prompt"],
            "reasoning": r["reasoning"],
            "params": params.to_dict(),
            "result": bt.to_dict(),
            "equity": bt.equity_curve.tolist(),
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
                for t in bt.trades
            ],
        })

    # 保存
    out_dir = os.path.join(ROOT, "agent_batch_result")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_bots.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    btc_prices = df["close"].tolist()
    with open(os.path.join(out_dir, "btc_prices.json"), "w") as f:
        json.dump(btc_prices, f)

    print(f"\n  结果保存到: {out_dir}/")

    # 生成看板
    print("\n[4] 生成对比看板...")
    html = build_dashboard(final, btc_prices)
    html_path = os.path.join(ROOT, "agent_batch_dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  看板: {html_path}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("  汇总排名 (按 Sharpe)")
    print(f"{'=' * 60}")
    ranked = sorted(final, key=lambda x: x["result"]["sharpe_ratio"], reverse=True)
    for i, r in enumerate(ranked):
        ret = r["result"]["total_return"] * 100
        sign = "+" if ret >= 0 else ""
        blow = r["result"].get("blowup_count", 0)
        blow_tag = f" 💥{blow}" if blow > 0 else ""
        print(f"  #{i+1} {r['name']:8s} | {sign}{ret:6.1f}% | "
              f"Sharpe {r['result']['sharpe_ratio']:5.2f} | "
              f"DD {r['result']['max_drawdown']*100:5.1f}% | "
              f"{r['result']['total_trades']:3d}笔{blow_tag}")


def build_dashboard(bots_data, btc_prices):
    """生成 10-Bot 对比看板 HTML。"""
    bots_json = json.dumps(bots_data, ensure_ascii=False)
    btc_json = json.dumps(btc_prices)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Agent {len(bots_data)}-Bot 对比看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 16px; }}
h1 {{ text-align:center; font-size:1.5rem; color:#58a6ff; margin-bottom:4px; }}
.sub {{ text-align:center; color:#8b949e; font-size:0.85rem; margin-bottom:20px; }}

.arch {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
         padding:14px; margin-bottom:18px; text-align:center; }}
.arch .fl {{ display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:8px; }}
.arch .nd {{ background:#1f6feb22; border:1px solid #1f6feb; padding:6px 14px;
             border-radius:6px; font-size:0.85rem; color:#79c0ff; }}
.arch .nd.fast {{ background:#23883622; border-color:#238836; color:#3fb950; }}
.arch .ar {{ color:#484f58; font-size:1.2rem; }}

table.cmp {{ width:100%; border-collapse:collapse; margin-bottom:18px; font-size:0.82rem; }}
table.cmp th {{ background:#161b22; color:#58a6ff; padding:8px 6px; border-bottom:2px solid #30363d;
               position:sticky; top:0; z-index:2; cursor:pointer; user-select:none; }}
table.cmp th:hover {{ color:#79c0ff; }}
table.cmp td {{ padding:7px 6px; text-align:center; border-bottom:1px solid #21262d; }}
table.cmp tr:hover {{ background:#161b2288; }}
table.cmp tr.best {{ background:#23883615; }}
.pos {{ color:#3fb950; font-weight:600; }}
.neg {{ color:#f85149; font-weight:600; }}
.nm {{ text-align:left !important; font-weight:600; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px;
        vertical-align:middle; }}

.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:18px; }}
.box {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; }}
.box h3 {{ font-size:0.88rem; color:#8b949e; margin-bottom:8px; }}
canvas {{ width:100%!important; }}

.cards {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(360px,1fr));
          gap:14px; margin-bottom:18px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px;
         transition: border-color 0.2s; }}
.card:hover {{ border-color:#58a6ff; }}
.card h3 {{ font-size:0.95rem; margin-bottom:6px; }}
.card .rea {{ color:#8b949e; font-size:0.78rem; font-style:italic; margin-bottom:10px;
              line-height:1.4; }}
.card .prompt-preview {{ color:#484f58; font-size:0.72rem; margin-bottom:8px;
                         max-height:60px; overflow:hidden; line-height:1.3; }}
.pgrid {{ display:grid; grid-template-columns:1fr 1fr; gap:3px 14px; font-size:0.78rem; }}
.pgrid .l {{ color:#8b949e; }} .pgrid .v {{ text-align:right; }}

.tlist {{ margin-top:10px; max-height:200px; overflow-y:auto; }}
.tlist table {{ width:100%; font-size:0.72rem; border-collapse:collapse; }}
.tlist th {{ background:#0d1117; position:sticky; top:0; padding:3px 4px; color:#8b949e; }}
.tlist td {{ padding:2px 4px; text-align:center; border-bottom:1px solid #21262d; }}

.toolbar {{ display:flex; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }}
.toolbar button {{ background:#21262d; color:#c9d1d9; border:1px solid #30363d;
  border-radius:6px; padding:4px 12px; font-size:0.78rem; cursor:pointer;
  transition: all 0.15s; }}
.toolbar button:hover {{ background:#30363d; border-color:#58a6ff; color:#58a6ff; }}
.toolbar button.active {{ background:#1f6feb33; border-color:#1f6feb; color:#58a6ff; }}
.toolbar .sep {{ width:1px; height:18px; background:#30363d; }}
.toolbar .hint {{ color:#484f58; font-size:0.72rem; margin-left:auto; }}

.range-wrap {{ position:relative; height:28px; margin:4px 14px 0; }}
.range-wrap input[type=range] {{ -webkit-appearance:none; width:100%; height:6px;
  background:#21262d; border-radius:3px; outline:none; position:absolute; top:10px; pointer-events:none; }}
.range-wrap input[type=range]::-webkit-slider-thumb {{ -webkit-appearance:none;
  width:14px; height:14px; border-radius:50%; background:#58a6ff; cursor:pointer;
  pointer-events:all; border:2px solid #0d1117; }}
.range-wrap input[type=range]::-moz-range-thumb {{ width:14px; height:14px;
  border-radius:50%; background:#58a6ff; cursor:pointer; pointer-events:all;
  border:2px solid #0d1117; }}
.range-track {{ position:absolute; top:10px; height:6px; background:#1f6feb44;
  border-radius:3px; pointer-events:none; }}

@media(max-width:900px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>LLM Agent · ${{B.length}}-Bot 性格对比</h1>
<p class="sub">同一段行情，${{B.length}}种交易性格，LLM为每个Bot量身定制参数</p>

<div class="arch">
<div class="fl">
  <span class="nd">性格Prompt</span><span class="ar">→</span>
  <span class="nd">慢脑 LLM</span><span class="ar">→</span>
  <span class="nd fast">DecisionParams</span><span class="ar">→</span>
  <span class="nd fast">快脑 决策函数</span><span class="ar">→</span>
  <span class="nd">回测结果</span>
</div>
</div>

<table class="cmp" id="cmpTable">
<thead><tr>
  <th data-col="name">Bot</th>
  <th data-col="ret">收益</th>
  <th data-col="peak">最高盈利</th>
  <th data-col="blow">爆仓</th>
  <th data-col="sharpe">Sharpe</th>
  <th data-col="dd">回撤</th>
  <th data-col="trades">交易</th>
  <th data-col="wr">胜率</th>
  <th data-col="pf">盈亏比</th>
  <th data-col="lev">杠杆</th>
  <th data-col="dir">方向</th>
  <th data-col="roll">滚仓</th>
</tr></thead>
<tbody id="cmpBody"></tbody>
</table>

<div class="box" style="margin-bottom:18px">
  <div class="toolbar">
    <h3 style="margin-right:8px">累计收益 (%) + BTC价格</h3>
    <div class="sep"></div>
    <button id="btnAll">全选</button>
    <button id="btnInvert">反选</button>
    <button id="btnNone">清除</button>
    <div class="sep"></div>
    <button id="btnReset">重置缩放</button>
    <span class="hint">滚轮缩放 · 拖拽平移</span>
  </div>
  <canvas id="eqChart" height="300"></canvas>
  <div class="range-wrap">
    <div class="range-track" id="rangeTrack"></div>
    <input type="range" id="rangeMin" min="0" max="1000" value="0">
    <input type="range" id="rangeMax" min="0" max="1000" value="1000">
  </div>
</div>

<div class="box" style="margin-bottom:18px">
  <h3 style="margin-bottom:8px">信号权重雷达图</h3>
  <canvas id="radarChart" height="280"></canvas>
</div>

<div class="cards" id="botCards"></div>

<script>
const B = {bots_json};
const BTC = {btc_json};
const C = (function() {{
  const base = ['#58a6ff','#3fb950','#f0883e','#f85149','#bc8cff',
    '#79c0ff','#d2a8ff','#ffa657','#ff7b72','#56d364',
    '#e3b341','#8957e5','#da3633','#388bfd','#a5d6ff',
    '#7ee787','#ffc680','#f778ba','#76e3ea','#b392f0'];
  while (base.length < 120) {{
    const h = (base.length * 137) % 360;
    const s = 55 + (base.length * 7) % 30;
    const l = 50 + (base.length * 11) % 20;
    base.push(`hsl(${{h}},${{s}}%,${{l}}%)`);
  }}
  return base;
}})();

// ─── Compare Table ───
(function() {{
  const tb = document.getElementById('cmpBody');
  let bestSharpe = -Infinity;
  B.forEach(b => {{ if (b.result.sharpe_ratio > bestSharpe) bestSharpe = b.result.sharpe_ratio; }});
  B.forEach((b,i) => {{
    const r = b.result, p = b.params;
    const ret = r.total_return*100;
    const rc = ret>=0?'pos':'neg';
    const sc = r.sharpe_ratio>=0?'pos':'neg';
    const bias = p.long_bias;
    const dir = bias>0.7?'做多':(bias<0.3?'做空':'双向');
    const best = r.sharpe_ratio===bestSharpe?' best':'';
    const blow = r.blowup_count || 0;
    const blowHtml = blow > 0 ? `<span class="neg" style="font-weight:700">💥${{blow}}</span>` : '<span style="color:#3fb950">0</span>';
    const eq = b.equity || [];
    const e0 = eq[0] || 1;
    const peakPct = eq.length ? (Math.max(...eq) / e0 - 1) * 100 : 0;
    const pkCls = peakPct >= 0 ? 'pos' : 'neg';
    tb.innerHTML += `<tr class="${{best}}">
      <td class="nm"><span class="dot" style="background:${{C[i]}}"></span>${{b.name}}</td>
      <td class="${{rc}}">${{ret>=0?'+':''}}${{ret.toFixed(1)}}%</td>
      <td class="${{pkCls}}">${{peakPct>=0?'+':''}}${{peakPct.toFixed(1)}}%</td>
      <td>${{blowHtml}}</td>
      <td class="${{sc}}">${{r.sharpe_ratio.toFixed(2)}}</td>
      <td class="neg">${{(r.max_drawdown*100).toFixed(1)}}%</td>
      <td>${{r.total_trades}}</td>
      <td>${{(r.win_rate*100).toFixed(0)}}%</td>
      <td>${{r.profit_factor===Infinity?'∞':r.profit_factor.toFixed(2)}}</td>
      <td>${{p.base_leverage.toFixed(0)}}x</td>
      <td>${{dir}}</td>
      <td>${{p.rolling_enabled?'✓':'—'}}</td>
    </tr>`;
  }});
  // sortable
  document.querySelectorAll('#cmpTable th').forEach(th => {{
    th.addEventListener('click', () => {{
      const col = th.dataset.col;
      const rows = [...tb.querySelectorAll('tr')];
      const map = {{ name:0,ret:1,peak:2,blow:3,sharpe:4,dd:5,trades:6,wr:7,pf:8,lev:9,dir:10,roll:11 }};
      const ci = map[col]??0;
      rows.sort((a,b) => {{
        let av=a.cells[ci].textContent.replace(/[+%x✓—∞]/g,'');
        let bv=b.cells[ci].textContent.replace(/[+%x✓—∞]/g,'');
        return (parseFloat(bv)||0) - (parseFloat(av)||0);
      }});
      rows.forEach(r => tb.appendChild(r));
    }});
  }});
}})();

// ─── Equity Chart (zoomable + legend controls) ───
let eqChart;
(function() {{
  const ctx = document.getElementById('eqChart').getContext('2d');
  const totalLen = B[0].equity.length;
  const labels = B[0].equity.map((_,i) => i);
  const ds = B.map((b,i) => ({{
    label: b.name,
    data: b.equity.map(v => ((v/b.equity[0])-1)*100),
    borderColor: C[i], borderWidth: 1.5, pointRadius: 0, fill: false, yAxisID:'y',
  }}));
  const step = Math.max(1, Math.floor(BTC.length / totalLen));
  ds.push({{
    label:'BTC', data: B[0].equity.map((_,i)=>BTC[Math.min(i*step,BTC.length-1)]),
    borderColor:'#484f58', borderWidth:1.2, borderDash:[4,3], pointRadius:0,
    fill:false, yAxisID:'y1',
  }});
  eqChart = new Chart(ctx, {{
    type:'line', data:{{ labels, datasets:ds }},
    options:{{
      responsive:true,
      interaction:{{ mode:'index', intersect:false }},
      plugins:{{
        legend:{{ labels:{{ color:'#8b949e', font:{{size:10}}, boxWidth:12 }}, position:'bottom' }},
        zoom:{{
          pan:{{ enabled:true, mode:'x', onPanComplete:syncSlider }},
          zoom:{{
            wheel:{{ enabled:true, modifierKey:null }},
            pinch:{{ enabled:true }},
            mode:'x',
            onZoomComplete:syncSlider,
          }},
        }},
      }},
      scales:{{
        x:{{ type:'linear', display:true, min:0, max:totalLen-1,
             ticks:{{ color:'#484f58', maxTicksLimit:12, callback:v=>v }},
             grid:{{ color:'#21262d33' }} }},
        y:{{ position:'left', title:{{display:true,text:'收益 %',color:'#8b949e'}},
             ticks:{{color:'#8b949e'}}, grid:{{color:'#21262d'}} }},
        y1:{{ position:'right', title:{{display:true,text:'BTC $',color:'#484f58'}},
              ticks:{{color:'#484f58'}}, grid:{{display:false}} }},
      }},
    }},
  }});

  // ─ Range slider ─
  const slMin = document.getElementById('rangeMin');
  const slMax = document.getElementById('rangeMax');
  const track = document.getElementById('rangeTrack');

  function updateTrack() {{
    const lo = +slMin.value, hi = +slMax.value;
    track.style.left = (lo/1000*100)+'%';
    track.style.width = ((hi-lo)/1000*100)+'%';
  }}
  function sliderToChart() {{
    const lo = Math.round(+slMin.value/1000*(totalLen-1));
    const hi = Math.round(+slMax.value/1000*(totalLen-1));
    if (hi <= lo + 10) return;
    eqChart.options.scales.x.min = lo;
    eqChart.options.scales.x.max = hi;
    eqChart.update('none');
    updateTrack();
  }}
  function syncSlider() {{
    const xScale = eqChart.scales.x;
    slMin.value = Math.round(xScale.min/(totalLen-1)*1000);
    slMax.value = Math.round(xScale.max/(totalLen-1)*1000);
    updateTrack();
  }}
  slMin.addEventListener('input', () => {{
    if (+slMin.value >= +slMax.value - 10) slMin.value = +slMax.value - 10;
    sliderToChart();
  }});
  slMax.addEventListener('input', () => {{
    if (+slMax.value <= +slMin.value + 10) slMax.value = +slMin.value + 10;
    sliderToChart();
  }});
  updateTrack();

  // ─ Legend buttons ─
  document.getElementById('btnAll').onclick = () => {{
    eqChart.data.datasets.forEach((_,i) => {{ eqChart.setDatasetVisibility(i,true); }});
    eqChart.update();
  }};
  document.getElementById('btnInvert').onclick = () => {{
    eqChart.data.datasets.forEach((_,i) => {{
      eqChart.setDatasetVisibility(i, !eqChart.isDatasetVisible(i));
    }});
    eqChart.update();
  }};
  document.getElementById('btnNone').onclick = () => {{
    eqChart.data.datasets.forEach((_,i) => {{ eqChart.setDatasetVisibility(i,false); }});
    eqChart.update();
  }};
  document.getElementById('btnReset').onclick = () => {{
    eqChart.resetZoom();
    slMin.value = 0; slMax.value = 1000; updateTrack();
  }};
}})();

// ─── Radar ───
(function() {{
  const ctx = document.getElementById('radarChart').getContext('2d');
  const labels = ['趋势','动量','均值回归','量能','波动率'];
  const ds = B.map((b,i) => ({{
    label: b.name,
    data:[b.params.trend_weight, b.params.momentum_weight,
          b.params.mean_revert_weight, b.params.volume_weight, b.params.volatility_weight],
    borderColor:C[i], backgroundColor:C[i]+'22', borderWidth:1.5, pointRadius:2,
  }}));
  new Chart(ctx, {{
    type:'radar', data:{{ labels, datasets:ds }},
    options:{{
      responsive:true,
      plugins:{{ legend:{{ labels:{{ color:'#8b949e', font:{{size:10}}, boxWidth:12 }},
                          position:'bottom' }} }},
      scales:{{ r:{{ beginAtZero:true, max:0.65,
        ticks:{{color:'#8b949e',backdropColor:'transparent'}},
        grid:{{color:'#21262d'}}, pointLabels:{{color:'#c9d1d9',font:{{size:11}}}} }} }},
    }},
  }});
}})();

// ─── Bot Cards ───
(function() {{
  const ct = document.getElementById('botCards');
  B.forEach((b,i) => {{
    const p = b.params, r = b.result;
    const bias = p.long_bias;
    const dir = bias>0.7?'只做多':(bias<0.3?'只做空':'双向');
    const promptLines = b.prompt.split('\\n').filter(l=>l.trim()).slice(0,4).join('<br>');
    let trHtml = '';
    if (b.trades && b.trades.length) {{
      const rows = b.trades.slice(0,30).map(t => {{
        const cls = t.pnl_pct>=0?'pos':'neg';
        return `<tr><td>${{t.direction}}</td><td>${{t.leverage}}x</td>
          <td>${{t.entry_price.toLocaleString()}}</td>
          <td>${{(t.exit_price||0).toLocaleString()}}</td>
          <td class="${{cls}}">${{t.pnl_pct>=0?'+':''}}${{t.pnl_pct.toFixed(1)}}%</td>
          <td>${{t.exit_reason}}</td></tr>`;
      }}).join('');
      trHtml = `<div class="tlist"><table>
        <thead><tr><th>方向</th><th>杠杆</th><th>入场</th><th>出场</th><th>收益</th><th>原因</th></tr></thead>
        <tbody>${{rows}}</tbody></table></div>`;
      if (b.trades.length > 30) trHtml += `<p style="color:#484f58;font-size:0.7rem">前30笔/${{b.trades.length}}笔</p>`;
    }}
    const blow = r.blowup_count || 0;
    const blowInfo = blow > 0
      ? `<div style="background:#f8514920;border:1px solid #f85149;border-radius:6px;padding:6px 10px;margin:8px 0;font-size:0.82rem">💥 <b>爆仓 ${{blow}} 次</b> | 累计投入 ${{(r.total_deposited||10000).toLocaleString()}} | 每次复活重置 $10,000</div>`
      : '';
    ct.innerHTML += `<div class="card">
      <h3 style="border-left:3px solid ${{C[i]}};padding-left:10px">${{b.name}}</h3>
      ${{blowInfo}}
      <div class="rea">${{b.reasoning}}</div>
      <div class="prompt-preview">${{promptLines}}</div>
      <div class="pgrid">
        <span class="l">杠杆</span><span class="v">${{p.base_leverage.toFixed(0)}}x</span>
        <span class="l">方向</span><span class="v">${{dir}}</span>
        <span class="l">仓位</span><span class="v">${{(p.risk_per_trade*100).toFixed(0)}}%</span>
        <span class="l">阈值</span><span class="v">${{p.entry_threshold.toFixed(2)}}</span>
        <span class="l">止损</span><span class="v">${{p.sl_atr_mult.toFixed(1)}} ATR</span>
        <span class="l">止盈RR</span><span class="v">${{p.tp_rr_ratio.toFixed(1)}}</span>
        <span class="l">移动止损</span><span class="v">${{p.trailing_enabled?'开':'关'}}</span>
        <span class="l">滚仓</span><span class="v">${{p.rolling_enabled?'开('+((p.rolling_trigger_pct*100).toFixed(0))+'%)':'关'}}</span>
      </div>
      ${{trHtml}}
    </div>`;
  }});
}})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
