# Trade Bot - 策略发现工厂

AI驱动的加密货币交易策略发现引擎。从真实行情数据中自动生成100个多样化的Bot策略配置。

## 快速开始

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
python main.py --symbols BTC/USDT --bots 100
```

## 产出

`profiles/` 目录下的JSON文件，每个描述一个bot的完整交易策略，可被openclaw等平台直接消费。
