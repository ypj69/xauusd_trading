# MT5账户配置
MT5_CONFIG = {
    'login': 52323915,
    'password': "a7PeK&rgZsN3rZ",
    'server': "ICMarketsSC-Demo",
    'max_reconnect_attempts': 10
}

# API配置
API_CONFIG = {
    'zhipu_api_key': "b72284b1cf294a74bc5d305a8d27eb1e.YmSfvU2KHtfjjkLT"
}

# 增强型技术分析特征计算提示词
FEATURE_PROMPT = """您是一位专业的XAUUSD（黄金）技术分析交易员。
请重点分析1小时时间周期的交易机会，同时参考更高时间周期作为趋势确认。

日线数据（用于趋势确认）:
{daily_data}

4小时数据（用于结构确认）:
{h4_data}

1小时数据（主要交易周期）:
{h1_data}

请提供以下分析：
1. 日线和4小时的主导趋势方向
2. 1小时图的市场结构分析
3. 1小时图的关键支撑/阻力位
4. 1小时图的供需区域和订单块
5. 1小时图的技术指标分析（RSI、EMA、WPR等）
6. 1小时图的成交量特征
7. 具体的交易机会和入场点

注意：所有交易信号必须在1小时图上产生，高时间周期仅作为趋势过滤。
"""

# 交易信号生成提示词
TRADING_PROMPT = """您是一位专注于1小时级别XAUUSD交易的机构交易员。请基于以下数据生成交易信号：

**市场环境**
日线趋势确认: {daily_data}
4小时结构确认: {h4_data}
1小时主要分析: {h1_data}

**技术分析汇总**
{technical_features}

**1小时图交易规则**
1. 趋势确认:
   - 日线定义主趋势方向
   - 4小时确认市场结构
   - 1小时产生具体信号
   
2. 1小时图入场条件:
   - 与主趋势方向一致
   - 在关键结构位置
   - RSI无超买超卖
   - EMA5/10/20提供动态支撑或阻力
   - WPR指标确认超买超卖区域(-20至-80区间)
   - 成交量确认

3. 风险管理:
   - 止损设置在1小时结构位之外
   - 每笔风险1%账户资金
   - 最小收益比1:2
   - 点差<35点

请生成明确的交易指令：
信号: [买入/卖出]
入场: [精确价格]
止损: [价格]
止盈: [价格]
仓位: [0.1-1手]

如不满足交易条件，请回复"不交易"并说明原因。
"""

# 合并后的统一市场分析提示词
MARKET_ANALYSIS_PROMPT = """您是一位专业的XAUUSD（黄金）技术分析交易员。
请分两部分完成分析：

第一部分 - 技术分析：
{feature_prompt}

第二部分 - 交易信号：
{trading_prompt}

请按以下格式返回结果：
---ANALYSIS_START---
[技术分析结果]
---ANALYSIS_END---
---SIGNAL_START---
[交易信号]
---SIGNAL_END---
"""

# 交易参数配置
TRADING_CONFIG = {
    'symbol': 'XAUUSD',
    'default_volume': 0.01,
    'max_spread': 35,
    'risk_percent': 1,
    'min_reward_ratio': 2,
    'magic_number': 234000
}

# 技术指标配置
INDICATOR_CONFIG = {
    'rsi_period': 14,
    'wpr_period': 14,
    'ema_periods': [5, 10, 20],
    'atr_period': 14,
    'wpr_oversold': -80,
    'wpr_overbought': -20
}
