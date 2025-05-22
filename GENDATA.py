import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os

# 连接MT5
def connect_to_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    return True

# 获取历史数据 - 专门针对XAUUSD优化
def get_historical_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, 
                        start_date="2020-01-01", end_date="2025-05-01"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 分批获取数据，避免MT5限制
    all_data = pd.DataFrame()
    current_start = start
    
    while current_start < end:
        current_end = current_start + timedelta(days=365)
        if current_end > end:
            current_end = end
            
        rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
        if rates is None or len(rates) == 0:
            current_start = current_end
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        all_data = pd.concat([all_data, df])
        current_start = current_end
    
    # 黄金数据特殊处理：调整点值和pip值
    if symbol == "XAUUSD":
        df['point'] = 0.01  # 黄金的点值是0.01
        df['pip'] = 0.1     # 黄金的pip是0.1
    
    return all_data

# 数据处理函数
def process_data(df, symbol="XAUUSD"):
    # 基本收益率
    df['returns'] = df['close'].pct_change()
    
    # 移动平均线 - 针对黄金波动特性优化
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # 布林带 - 黄金波动较大，调整参数
    df['MA20_std'] = df['close'].rolling(window=20).std()
    df['UpperBand'] = df['MA20'] + (df['MA20_std'] * 2.5)  # 增加带宽以适应黄金波动
    df['LowerBand'] = df['MA20'] - (df['MA20_std'] * 2.5)
    
    # RSI指标
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # ATR指标 - 衡量黄金市场波动性
    df['HL'] = df['high'] - df['low']
    df['HC'] = np.abs(df['high'] - df['close'].shift())
    df['LC'] = np.abs(df['low'] - df['close'].shift())
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 黄金价格水平分类 - 帮助模型识别关键价格区域
    if symbol == "XAUUSD":
        df['price_level'] = pd.cut(df['close'], bins=[0, 1200, 1400, 1600, 1800, 2000, 2200, 3000, np.inf],
                                   labels=[1, 2, 3, 4, 5, 6, 7, 8])
        df['price_level'] = df['price_level'].astype(float)
    
    # 移除NaN值
    df.dropna(inplace=True)
    
    return df

# 主函数：获取并处理数据
def prepare_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, save_raw=True):
    if not connect_to_mt5():
        return None, None, None, None
    
    # 创建保存目录
    data_dir = f"{symbol}_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 获取训练集数据
    print("正在获取训练集数据...")
    train_data = get_historical_data(symbol, timeframe, "2020-01-01", "2024-01-01")
    print(f"训练集数据大小: {len(train_data)}")
    
    # 获取测试集数据
    print("正在获取测试集数据...")
    test_data = get_historical_data(symbol, timeframe, "2024-01-02", "2025-05-01")
    print(f"测试集数据大小: {len(test_data)}")
    
    # 断开MT5连接
    mt5.shutdown()
    
    # 保存原始数据（DataFrame格式）
    if save_raw:
        print("正在保存原始数据...")
        train_data.to_csv(f"{data_dir}/{symbol}_train_raw.csv", index=False)
        test_data.to_csv(f"{data_dir}/{symbol}_test_raw.csv", index=False)
        print("原始数据已保存")
    
    # 处理数据
    print("正在处理数据...")
    train_features = process_data(train_data.copy(), symbol)  # 使用.copy()避免修改原始数据
    test_features = process_data(test_data.copy(), symbol)
    print("数据处理完成")
    
    # 保存处理后的数据
    print("正在保存处理后的数据...")
    train_features.to_csv(f"{data_dir}/{symbol}_train_processed.csv", index=False)
    test_features.to_csv(f"{data_dir}/{symbol}_test_processed.csv", index=False)
    print("处理后的数据已保存")
    
    return train_features, test_features

# 加载数据
def load_data(symbol="XAUUSD", use_processed=True):
    data_dir = f"{symbol}_data"
    
    if use_processed:
        try:
            train_data = pd.read_csv(f"{data_dir}/{symbol}_train_processed.csv")
            test_data = pd.read_csv(f"{data_dir}/{symbol}_test_processed.csv")
            return train_data, test_data
        except FileNotFoundError:
            print("处理后的数据不存在，尝试加载原始数据...")
    
    # 加载原始数据并处理
    try:
        train_data = pd.read_csv(f"{data_dir}/{symbol}_train_raw.csv")
        test_data = pd.read_csv(f"{data_dir}/{symbol}_test_raw.csv")
        
        print("正在处理数据...")
        train_features = process_data(train_data.copy(), symbol)
        test_features = process_data(test_data.copy(), symbol)
        print("数据处理完成")
        
        # 保存处理后的数据
        print("正在保存处理后的数据...")
        train_features.to_csv(f"{data_dir}/{symbol}_train_processed.csv", index=False)
        test_features.to_csv(f"{data_dir}/{symbol}_test_processed.csv", index=False)
        print("处理后的数据已保存")
        
        return train_features, test_features
    except FileNotFoundError:
        print("原始数据也不存在，请先运行prepare_data函数获取数据")
        return None, None

# 可视化黄金数据
def visualize_data(df, symbol="XAUUSD"):
    plt.figure(figsize=(14, 10))
    
    # 绘制价格和移动平均线
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['close'], label='Close Price')
    plt.plot(df['time'], df['MA5'], label='MA5')
    plt.plot(df['time'], df['MA20'], label='MA20')
    plt.plot(df['time'], df['MA50'], label='MA50')
    plt.title(f'{symbol} Price with Moving Averages')
    plt.legend()
    plt.grid(True)
    
    # 绘制技术指标
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['RSI'], label='RSI')
    plt.plot(df['time'], df['MACD'], label='MACD')
    plt.plot(df['time'], df['Signal'], label='Signal')
    plt.axhline(y=70, color='r', linestyle='-')
    plt.axhline(y=30, color='g', linestyle='-')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title(f'{symbol} Technical Indicators')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 示例用法
if __name__ == "__main__":
    symbol = "XAUUSD"
    train_data, test_data = prepare_data(symbol, save_raw=True)
    
    if train_data is not None and test_data is not None:
        visualize_data(train_data, symbol)