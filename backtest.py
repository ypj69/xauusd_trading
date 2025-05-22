import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from main import XAUUSDTradingBot, FEATURE_PROMPT, TRADING_PROMPT

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class XAUUSDBacktest(XAUUSDTradingBot):
    def __init__(self, api_key, initial_balance=10000):
        super().__init__(api_key)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, start_date, end_date):
        """运行回测"""
        try:
            self.initialize_mt5()
            print(f"开始回测 {start_date} 到 {end_date}")
            
            # 获取历史数据
            rates = mt5.copy_rates_range(
                "XAUUSD",
                mt5.TIMEFRAME_H1,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                raise Exception("无法获取回测数据")
            
            # 创建完整的DataFrame并计算技术指标
            full_df = pd.DataFrame(rates)
            full_df['time'] = pd.to_datetime(full_df['time'], unit='s')
            full_df.set_index('time', inplace=True)  # 设置时间索引
            full_df = self.calculate_indicators(full_df)
            
            # 准备日线和4小时数据
            daily_df = full_df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum',
                'spread': 'mean',
                'real_volume': 'sum'
            })
            daily_df = self.calculate_indicators(daily_df)
            
            h4_df = full_df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum',
                'spread': 'mean',
                'real_volume': 'sum'
            })
            h4_df = self.calculate_indicators(h4_df)
            
            # 重置索引以便于后续处理
            full_df = full_df.reset_index()
            daily_df = daily_df.reset_index()
            h4_df = h4_df.reset_index()
            
            # 按时间顺序遍历每个小时
            for i in range(len(full_df)-10):  # 留出10根K线作为分析窗口
                current_time = full_df.iloc[i]['time']
                print(f"分析时间点: {current_time}")  # 添加进度显示
                
                # 准备各个时间周期的分析数据
                h1_window = full_df[i:i+10].copy()
                h4_window = h4_df[h4_df['time'] <= current_time].tail(10).copy()
                daily_window = daily_df[daily_df['time'] <= current_time].tail(10).copy()
                
                # 准备市场数据
                market_data = {
                    'D1': self.prepare_data_string(daily_window, 'D1'),
                    'H4': self.prepare_data_string(h4_window, 'H4'),
                    'H1': self.prepare_data_string(h1_window, 'H1')
                }
                
                # 计算技术特征
                features = self.calculate_features(market_data)
                if not features:
                    continue
                
                # 生成交易信号
                signal = self.generate_trading_signal(market_data, features)
                signal_dict = self.parse_trading_signal(signal)
                
                if signal_dict:
                    # 模拟交易执行
                    self.simulate_trade(signal_dict, full_df.iloc[i+1], current_time)
                    print(f"信号详情: {signal_dict}")  # 添加信号详情输出
                
                # 每处理100个时间点显示一次进度
                if i % 100 == 0:
                    print(f"已处理 {i}/{len(full_df)} 个时间点")
                    
                # 更新权益曲线
                self.update_equity(full_df.iloc[i+1])
            
            # 分析回测结果
            self.analyze_results()
            
        except Exception as e:
            print(f"回测错误: {str(e)}")
            raise  # 添加这行以显示完整的错误堆栈
        finally:
            mt5.shutdown()
    
    def simulate_trade(self, signal_dict, next_bar, timestamp):
        """模拟交易执行"""
        try:
            action = signal_dict['action']
            entry_price = float(signal_dict['entry_price'])
            stop_loss = float(signal_dict['stop_loss'])
            take_profit = float(signal_dict['take_profit'])
            volume = float(signal_dict['volume'])
            
            # 计算下一根K线是否触发入场
            if (action == "BUY" and next_bar['low'] <= entry_price <= next_bar['high']) or \
               (action == "SELL" and next_bar['low'] <= entry_price <= next_bar['high']):
                
                # 记录交易
                trade = {
                    'timestamp': timestamp,
                    'action': action,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'volume': volume,
                    'status': 'open'
                }
                self.trades.append(trade)
                print(f"开仓: {action} @ {entry_price}")
        
        except Exception as e:
            print(f"模拟交易错误: {str(e)}")
    
    def update_equity(self, current_bar):
        """更新权益曲线"""
        for trade in self.trades:
            if trade['status'] == 'open':
                # 计算当前收益
                if trade['action'] == 'BUY':
                    pnl = (current_bar['close'] - trade['entry_price']) * trade['volume'] * 100
                else:  # SELL
                    pnl = (trade['entry_price'] - current_bar['close']) * trade['volume'] * 100
                
                # 检查是否触及止损或止盈
                if trade['action'] == 'BUY':
                    if current_bar['low'] <= trade['stop_loss']:
                        pnl = (trade['stop_loss'] - trade['entry_price']) * trade['volume'] * 100
                        trade['status'] = 'closed'
                        trade['pnl'] = pnl
                    elif current_bar['high'] >= trade['take_profit']:
                        pnl = (trade['take_profit'] - trade['entry_price']) * trade['volume'] * 100
                        trade['status'] = 'closed'
                        trade['pnl'] = pnl
                else:  # SELL
                    if current_bar['high'] >= trade['stop_loss']:
                        pnl = (trade['entry_price'] - trade['stop_loss']) * trade['volume'] * 100
                        trade['status'] = 'closed'
                        trade['pnl'] = pnl
                    elif current_bar['low'] <= trade['take_profit']:
                        pnl = (trade['entry_price'] - trade['take_profit']) * trade['volume'] * 100
                        trade['status'] = 'closed'
                        trade['pnl'] = pnl
                
                self.current_balance += pnl
        
        self.equity_curve.append(self.current_balance)
    
    def analyze_results(self):
        """详细分析回测结果"""
        if len(self.trades) == 0:
            print("回测期间没有产生交易")
            return
            
        # 基础统计
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.get('status') == 'closed']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        # 计算关键指标
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / \
                          sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        # 计算回撤
        max_drawdown = 0
        peak_balance = self.initial_balance
        for balance in self.equity_curve:
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算月度收益
        monthly_returns = []
        for trade in closed_trades:
            month = pd.to_datetime(trade['timestamp']).strftime('%Y-%m')
            monthly_returns.append((month, trade['pnl']))
        
        monthly_pnl = pd.DataFrame(monthly_returns, columns=['month', 'pnl'])\
                       .groupby('month')['pnl'].sum()
        
        # 输出详细报告
        print("\n====== 回测详细报告 ======")
        print(f"\n基础统计:")
        print(f"总交易次数: {total_trades}")
        print(f"完成交易: {len(closed_trades)}")
        print(f"盈利交易: {len(winning_trades)}")
        print(f"亏损交易: {len(losing_trades)}")
        
        print(f"\n盈利能力分析:")
        print(f"胜率: {win_rate:.2f}%")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "盈亏比: ∞")
        print(f"盈利因子: {profit_factor:.2f}")
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        print(f"\n收益统计:")
        print(f"起始资金: ${self.initial_balance:.2f}")
        print(f"最终权益: ${self.current_balance:.2f}")
        print(f"总收益率: {((self.current_balance - self.initial_balance) / self.initial_balance * 100):.2f}%")
        print(f"月均收益率: {(monthly_pnl.mean()):.2f}$")
        
        print("\n月度收益详情:")
        for month, pnl in monthly_pnl.items():
            print(f"{month}: {'+'if pnl > 0 else ''}{pnl:.2f}$")
        
        # 绘制分析图表
        self.plot_analysis_charts()

    def plot_analysis_charts(self):
        """绘制多个分析图表"""
        plt.figure(figsize=(15, 10))
        
        # 1. 权益曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.equity_curve)
        plt.title('账户权益曲线')
        plt.xlabel('时间')        
        plt.ylabel('权益')
        plt.grid(True)
        
        # 2. 盈亏分布
        plt.subplot(2, 2, 2)
        pnls = [t.get('pnl', 0) for t in self.trades if t.get('status') == 'closed']
        plt.hist(pnls, bins=50)
        plt.title('交易盈亏分布')
        plt.grid(True)
        
        # 3. 每月盈亏
        plt.subplot(2, 2, 3)
        monthly_data = []
        for trade in self.trades:
            if trade.get('status') == 'closed':
                month = pd.to_datetime(trade['timestamp']).strftime('%Y-%m')
                monthly_data.append((month, trade['pnl']))
        
        monthly_pnl = pd.DataFrame(monthly_data, columns=['month', 'pnl'])\
                       .groupby('month')['pnl'].sum()
        monthly_pnl.plot(kind='bar')
        plt.title('月度盈亏')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 4. 连续盈亏
        plt.subplot(2, 2, 4)
        streak = 0
        streaks = []
        for trade in self.trades:
            if trade.get('status') == 'closed':
                if trade['pnl'] > 0:
                    if streak >= 0:
                        streak += 1
                    else:
                        streaks.append(streak)
                        streak = 1
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        streaks.append(streak)
                        streak = -1
        if streak != 0:
            streaks.append(streak)
            
        plt.hist(streaks, bins=range(min(streaks)-1, max(streaks)+2, 1))
        plt.title('连续盈亏次数分布')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_trading_signal(self, market_data, technical_features):
        """Generate trading signal using LLM"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "You are an institutional XAUUSD trader focusing on 1-hour timeframe trading."},
                    {"role": "user", "content": f"""Market Context:
                    Daily Structure: {market_data['D1']}
                    4H Order Blocks: {market_data['H4']}
                    1H Analysis: {market_data['H1']}
                    
                    Technical Analysis:
                    {technical_features}
                    
                    Please generate a detailed trading signal with:
                    1. Entry price (must be a specific number)
                    2. Stop loss level (must be a specific number)
                    3. Take profit level (must be a specific number)
                    4. Position size (0.1-1 lots)
                    
                    Format your response exactly like this:
                    信号: [买入/卖出]
                    入场: [价格]
                    止损: [价格]
                    止盈: [价格]
                    仓位: [数量]
                    
                    If no valid setup exists, respond with: "不交易" """}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None

def main():
    # 使用示例
    api_key = "b72284b1cf294a74bc5d305a8d27eb1e.YmSfvU2KHtfjjkLT"
    backtest = XAUUSDBacktest(api_key)
    
    # 设置回测时间范围
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 5)

    # 运行回测
    backtest.run_backtest(start_date, end_date)

if __name__ == "__main__":
    main()
