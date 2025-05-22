import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zhipuai import ZhipuAI
import json
import time
import logging
import os
from config import (
    MT5_CONFIG, 
    TRADING_CONFIG,
    INDICATOR_CONFIG,
    FEATURE_PROMPT, 
    TRADING_PROMPT, 
    MARKET_ANALYSIS_PROMPT
)

class AuditLogger:
    def __init__(self, log_dir="audit_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 设置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/trading_audit.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_trade_decision(self, decision_data):
        """记录交易决策"""
        self.logger.info(f"Trade Decision: {json.dumps(decision_data, ensure_ascii=False, indent=2)}")
    
    def log_parameter_change(self, param_type, old_value, new_value):
        """记录参数变更"""
        self.logger.info(f"Parameter Change - {param_type}: {old_value} -> {new_value}")
    
    def log_risk_check(self, risk_data):
        """记录风险检查"""
        self.logger.info(f"Risk Check: {json.dumps(risk_data, ensure_ascii=False, indent=2)}")

class XAUUSDTradingBot:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)
        self.mt5_login = MT5_CONFIG['login']
        self.mt5_password = MT5_CONFIG['password']
        self.mt5_server = MT5_CONFIG['server']
        self.max_reconnect_attempts = MT5_CONFIG['max_reconnect_attempts']
        
        self.timeframes = {
            'D1': mt5.TIMEFRAME_D1,
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1
        }
        
        self.audit_logger = AuditLogger()
        self.mt5_connected = False

    def check_mt5_connection(self):
        """检查MT5连接状态"""
        return mt5.terminal_info() is not None and mt5.account_info() is not None
    
    def initialize_mt5(self):
        """初始化并登录MT5账户，失败时自动重试"""
        attempt = 0
        while attempt < self.max_reconnect_attempts:
            try:
                if not mt5.initialize():
                    print(f"MT5初始化失败，第{attempt + 1}次重试...")
                    time.sleep(30)
                    attempt += 1
                    continue
                
                # 登录MT5账户
                if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                    print(f"MT5登录失败，错误代码: {mt5.last_error()}，第{attempt + 1}次重试...")
                    mt5.shutdown()
                    time.sleep(30)
                    attempt += 1
                    continue
                
                self.mt5_connected = True
                print("MT5登录成功!")
                return True
                
            except Exception as e:
                print(f"MT5连接错误: {str(e)}，第{attempt + 1}次重试...")
                if mt5.initialized():
                    mt5.shutdown()
                time.sleep(30)
                attempt += 1
        
        raise Exception("MT5连接失败，已达到最大重试次数")
    
    def ensure_mt5_connection(self):
        """确保MT5连接状态，如果断开则尝试重连"""
        if not self.check_mt5_connection():
            print("MT5连接已断开，尝试重新连接...")
            self.mt5_connected = False
            try:
                mt5.shutdown()
            except:
                pass
            return self.initialize_mt5()
        return True

    def execute_trade(self, signal_dict):
        """执行交易指令"""
        try:
            # 记录交易决策
            self.audit_logger.log_trade_decision({
                'type': 'trade_execution',
                'signal': signal_dict,
                'timestamp': datetime.now().isoformat()
            })
            
            if not self.ensure_mt5_connection():
                return False
                
            if not signal_dict or 'action' not in signal_dict:
                return False
                
            symbol = "XAUUSD"
            action = signal_dict['action']
            volume = float(signal_dict.get('volume', 0.01))
            price = float(signal_dict.get('entry_price', 0))
            sl = float(signal_dict.get('stop_loss', 0))
            tp = float(signal_dict.get('take_profit', 0))
            
            # 获取当前市价
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print("无法获取交易品种信息")
                return False
                
            current_price = symbol_info.bid if action == "SELL" else symbol_info.ask
            
            # 验证限价单价格是否合理
            price_diff = abs(price - current_price)
            max_deviation = symbol_info.point * 1000  # 允许最大偏差1000个点
            
            if price_diff > max_deviation:
                print(f"入场价格 {price} 与当前市价 {current_price} 偏差过大")
                return False
                
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY_LIMIT if action == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 35,
                "magic": 234000,
                "comment": "python trading bot limit order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.audit_logger.log_trade_decision({
                    'type': 'trade_failed',
                    'error_code': result.retcode,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"交易执行失败，错误代码: {result.retcode}")
                return False
            
            self.audit_logger.log_trade_decision({
                'type': 'trade_success',
                'ticket': result.order,
                'timestamp': datetime.now().isoformat()
            })
            print(f"交易执行成功! Ticket: {result.order}")
            return True
            
        except Exception as e:
            self.audit_logger.log_trade_decision({
                'type': 'trade_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print(f"交易执行错误: {str(e)}")
            self.mt5_connected = False
            return False

    def parse_trading_signal(self, signal_text):
        """解析AI生成的交易信号为可执行的交易指令"""
        try:
            # 简单的信号解析示例
            signal_dict = {}
            
            if "买入" in signal_text.upper() or "BUY" in signal_text.upper():
                signal_dict['action'] = "BUY"
            elif "卖出" in signal_text.upper() or "SELL" in signal_text.upper():
                signal_dict['action'] = "SELL"
            else:
                return None
                
            # 解析价格值（这里需要根据实际的AI输出格式调整）
            import re
            
            # 寻找入场价格
            entry_match = re.search(r'入场[：:]\s*(\d+\.?\d*)', signal_text)
            if entry_match:
                signal_dict['entry_price'] = float(entry_match.group(1))
                
            # 寻找止损价格
            sl_match = re.search(r'止损[：:]\s*(\d+\.?\d*)', signal_text)
            if sl_match:
                signal_dict['stop_loss'] = float(sl_match.group(1))
                
            # 寻找止盈价格
            tp_match = re.search(r'止盈[：:]\s*(\d+\.?\d*)', signal_text)
            if tp_match:
                signal_dict['take_profit'] = float(tp_match.group(1))
                
            # 寻找交易量
            volume_match = re.search(r'仓位[：:]\s*(\d+\.?\d*)', signal_text)
            if volume_match:
                signal_dict['volume'] = float(volume_match.group(1))
            else:
                signal_dict['volume'] = 0.01  # 默认交易量
                
            return signal_dict
            
        except Exception as e:
            print(f"信号解析错误: {str(e)}")
            return None

    def calculate_wpr(self, data, periods=14):
        """计算Williams Percent Range指标"""
        high = data['high'].rolling(window=periods).max()
        low = data['low'].rolling(window=periods).min()
        close = data['close']
        wpr = -100 * ((high - close) / (high - low))
        return wpr
    
    def calculate_indicators(self, df):
        """Calculate multiple technical indicators"""
        # RSI
        df['rsi'] = self.calculate_rsi(df)
        
        # Moving Averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()


        # Average True Range (ATR)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 添加WPR指标
        df['wpr'] = self.calculate_wpr(df)
        
        return df
    
    def calculate_rsi(self, data, periods=14):
        """Calculate RSI indicator"""
        close_delta = data['close'].diff()
        gains = close_delta.clip(lower=0)
        losses = -1 * close_delta.clip(upper=0)
        avg_gains = gains.rolling(window=periods, min_periods=periods).mean()
        avg_losses = losses.rolling(window=periods, min_periods=periods).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_market_data(self, timeframe, bars=100):
        """Get market data for specified timeframe"""
        try:
            if not self.ensure_mt5_connection():
                return None
                
            # 设置固定的结束日期为2025年3月17日
            end_date = datetime(2025, 3, 17)
            start_date = end_date - timedelta(days=bars)
            
            rates = mt5.copy_rates_range(
                "XAUUSD",
                timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                print(f"No data received for timeframe {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            if df.empty:
                print("Empty dataframe received")
                return None
                
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting market data: {str(e)}")
            self.mt5_connected = False
            return None
    
    def prepare_data_string(self, df, timeframe_name):
        """Prepare formatted data string for analysis"""
        recent_data = df.tail(20).copy()
        data_str = f"\nRecent XAUUSD {timeframe_name} candles:\n"
        for _, row in recent_data.iterrows():
            data_str += (
                f"Time: {row['time']}, Open: {row['open']:.2f}, High: {row['high']:.2f}, "
                f"Low: {row['low']:.2f}, Close: {row['close']:.2f}, RSI: {row['rsi']:.2f}, "
                f"EMA5: {row['ema_5']:.2f}, EMA10: {row['ema_10']:.2f}, EMA20: {row['ema_20']:.2f}, ATR: {row['atr']:.2f}, WPR: {row['wpr']:.2f}\n"
            )
        return data_str
    
    def analyze_market(self, market_data):
        """统一的市场分析函数，合并特征计算和信号生成"""
        try:
            # 构建完整的提示词
            prompt = MARKET_ANALYSIS_PROMPT.format(
                feature_prompt=FEATURE_PROMPT.format(
                    daily_data=market_data['D1'],
                    h4_data=market_data['H4'],
                    h1_data=market_data['H1']
                ),
                trading_prompt=TRADING_PROMPT.format(
                    daily_data=market_data['D1'],
                    h4_data=market_data['H4'],
                    h1_data=market_data['H1'],
                    technical_features="[将由第一部分分析结果填充]"
                )
            )
              # 打印实际的提示词内容
            print("发送给AI的提示词：")
            print(prompt)
            # 单次调用LLM
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "You are a professional XAUUSD trader."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content

            # 解析返回结果
            analysis = self._extract_content(result, "ANALYSIS_START", "ANALYSIS_END")
            signal = self._extract_content(result, "SIGNAL_START", "SIGNAL_END")

            # 记录分析结果时确保中文正确显示
            self.audit_logger.log_risk_check({
                'market_data': {k: str(v)[:100] + '...' for k, v in market_data.items()},
                'analysis': analysis[:100] + '...',
                'timestamp': datetime.now().isoformat(),
                'analysis_content': {
                    'technical_analysis': analysis,
                    'trading_signal': signal
                }
            })

            return analysis, signal

        except Exception as e:
            print(f"Error in market analysis: {e}")
            return None, None

    def _extract_content(self, text, start_marker, end_marker):
        """从文本中提取标记之间的内容"""
        try:
            start_idx = text.index(start_marker) + len(start_marker)
            end_idx = text.index(end_marker)
            return text[start_idx:end_idx].strip()
        except ValueError:
            return ""

    def run_analysis(self):
        """更新后的主分析方法"""
        try:
            if not self.ensure_mt5_connection():
                return None
                
            # Get market data for all timeframes
            market_data = {}
            for tf_name, tf_value in self.timeframes.items():
                df = self.get_market_data(tf_value)
                if df is not None:
                    market_data[tf_name] = self.prepare_data_string(df, tf_name)

            # 统一进行市场分析
            features, signal = self.analyze_market(market_data)
            if not features or not signal:
                return None

            # 解析交易信号
            signal_dict = self.parse_trading_signal(signal)
            
            # 如果有有效的交易信号，执行交易
            if signal_dict:
                self.execute_trade(signal_dict)
            
            # Check current spread
            symbol_info = mt5.symbol_info("XAUUSD")
            current_spread = symbol_info.spread if symbol_info else None
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "technical_features": features,
                "trading_signal": signal,
                "current_spread": current_spread,
                "signal_dict": signal_dict
            }
            
            return result
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            self.mt5_connected = False
            return None
        finally:
            try:
                mt5.shutdown()
            except:
                pass