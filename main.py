from trading_bot import XAUUSDTradingBot
from config import API_CONFIG, TRADING_CONFIG
import time

def main():
    bot = XAUUSDTradingBot(api_key=API_CONFIG['zhipu_api_key'])
    
    while True:
        try:
            result = bot.run_analysis()
            
            if result:
                print("\n=== Analysis Results ===")
                print("\nCurrent Spread:", result["current_spread"])
                
                print("\nMarket Data Analysis:")
                for timeframe, data in result["market_data"].items():
                    print(f"\n{timeframe} Timeframe:")
                    print(data)
                
                print("\nTechnical Analysis:")
                print(result["technical_features"])
                
                print("\nTrading Signal:")
                print(result["trading_signal"])
                print("\n=====================")
            else:
                print("No valid analysis generated")
            
            print("\nWaiting 30 minutes for next analysis...")
            time.sleep(1800)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()
