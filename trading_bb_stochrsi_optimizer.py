"""
BACKTEST OPTIMIZER: BB + Stochastic RSI + Pinbar Strategy
Test nhiều tổ hợp tham số và tìm cấu hình tối ưu
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from colorama import Fore, Style, init
import os
import time
from itertools import product

# Init màu console
init(autoreset=True)

# --- CẤU HÌNH CỐ ĐỊNH ---
SYMBOLS = ["btcusdt","ethusdt","solusdt", "bnbusdt", "zecusdt","dogeusdt","xrpusdt"]
BACKTEST_DAYS = 7
INIT_CAPITAL = 200.0
LEVERAGE = 20
FEE_RATE = 0.05 / 100

# --- CÁC TỔ HỢP THAM SỐ CẦN TEST ---
PARAM_GRID = {
    'timeframe': ['3m', '5m', '15m',"30m","1h"],  # Khung giờ
    'margin': [10, 20],              # Margin mỗi lệnh
    'stop_loss': [20, 50, 100],        # Stop Loss (USDT)
    'trailing_trigger': [0.3, 0.5, 1.0],  # Trailing trigger (%)
    'trailing_callback': [0.1],   # Trailing callback (%)
    'bb_period': [20, 30],             # BB period
    'bb_std': [2, 2.5],              # BB std deviation
    'rsi_period': [7, 9, 14],             # RSI period
    'stoch_oversold': [25],        # StochRSI oversold threshold
    'stoch_overbought': [75],      # StochRSI overbought threshold
    'max_dca': [10],              # Số lần DCA tối đa (0 = tắt DCA)
}


class OptimizedTrader:
    """Trader với tham số cấu hình linh hoạt"""
    
    def __init__(self, symbol, config, data_cache):
        self.symbol = symbol
        self.config = config
        self.df = data_cache.get(symbol, pd.DataFrame())
        self.position = {
            'type': None, 'entry_price': 0.0, 'size': 0.0, 'margin': 0.0,
            'highest_price_move': -999.0, 'entry_time': None, 'dca_count': 0
        }
        self.balance = INIT_CAPITAL
        self.initial_balance = INIT_CAPITAL
        self.peak_balance = INIT_CAPITAL
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.dca_trades = 0
        
    def update_indicators(self):
        """Tính indicators với config hiện tại"""
        if self.df.empty:
            return
            
        df = self.df.copy()
        bb_period = self.config['bb_period']
        bb_std = self.config['bb_std']
        rsi_period = self.config['rsi_period']
        
        # Bollinger Bands
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['ma'] = df['tp'].rolling(window=bb_period).mean()
        df['std'] = df['tp'].rolling(window=bb_period).std()
        df['upper_bb'] = df['ma'] + (df['std'] * bb_std)
        df['lower_bb'] = df['ma'] - (df['std'] * bb_std)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        min_rsi = df['rsi'].rolling(window=14).min()
        max_rsi = df['rsi'].rolling(window=14).max()
        df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi) * 100
        df['k'] = df['stoch_rsi'].rolling(window=3).mean()
        df['d'] = df['k'].rolling(window=3).mean()
        
        self.df = df

    def calc_unrealized_pnl(self, current_price):
        if self.position['type'] == 'LONG':
            return (current_price - self.position['entry_price']) * self.position['size']
        elif self.position['type'] == 'SHORT':
            return (self.position['entry_price'] - current_price) * self.position['size']
        return 0.0

    def close_position(self, price):
        pnl = self.calc_unrealized_pnl(price)
        order_size = self.config['margin'] * LEVERAGE
        fee = (self.position['size'] * price) * FEE_RATE
        realized_pnl = pnl - fee
        
        self.balance += realized_pnl
        self.trades_count += 1
        self.total_pnl += realized_pnl
        
        if realized_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Update drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        self.position = {
            'type': None, 'entry_price': 0, 'size': 0, 'margin': 0,
            'highest_price_move': -999, 'entry_time': None, 'dca_count': 0
        }

    def execute_trade(self, signal_type, price):
        order_size = self.config['margin'] * LEVERAGE
        quantity = order_size / price
        fee = order_size * FEE_RATE
        
        # DCA Logic
        if self.position['type'] is not None:
            if self.config['max_dca'] == 0:
                return False
            
            current_pnl = self.calc_unrealized_pnl(price)
            if current_pnl < 0 and self.position['type'] == signal_type and self.position['dca_count'] < self.config['max_dca']:
                new_size = self.position['size'] + quantity
                avg_entry = ((self.position['entry_price'] * self.position['size']) + (price * quantity)) / new_size
                self.position['entry_price'] = avg_entry
                self.position['size'] = new_size
                self.position['margin'] += self.config['margin']
                self.position['dca_count'] += 1
                self.balance -= fee
                self.dca_trades += 1
                return True
            return False

        # New position
        self.position = {
            'type': signal_type,
            'entry_price': price,
            'size': quantity,
            'margin': self.config['margin'],
            'highest_price_move': -999.0,
            'entry_time': None,
            'dca_count': 0
        }
        self.balance -= fee
        return True

    def run_backtest(self):
        """Chạy backtest với config hiện tại"""
        if self.df.empty:
            return
            
        self.update_indicators()
        
        min_period = max(self.config['bb_period'], self.config['rsi_period'] + 14) + 50
        if len(self.df) < min_period:
            return
        
        trailing_trigger = self.config['trailing_trigger'] / 100
        trailing_callback = self.config['trailing_callback'] / 100
        stoch_oversold = self.config['stoch_oversold']
        stoch_overbought = self.config['stoch_overbought']
        stop_loss_usdt = self.config['stop_loss']
        
        for i in range(min_period, len(self.df)):
            current = self.df.iloc[i]
            close_price = current['close']
            
            # Quản lý vị thế đang mở
            if self.position['type'] is not None:
                if self.position['type'] == 'LONG':
                    price_move_pct = (close_price - self.position['entry_price']) / self.position['entry_price']
                else:
                    price_move_pct = (self.position['entry_price'] - close_price) / self.position['entry_price']

                if price_move_pct > self.position['highest_price_move']:
                    self.position['highest_price_move'] = price_move_pct

                unrealized_pnl = self.calc_unrealized_pnl(close_price)
                if unrealized_pnl <= -stop_loss_usdt:
                    self.close_position(close_price)
                    continue

                if self.position['highest_price_move'] >= trailing_trigger:
                    if (self.position['highest_price_move'] - price_move_pct) >= trailing_callback:
                        self.close_position(close_price)
                        continue
            
            # Tính pinbar
            candle = current
            total_size = candle['high'] - candle['low'] if (candle['high'] - candle['low']) > 0 else 0.00001
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            
            is_bullish_pinbar = (lower_wick / total_size) > 0.6
            is_bearish_pinbar = (upper_wick / total_size) > 0.6
            is_green = candle['close'] > candle['open']
            is_red = candle['close'] < candle['open']
            
            # LONG Signal
            l_cond1 = candle['low'] <= current['lower_bb']
            l_cond2 = current['k'] < stoch_oversold and current['d'] < stoch_oversold and current['k'] > current['d']
            l_cond3 = is_bullish_pinbar or (is_green and candle['close'] > current['lower_bb'])
            
            if l_cond1 and l_cond2 and l_cond3:
                self.execute_trade('LONG', close_price)
                continue
            
            # SHORT Signal
            s_cond1 = candle['high'] >= current['upper_bb']
            s_cond2 = current['k'] > stoch_overbought and current['d'] > stoch_overbought and current['k'] < current['d']
            s_cond3 = is_bearish_pinbar or (is_red and candle['close'] < current['upper_bb'])
            
            if s_cond1 and s_cond2 and s_cond3:
                self.execute_trade('SHORT', close_price)
        
        # Đóng vị thế cuối
        if self.position['type'] is not None:
            self.close_position(self.df.iloc[-1]['close'])


def fetch_data_for_symbols(symbols, start_time, end_time, timeframe):
    """Tải dữ liệu 1 lần cho tất cả symbols"""
    data_cache = {}
    
    for symbol in symbols:
        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        print(f"{Fore.CYAN}Đang tải dữ liệu {symbol}...{Style.RESET_ALL}", end=" ")
        
        while current_start < end_ts:
            try:
                url = "https://fapi.binance.com/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1500
                }
                res = requests.get(url, params=params).json()
                
                if not res or 'code' in res:
                    break
                    
                for k in res:
                    all_data.append({
                        'time': datetime.fromtimestamp(k[0]/1000),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                
                if len(res) < 1500:
                    break
                    
                current_start = res[-1][0] + 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"{Fore.RED}Lỗi: {e}{Style.RESET_ALL}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
            data_cache[symbol] = df
            print(f"{Fore.GREEN}{len(df)} nến{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Không có dữ liệu!{Style.RESET_ALL}")
        
        time.sleep(0.3)
    
    return data_cache


def run_single_config(config, symbols, data_cache):
    """Chạy backtest với 1 config cụ thể"""
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    total_dca = 0
    max_dd = 0.0
    
    for symbol in symbols:
        if symbol not in data_cache:
            continue
            
        trader = OptimizedTrader(symbol, config, data_cache)
        trader.run_backtest()
        
        total_trades += trader.trades_count
        total_wins += trader.win_count
        total_losses += trader.loss_count
        total_pnl += trader.total_pnl
        total_dca += trader.dca_trades
        if trader.max_drawdown > max_dd:
            max_dd = trader.max_drawdown
    
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    roi = (total_pnl / INIT_CAPITAL) * 100
    
    return {
        'trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': win_rate,
        'pnl': total_pnl,
        'roi': roi,
        'max_dd': max_dd * 100,
        'dca_count': total_dca
    }


def run_optimizer():
    """Chạy tối ưu hóa tham số"""
    print(f"\n{Fore.YELLOW}{'='*80}")
    print(f"{'='*80}")
    print(f"       🔬 PARAMETER OPTIMIZER - BB + Stoch RSI + Pinbar")
    print(f"{'='*80}")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Tính tổng số combinations
    total_combinations = 1
    for key, values in PARAM_GRID.items():
        total_combinations *= len(values)
    
    print(f"{Fore.CYAN}📊 Tổng số tổ hợp cần test: {total_combinations}")
    print(f"📈 Coins: {', '.join(SYMBOLS)}")
    print(f"⏱️  Timeframes: {PARAM_GRID.get('timeframe', ['3m'])} | Days: {BACKTEST_DAYS}{Style.RESET_ALL}\n")
    
    # Tải dữ liệu cho tất cả timeframe
    print(f"{Fore.YELLOW}📥 Đang tải dữ liệu lịch sử...{Style.RESET_ALL}")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=BACKTEST_DAYS)
    
    # Cache dữ liệu theo timeframe
    timeframes = PARAM_GRID.get('timeframe', ['3m'])
    data_cache_by_tf = {}
    for tf in timeframes:
        print(f"\n{Fore.YELLOW}--- Timeframe: {tf} ---{Style.RESET_ALL}")
        data_cache_by_tf[tf] = fetch_data_for_symbols(SYMBOLS, start_time, end_time, tf)
    
    if not any(data_cache_by_tf.values()):
        print(f"{Fore.RED}❌ Không có dữ liệu để test!{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.YELLOW}🔄 Bắt đầu chạy optimization...{Style.RESET_ALL}\n")
    
    # Generate all combinations
    param_keys = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(product(*param_values))
    
    results = []
    
    for idx, combo in enumerate(combinations, 1):
        config = dict(zip(param_keys, combo))
        
        # Progress
        if idx % 10 == 0 or idx == 1:
            print(f"{Fore.CYAN}  Testing {idx}/{total_combinations}...{Style.RESET_ALL}")
        
        # Lấy data cache cho timeframe tương ứng
        tf = config.get('timeframe', '3m')
        data_cache = data_cache_by_tf.get(tf, {})
        
        result = run_single_config(config, SYMBOLS, data_cache)
        result['config'] = config
        results.append(result)
    
    # Sắp xếp theo ROI
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    # Hiển thị Top 20 config tốt nhất
    print(f"\n\n{Fore.GREEN}{'='*100}")
    print(f"{'='*100}")
    print(f"                    📊 TOP 20 CẤU HÌNH TỐT NHẤT (THEO ROI)")
    print(f"{'='*100}")
    print(f"{'='*100}{Style.RESET_ALL}\n")
    
    print(f"{'Rank':<5} | {'ROI %':<10} | {'WinRate %':<10} | {'Trades':<7} | {'MaxDD %':<8} | {'PnL':<12} | Config")
    print("-" * 120)
    
    for rank, r in enumerate(results[:20], 1):
        c = r['config']
        tf = c.get('timeframe', '3m')
        config_str = f"TF={tf}, M={c['margin']}, SL={c['stop_loss']}, TT={c['trailing_trigger']}%, BB={c['bb_period']}, RSI={c['rsi_period']}, DCA={c['max_dca']}"
        
        roi_color = Fore.GREEN if r['roi'] >= 0 else Fore.RED
        print(f"{rank:<5} | {roi_color}{r['roi']:>+8.2f}%{Style.RESET_ALL} | {r['win_rate']:>8.1f}% | {r['trades']:<7} | {r['max_dd']:>6.2f}% | {r['pnl']:>+10.2f}u | {config_str}")
    
    # Lưu kết quả ra file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"optimizer_results_{timestamp}.csv")
    
    # Tạo DataFrame để lưu
    df_results = []
    for r in results:
        row = {
            'rank': results.index(r) + 1,
            'roi_pct': r['roi'],
            'win_rate_pct': r['win_rate'],
            'total_trades': r['trades'],
            'wins': r['wins'],
            'losses': r['losses'],
            'pnl_usdt': r['pnl'],
            'max_drawdown_pct': r['max_dd'],
            'dca_count': r['dca_count'],
            **r['config']
        }
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{Fore.GREEN}✅ Đã lưu kết quả: {result_file}{Style.RESET_ALL}")
    
    # Phân tích config tốt nhất
    if results:
        best = results[0]
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"       🏆 CẤU HÌNH TỐT NHẤT")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        c = best['config']
        print(f"\n{Fore.GREEN}📈 Kết quả:")
        print(f"    ROI: {best['roi']:+.2f}%")
        print(f"    Win Rate: {best['win_rate']:.1f}%")
        print(f"    Total Trades: {best['trades']}")
        print(f"    PnL: {best['pnl']:+.2f}u")
        print(f"    Max Drawdown: {best['max_dd']:.2f}%")
        
        print(f"\n{Fore.CYAN}⚙️  Cấu hình:")
        print(f"    TIMEFRAME = '{c.get('timeframe', '3m')}'")
        print(f"    MARGIN_PER_ORDER = {c['margin']}")
        print(f"    STOP_LOSS_USDT = {c['stop_loss']}")
        print(f"    TRAILING_TRIGGER_PCT = {c['trailing_trigger']} / 100")
        print(f"    TRAILING_CALLBACK_PCT = {c['trailing_callback']} / 100")
        print(f"    BB_PERIOD = {c['bb_period']}")
        print(f"    BB_STD = {c['bb_std']}")
        print(f"    RSI_PERIOD = {c['rsi_period']}")
        print(f"    STOCH_OVERSOLD = {c['stoch_oversold']}")
        print(f"    STOCH_OVERBOUGHT = {c['stoch_overbought']}")
        print(f"    MAX_DCA_COUNT = {c['max_dca']}{Style.RESET_ALL}")
    
    # Phân tích thống kê
    print(f"\n{Fore.YELLOW}{'='*80}")
    print(f"       📊 THỐNG KÊ PHÂN TÍCH")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    profitable = [r for r in results if r['roi'] > 0]
    print(f"\n{Fore.CYAN}Tổng quan:")
    print(f"    Số config có lãi: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)")
    print(f"    Avg ROI: {np.mean([r['roi'] for r in results]):.2f}%")
    print(f"    Best ROI: {results[0]['roi']:.2f}%")
    print(f"    Worst ROI: {results[-1]['roi']:.2f}%{Style.RESET_ALL}")


if __name__ == "__main__":
    print(f"\n{Fore.YELLOW}╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║         PARAMETER OPTIMIZER - Find Best Trading Config            ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    run_optimizer()
    
    print(f"\n{Fore.GREEN}✅ Optimization hoàn tất!{Style.RESET_ALL}")
