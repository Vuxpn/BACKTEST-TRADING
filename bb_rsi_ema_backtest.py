"""
BACKTEST: BB(20) + RSI(9) + EMA(200) + Divergence Strategy
Sử dụng dữ liệu lịch sử thật từ Binance Futures
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from colorama import Fore, Style, init
import os
import time

# Init màu console
init(autoreset=True)

# --- CẤU HÌNH BACKTEST ---
SYMBOLS = [
    'BEATUSDT', 'POWERUSDT', 'ACTUSDT', 'RAVEUSDT', 'BUSDT', 
    
] 
TIMEFRAME = '3m'

# Thời gian backtest (số ngày lùi về trước)
BACKTEST_DAYS = 30

# Chỉ báo tối ưu cho khung 3m
BB_PERIOD = 20      
BB_STD = 2.0      
RSI_PERIOD = 9      
EMA_FILTER = 200    

# Quản lý vốn
INIT_CAPITAL = 200.0
LEVERAGE = 20
MARGIN_PER_ORDER = 2
ORDER_SIZE_USDT = MARGIN_PER_ORDER * LEVERAGE 
FEE_RATE = 0.05 / 100
STOP_LOSS_USDT = 20

# Trailing Stop
TRAILING_TRIGGER_PCT = 0.5 / 100
TRAILING_CALLBACK_PCT = 0.2 / 100

# DCA (Dollar Cost Averaging) - Nhồi lệnh
MAX_DCA_ENTRIES = 5          # Số lần nhồi tối đa (không tính lệnh gốc)
DCA_PRICE_DROP_PCT = 0.5 / 100  # % giá giảm/tăng để được nhồi tiếp


class BacktestPortfolio:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.balance_history = [(datetime.now(), initial_balance)]

    def update_balance(self, amount, timestamp=None):
        self.balance += amount
        if timestamp:
            self.balance_history.append((timestamp, self.balance))

    def get_balance(self):
        return self.balance


class BacktestTrader:
    def __init__(self, symbol, portfolio):
        self.symbol = symbol
        self.portfolio = portfolio
        self.df = pd.DataFrame()
        self.position = {'type': None, 'entry_price': 0.0, 'size': 0.0, 'margin': 0.0, 
                         'highest_price_move': -999.0, 'entry_time': None, 'dca_count': 0}
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.trades_history = []
        self.max_drawdown = 0.0
        self.peak_balance = INIT_CAPITAL
        self.dca_trades = 0  # Thống kê số lần nhồi
        
    def log(self, msg, color=Fore.WHITE):
        print(f"{color}[{self.symbol}] {msg}{Style.RESET_ALL}")

    def fetch_historical_data(self, start_time, end_time):
        """Lấy dữ liệu lịch sử từ Binance"""
        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        print(f"{Fore.CYAN}[{self.symbol}] Đang tải dữ liệu từ {start_time} đến {end_time}...{Style.RESET_ALL}")
        
        while current_start < end_ts:
            try:
                url = f"https://fapi.binance.com/fapi/v1/klines"
                params = {
                    'symbol': self.symbol,
                    'interval': TIMEFRAME,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1500
                }
                res = requests.get(url, params=params).json()
                
                if not res or 'code' in res:
                    print(f"{Fore.RED}[{self.symbol}] Lỗi API: {res}{Style.RESET_ALL}")
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
                time.sleep(0.1)  # Rate limit
                
            except Exception as e:
                print(f"{Fore.RED}[{self.symbol}] Lỗi tải dữ liệu: {e}{Style.RESET_ALL}")
                break
        
        if all_data:
            self.df = pd.DataFrame(all_data)
            self.df = self.df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
            print(f"{Fore.GREEN}[{self.symbol}] Đã tải {len(self.df)} nến{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[{self.symbol}] Không có dữ liệu!{Style.RESET_ALL}")
        
        return len(self.df) > 0

    def update_indicators(self, df):
        """Cập nhật các chỉ báo kỹ thuật"""
        df = df.copy()
        df['ma'] = df['close'].rolling(window=BB_PERIOD).mean()
        df['std'] = df['close'].rolling(window=BB_PERIOD).std()
        df['upper_bb'] = df['ma'] + (df['std'] * BB_STD)
        df['lower_bb'] = df['ma'] - (df['std'] * BB_STD)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['ema_200'] = df['close'].ewm(span=EMA_FILTER, adjust=False).mean()
        
        return df

    def check_divergence(self, df_subset, side):
        """Kiểm tra phân kỳ RSI"""
        try:
            if len(df_subset) < 15:
                return False
                
            if side == 'SHORT':
                last_high = df_subset['high'].iloc[-1]
                last_rsi = df_subset['rsi'].iloc[-1]
                prev_high = df_subset['high'].iloc[:-2].max()
                prev_rsi = df_subset['rsi'].iloc[:-2].max()
                return last_high >= prev_high and last_rsi < prev_rsi
            else:
                last_low = df_subset['low'].iloc[-1]
                last_rsi = df_subset['rsi'].iloc[-1]
                prev_low = df_subset['low'].iloc[:-2].min()
                prev_rsi = df_subset['rsi'].iloc[:-2].min()
                return last_low <= prev_low and last_rsi > prev_rsi
        except:
            return False

    def calc_unrealized_pnl(self, current_price):
        if self.position['type'] == 'LONG':
            return (current_price - self.position['entry_price']) * self.position['size']
        if self.position['type'] == 'SHORT':
            return (self.position['entry_price'] - current_price) * self.position['size']
        return 0.0

    def close_position(self, price, reason, timestamp):
        pnl = self.calc_unrealized_pnl(price)
        fee = (self.position['size'] * price) * FEE_RATE
        realized_pnl = pnl - fee
        
        self.portfolio.update_balance(realized_pnl, timestamp)
        
        self.trades_count += 1
        self.total_pnl += realized_pnl
        if realized_pnl > 0: 
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Cập nhật max drawdown
        current_balance = self.portfolio.get_balance()
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Lưu lịch sử giao dịch
        self.trades_history.append({
            'symbol': self.symbol,
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'size': self.position['size'],
            'margin': self.position['margin'],
            'dca_count': self.position['dca_count'],
            'pnl': realized_pnl,
            'fee': fee,
            'reason': reason,
            'balance_after': current_balance,
            'duration': str(timestamp - self.position['entry_time']) if self.position['entry_time'] else 'N/A'
        })
        
        self.position = {'type': None, 'entry_price': 0, 'size': 0, 'margin': 0, 
                        'highest_price_move': -999, 'entry_time': None, 'dca_count': 0}
        
        return realized_pnl

    def execute_trade(self, signal_type, price, timestamp):
        if self.position['type'] is not None: 
            return False
        
        quantity = ORDER_SIZE_USDT / price
        fee = ORDER_SIZE_USDT * FEE_RATE
        
        self.position = {
            'type': signal_type, 
            'entry_price': price, 
            'size': quantity, 
            'margin': MARGIN_PER_ORDER, 
            'highest_price_move': -999.0,
            'entry_time': timestamp,
            'dca_count': 0
        }
        self.portfolio.update_balance(-fee, timestamp)
        return True

    def execute_dca(self, price, timestamp):
        """Thực hiện nhồi lệnh (DCA) - thêm vốn vào vị thế hiện tại"""
        if self.position['type'] is None:
            return False
        
        if self.position['dca_count'] >= MAX_DCA_ENTRIES:
            return False
        
        # Kiểm tra giá đã giảm/tăng đủ % so với entry để được nhồi
        price_change_pct = 0.0
        if self.position['type'] == 'LONG':
            price_change_pct = (self.position['entry_price'] - price) / self.position['entry_price']
        else:  # SHORT
            price_change_pct = (price - self.position['entry_price']) / self.position['entry_price']
        
        # Chỉ nhồi khi giá đi ngược xu hướng đủ mức
        if price_change_pct < DCA_PRICE_DROP_PCT:
            return False
        
        # Tính toán DCA
        add_quantity = ORDER_SIZE_USDT / price
        fee = ORDER_SIZE_USDT * FEE_RATE
        
        # Cập nhật giá trung bình
        total_value = (self.position['entry_price'] * self.position['size']) + (price * add_quantity)
        new_size = self.position['size'] + add_quantity
        new_avg_price = total_value / new_size
        
        self.position['entry_price'] = new_avg_price
        self.position['size'] = new_size
        self.position['margin'] += MARGIN_PER_ORDER
        self.position['dca_count'] += 1
        self.position['highest_price_move'] = -999.0  # Reset trailing trigger sau DCA
        
        self.portfolio.update_balance(-fee, timestamp)
        self.dca_trades += 1
        
        return True

    def run_backtest(self):
        """Chạy backtest trên dữ liệu đã tải"""
        if len(self.df) < EMA_FILTER + 50:
            print(f"{Fore.RED}[{self.symbol}] Không đủ dữ liệu để backtest!{Style.RESET_ALL}")
            return
        
        # Tính toán indicators cho toàn bộ dữ liệu
        self.df = self.update_indicators(self.df)
        
        # Bắt đầu từ candle thứ EMA_FILTER + 50 để có đủ dữ liệu
        start_idx = EMA_FILTER + 50
        
        print(f"{Fore.CYAN}[{self.symbol}] Bắt đầu backtest từ index {start_idx}...{Style.RESET_ALL}")
        
        for i in range(start_idx, len(self.df)):
            current = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            timestamp = current['time']
            close_price = current['close']
            
            # Xử lý vị thế đang mở (check stop loss và trailing)
            if self.position['type'] is not None:
                price_move_pct = 0.0
                if self.position['type'] == 'LONG':
                    price_move_pct = (close_price - self.position['entry_price']) / self.position['entry_price']
                else:
                    price_move_pct = (self.position['entry_price'] - close_price) / self.position['entry_price']

                if price_move_pct > self.position['highest_price_move']:
                    self.position['highest_price_move'] = price_move_pct

                unrealized_pnl = self.calc_unrealized_pnl(close_price)
                if unrealized_pnl <= -STOP_LOSS_USDT:
                    self.close_position(close_price, f"STOP LOSS ({unrealized_pnl:.2f}u)", timestamp)
                    continue

                if self.position['highest_price_move'] >= TRAILING_TRIGGER_PCT:
                    if (self.position['highest_price_move'] - price_move_pct) >= TRAILING_CALLBACK_PCT:
                        self.close_position(close_price, "Trailing Stop", timestamp)
                        continue
            
            # Lấy 15 nến gần nhất cho divergence check
            df_subset = self.df.iloc[max(0, i-14):i+1]
            
            # Kiểm tra tín hiệu LONG
            long_signal = False
            if current['close'] > current['ema_200']:
                if current['low'] <= current['lower_bb'] and current['rsi'] < 35:
                    if self.check_divergence(df_subset, 'LONG'):
                        if current['close'] > prev['close']:
                            long_signal = True
            
            # Kiểm tra tín hiệu SHORT
            short_signal = False
            if current['close'] < current['ema_200']:
                if current['high'] >= current['upper_bb'] and current['rsi'] > 65:
                    if self.check_divergence(df_subset, 'SHORT'):
                        if current['close'] < prev['close']:
                            short_signal = True
            
            # Xử lý tín hiệu
            if self.position['type'] is None:
                # Chưa có vị thế - mở lệnh mới
                if long_signal:
                    self.execute_trade('LONG', close_price, timestamp)
                elif short_signal:
                    self.execute_trade('SHORT', close_price, timestamp)
            else:
                # Đã có vị thế - kiểm tra nhồi lệnh (DCA)
                if self.position['type'] == 'LONG' and long_signal:
                    self.execute_dca(close_price, timestamp)
                elif self.position['type'] == 'SHORT' and short_signal:
                    self.execute_dca(close_price, timestamp)
        
        # Đóng vị thế cuối cùng nếu còn
        if self.position['type'] is not None:
            last_row = self.df.iloc[-1]
            self.close_position(last_row['close'], "End of Backtest", last_row['time'])


def run_full_backtest():
    """Chạy backtest cho tất cả symbols"""
    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"       🚀 BACKTEST: BB(20) + RSI(9) + EMA(200) + Divergence")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Thiết lập thời gian backtest
    end_time = datetime.now()
    start_time = end_time - timedelta(days=BACKTEST_DAYS)
    
    print(f"{Fore.CYAN}⏱️  Khoảng thời gian: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 Khung thời gian: {TIMEFRAME}")
    print(f"💰 Vốn ban đầu: {INIT_CAPITAL}u | Leverage: {LEVERAGE}x")
    print(f"📈 Chỉ báo: BB({BB_PERIOD}, {BB_STD}) | RSI({RSI_PERIOD}) | EMA({EMA_FILTER})")
    print(f"🛡️  Stop Loss: {STOP_LOSS_USDT}u | Trailing: {TRAILING_TRIGGER_PCT*100:.1f}%/{TRAILING_CALLBACK_PCT*100:.1f}%")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    portfolio = BacktestPortfolio(INIT_CAPITAL)
    traders = {}
    all_trades = []
    
    # Backtest từng symbol
    for symbol in SYMBOLS:
        print(f"\n{Fore.YELLOW}{'='*50}")
        print(f"       Backtest: {symbol}")
        print(f"{'='*50}{Style.RESET_ALL}")
        
        trader = BacktestTrader(symbol, portfolio)
        
        if trader.fetch_historical_data(start_time, end_time):
            trader.run_backtest()
            traders[symbol] = trader
            all_trades.extend(trader.trades_history)
        
        time.sleep(0.5)  # Rate limit between symbols
    
    # Tổng hợp kết quả
    generate_backtest_report(traders, all_trades, portfolio, start_time, end_time)


def generate_backtest_report(traders, all_trades, portfolio, start_time, end_time):
    """Tạo báo cáo backtest chi tiết"""
    
    # Tính toán thống kê tổng
    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t['pnl'] > 0]
    losing_trades = [t for t in all_trades if t['pnl'] <= 0]
    total_wins = len(winning_trades)
    total_losses = len(losing_trades)
    
    total_pnl = sum(t['pnl'] for t in all_trades)
    total_fees = sum(t['fee'] for t in all_trades)
    
    final_balance = portfolio.get_balance()
    balance_pnl = final_balance - INIT_CAPITAL
    roi = (balance_pnl / INIT_CAPITAL) * 100
    
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    # Tính metrics bổ sung
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else np.inf
    
    max_win = max(t['pnl'] for t in all_trades) if all_trades else 0
    max_loss = min(t['pnl'] for t in all_trades) if all_trades else 0
    
    # Tính max drawdown
    max_drawdown = max([trader.max_drawdown for trader in traders.values()]) if traders else 0
    
    # Thống kê theo loại lệnh
    long_trades = [t for t in all_trades if t['type'] == 'LONG']
    short_trades = [t for t in all_trades if t['type'] == 'SHORT']
    
    long_wins = len([t for t in long_trades if t['pnl'] > 0])
    short_wins = len([t for t in short_trades if t['pnl'] > 0])
    
    long_pnl = sum(t['pnl'] for t in long_trades)
    short_pnl = sum(t['pnl'] for t in short_trades)
    
    # Thống kê theo lý do đóng lệnh
    trailing_closes = len([t for t in all_trades if 'Trailing' in t['reason']])
    stoploss_closes = len([t for t in all_trades if 'STOP LOSS' in t['reason']])
    
    # In báo cáo
    print(f"\n\n{Fore.YELLOW}{'='*70}")
    print(f"{'='*70}")
    print(f"           📊 BÁO CÁO BACKTEST CHI TIẾT")
    print(f"{'='*70}")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}⏱️  THÔNG TIN BACKTEST:")
    print(f"    Khoảng thời gian: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"    Số ngày: {BACKTEST_DAYS}")
    print(f"    Khung thời gian: {TIMEFRAME}")
    print(f"    Số coins: {len(traders)}")
    
    print(f"\n{Fore.GREEN}💰 THỐNG KÊ TÀI KHOẢN:")
    print(f"    Vốn ban đầu: {INIT_CAPITAL:.2f}u")
    print(f"    Balance cuối: {final_balance:.2f}u")
    balance_color = Fore.GREEN if balance_pnl >= 0 else Fore.RED
    print(f"    PNL: {balance_color}{balance_pnl:+.2f}u{Fore.GREEN}")
    print(f"    ROI: {balance_color}{roi:+.2f}%{Fore.GREEN}")
    print(f"    Max Drawdown: {Fore.RED}{max_drawdown*100:.2f}%{Style.RESET_ALL}")
    
    print(f"\n{Fore.MAGENTA}📈 THỐNG KÊ GIAO DỊCH:")
    print(f"    Tổng số lệnh: {total_trades}")
    print(f"    Số lệnh thắng: {Fore.GREEN}{total_wins}{Fore.MAGENTA}")
    print(f"    Số lệnh thua: {Fore.RED}{total_losses}{Fore.MAGENTA}")
    print(f"    Win Rate: {win_rate:.2f}%")
    print(f"    Profit Factor: {profit_factor:.2f}")
    print(f"    Tổng Fee: {total_fees:.4f}u")
    
    print(f"\n{Fore.BLUE}📊 PHÂN TÍCH CHI TIẾT:")
    print(f"    Avg Win: {Fore.GREEN}{avg_win:+.4f}u{Fore.BLUE}")
    print(f"    Avg Loss: {Fore.RED}{avg_loss:+.4f}u{Fore.BLUE}")
    print(f"    Max Win: {Fore.GREEN}{max_win:+.4f}u{Fore.BLUE}")
    print(f"    Max Loss: {Fore.RED}{max_loss:+.4f}u{Fore.BLUE}")
    
    print(f"\n{Fore.CYAN}📊 THỐNG KÊ THEO LOẠI:")
    print(f"    LONG: {len(long_trades)} lệnh | Win: {long_wins} | WR: {long_wins/len(long_trades)*100 if long_trades else 0:.1f}% | PNL: {long_pnl:+.4f}u")
    print(f"    SHORT: {len(short_trades)} lệnh | Win: {short_wins} | WR: {short_wins/len(short_trades)*100 if short_trades else 0:.1f}% | PNL: {short_pnl:+.4f}u")
    
    print(f"\n{Fore.YELLOW}🔔 THỐNG KÊ ĐÓNG LỆNH:")
    print(f"    Trailing Stop: {trailing_closes} lệnh")
    print(f"    Stop Loss: {stoploss_closes} lệnh")
    print(f"    Khác (End backtest): {total_trades - trailing_closes - stoploss_closes} lệnh")
    
    print(f"\n{Fore.BLUE}🪙 THỐNG KÊ THEO COIN:")
    for symbol, trader in sorted(traders.items(), key=lambda x: x[1].total_pnl, reverse=True):
        if trader.trades_count > 0:
            coin_win_rate = (trader.win_count / trader.trades_count * 100)
            pnl_color = Fore.GREEN if trader.total_pnl >= 0 else Fore.RED
            print(f"    [{symbol}] Trades: {trader.trades_count} | W: {trader.win_count} | L: {trader.loss_count} | "
                  f"WR: {coin_win_rate:.1f}% | PNL: {pnl_color}{trader.total_pnl:+.4f}u{Fore.BLUE}")
    
    print(f"\n{Style.RESET_ALL}")
    
    # Lưu file báo cáo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_txt_file = os.path.join(report_dir, f"backtest_report_{timestamp}.txt")
    report_csv_file = os.path.join(report_dir, f"backtest_trades_{timestamp}.csv")
    
    # Ghi file TXT
    try:
        with open(report_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("       📊 BÁO CÁO BACKTEST: BB(20) + RSI(9) + EMA(200) + Divergence\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("⏱️ THÔNG TIN BACKTEST:\n")
            f.write(f"    Khoảng thời gian: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"    Số ngày: {BACKTEST_DAYS}\n")
            f.write(f"    Khung thời gian: {TIMEFRAME}\n")
            f.write(f"    Số coins: {len(traders)}\n\n")
            
            f.write("💰 THỐNG KÊ TÀI KHOẢN:\n")
            f.write(f"    Vốn ban đầu: {INIT_CAPITAL:.2f}u\n")
            f.write(f"    Balance cuối: {final_balance:.2f}u\n")
            f.write(f"    PNL: {balance_pnl:+.2f}u\n")
            f.write(f"    ROI: {roi:+.2f}%\n")
            f.write(f"    Max Drawdown: {max_drawdown*100:.2f}%\n\n")
            
            f.write("📈 THỐNG KÊ GIAO DỊCH:\n")
            f.write(f"    Tổng số lệnh: {total_trades}\n")
            f.write(f"    Số lệnh thắng: {total_wins}\n")
            f.write(f"    Số lệnh thua: {total_losses}\n")
            f.write(f"    Win Rate: {win_rate:.2f}%\n")
            f.write(f"    Profit Factor: {profit_factor:.2f}\n")
            f.write(f"    Tổng Fee: {total_fees:.4f}u\n\n")
            
            f.write("📊 PHÂN TÍCH CHI TIẾT:\n")
            f.write(f"    Avg Win: {avg_win:+.4f}u\n")
            f.write(f"    Avg Loss: {avg_loss:+.4f}u\n")
            f.write(f"    Max Win: {max_win:+.4f}u\n")
            f.write(f"    Max Loss: {max_loss:+.4f}u\n\n")
            
            f.write("📊 THỐNG KÊ THEO LOẠI:\n")
            f.write(f"    LONG: {len(long_trades)} lệnh | Win: {long_wins} | WR: {long_wins/len(long_trades)*100 if long_trades else 0:.1f}% | PNL: {long_pnl:+.4f}u\n")
            f.write(f"    SHORT: {len(short_trades)} lệnh | Win: {short_wins} | WR: {short_wins/len(short_trades)*100 if short_trades else 0:.1f}% | PNL: {short_pnl:+.4f}u\n\n")
            
            f.write("🔔 THỐNG KÊ ĐÓNG LỆNH:\n")
            f.write(f"    Trailing Stop: {trailing_closes} lệnh\n")
            f.write(f"    Stop Loss: {stoploss_closes} lệnh\n")
            f.write(f"    Khác: {total_trades - trailing_closes - stoploss_closes} lệnh\n\n")
            
            f.write("🪙 THỐNG KÊ THEO COIN:\n")
            for symbol, trader in sorted(traders.items(), key=lambda x: x[1].total_pnl, reverse=True):
                if trader.trades_count > 0:
                    coin_win_rate = (trader.win_count / trader.trades_count * 100)
                    f.write(f"    [{symbol}] Trades: {trader.trades_count} | W: {trader.win_count} | L: {trader.loss_count} | "
                            f"WR: {coin_win_rate:.1f}% | PNL: {trader.total_pnl:+.4f}u\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Chi tiết giao dịch xem file CSV\n")
        
        print(f"{Fore.GREEN}✅ Đã lưu báo cáo: {report_txt_file}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Lỗi ghi file TXT: {e}{Style.RESET_ALL}")
    
    # Ghi file CSV
    try:
        if all_trades:
            df = pd.DataFrame(all_trades)
            df = df.sort_values('entry_time')
            df.to_csv(report_csv_file, index=False, encoding='utf-8-sig')
            print(f"{Fore.GREEN}✅ Đã lưu lịch sử giao dịch: {report_csv_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Không có giao dịch nào để lưu{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Lỗi ghi file CSV: {e}{Style.RESET_ALL}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'roi': roi,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }


if __name__ == "__main__":
    print(f"\n{Fore.YELLOW}╔══════════════════════════════════════════════════════════════╗")
    print(f"║        BACKTEST ENGINE - BB + RSI + EMA + Divergence         ║")
    print(f"╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    run_full_backtest()
    
    print(f"\n{Fore.GREEN}✅ Backtest hoàn tất!{Style.RESET_ALL}")
