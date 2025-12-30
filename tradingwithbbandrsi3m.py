import websocket
import json
import pandas as pd
import numpy as np
import requests
import threading
import signal
import sys
import time
import os
from datetime import datetime
from colorama import Fore, Style, init

# Init màu console
init(autoreset=True)

# --- CẤU HÌNH GLOBAL ---
SYMBOLS = ["tradoorusdt","beatusdt","basusdt","husdt", "jellyjellyusdt","mmtusdt", "grassusdt", "1000pepeusdt",  "1000bonkusdt", "aaveusdt", "giggleusdt", "atomusdt", "galausdt", "aptusdt", "trustusdt", "opusdt", "injusdt", "pythusdt",  "zkusdt", "avaxusdt", "arbusdt", "ybusdt", "linkusdt", "zorausdt", "solusdt", "arusdt", "bchusdt", "ethusdt", "nearusdt", "dogeusdt", "adausdt", "1000flokiusdt", "xrpusdt", "etcusdt", "1000shibusdt", "btcusdt", "xplusdt", "ltcusdt", "tonusdt", "bnbusdt", "trxusdt"]
TIMEFRAME = '3m'

# Vốn & Risk Management
INIT_CAPITAL = 200.0   # Tổng vốn
LEVERAGE = 20
MARGIN_PER_ORDER = 5 # Margin gốc (Chưa nhân bẫy)
ORDER_SIZE_USDT = MARGIN_PER_ORDER * LEVERAGE # Volume (10u)
FEE_RATE = 0.05 / 100
STOP_LOSS_USDT = 100  # Stop Loss: Đóng lệnh khi lỗ >= 50u

# Cấu hình Chỉ báo
BB_PERIOD = 20    # Bollinger Bands Period
BB_STD = 2.5      # Độ lệch chuẩn
RSI_PERIOD = 9
STOCH_PERIOD = 14
STOCH_K = 3
STOCH_D = 3

# Trailing Stop (Tính theo % Giá chạy thực tế của coin - Chưa nhân bẫy)
TRAILING_TRIGGER_PCT = 0,5 / 100  # Giá chạy 0.5% -> Kích hoạt
TRAILING_CALLBACK_PCT = 0.1 / 100 # Giá tụt 0.2% từ đỉnh -> Chốt

# Log interval (seconds)
PNL_LOG_INTERVAL = 60  # 1 phút

# Biến lưu giá hiện tại
current_prices = {}

# --- CLASS QUẢN LÝ VỐN ---
class Portfolio:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.lock = threading.Lock()

    def update_balance(self, amount):
        with self.lock:
            self.balance += amount

    def get_balance(self):
        with self.lock:
            return self.balance

# --- CLASS TRADER CHO TỪNG COIN ---
class SymbolTrader:
    def __init__(self, symbol, portfolio):
        self.symbol = symbol
        self.portfolio = portfolio
        self.df = pd.DataFrame()
        self.position = {
            'type': None, 
            'entry_price': 0.0, 
            'size': 0.0, 
            'margin': 0.0,
            'highest_price_move': -999.0
        }
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.trades_history = []  # Lưu lịch sử giao dịch
        # Random màu để dễ phân biệt các coin
        self.log_color = np.random.choice([Fore.CYAN, Fore.YELLOW, Fore.MAGENTA, Fore.BLUE, Fore.LIGHTGREEN_EX])
        
        self.fetch_historical_data()

    def get_time(self):
        # Hàm lấy thời gian hiện tại
        return datetime.now().strftime("%H:%M:%S %d/%m")

    def log(self, msg):
        print(f"{self.log_color}[{self.symbol}] {msg}{Style.RESET_ALL}")

    def fetch_historical_data(self):
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval={TIMEFRAME}&limit=100"
            res = requests.get(url).json()
            data = []
            for k in res:
                data.append({
                    'time': datetime.fromtimestamp(k[0]/1000),
                    'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
                })
            self.df = pd.DataFrame(data)
            self.update_indicators()
            self.log(f"Sẵn sàng. (Data: {len(self.df)} nến)")
        except Exception as e:
            self.log(f"{Fore.RED}Lỗi tải history: {e}")

    def update_indicators(self):
        self.df['tp'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['ma'] = self.df['tp'].rolling(window=BB_PERIOD).mean()
        self.df['std'] = self.df['tp'].rolling(window=BB_PERIOD).std()
        self.df['upper_bb'] = self.df['ma'] + (self.df['std'] * BB_STD)
        self.df['lower_bb'] = self.df['ma'] - (self.df['std'] * BB_STD)

        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        min_rsi = self.df['rsi'].rolling(window=STOCH_PERIOD).min()
        max_rsi = self.df['rsi'].rolling(window=STOCH_PERIOD).max()
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / (max_rsi - min_rsi) * 100
        self.df['k'] = self.df['stoch_rsi'].rolling(window=STOCH_K).mean()
        self.df['d'] = self.df['k'].rolling(window=STOCH_D).mean()

    def process_tick(self, close_price):
        if self.position['type'] is None: return

        # 1. Tính % Giá chạy (Unleveraged Price Move)
        price_move_pct = 0.0
        if self.position['type'] == 'LONG':
            price_move_pct = (close_price - self.position['entry_price']) / self.position['entry_price']
        elif self.position['type'] == 'SHORT':
            price_move_pct = (self.position['entry_price'] - close_price) / self.position['entry_price']

        # 2. Update đỉnh
        if price_move_pct > self.position['highest_price_move']:
            self.position['highest_price_move'] = price_move_pct

        # 3. Log PnL Realtime nếu lãi > 0.1% giá (để đỡ spam console)
        # if price_move_pct > 0.001:
        #    real_pnl = self.calc_unrealized_pnl(close_price)
        #    print(f"\r[{self.symbol}] Move: {price_move_pct*100:.2f}% | PnL: {real_pnl:.4f}u", end="")

        # 4. Check Stop Loss (50u)
        unrealized_pnl = self.calc_unrealized_pnl(close_price)
        if unrealized_pnl <= -STOP_LOSS_USDT:
            reason = f"STOP LOSS (Lỗ: {unrealized_pnl:.2f}u >= {STOP_LOSS_USDT}u)"
            self.close_position(close_price, reason)
            return

        # 5. Check Trailing Stop
        if self.position['highest_price_move'] >= TRAILING_TRIGGER_PCT:
            drawdown = self.position['highest_price_move'] - price_move_pct
            if drawdown >= TRAILING_CALLBACK_PCT:
                reason = f"Trailing Stop (Đỉnh: {self.position['highest_price_move']*100:.2f}%)"
                self.close_position(close_price, reason)

    def process_candle_close(self, candle):
        new_row = pd.DataFrame([candle])
        self.df = pd.concat([self.df, new_row], ignore_index=True).tail(100)
        self.update_indicators()
        last_row = self.df.iloc[-1]
        
        # Logic Nến & Pinbar
        is_green = candle['close'] > candle['open']
        is_red = candle['close'] < candle['open']
        total_size = candle['high'] - candle['low'] if (candle['high'] - candle['low']) > 0 else 0.00001
        
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        
        is_bullish_pinbar = (lower_wick / total_size) > 0.6
        is_bearish_pinbar = (upper_wick / total_size) > 0.6

        # Rule A: LONG
        l_cond1 = candle['low'] <= last_row['lower_bb']
        l_cond2 = last_row['k'] < 25 and last_row['d'] < 25 and last_row['k'] > last_row['d']
        l_cond3 = is_bullish_pinbar or (is_green and candle['close'] > last_row['lower_bb'])
        
        if l_cond1 and l_cond2 and l_cond3:
            self.execute_trade('LONG', candle['close'])
            return

        # Rule B: SHORT
        s_cond1 = candle['high'] >= last_row['upper_bb']
        s_cond2 = last_row['k'] > 75 and last_row['d'] > 75 and last_row['k'] < last_row['d']
        s_cond3 = is_bearish_pinbar or (is_red and candle['close'] < last_row['upper_bb'])

        if s_cond1 and s_cond2 and s_cond3:
            self.execute_trade('SHORT', candle['close'])

    def execute_trade(self, signal, price):
        quantity = ORDER_SIZE_USDT / price
        fee = ORDER_SIZE_USDT * FEE_RATE
        time_now = self.get_time()

        # --- LOGIC NHỒI LỆNH (DCA) ---
        if self.position['type'] is not None:
            current_pnl = self.calc_unrealized_pnl(price)
            # Chỉ nhồi nếu đang Lỗ và đúng tín hiệu
            if current_pnl < 0 and self.position['type'] == signal:
                self.log(f"{Fore.RED}>>> [{time_now}] NHỒI LỆNH {signal} (Giá: {price})")
                
                new_size = self.position['size'] + quantity
                new_margin = self.position['margin'] + MARGIN_PER_ORDER
                avg_entry = ((self.position['entry_price'] * self.position['size']) + (price * quantity)) / new_size
                
                self.position['entry_price'] = avg_entry
                self.position['size'] = new_size
                self.position['margin'] = new_margin
                self.portfolio.update_balance(-fee)
                
                print(f"    Avg Entry Mới: {avg_entry:.4f} | Margin Tổng: {new_margin}u")
            return

        # --- LOGIC MỞ LỆNH MỚI ---
        self.log(f"{Fore.GREEN}>>> [{time_now}] VÀO LỆNH {signal} (Giá: {price})")
        self.position = {
            'type': signal, 'entry_price': price, 'size': quantity, 
            'margin': MARGIN_PER_ORDER, 'highest_price_move': -999.0
        }
        self.portfolio.update_balance(-fee)
        print(f"    Vol: {ORDER_SIZE_USDT}u (Leverage x{LEVERAGE}) | Balance: {self.portfolio.get_balance():.2f}u")

    def close_position(self, price, reason):
        time_now = self.get_time()
        pnl = self.calc_unrealized_pnl(price)
        fee = (self.position['size'] * price) * FEE_RATE
        realized_pnl = pnl - fee
        
        self.portfolio.update_balance(realized_pnl)
        
        self.trades_count += 1
        self.total_pnl += realized_pnl
        if realized_pnl > 0: 
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Lưu lịch sử giao dịch
        self.trades_history.append({
            'time': time_now,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'size': self.position['size'],
            'margin': self.position['margin'],
            'pnl': realized_pnl,
            'fee': fee,
            'reason': reason,
            'balance_after': self.portfolio.get_balance()
        })
        
        color = Fore.GREEN if realized_pnl > 0 else Fore.RED
        print(f"\n{self.log_color}=========================================")
        print(f"[{self.symbol}] [{time_now}] ĐÓNG LỆNH ({reason})")
        print(f"PnL: {color}{realized_pnl:.4f} USDT{self.log_color} (Fee: {fee:.4f})")
        print(f"Tổng Balance: {self.portfolio.get_balance():.2f} USDT")
        print(f"========================================={Style.RESET_ALL}")
        
        self.position = {'type': None, 'entry_price': 0, 'size': 0, 'margin': 0, 'highest_price_move': -999}

    def calc_unrealized_pnl(self, current_price):
        if self.position['type'] == 'LONG':
            return (current_price - self.position['entry_price']) * self.position['size']
        elif self.position['type'] == 'SHORT':
            return (self.position['entry_price'] - current_price) * self.position['size']
        return 0.0

# --- MAIN RUN ---
portfolio = Portfolio(INIT_CAPITAL)
traders = {}
start_time = datetime.now()
pnl_log_running = True

print(f"{Fore.YELLOW}=== KHỞI TẠO MULTI-TOKEN BOT (BB 30, STD 2.5) ==={Style.RESET_ALL}")
for s in SYMBOLS:
    traders[s] = SymbolTrader(s, portfolio)

def log_pnl_positions():
    """Log PNL của các vị thế đang mở và balance mỗi phút"""
    global pnl_log_running
    while pnl_log_running:
        try:
            time.sleep(PNL_LOG_INTERVAL)
            if not pnl_log_running:
                break
            
            time_now = datetime.now().strftime("%H:%M:%S %d/%m")
            total_unrealized_pnl = 0.0
            open_positions = []
            
            for symbol, trader in traders.items():
                if trader.position['type'] is not None:
                    # Lấy giá hiện tại từ current_prices
                    current_price = current_prices.get(symbol, trader.position['entry_price'])
                    unrealized = trader.calc_unrealized_pnl(current_price)
                    total_unrealized_pnl += unrealized
                    open_positions.append({
                        'symbol': symbol,
                        'type': trader.position['type'],
                        'entry': trader.position['entry_price'],
                        'current': current_price,
                        'pnl': unrealized,
                        'margin': trader.position['margin']
                    })
            
            # Tính balance PNL
            current_balance = portfolio.get_balance()
            balance_pnl = current_balance - INIT_CAPITAL
            
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"[{time_now}] 📊 BÁO CÁO PNL ĐỊNH KỲ")
            print(f"{'='*60}")
            print(f"💰 BALANCE: {current_balance:.2f}u | PNL Balance: {Fore.GREEN if balance_pnl >= 0 else Fore.RED}{balance_pnl:+.2f}u{Fore.CYAN}")
            
            if open_positions:
                print(f"\n📈 VỊ THẾ ĐANG MỞ ({len(open_positions)}):")
                for pos in open_positions:
                    pnl_color = Fore.GREEN if pos['pnl'] >= 0 else Fore.RED
                    print(f"   [{pos['symbol']}] {pos['type']} | Entry: {pos['entry']:.6f} | "
                          f"Current: {pos['current']:.6f} | PNL: {pnl_color}{pos['pnl']:+.4f}u{Fore.CYAN} | Margin: {pos['margin']}u")
                
                print(f"\n   📊 Tổng Unrealized PNL: {Fore.GREEN if total_unrealized_pnl >= 0 else Fore.RED}{total_unrealized_pnl:+.4f}u{Fore.CYAN}")
            else:
                print(f"\n   Không có vị thế nào đang mở")
            
            print(f"{'='*60}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Lỗi log PNL: {e}{Style.RESET_ALL}")

def generate_final_report():
    """Tạo báo cáo thống kê và xuất file khi kết thúc"""
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Thu thập thống kê
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    all_trades_history = []
    
    for symbol, trader in traders.items():
        total_trades += trader.trades_count
        total_wins += trader.win_count
        total_losses += trader.loss_count
        total_pnl += trader.total_pnl
        
        for trade in trader.trades_history:
            trade['symbol'] = symbol
            all_trades_history.append(trade)
    
    final_balance = portfolio.get_balance()
    balance_pnl = final_balance - INIT_CAPITAL
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    # Tạo tên file với timestamp
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_txt_file = os.path.join(report_dir, f"trading_report_{timestamp}.txt")
    report_csv_file = os.path.join(report_dir, f"trades_history_{timestamp}.csv")
    
    # In ra console
    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"{'='*70}")
    print(f"           📊 BÁO CÁO THỐNG KÊ CUỐI CÙNG")
    print(f"{'='*70}")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}⏱️  THỜI GIAN CHẠY:")
    print(f"    Bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Tổng thời gian: {duration}")
    
    print(f"\n{Fore.GREEN}💰 THỐNG KÊ TÀI KHOẢN:")
    print(f"    Vốn ban đầu: {INIT_CAPITAL:.2f}u")
    print(f"    Balance cuối: {final_balance:.2f}u")
    balance_color = Fore.GREEN if balance_pnl >= 0 else Fore.RED
    print(f"    PNL Balance: {balance_color}{balance_pnl:+.2f}u ({balance_pnl/INIT_CAPITAL*100:+.2f}%){Style.RESET_ALL}")
    
    print(f"\n{Fore.MAGENTA}📈 THỐNG KÊ GIAO DỊCH:")
    print(f"    Tổng số lệnh: {total_trades}")
    print(f"    Số lệnh thắng: {Fore.GREEN}{total_wins}{Fore.MAGENTA}")
    print(f"    Số lệnh thua: {Fore.RED}{total_losses}{Fore.MAGENTA}")
    print(f"    Win Rate: {win_rate:.2f}%")
    pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(f"    Tổng PNL: {pnl_color}{total_pnl:+.4f}u{Style.RESET_ALL}")
    
    # Thống kê theo từng coin
    print(f"\n{Fore.BLUE}🪙 THỐNG KÊ THEO COIN:")
    for symbol, trader in traders.items():
        if trader.trades_count > 0:
            coin_win_rate = (trader.win_count / trader.trades_count * 100) if trader.trades_count > 0 else 0
            pnl_color = Fore.GREEN if trader.total_pnl >= 0 else Fore.RED
            print(f"    [{symbol}] Trades: {trader.trades_count} | W: {trader.win_count} | L: {trader.loss_count} | "
                  f"WR: {coin_win_rate:.1f}% | PNL: {pnl_color}{trader.total_pnl:+.4f}u{Fore.BLUE}")
    
    print(f"\n{Style.RESET_ALL}")
    
    # Ghi file TXT
    try:
        with open(report_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("           📊 BÁO CÁO THỐNG KÊ GIAO DỊCH\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("⏱️ THỜI GIAN CHẠY:\n")
            f.write(f"    Bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"    Kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"    Tổng thời gian: {duration}\n\n")
            
            f.write("💰 THỐNG KÊ TÀI KHOẢN:\n")
            f.write(f"    Vốn ban đầu: {INIT_CAPITAL:.2f}u\n")
            f.write(f"    Balance cuối: {final_balance:.2f}u\n")
            f.write(f"    PNL Balance: {balance_pnl:+.2f}u ({balance_pnl/INIT_CAPITAL*100:+.2f}%)\n\n")
            
            f.write("📈 THỐNG KÊ GIAO DỊCH:\n")
            f.write(f"    Tổng số lệnh: {total_trades}\n")
            f.write(f"    Số lệnh thắng: {total_wins}\n")
            f.write(f"    Số lệnh thua: {total_losses}\n")
            f.write(f"    Win Rate: {win_rate:.2f}%\n")
            f.write(f"    Tổng PNL: {total_pnl:+.4f}u\n\n")
            
            f.write("🪙 THỐNG KÊ THEO COIN:\n")
            for symbol, trader in traders.items():
                if trader.trades_count > 0:
                    coin_win_rate = (trader.win_count / trader.trades_count * 100) if trader.trades_count > 0 else 0
                    f.write(f"    [{symbol}] Trades: {trader.trades_count} | W: {trader.win_count} | L: {trader.loss_count} | "
                            f"WR: {coin_win_rate:.1f}% | PNL: {trader.total_pnl:+.4f}u\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Chi tiết giao dịch xem file CSV\n")
        
        print(f"{Fore.GREEN}✅ Đã lưu báo cáo: {report_txt_file}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Lỗi ghi file TXT: {e}{Style.RESET_ALL}")
    
    # Ghi file CSV với lịch sử trades
    try:
        if all_trades_history:
            df = pd.DataFrame(all_trades_history)
            df.to_csv(report_csv_file, index=False, encoding='utf-8-sig')
            print(f"{Fore.GREEN}✅ Đã lưu lịch sử giao dịch: {report_csv_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Không có giao dịch nào để lưu{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Lỗi ghi file CSV: {e}{Style.RESET_ALL}")

def signal_handler(sig, frame):
    """Xử lý tín hiệu dừng chương trình"""
    global pnl_log_running
    print(f"\n{Fore.YELLOW}⚠️ Nhận tín hiệu dừng. Đang tạo báo cáo...{Style.RESET_ALL}")
    pnl_log_running = False
    generate_final_report()
    sys.exit(0)

def on_message(ws, message):
    try:
        msg = json.loads(message)
        data = msg['data']
        kline = data['k']
        symbol = kline['s']
        close_price = float(kline['c'])
        is_closed = kline['x']
        
        # Cập nhật giá hiện tại
        current_prices[symbol] = close_price

        if symbol in traders:
            trader = traders[symbol]
            # 1. Update Tick (Trailing Stop + Stop Loss)
            trader.process_tick(close_price)
            # 2. Update Candle (Entry)
            if is_closed:
                new_candle = {
                    'time': datetime.fromtimestamp(kline['T']/1000),
                    'open': float(kline['o']), 'high': float(kline['h']), 
                    'low': float(kline['l']), 'close': float(kline['c']), 
                    'volume': float(kline['v'])
                }
                trader.process_candle_close(new_candle)
    except:
        pass

def on_open(ws):
    print(f"{Fore.GREEN}>> Đã kết nối WebSocket! Đang theo dõi {len(SYMBOLS)} coins...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📊 Log PNL định kỳ: Mỗi {PNL_LOG_INTERVAL}s | Stop Loss mỗi lệnh: {STOP_LOSS_USDT}u{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}💡 Nhấn Ctrl+C để dừng và xem báo cáo thống kê{Style.RESET_ALL}")

if __name__ == "__main__":
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Bắt đầu thread log PNL định kỳ
    pnl_thread = threading.Thread(target=log_pnl_positions, daemon=True)
    pnl_thread.start()
    
    stream_list = [f"{s.lower()}@kline_{TIMEFRAME}" for s in SYMBOLS]
    socket_url = f"wss://fstream.binance.com/stream?streams={'/'.join(stream_list)}"
    ws = websocket.WebSocketApp(socket_url, on_open=on_open, on_message=on_message)
    
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        signal_handler(None, None)