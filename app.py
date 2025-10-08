from flask import Flask, render_template
import pandas as pd
import yfinance as yf
from functools import lru_cache
import numpy as np

# 建立 Flask 網站應用程式
app = Flask(__name__)

# --- 全域常數 ---
EXCEL_PATH = '2025_09.xlsx'

@lru_cache(maxsize=1)
def get_usd_twd_rate():
    """抓取最新的美元兌台幣匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        data = ticker.history(period="1d")
        return round(data['Close'].iloc[-1], 2) if not data.empty else 32.5
    except Exception:
        return 32.5

# --- 資料處理函式 (與前一版相同) ---

@lru_cache(maxsize=500)
def get_stock_info(ticker):
    """抓取股價和產業別，回傳一個字典"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
        sector = info.get('sector', 'N/A')
        return {'price': price, 'sector': sector}
    except Exception:
        return {'price': 0, 'sector': 'N/A'}

@lru_cache(maxsize=500)
def get_dividend_info(ticker):
    """抓取最近一次的現金股利和年化配息次數"""
    try:
        stock = yf.Ticker(ticker)
        # 抓取近一年的配息數據
        dividends = stock.dividends.last('365d')
        if dividends.empty:
            return {'last_dividend': 0, 'payouts_per_year': 0}
        
        last_dividend = dividends.iloc[-1]
        payouts_per_year = len(dividends)
        return {'last_dividend': last_dividend, 'payouts_per_year': payouts_per_year}
    except Exception:
        return {'last_dividend': 0, 'payouts_per_year': 0}

def get_tw_holdings_details():
    """讀取 Excel 檔案，獲取高股息持股的詳細清單 (返回原始數值)"""
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name='高股息')

        required_cols = ['股號', '目前股價', '持股', '市值']
        if not all(col in df.columns for col in required_cols):
            return []

        df['股號'] = df['股號'].astype(str)
        for col in ['市值', '持股', '目前股價']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        holdings = []
        for index, row in df.iterrows():
            if row['市值'] == 0:
                continue
            
            dividend_info = get_dividend_info(row['股號'])
            
            annual_income = dividend_info['last_dividend'] * row['持股'] * dividend_info['payouts_per_year']
            
            holdings.append({
                'ticker': row['股號'],
                'price': row['目前股價'],
                'shares': row['持股'],
                'market_value': row['市值'],
                'last_dividend': dividend_info['last_dividend'],
                'payouts_per_year': dividend_info['payouts_per_year'],
                'monthly_income': annual_income / 12
            })
        return holdings
    except Exception as e:
        import traceback
        print(f"處理台股詳細持股時發生預期外的錯誤: {e}")
        print(traceback.format_exc())
        return []

# --- Flask 的核心路由 (Route) ---
@app.route('/')
def dashboard():
    """網站首頁的處理函式"""
    try:
        # --- 讀取資料 ---
        broker_report_path = '複委託庫存20251006174326.csv'
        df_us = pd.read_csv(broker_report_path)
        for col in ['均價', '目前庫存']:
            if df_us[col].dtype == 'object':
                df_us[col] = df_us[col].str.replace(',', '', regex=False).astype(float)
        df_us = df_us[df_us['目前庫存'] > 0].copy()

        # --- 新增：合併相同股號的資料 ---
        # 先計算每筆的總成本
        df_us['總成本'] = df_us['均價'] * df_us['目前庫存']
        
        # 按股號分組並加總
        agg_funcs = {
            '目前庫存': 'sum',
            '總成本': 'sum',
            '股票名稱': 'first', # 保留第一個出現的名稱
        }
        df_us_agg = df_us.groupby('代號').agg(agg_funcs)
        
        # 計算加權平均均價
        df_us_agg['均價'] = df_us_agg['總成本'] / df_us_agg['目前庫存']
        
        # 重設索引，讓'代號'變回欄位
        df_us = df_us_agg.reset_index()
        # --- 合併資料結束 ---

        tw_holdings_raw = get_tw_holdings_details() # 取得台股原始數據

        # --- 處理美股資料 ---
        usd_rate = get_usd_twd_rate()
        holdings_data = []
        us_stock_total_value = 0
        for _, row in df_us.iterrows():
            ticker = row['代號']
            shares = row['目前庫存']
            avg_cost = row['均價']
            stock_info = get_stock_info(ticker)
            current_price = stock_info['price']
            market_value_twd = (current_price * shares) * usd_rate
            profit_loss_twd = (market_value_twd - (avg_cost * shares * usd_rate))
            holdings_data.append({
                'ticker': ticker,
                'sector': stock_info['sector'],
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value_twd': market_value_twd,
                'profit_loss_twd': profit_loss_twd
            })
            us_stock_total_value += market_value_twd

        # --- 處理台股資料 ---
        tw_stock_total_value = sum(item['market_value'] for item in tw_holdings_raw)

        # --- 計算總覽數據 ---
        total_profit_loss_twd = sum(item['profit_loss_twd'] for item in holdings_data)
        grand_total_value = us_stock_total_value + tw_stock_total_value
        summary = {
            'total_market_value': f"{grand_total_value:,.0f}",
            'total_profit_loss': f"{total_profit_loss_twd:,.0f}",
            'usd_rate': usd_rate
        }
        
        # --- 準備前端圖表資料 ---
        asset_allocation_data = {
            'labels': ['美股', '台股高股息'],
            'values': [us_stock_total_value, tw_stock_total_value]
        }
        us_stock_allocation = {
            'labels': [item['ticker'] for item in holdings_data],
            'values': [item['market_value_twd'] for item in holdings_data]
        }
        tw_stock_allocation = {
            'labels': [item['ticker'] for item in tw_holdings_raw],
            'values': [item['market_value'] for item in tw_holdings_raw]
        }

        # --- 格式化前端表格資料 ---
        holdings_data_formatted = [
            {
                **item, # 解構 item 字典
                'avg_cost': f"{item['avg_cost']:,.2f}",
                'current_price': f"{item['current_price']:,.2f}",
                'position_percentage': f"{(item['market_value_twd'] / us_stock_total_value * 100) if us_stock_total_value > 0 else 0:.2f}%",
                'market_value_twd': f"{item['market_value_twd']:,.0f}",
                'profit_loss_twd': f"{item['profit_loss_twd']:,.2f}",
                'roi': f"{(item['profit_loss_twd'] / (item['avg_cost'] * item['shares'] * usd_rate) * 100) if item['avg_cost'] > 0 else 0:.2f}%"
            } for item in holdings_data
        ]
        tw_holdings_formatted = [
            {
                **item,
                'price': f"{item['price']:,.2f}",
                'shares': f"{item['shares']:,.0f}",
                'market_value': f"{item['market_value']:,.0f}",
                'last_dividend': f"{item['last_dividend']:,.4f}",
                'payouts_per_year': item['payouts_per_year'],
                'monthly_income': f"{item['monthly_income']:,.0f}"
            } for item in tw_holdings_raw
        ]

        # --- 最終渲染 ---
        return render_template('index.html', 
                               holdings=holdings_data_formatted, 
                               summary=summary,
                               asset_allocation=asset_allocation_data,
                               tw_holdings=tw_holdings_formatted,
                               us_stock_allocation=us_stock_allocation,
                               tw_stock_allocation=tw_stock_allocation)

    except FileNotFoundError as e:
        return f"錯誤：找不到資料檔案。請確認 '{e.filename}' 存在。", 500
    except Exception as e:
        import traceback
        return f"處理資料或渲染網頁時發生預期外的錯誤: {e}<br><pre>{traceback.format_exc()}</pre>", 500

# --- 讓網站跑起來 ---
if __name__ == '__main__':
    app.run(debug=True)