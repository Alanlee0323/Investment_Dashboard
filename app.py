
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import yfinance as yf
from functools import lru_cache
import numpy as np
import os
import io
import logging

# 建立 Flask 網站應用程式
app = Flask(__name__)

# --- 設定日誌 (Logging) ---
# 建立一個日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 建立一個檔案處理器，將日誌寫入 app.log
# mode='a' 表示附加模式，'utf-8' 確保能處理中文
file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 建立一個控制台處理器，將日誌輸出到控制台
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# 定義日誌格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# 將處理器加入到日誌記錄器中
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Flask 應用程式啟動")


# 設定 Session 的密鑰。在生產環境中，這應該是一個更複雜且來自環境變數的值。
app.secret_key = os.urandom(24)

# --- 外部 API 和資料處理函式 ---

@lru_cache(maxsize=1)
def get_usd_twd_rate():
    """抓取最新的美元兌台幣匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        data = ticker.history(period="1d")
        return round(data['Close'].iloc[-1], 2) if not data.empty else 32.5
    except Exception:
        return 32.5

@lru_cache(maxsize=500)
def get_stock_info(ticker):
    """抓取股價和產業別"""
    logger.info(f"嘗試抓取股票資訊: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
        sector = info.get('sector', 'ETF')
        
        if price is None or price == 0:
            logger.warning(f"無法找到 {ticker} 的股價。yfinance 回傳的 info: {info}")
            return {'price': 0, 'sector': 'N/A'}

        logger.info(f"成功抓取 {ticker} 資訊: 價格={price}, 產業={sector}")
        return {'price': price, 'sector': sector}
    except Exception as e:
        logger.error(f"抓取 {ticker} 資訊時發生嚴重錯誤: {e}", exc_info=True)
        return {'price': 0, 'sector': 'N/A'}

@lru_cache(maxsize=500)
def get_dividend_info(ticker):
    """抓取最近一次的現金股利和年化配息次數"""
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends.last('365d')
        if dividends.empty:
            return {'last_dividend': 0, 'payouts_per_year': 0}
        last_dividend = dividends.iloc[-1]
        payouts_per_year = len(dividends)
        return {'last_dividend': last_dividend, 'payouts_per_year': payouts_per_year}
    except Exception:
        return {'last_dividend': 0, 'payouts_per_year': 0}

def process_data_files(us_stock_file, tw_stock_file):
    """
    核心處理函式：接收上傳的檔案內容，進行所有計算，並返回一個包含所有頁面所需資料的字典。
    """
    # --- 讀取資料 ---
    df_us = pd.read_csv(us_stock_file)
    df_tw = pd.read_excel(tw_stock_file, sheet_name='高股息')

    # --- 處理美股資料 ---
    for col in ['均價', '目前庫存']:
        if df_us[col].dtype == 'object':
            df_us[col] = df_us[col].str.replace(',', '', regex=False).astype(float)
    df_us = df_us[df_us['目前庫存'] > 0].copy()

    df_us['總成本'] = df_us['均價'] * df_us['目前庫存']
    agg_funcs = {'目前庫存': 'sum', '總成本': 'sum', '股票名稱': 'first'}
    df_us_agg = df_us.groupby('代號').agg(agg_funcs)
    df_us_agg['均價'] = df_us_agg['總成本'] / df_us_agg['目前庫存']
    df_us = df_us_agg.reset_index()

    # --- 全面批次抓取 (美股 + 台股) ---
    us_tickers = df_us['代號'].tolist()
    tw_tickers = [str(t) for t in df_tw['股號'].dropna().unique()]
    all_tickers = list(set(us_tickers + tw_tickers))

    all_stock_info = {}
    all_dividend_info = {}

    if all_tickers:
        logger.info(f"準備全面批次抓取 {len(all_tickers)} 支股票的資料: {all_tickers}")
        try:
            tickers_data = yf.Tickers(' '.join(all_tickers))
            
            for ticker_symbol in all_tickers:
                ticker_obj = tickers_data.tickers.get(ticker_symbol)
                if not ticker_obj:
                    logger.warning(f"找不到 {ticker_symbol} 的 Ticker 物件")
                    continue

                # 抓取價格和產業別 (主要為美股)
                info = ticker_obj.info
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                if not price:
                    logger.warning(f"批次抓取中，找不到 {ticker_symbol} 的股價。yfinance 回傳: {info}")
                    price = 0
                all_stock_info[ticker_symbol] = {
                    'price': price,
                    'sector': info.get('sector', 'N/A')
                }

                # 抓取股息資訊 (主要為台股)
                dividends = ticker_obj.dividends.last('365d')
                if dividends.empty:
                    all_dividend_info[ticker_symbol] = {'last_dividend': 0, 'payouts_per_year': 0}
                else:
                    all_dividend_info[ticker_symbol] = {
                        'last_dividend': dividends.iloc[-1],
                        'payouts_per_year': len(dividends)
                    }

            logger.info("全面批次抓取資料成功")

        except Exception as e:
            logger.error(f"全面批次抓取資料時發生嚴重錯誤: {e}", exc_info=True)
            # 如果批次抓取失敗，則不進行任何資料處理，返回空值
            all_stock_info = {}
            all_dividend_info = {}

    # --- 處理美股資料 ---
    usd_rate = get_usd_twd_rate()
    holdings_data = []
    us_stock_total_value = 0
    for _, row in df_us.iterrows():
        ticker = row['代號']
        shares = row['目前庫存']
        avg_cost = row['均價']
        
        stock_info = all_stock_info.get(ticker, {'price': 0, 'sector': 'N/A'})
        current_price = stock_info['price']
        
        market_value_twd = (current_price * shares) * usd_rate
        profit_loss_twd = (market_value_twd - (avg_cost * shares * usd_rate))
        raw_roi = (profit_loss_twd / (avg_cost * shares * usd_rate) * 100) if avg_cost > 0 else 0
        
        holdings_data.append({
            'ticker': ticker, 'sector': stock_info['sector'], 'shares': shares,
            'avg_cost': avg_cost, 'current_price': current_price,
            'market_value_twd': market_value_twd, 'profit_loss_twd': profit_loss_twd,
            'raw_roi': raw_roi,
        })
        us_stock_total_value += market_value_twd

    for item in holdings_data:
        item['raw_position_percentage'] = (item['market_value_twd'] / us_stock_total_value * 100) if us_stock_total_value > 0 else 0

    # --- 處理台股資料 ---
    tw_holdings_raw = []
    for index, row in df_tw.iterrows():
        if pd.to_numeric(row.get('市值', 0), errors='coerce') == 0:
            continue
        
        ticker_str = str(row['股號'])
        dividend_info = all_dividend_info.get(ticker_str, {'last_dividend': 0, 'payouts_per_year': 0})
        
        annual_income = dividend_info['last_dividend'] * row['持股'] * dividend_info['payouts_per_year']
        tw_holdings_raw.append({
            'ticker': str(row['股號']), 'price': row['目前股價'], 'shares': row['持股'],
            'market_value': row['市值'], 'last_dividend': dividend_info['last_dividend'],
            'payouts_per_year': dividend_info['payouts_per_year'],
            'monthly_income': annual_income / 12
        })

    tw_stock_total_value = sum(item['market_value'] for item in tw_holdings_raw)
    tw_total_monthly_income = sum(item['monthly_income'] for item in tw_holdings_raw)

    # --- 計算總覽數據 ---
    total_profit_loss_twd = sum(item['profit_loss_twd'] for item in holdings_data)
    grand_total_value = us_stock_total_value + tw_stock_total_value
    summary = {
        'total_market_value': f"{grand_total_value:,.0f}",
        'total_profit_loss': f"{total_profit_loss_twd:,.0f}",
        'usd_rate': usd_rate,
        'monthly_cash_flow': f"NT$ {tw_total_monthly_income:,.0f}"
    }
    
    # --- 準備圖表資料 ---
    asset_allocation_data = {'labels': ['美股', '台股高股息'], 'values': [us_stock_total_value, tw_stock_total_value]}
    us_stock_allocation = {'labels': [item['ticker'] for item in holdings_data], 'values': [item['market_value_twd'] for item in holdings_data]}
    tw_stock_allocation = {'labels': [item['ticker'] for item in tw_holdings_raw], 'values': [item['market_value'] for item in tw_holdings_raw]}

    # --- 格式化表格資料 ---
    holdings_data_formatted = [{**item, 'avg_cost': f"{item['avg_cost']:,.2f}", 'current_price': f"{item['current_price']:,.2f}", 'position_percentage': f"{item['raw_position_percentage']:.2f}%", 'market_value_twd': f"{item['market_value_twd']:,.0f}", 'profit_loss_twd': f"{item['profit_loss_twd']:,.2f}", 'roi': f"{item['raw_roi']:.2f}%"} for item in holdings_data]
    tw_holdings_formatted = [{**item, 'price': f"{item['price']:,.2f}", 'shares': f"{item['shares']:,.0f}", 'market_value': f"{item['market_value']:,.0f}", 'last_dividend': f"{item['last_dividend']:,.4f}", 'monthly_income': f"{item['monthly_income']:,.0f}"} for item in tw_holdings_raw]

    return {
        "holdings": holdings_data_formatted,
        "summary": summary,
        "asset_allocation": asset_allocation_data,
        "tw_holdings": tw_holdings_formatted,
        "us_stock_allocation": us_stock_allocation,
        "tw_stock_allocation": tw_stock_allocation,
        "raw_holdings": holdings_data # 未格式化的原始數據，用於排序
    }

# --- Flask 的路由 (Routes) ---

@app.route('/')
def dashboard():
    """網站首頁：根據 session 決定顯示儀表板還是上傳頁面"""
    page_data = session.get('page_data', None)
    
    if not page_data:
        return render_template('index.html') # Session 中沒資料，顯示上傳頁面

    # --- 排序邏輯 ---
    sort_by = request.args.get('sort_by', 'market_value_twd')
    sort_order = request.args.get('sort_order', 'desc')
    reverse = (sort_order == 'desc')
    
    # 從 session 中取出未經格式化的原始數據進行排序
    holdings_to_sort = page_data['raw_holdings']
    
    sort_key_map = {
        'position_percentage': 'raw_position_percentage',
        'profit_loss_twd': 'profit_loss_twd',
        'roi': 'raw_roi',
        'market_value_twd': 'market_value_twd'
    }
    sort_key = sort_key_map.get(sort_by, 'market_value_twd')
    
    holdings_to_sort.sort(key=lambda x: x.get(sort_key, 0), reverse=reverse)
    
    # 排序後重新格式化
    page_data['holdings'] = [{**item, 'avg_cost': f"{item['avg_cost']:,.2f}", 'current_price': f"{item['current_price']:,.2f}", 'position_percentage': f"{item.get('raw_position_percentage', 0):.2f}%", 'market_value_twd': f"{item['market_value_twd']:,.0f}", 'profit_loss_twd': f"{item['profit_loss_twd']:,.2f}", 'roi': f"{item.get('raw_roi', 0):.2f}%"} for item in holdings_to_sort]

    return render_template('index.html', 
                           **page_data,
                           sort_by=sort_by,
                           sort_order=sort_order)

@app.route('/upload', methods=['POST'])
def upload_files():
    """處理檔案上傳，計算數據並存入 session"""
    logger.info("接收到檔案上傳請求")
    if 'us_stock_file' not in request.files or 'tw_stock_file' not in request.files:
        logger.error("上傳錯誤：請求中缺少 us_stock_file 或 tw_stock_file")
        return "錯誤：缺少檔案", 400
    
    us_file = request.files['us_stock_file']
    tw_file = request.files['tw_stock_file']

    if us_file.filename == '' or tw_file.filename == '':
        logger.error("上傳錯誤：使用者未選擇檔案")
        return "錯誤：未選擇檔案", 400

    logger.info(f"上傳的檔案: 美股='{us_file.filename}', 台股='{tw_file.filename}'")

    try:
        # 將檔案讀入記憶體中的 BytesIO 物件
        us_stock_content = io.BytesIO(us_file.read())
        tw_stock_content = io.BytesIO(tw_file.read())

        # 處理數據
        logger.info("開始進行資料處理...")
        page_data = process_data_files(us_stock_content, tw_stock_content)
        logger.info("資料處理完成")
        
        # 將處理好的數據存入 session
        session['page_data'] = page_data
        logger.info("處理完成的資料已存入 Session")
        
    except Exception as e:
        logger.error("處理上傳檔案時發生錯誤", exc_info=True)
        import traceback
        return f"處理上傳檔案時發生錯誤: {e}<br><pre>{traceback.format_exc()}</pre>", 500

    return redirect(url_for('dashboard'))

@app.route('/reset')
def reset_session():
    """清除 session 並返回首頁"""
    session.pop('page_data', None)
    return redirect(url_for('dashboard'))

# --- 讓網站跑起來 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
