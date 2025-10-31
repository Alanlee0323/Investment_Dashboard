from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import yfinance as yf
from functools import lru_cache
import numpy as np
import os
import io
import logging
import time
import pickle
from datetime import datetime, timedelta

# 建立 Flask 網站應用程式
app = Flask(__name__)

# --- 設定日誌 (Logging) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Flask 應用程式啟動")

# 設定 Session 的密鑰
app.secret_key = os.urandom(24)

# --- 快取設定 ---
CACHE_FILE = 'stock_cache.pkl'
CACHE_DURATION = timedelta(minutes=30)

def load_cache():
    """載入快取檔案"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"載入快取失敗: {e}")
            return {}
    return {}

def save_cache(cache):
    """儲存快取檔案"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.error(f"儲存快取失敗: {e}")

# --- 外部 API 和資料處理函式 ---

@lru_cache(maxsize=1)
def get_usd_twd_rate():
    """抓取最新的美元兌台幣匯率"""
    cache = load_cache()
    cache_key = "usd_twd_rate"
    
    if cache_key in cache:
        cached_time, cached_rate = cache[cache_key]
        if datetime.now() - cached_time < timedelta(hours=1):
            logger.info(f"使用匯率快取: {cached_rate}")
            return cached_rate
    
    try:
        ticker = yf.Ticker("USDTWD=X")
        data = ticker.history(period="1d")
        rate = round(data['Close'].iloc[-1], 2) if not data.empty else 32.5
        
        cache[cache_key] = (datetime.now(), rate)
        save_cache(cache)
        return rate
    except Exception as e:
        logger.error(f"抓取匯率時發生錯誤: {e}")
        return 32.5

def batch_fetch_with_cache_and_retry(tickers_list, batch_size=5, batch_delay=3, max_retries=3):
    """
    分批抓取股票資料，使用快取並帶重試機制
    
    Args:
        tickers_list: 股票代號列表
        batch_size: 每批處理的股票數量
        batch_delay: 每批之間的延遲秒數
        max_retries: 最大重試次數
    
    Returns:
        tuple: (all_stock_info, all_dividend_info)
    """
    cache = load_cache()
    all_stock_info = {}
    all_dividend_info = {}
    
    # 先從快取中載入
    uncached_tickers = []
    for ticker in tickers_list:
        stock_cache_key = f"stock_{ticker}"
        div_cache_key = f"div_{ticker}"
        
        # 檢查股價快取
        if stock_cache_key in cache:
            cached_time, cached_data = cache[stock_cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                all_stock_info[ticker] = cached_data
            else:
                uncached_tickers.append(ticker)
        else:
            uncached_tickers.append(ticker)
        
        # 檢查股息快取（即使股價有快取，股息也可能需要更新）
        if div_cache_key in cache:
            cached_time, cached_data = cache[div_cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                all_dividend_info[ticker] = cached_data
    
    if uncached_tickers:
        logger.info(f"從快取載入 {len(tickers_list) - len(uncached_tickers)} 支股票")
        logger.info(f"需重新抓取 {len(uncached_tickers)} 支股票: {uncached_tickers}")
    else:
        logger.info(f"全部 {len(tickers_list)} 支股票都從快取載入")
        return all_stock_info, all_dividend_info
    
    # 分批處理未快取的股票
    total_batches = (len(uncached_tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(uncached_tickers), batch_size):
        batch = uncached_tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"正在處理第 {batch_num}/{total_batches} 批，共 {len(batch)} 支股票: {batch}")
        
        # 重試機制
        for attempt in range(max_retries):
            try:
                tickers_data = yf.Tickers(' '.join(batch))
                
                for ticker_symbol in batch:
                    try:
                        ticker_obj = tickers_data.tickers.get(ticker_symbol)
                        if not ticker_obj:
                            logger.warning(f"找不到 {ticker_symbol} 的 Ticker 物件")
                            continue

                        # 抓取價格和產業別
                        info = ticker_obj.info
                        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                        if not price:
                            logger.warning(f"找不到 {ticker_symbol} 的股價")
                            price = 0
                        
                        stock_info = {
                            'price': price,
                            'sector': info.get('sector', 'N/A')
                        }
                        all_stock_info[ticker_symbol] = stock_info
                        
                        # 儲存股價到快取
                        cache[f"stock_{ticker_symbol}"] = (datetime.now(), stock_info)

                        # 抓取股息資訊
                        try:
                            dividends = ticker_obj.dividends.last('365d')
                            if dividends.empty:
                                div_info = {'last_dividend': 0, 'payouts_per_year': 0}
                            else:
                                div_info = {
                                    'last_dividend': dividends.iloc[-1],
                                    'payouts_per_year': len(dividends)
                                }
                            all_dividend_info[ticker_symbol] = div_info
                            
                            # 儲存股息到快取
                            cache[f"div_{ticker_symbol}"] = (datetime.now(), div_info)
                        except Exception as e:
                            logger.warning(f"抓取 {ticker_symbol} 股息時發生錯誤: {e}")
                            all_dividend_info[ticker_symbol] = {'last_dividend': 0, 'payouts_per_year': 0}

                    except Exception as e:
                        logger.error(f"處理 {ticker_symbol} 時發生錯誤: {e}")
                        all_stock_info[ticker_symbol] = {'price': 0, 'sector': 'N/A'}
                        all_dividend_info[ticker_symbol] = {'last_dividend': 0, 'payouts_per_year': 0}
                
                # 成功完成這批，儲存快取並跳出重試迴圈
                save_cache(cache)
                logger.info(f"第 {batch_num} 批處理成功")
                break
                
            except Exception as e:
                error_msg = str(e)
                if "Rate limit" in error_msg or "Too Many Requests" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = batch_delay * (2 ** attempt)  # 指數退避
                        logger.warning(f"第 {batch_num} 批遇到限流，等待 {wait_time} 秒後重試... (第 {attempt + 1}/{max_retries} 次)")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"第 {batch_num} 批達到最大重試次數，標記為失敗")
                        # 將這批股票標記為無資料
                        for ticker_symbol in batch:
                            if ticker_symbol not in all_stock_info:
                                all_stock_info[ticker_symbol] = {'price': 0, 'sector': 'N/A'}
                            if ticker_symbol not in all_dividend_info:
                                all_dividend_info[ticker_symbol] = {'last_dividend': 0, 'payouts_per_year': 0}
                else:
                    logger.error(f"第 {batch_num} 批發生非限流錯誤: {e}")
                    # 將這批股票標記為無資料
                    for ticker_symbol in batch:
                        if ticker_symbol not in all_stock_info:
                            all_stock_info[ticker_symbol] = {'price': 0, 'sector': 'N/A'}
                        if ticker_symbol not in all_dividend_info:
                            all_dividend_info[ticker_symbol] = {'last_dividend': 0, 'payouts_per_year': 0}
                    break
        
        # 每批之間延遲（除了最後一批）
        if i + batch_size < len(uncached_tickers):
            logger.info(f"等待 {batch_delay} 秒後處理下一批...")
            time.sleep(batch_delay)
    
    logger.info(f"批次抓取完成，共處理 {len(all_stock_info)} 支股票資訊和 {len(all_dividend_info)} 支股息資訊")
    return all_stock_info, all_dividend_info

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

    # --- 收集所有需要抓取的股票代號 ---
    us_tickers = df_us['代號'].tolist()
    tw_tickers = [str(t) for t in df_tw['股號'].dropna().unique() if pd.to_numeric(df_tw[df_tw['股號'] == t].get('市值', 0).iloc[0] if len(df_tw[df_tw['股號'] == t]) > 0 else 0, errors='coerce') != 0]
    all_tickers = list(set(us_tickers + tw_tickers))

    # --- 分批抓取所有股票資料（帶快取和重試） ---
    all_stock_info = {}
    all_dividend_info = {}

    if all_tickers:
        logger.info(f"準備抓取 {len(all_tickers)} 支股票的資料")
        logger.info(f"美股: {len(us_tickers)} 支, 台股: {len(tw_tickers)} 支")
        
        all_stock_info, all_dividend_info = batch_fetch_with_cache_and_retry(
            all_tickers, 
            batch_size=5,      # 每批 5 支股票
            batch_delay=3,     # 每批間隔 3 秒
            max_retries=3      # 最多重試 3 次
        )

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
            'ticker': ticker_str, 'price': row['目前股價'], 'shares': row['持股'],
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
        "raw_holdings": holdings_data
    }

# --- Flask 的路由 (Routes) ---

@app.route('/')
def dashboard():
    """網站首頁：根據 session 決定顯示儀表板還是上傳頁面"""
    page_data = session.get('page_data', None)
    
    if not page_data:
        return render_template('index.html')

    # --- 排序邏輯 ---
    sort_by = request.args.get('sort_by', 'market_value_twd')
    sort_order = request.args.get('sort_order', 'desc')
    reverse = (sort_order == 'desc')
    
    holdings_to_sort = page_data['raw_holdings']
    
    sort_key_map = {
        'position_percentage': 'raw_position_percentage',
        'profit_loss_twd': 'profit_loss_twd',
        'roi': 'raw_roi',
        'market_value_twd': 'market_value_twd'
    }
    sort_key = sort_key_map.get(sort_by, 'market_value_twd')
    
    holdings_to_sort.sort(key=lambda x: x.get(sort_key, 0), reverse=reverse)
    
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
        logger.error("上傳錯誤：請求中缺少檔案")
        return "錯誤：缺少檔案", 400
    
    us_file = request.files['us_stock_file']
    tw_file = request.files['tw_stock_file']

    if us_file.filename == '' or tw_file.filename == '':
        logger.error("上傳錯誤：使用者未選擇檔案")
        return "錯誤：未選擇檔案", 400

    logger.info(f"上傳的檔案: 美股='{us_file.filename}', 台股='{tw_file.filename}'")

    try:
        us_stock_content = io.BytesIO(us_file.read())
        tw_stock_content = io.BytesIO(tw_file.read())

        logger.info("開始進行資料處理...")
        page_data = process_data_files(us_stock_content, tw_stock_content)
        logger.info("資料處理完成")
        
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

@app.route('/clear-cache')
def clear_cache():
    """清除快取檔案"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            logger.info("快取已清除")
            return "快取已清除", 200
        else:
            return "快取檔案不存在", 404
    except Exception as e:
        logger.error(f"清除快取時發生錯誤: {e}")
        return f"清除快取失敗: {e}", 500

# --- 讓網站跑起來 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)