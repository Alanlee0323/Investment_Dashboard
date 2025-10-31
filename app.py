from flask import Flask, render_template, request, session, redirect, url_for, jsonify
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
import threading

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

# 用於追蹤處理狀態的全域字典
processing_status = {}

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

def get_usd_twd_rate():
    """抓取最新的美元兌台幣匯率（帶快取）"""
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

def fetch_single_stock_with_retry(ticker, max_retries=3, base_delay=2):
    """
    逐一抓取單支股票，帶重試機制
    使用 history() 而非 info 來避免觸發限流
    
    Returns:
        tuple: (stock_info, dividend_info)
    """
    stock_info = {'price': 0, 'sector': 'N/A'}
    dividend_info = {'last_dividend': 0, 'payouts_per_year': 0}
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # 使用 history() 獲取價格（輕量級操作，較不易觸發限流）
            data = stock.history(period="1d")
            if data.empty:
                logger.warning(f"找不到 {ticker} 的股價資料")
                price = 0
            else:
                # 處理 MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                price = round(data['Close'].iloc[-1], 2)
                logger.info(f"成功抓取 {ticker}: 價格={price}")
            
            # 只在有價格時才嘗試獲取產業別（使用 info）
            sector = 'N/A'
            if price > 0:
                try:
                    # 使用 get_info() 如果可用，否則使用 info
                    if hasattr(stock, "get_info"):
                        info = stock.get_info()
                    else:
                        info = stock.info
                    sector = info.get('sector', 'N/A')
                except Exception as e:
                    logger.warning(f"無法獲取 {ticker} 的產業別: {e}")
                    sector = 'N/A'
            
            stock_info = {
                'price': price,
                'sector': sector
            }
            
            # 抓取股息資訊
            try:
                # 使用 .loc 基於時間的索引來取代 .last()
                today = pd.to_datetime(datetime.now().date())
                start_date = today - pd.Timedelta(days=365)
                dividends = stock.dividends.loc[stock.dividends.index >= start_date]

                if not dividends.empty:
                    dividend_info = {
                        'last_dividend': dividends.iloc[-1],
                        'payouts_per_year': len(dividends)
                    }
                    logger.info(f"成功抓取 {ticker} 股息: {dividend_info}")
            except Exception as e:
                logger.warning(f"抓取 {ticker} 股息時發生錯誤: {e}")
            
            return stock_info, dividend_info
            
        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg or "Too Many Requests" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # 指數退避: 2, 4, 8 秒
                    logger.warning(f"{ticker} 遇到限流，等待 {wait_time} 秒後重試... (第 {attempt + 1}/{max_retries} 次)")
                    time.sleep(wait_time)
                else:
                    logger.error(f"{ticker} 達到最大重試次數")
                    return stock_info, dividend_info
            else:
                logger.error(f"抓取 {ticker} 時發生錯誤: {e}")
                return stock_info, dividend_info
    
    return stock_info, dividend_info

def sequential_fetch_with_cache(tickers_list, request_delay=1.0):
    """
    逐一抓取股票資料，使用快取並加入延遲
    
    Args:
        tickers_list: 股票代號列表
        request_delay: 每次請求之間的延遲秒數
    
    Returns:
        tuple: (all_stock_info, all_dividend_info)
    """
    cache = load_cache()
    all_stock_info = {}
    all_dividend_info = {}
    
    # 先從快取中載入
    uncached_tickers = []
    cached_count = 0
    
    for ticker in tickers_list:
        stock_cache_key = f"stock_{ticker}"
        div_cache_key = f"div_{ticker}"
        
        stock_cached = False
        div_cached = False
        
        # 檢查股價快取
        if stock_cache_key in cache:
            cached_time, cached_data = cache[stock_cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                all_stock_info[ticker] = cached_data
                stock_cached = True
        
        # 檢查股息快取
        if div_cache_key in cache:
            cached_time, cached_data = cache[div_cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                all_dividend_info[ticker] = cached_data
                div_cached = True
        
        if stock_cached and div_cached:
            cached_count += 1
        else:
            uncached_tickers.append(ticker)
    
    logger.info(f"從快取載入 {cached_count} 支股票")
    
    if uncached_tickers:
        logger.info(f"需重新抓取 {len(uncached_tickers)} 支股票")
        logger.info(f"預估耗時約 {len(uncached_tickers) * request_delay:.0f} 秒")
        
        for i, ticker in enumerate(uncached_tickers, 1):
            logger.info(f"[{i}/{len(uncached_tickers)}] 正在抓取 {ticker}...")
            
            stock_info, dividend_info = fetch_single_stock_with_retry(ticker)
            
            all_stock_info[ticker] = stock_info
            all_dividend_info[ticker] = dividend_info
            
            # 儲存到快取
            cache[f"stock_{ticker}"] = (datetime.now(), stock_info)
            cache[f"div_{ticker}"] = (datetime.now(), dividend_info)
            
            # 每 5 支股票儲存一次快取
            if i % 5 == 0:
                save_cache(cache)
                logger.info(f"已儲存前 {i} 支股票到快取")
            
            # 加入延遲（最後一支不需要延遲）
            if i < len(uncached_tickers):
                time.sleep(request_delay)
        
        # 最後儲存一次快取
        save_cache(cache)
        logger.info(f"所有 {len(uncached_tickers)} 支股票抓取完成並已快取")
    else:
        logger.info("全部股票都從快取載入，無需抓取")
    
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
    # 台股代號需要加上 .TW 後綴
    tw_tickers = [f"{str(t)}.TW" if not str(t).lower().endswith('.tw') else str(t) 
                  for t in df_tw['股號'].dropna().unique() 
                  if pd.notna(t)]
    
    all_tickers = list(set(us_tickers + tw_tickers))
    
    logger.info(f"收集到的股票: 美股 {len(us_tickers)} 支, 台股 {len(tw_tickers)} 支")
    logger.info(f"美股代號: {us_tickers}")
    logger.info(f"台股代號: {tw_tickers}")

    # --- 逐一抓取所有股票資料（帶快取和重試） ---
    all_stock_info = {}
    all_dividend_info = {}

    if all_tickers:
        logger.info(f"準備抓取 {len(all_tickers)} 支美股的資料")
        
        # 使用逐一抓取，每次請求間隔 1 秒
        all_stock_info, all_dividend_info = sequential_fetch_with_cache(
            all_tickers, 
            request_delay=1.0
        )
    else:
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

    # --- 處理台股資料（使用檔案中的資料，不呼叫 API） ---
    tw_holdings_raw = []
    for index, row in df_tw.iterrows():
        if pd.to_numeric(row.get('市值', 0), errors='coerce') == 0:
            continue
        
        ticker_str = str(row['股號'])
        
        # 台股股息資訊從檔案中計算或設為 0
        # 因為 yfinance 對台股的股息資料不完整
        tw_holdings_raw.append({
            'ticker': ticker_str, 
            'price': row['目前股價'], 
            'shares': row['持股'],
            'market_value': row['市值'], 
            'last_dividend': 0,  # 使用檔案資料或設為 0
            'payouts_per_year': 0,
            'monthly_income': 0  # 如果檔案中有股息資料，可以在這裡計算
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
    """處理檔案上傳，啟動背景處理任務"""
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
        # 將檔案讀入記憶體
        us_stock_content = io.BytesIO(us_file.read())
        tw_stock_content = io.BytesIO(tw_file.read())
        
        # 生成唯一的任務 ID
        task_id = f"task_{int(time.time())}"
        session['task_id'] = task_id
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '正在處理資料...'
        }
        
        # 在背景執行緒中處理資料
        def process_in_background():
            try:
                logger.info("開始進行資料處理...")
                page_data = process_data_files(us_stock_content, tw_stock_content)
                logger.info("資料處理完成")
                
                processing_status[task_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'message': '處理完成',
                    'data': page_data
                }
                
            except Exception as e:
                logger.error("處理上傳檔案時發生錯誤", exc_info=True)
                processing_status[task_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'處理失敗: {str(e)}'
                }
        
        # 啟動背景執行緒
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # 立即返回處理中頁面
        return render_template('processing.html', task_id=task_id)
        
    except Exception as e:
        logger.error("處理上傳檔案時發生錯誤", exc_info=True)
        import traceback
        return f"處理上傳檔案時發生錯誤: {e}<br><pre>{traceback.format_exc()}</pre>", 500

@app.route('/check-status/<task_id>')
def check_status(task_id):
    """檢查處理狀態的 API"""
    status = processing_status.get(task_id, {'status': 'unknown', 'message': '找不到任務'})
    
    # 如果處理完成，將資料存入 session
    if status['status'] == 'completed' and 'data' in status:
        session['page_data'] = status['data']
        # 清理處理狀態（保留一段時間以防重複請求）
        del processing_status[task_id]['data']
    
    return jsonify(status)

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