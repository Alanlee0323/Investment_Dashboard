import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_chinese_font(font_path, font_name):
    try:
        if font_path and fm.findfont(font_name, fallback_to_default=False) is None:
            fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False 
        print(f"Matplotlib 中文字型設置成功：{font_name}")
    except Exception as e:
        print(f"中文字型設置失敗：{e}。請檢查 font_path 和 font_name。")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = True

def load_config(filename='config.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"配置載入錯誤: {e}")
        return None

def run_actual_backtest_and_visualize():
    
    config = load_config()
    if not config:
        return

    # --- 參數提取 ---
    P0 = config['initial_capital']
    ticker = config['ticker']
    start_date = config['start_date']
    end_date = config['end_date']
    L_low = config['leverage_low']
    L_high = config['leverage_high']
    C_fric_low = config['friction_cost_low']
    C_fric_high = config['friction_cost_high']
    MA_period = config['ma_period']
    F_days = config['frequency_days']

    setup_chinese_font(config.get('font_path'), config.get('font_name'))

    # 獲取歷史數據
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty: return

    # 計算報酬率和 MA (季線)
    data['Returns'] = data['Adj Close'].pct_change()
    data['MA'] = data['Adj Close'].rolling(window=MA_period).mean() 
    
    # 清理數據
    data = data.dropna()
    
    daily_returns = data['Returns'] 
    
    # 數據清洗
    data_error_threshold = -0.50 
    extreme_error_days = daily_returns[daily_returns < data_error_threshold]
    if not extreme_error_days.empty:
        print("\n*** 數據清洗警告：發現並修正異常報酬點 (設為 0) ***")
        daily_returns.loc[extreme_error_days.index] = 0.0

    # --- 核心回測邏輯 ---

    daily_friction_low = C_fric_low / config['frequency_days']
    daily_friction_high = C_fric_high / config['frequency_days']
    
    # 初始化 Series
    portfolio_1x = pd.Series(index=daily_returns.index, dtype=float)
    portfolio_dynamic = pd.Series(index=daily_returns.index, dtype=float)
    leverage_status = pd.Series(index=daily_returns.index, dtype=float)

    portfolio_1x.iloc[0] = P0
    portfolio_dynamic.iloc[0] = P0
    leverage_status.iloc[0] = L_low
    
    for i in range(1, len(daily_returns)):
        R_base_day = daily_returns.iloc[i]
        current_date = daily_returns.index[i]
        
        # 1. 策略一 (1.0x 基準)
        portfolio_1x.iloc[i] = portfolio_1x.iloc[i-1] * (1 + R_base_day)

        # 2. 策略二 (動態槓桿)
        current_price = data['Adj Close'].loc[current_date].item()
        current_ma = data['MA'].loc[current_date].item()

        # 核心策略切換點
        if current_price > current_ma:
            L_current = L_high
            C_fric_day = daily_friction_high
        else:
            L_current = L_low
            C_fric_day = daily_friction_low
            
        R_strategy_dynamic = (R_base_day * L_current) - C_fric_day 
        leverage_status.iloc[i] = L_current
        portfolio_dynamic.iloc[i] = portfolio_dynamic.iloc[i-1] * (1 + R_strategy_dynamic)

    # --- 視覺化結果（關鍵改進：分段繪製）---
    
    total_days = len(daily_returns)
    total_years = total_days / F_days
    
    cagr_1x = (portfolio_1x.iloc[-1] / P0) ** (1 / total_years) - 1
    cagr_dynamic = (portfolio_dynamic.iloc[-1] / P0) ** (1 / total_years) - 1

    plt.figure(figsize=(14, 8))
    
    # 1. 繪製 1.0x 基準線
    plt.plot(portfolio_1x.index, portfolio_1x.values, 
             label=f'1.0x 基準 (CAGR: {cagr_1x:.2%})', 
             linewidth=2, color='steelblue', alpha=0.7)
    
    # --- 【關鍵改進：連續區間分段繪製】 ---
    
    # 找出槓桿切換點，分段繪製保持連續性
    leverage_changes = leverage_status != leverage_status.shift(1)
    segment_starts = [0] + list(leverage_changes[leverage_changes].index)
    
    # 添加標記變量，確保圖例只出現一次
    low_legend_added = False
    high_legend_added = False
    
    for i in range(len(segment_starts)):
        start_idx = segment_starts[i]
        end_idx = segment_starts[i + 1] if i + 1 < len(segment_starts) else len(portfolio_dynamic)
        
        # 獲取該區間的槓桿狀態
        segment_leverage = leverage_status.iloc[start_idx]
        
        # 為了保持連續性，包含前一個點
        if i > 0:
            plot_start = start_idx - 1
        else:
            plot_start = start_idx
            
        segment_data = portfolio_dynamic.iloc[plot_start:end_idx]
        
        if segment_leverage == L_low:
            # 低槓桿區間：橙色實線
            label = f'動態 ({L_low}x 低曝險)' if not low_legend_added else None
            plt.plot(segment_data.index, segment_data.values, 
                     color='darkorange', linewidth=2.5, linestyle='-', 
                     label=label, alpha=0.9)
            low_legend_added = True
        else:
            # 高槓桿區間：紅色虛線
            label = f'動態 ({L_high}x 高曝險)' if not high_legend_added else None
            plt.plot(segment_data.index, segment_data.values, 
                     color='crimson', linewidth=2.5, linestyle='--', 
                     label=label, alpha=0.9)
            high_legend_added = True
    
    # 圖表設置
    plt.title(f'{ticker} 歷史回測：1.0x vs. 動態槓桿 (MA{MA_period}) | 動態 CAGR: {cagr_dynamic:.2%}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('日期', fontsize=13)
    plt.ylabel('投資組合價值 (台幣)', fontsize=13)
    
    # 終值標註
    plt.annotate(f'終值: NT${portfolio_1x.iloc[-1]:,.0f}', 
                 (portfolio_1x.index[-1], portfolio_1x.iloc[-1]), 
                 textcoords="offset points", xytext=(-60, -15), 
                 ha='right', fontsize=10, color='steelblue',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
                 
    plt.annotate(f'終值: NT${portfolio_dynamic.iloc[-1]:,.0f}', 
                 (portfolio_dynamic.index[-1], portfolio_dynamic.iloc[-1]), 
                 textcoords="offset points", xytext=(-60, 15), 
                 ha='right', fontsize=10, color='crimson', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
                 
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=11, loc='upper left', framealpha=0.95)
    plt.tight_layout()
    plt.show()
    
    # --- 統計摘要 ---
    low_days = (leverage_status == L_low).sum()
    high_days = (leverage_status == L_high).sum()
    total = len(leverage_status)
    
    print("\n" + "="*60)
    print(f"回測期間統計 ({start_date} 至 {end_date})")
    print("="*60)
    print(f"總交易日數: {total} 天")
    print(f"低槓桿 ({L_low}x) 天數: {low_days} 天 ({low_days/total*100:.1f}%)")
    print(f"高槓桿 ({L_high}x) 天數: {high_days} 天 ({high_days/total*100:.1f}%)")
    print(f"\n1.0x 基準最終價值: NT${portfolio_1x.iloc[-1]:,.0f} (CAGR: {cagr_1x:.2%})")
    print(f"動態槓桿最終價值: NT${portfolio_dynamic.iloc[-1]:,.0f} (CAGR: {cagr_dynamic:.2%})")
    print(f"超額報酬: {(cagr_dynamic - cagr_1x)*100:.2f}% 年化")
    print("="*60)

if __name__ == "__main__":
    run_actual_backtest_and_visualize()