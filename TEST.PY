import pandas as pd
import openpyxl
import openpyxl.utils
import yfinance as yf
import os
from functools import lru_cache

# 快取功能，避免重複抓取同一支股票的資料
@lru_cache(maxsize=500)
def get_stock_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 優先使用 'sector'，如果沒有則回傳 'N/A'
        return stock.info.get('sector', 'N/A')
    except Exception:
        return 'N/A'

@lru_cache(maxsize=1)
def get_usd_twd_rate():
    """自動抓取最新的美元兌台幣匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        data = ticker.history(period="1d")
        return round(data['Close'].iloc[-1], 4) if not data.empty else 32.5
    except Exception as e:
        print(f"  [警告] 無法獲取美元匯率，將使用預設值 32.5。錯誤: {e}")
        return 32.5 # 提供一個合理的備用值

def smart_update_holdings_with_sector(
    excel_file_path,
    csv_file_path,
    output_file_path,
    sheet_name='美股',
    header_row=3 # <--- 新增：指定標題列所在的行數
):
    """
    智慧更新Excel持股狀況，並自動 enrich 產業類別資訊。
    """
    if not os.path.exists(excel_file_path):
        print(f"錯誤：找不到主要的Excel檔案 '{excel_file_path}'")
        return
    if not os.path.exists(csv_file_path):
        print(f"錯誤：找不到券商的CSV檔案 '{csv_file_path}'")
        return

    print("--- 開始智慧更新流程 (包含產業類別) ---")

    try:
        # 1. 讀取券商CSV並清理
        holdings_df = pd.read_csv(csv_file_path)
        print(f"成功讀取券商CSV檔案: '{csv_file_path}'")
        # 清理數字欄位中的千分位逗號，並轉換為浮點數
        for col in ['最新價', '均價', '目前庫存', '報酬率']:
            if holdings_df[col].dtype == 'object':
                holdings_df[col] = holdings_df[col].astype(str).str.replace('[%,]', '', regex=True).astype(float)

        # 2. 載入Excel工作簿
        workbook = openpyxl.load_workbook(excel_file_path)
        if sheet_name not in workbook.sheetnames:
            print(f"錯誤：找不到名為 '{sheet_name}' 的工作表。")
            return
        worksheet = workbook[sheet_name]

        # 3. 建立欄位映射
        headers = [cell.value for cell in worksheet[header_row]]
        column_map = {name: i + 1 for i, name in enumerate(headers) if name is not None}
        
        # 確保必要的欄位存在，若不存在則在最後新增
        required_cols = ['產業類別', '台幣市值', '台幣損益', '美元匯率']
        last_col_num = worksheet.max_column
        for col_name in required_cols:
            if col_name not in column_map:
                last_col_num += 1
                worksheet.cell(row=header_row, column=last_col_num).value = col_name
                column_map[col_name] = last_col_num
                print(f"  [提示] 已自動新增 '{col_name}' 欄位。")

        # 抓取並寫入即時匯率
        usd_rate = get_usd_twd_rate()
        # 將匯率寫入總計區 (假設在第2行)
        if '美元匯率' in column_map:
            worksheet.cell(row=2, column=column_map['美元匯率']).value = usd_rate
            print(f"成功獲取即時美元匯率: {usd_rate}")
        else:
            print("not found 美元匯率")

        # 為了方便計算，將匯率儲存格定義為一個名稱
        rate_cell_address = f"'{sheet_name}'!{worksheet.cell(row=2, column=column_map['美元匯率']).coordinate}"
        if 'USD_RATE' not in workbook.defined_names:
            # 修正：使用字典賦值語法來新增 DefinedName
            # 並且修正 DefinedName 的參數，'name' 是名稱, 'attr_text' 是範圍
            defined_name_obj = openpyxl.workbook.defined_name.DefinedName(name='USD_RATE', attr_text=rate_cell_address)
            workbook.defined_names['USD_RATE'] = defined_name_obj
 
        # 4. 建立股票代號到行數的映射
        ticker_to_row_map = {}
        last_data_row = 1
        # 修正：從標題列的下一行開始遍歷資料
        for row_num in range(header_row + 1, worksheet.max_row + 1):
            # 檢查 '股號' 欄位是否存在
            if '股號' not in column_map:
                raise KeyError("錯誤：在 Excel 的標題列中找不到 '股號' 欄位。請檢查拼寫或 header_row 設定。")
            ticker_cell = worksheet.cell(row=row_num, column=column_map.get('股號'))
        
            if ticker_cell.value and str(ticker_cell.value).strip() != "":
                ticker_str = str(ticker_cell.value).strip()
                ticker_to_row_map[ticker_str] = row_num
                last_data_row = row_num

        # 5. 核心更新與新增邏輯
        update_count = 0
        add_count = 0
        next_new_row = last_data_row + 1

        for _, holding_row in holdings_df.iterrows():
            ticker = str(holding_row['代號']).strip()
            
            # 抓取產業類別
            sector = get_stock_sector(ticker)

            if ticker in ticker_to_row_map:
                row_to_update = ticker_to_row_map[ticker]
                # 更新資料...
                worksheet.cell(row=row_to_update, column=column_map['目前股價']).value = holding_row['最新價']
                worksheet.cell(row=row_to_update, column=column_map['買入價格']).value = holding_row['均價']
                worksheet.cell(row=row_to_update, column=column_map['目前持股']).value = holding_row['目前庫存']
                worksheet.cell(row=row_to_update, column=column_map['投報率']).value = holding_row['報酬率']
                worksheet.cell(row=row_to_update, column=column_map['產業類別']).value = sector
                # 新增：寫入台幣市值和損益的公式
                price_cell = worksheet.cell(row=row_to_update, column=column_map['目前股價']).coordinate
                qty_cell = worksheet.cell(row=row_to_update, column=column_map['目前持股']).coordinate
                worksheet.cell(row=row_to_update, column=column_map['台幣市值']).value = f"={price_cell}*{qty_cell}*USD_RATE"
                print(f"  [更新] {ticker} (產業: {sector})")
                update_count += 1
            else:
                # 新增資料...
                worksheet.cell(row=next_new_row, column=column_map['股號']).value = ticker
                worksheet.cell(row=next_new_row, column=column_map['目前股價']).value = holding_row['最新價']
                worksheet.cell(row=next_new_row, column=column_map['買入價格']).value = holding_row['均價']
                worksheet.cell(row=next_new_row, column=column_map['目前持股']).value = holding_row['目前庫存']
                worksheet.cell(row=next_new_row, column=column_map['投報率']).value = holding_row['報酬率']
                worksheet.cell(row=next_new_row, column=column_map['產業類別']).value = sector
                # 新增：寫入台幣市值和損益的公式
                price_cell = worksheet.cell(row=next_new_row, column=column_map['目前股價']).coordinate
                qty_cell = worksheet.cell(row=next_new_row, column=column_map['目前持股']).coordinate
                worksheet.cell(row=next_new_row, column=column_map['台幣市值']).value = f"={price_cell}*{qty_cell}*USD_RATE"
                print(f"  [新增] {ticker} (產業: {sector})")
                add_count += 1
                next_new_row += 1

        # 6. 儲存結果
        # 6. 移除已出售的股票 (持股為 0)
        # 找出 Excel 中存在但新 CSV 中已不存在的股票
        excel_tickers = set(ticker_to_row_map.keys())
        csv_tickers = set(holdings_df['代號'].astype(str).str.strip())
        sold_tickers = excel_tickers - csv_tickers
        
        delete_count = 0
        if sold_tickers:
            rows_to_delete = [ticker_to_row_map[ticker] for ticker in sold_tickers]
            # 關鍵：必須從後往前刪除，才不會影響前面的行號索引
            rows_to_delete.sort(reverse=True)
            
            print("\n--- 開始移除已出售的股票 ---")
            for row_num in rows_to_delete:
                ticker_to_remove = [t for t, r in ticker_to_row_map.items() if r == row_num][0]
                worksheet.delete_rows(row_num)
                print(f"  [移除] {ticker_to_remove} (原行號: {row_num})")
                delete_count += 1

        # 7. 更新總計欄位的公式 (假設總計欄位在第2行)
        # 假設資料從第4行開始 (給總計區留出空間)
        data_start_row = 4 
        last_row = worksheet.max_row
        if '台幣總市值' in column_map and last_row >= data_start_row:
            twd_value_col = openpyxl.utils.get_column_letter(column_map['台幣市值'])
            twd_invest_col = openpyxl.utils.get_column_letter(column_map['台幣總投入'])
            
            worksheet.cell(row=2, column=column_map['台幣總市值']).value = f"=SUM({twd_value_col}{data_start_row}:{twd_value_col}{last_row})"
            # 總投入的計算需要您在Excel中手動維護或提供成本欄位，這裡暫時留空
            worksheet.cell(row=2, column=column_map['台幣總投入']).value = f"=SUM({twd_invest_col}{data_start_row}:{twd_invest_col}{last_row})"
            worksheet.cell(row=2, column=column_map['台幣總利潤']).value = f"={twd_value_col}2-{twd_invest_col}2"
            worksheet.cell(row=2, column=column_map['台幣總投報率']).value = f"=IFERROR({twd_value_col}2/{twd_invest_col}2-1, 0)"

        # 8. 儲存結果
        workbook.save(output_file_path)
        print("\n--- Python 腳本執行完成 ---")
        print(f"成功更新 {update_count} 筆, 新增 {add_count} 筆, 移除 {delete_count} 筆。")
        print(f"已產出 Tableau 可用的乾淨數據檔案: '{output_file_path}'")

    except Exception as e:
        print(f"處理過程中發生嚴重錯誤：{e}")


if __name__ == "__main__":
    # --- 請修改成你的檔案名稱 ---
    main_excel_file = '2025_09.xlsx'
    broker_csv_file = '複委託庫存20251006174326.csv'
    # 這是最終產出的檔案，將會給 Tableau 使用
    output_for_tableau = 'data_for_tableau.xlsx' 
    
    smart_update_holdings_with_sector(
        excel_file_path=main_excel_file,
        csv_file_path=broker_csv_file,
        output_file_path=output_for_tableau,
        header_row=13 # <--- 請根據您的Excel檔案，設定標題列所在的正確行數
    )