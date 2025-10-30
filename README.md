# 投資組合儀表板 (Investment Portfolio Dashboard)

這是一個使用 Python Flask 和 Pandas 建立的網頁應用程式，讓使用者可以上傳自己的投資報表，並以視覺化的方式呈現投資組合的詳細資訊。

---

### ✨ **[線上預覽 Live Demo](https://investment-dashboard-5lzr.onrender.com)** ✨

---

## 專案特色 (Features)

- **動態檔案上傳**: 無需在程式碼中寫死檔案路徑，直接在網頁上傳您的報表，安全又方便。
- **隱私安全**: 您的報表資料僅在當前的瀏覽階段 (Session) 中處理，不會被儲存到伺服器或版本控制中。
- **多市場資產配置**: 以圓餅圖清晰呈現美股與台股的資產分佈。
- **即時股價更新**: 透過 `yfinance` 函式庫抓取最新的股票價格與匯率。
- **詳細持股分析**: 
    - **美股**: 提供包含市值、部位佔比、損益、投報率的詳細表格。
    - **台股**: 提供高股息部位的現金流分析。
- **互動式表格**: 可點擊表頭對市值、損益、投報率等欄位進行排序。
- **一鍵部署**: 使用 Docker 容器化，並設定在 [Render](https://render.com/) 平台上一鍵部署。

## 技術棧 (Technology Stack)

- **後端 (Backend)**: Python, Flask, Gunicorn
- **資料處理 (Data Processing)**: Pandas
- **前端 (Frontend)**: HTML, CSS, JavaScript
- **圖表 (Charting)**: Chart.js
- **部署 (Deployment)**: Docker, Render

## 如何在本地端執行 (Local Development)

1.  **複製專案**:
    ```bash
    git clone <your-repository-url>
    cd investment_dashboard
    ```

2.  **建立並啟用虛擬環境**:
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安裝所需套件**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **執行應用程式**:
    ```bash
    python app.py
    ```

5.  **開啟瀏覽器**:
    前往 `http://127.0.0.1:5001` 即可看到上傳頁面。

## 部署 (Deployment)

本專案已設定為可透過 Docker 自動部署於 Render 平台。

- 每當有新的 commit 推送到 `main` 分支時，Render 都會自動觸發新的部署。
- 詳細的部署步驟請參考 `deploy.md` 檔案。
