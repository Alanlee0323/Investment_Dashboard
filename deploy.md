    # Flask 應用程式部署指南 (使用 Docker + Render)

本文件將指導您如何將這個 Python Flask 儀表板應用程式部署到雲端，讓您可以透過公開網址隨時隨地存取。

## 核心概念

- **靜態 vs. 動態網站**：GitHub Pages 僅能託管靜態網站 (HTML/CSS/JS)，它無法執行像 `app.py` 這樣的後端 Python 伺服器。
- **PaaS 平台**：我們需要一個能執行後端程式的服務，稱為 PaaS (Platform as a Service)。本教學推薦使用 [Render.com](https://render.com/)，它提供免費方案且對開發者非常友善。
- **Docker 容器化**：我們將使用 Docker 將應用程式及其所有依賴項打包成一個標準化的「映像檔」(Image)。這確保了開發環境和生產環境的一致性，是現代網站部署的最佳實踐。

---

## 部署流程概覽

1.  **準備您的應用程式**：建立 `requirements.txt` 和 `gunicorn` 設定。
2.  **撰寫 Dockerfile**：建立一個 Docker 設定檔，告訴 Docker 如何打包您的應用程式。
3.  **推送到 GitHub**：將所有程式碼（包含新的 `Dockerfile`）推送到您的 GitHub 倉庫。
4.  **在 Render 上建立服務**：連接您的 GitHub 帳號，並設定一個新的 Web Service 來運行您的 Docker 容器。
5.  **處理數據文件**：理解部署後如何更新您的 Excel/CSV 報表。

---

## 步驟一：準備您的應用程式

在部署之前，我們需要新增幾個檔案來讓應用程式「生產就緒」。

### 1. 建立 `requirements.txt`

這個檔案會列出您的應用程式需要的所有 Python 函式庫。在您的專案根目錄下，開啟終端機並執行：

```bash
pip freeze > requirements.txt
```

這會自動偵測您目前環境中安裝的套件並寫入檔案。請檢查 `requirements.txt` 的內容，至少應包含 `flask` 和 `pandas` (以及讀取 excel 需要的 `openpyxl`)。

**重要：** 我們還需要一個生產級的網頁伺服器來取代 Flask 內建的開發伺服器。`gunicorn` 是最常見的選擇。請手動在 `requirements.txt` 中加入 `gunicorn`。

您的 `requirements.txt` 看起來應該像這樣 (版本號可能不同)：

```
flask==2.3.2
pandas==1.5.3
openpyxl==3.1.2
gunicorn==20.1.0
# ... 其他您可能用到的函式庫
```

### 2. 修改 `.gitignore`

確保您的虛擬環境資料夾 (如 `venv/` 或 `.venv/`) 和 `__pycache__` 被包含在 `.gitignore` 中，避免將它們上傳到 GitHub。

```
# .gitignore
__pycache__/
*.pyc
.venv
venv
```

---

## 步驟二：撰寫 Dockerfile

在您的專案根目錄下，建立一個名為 `Dockerfile` (沒有副檔名) 的新檔案，並貼上以下內容：

```Dockerfile
# 1. 使用官方的 Python 映像檔作為基礎
FROM python:3.9-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 複製整個專案的程式碼到工作目錄
COPY . .

# 5. 設定環境變數 (Render 會自動提供 PORT)
# 我們預設一個端口 10000，但 Render 會用它自己的
ENV PORT 10000

# 6. 執行 Gunicorn 伺服器
#    - "app:app" 指的是 "執行 app.py 檔案中的 app 變數"
#    - "--bind 0.0.0.0:${PORT}" 讓伺服器監聽所有網路介面
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
```

---

## 步驟三：推送到 GitHub

現在您的專案結構應該像這樣：

```
.
├── .gitignore
├── app.py
├── Dockerfile         <-- 新增
├── requirements.txt   <-- 新增
├── static/
│   └── style.css
└── templates/
    └── index.html
```

將所有變更 commit 並 push 到您的 GitHub 倉庫。

```bash
git add .
git commit -m "Feat: Add Dockerfile and config for deployment"
git push origin main
```

---

## 步驟四：在 Render 上建立服務

1.  **註冊/登入 Render**：前往 [Render.com](https://render.com/) 並用您的 GitHub 帳號登入。
2.  **建立新服務**：在儀表板 (Dashboard) 點擊 "New +" -> "Web Service"。
3.  **連接倉庫**：選擇 "Build and deploy from a Git repository"，然後選擇您剛剛推送的 GitHub 倉庫。
4.  **設定服務**：
    -   **Name**：為您的服務取一個名字 (例如 `investment-dashboard`)。
    -   **Environment**：選擇 **`Docker`**。Render 會自動偵測到您專案中的 `Dockerfile`。
    -   **Region**：選擇離您最近的地區。
    -   **Instance Type**：選擇 `Free` (免費方案)。
5.  **建立服務**：點擊 "Create Web Service"。

Render 會自動從 GitHub 拉取您的程式碼，使用 `Dockerfile` 建立映像檔，並將其部署。您可以在 "Logs" 分頁看到部署進度。第一次部署大約需要幾分鐘。

部署成功後，Render 會提供給您一個公開的 `.onrender.com` 網址，您的儀表板就上線了！

---

## 步驟五：處理數據文件 (重要！)

您的應用程式目前是直接讀取專案內的 `.csv` 和 `.xlsx` 檔案。當部署到 Render 後，它讀取的會是您 **commit 到 GitHub 上的那個版本** 的檔案。

**如何更新數據？**

最直接的方法是：

1.  在您的**本機電腦**上更新 `.csv` / `.xlsx` 檔案。
2.  將更新後的檔案 `git add`, `git commit`, `git push` 到 GitHub。
3.  Render 會偵測到新的 commit，並**自動重新部署**您的應用程式。新的儀表板就會讀取到新的數據。

**進階方案 (未來可以考慮)**：

-   **資料庫**：將數據存儲在 Render 提供的免費 PostgreSQL 資料庫中。
-   **雲端儲存**：將報表上傳到 AWS S3 或類似的服務，並讓您的應用程式從那裡讀取。
-   **檔案上傳介面**：在您的網站上新增一個頁面，讓您可以直接上傳新的報表檔案。
