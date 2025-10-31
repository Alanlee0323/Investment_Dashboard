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
CMD ["/bin/sh", "-c", "gunicorn", "--timeout", "180", "--bind", "0.0.0.0:${PORT}", "app:app"]