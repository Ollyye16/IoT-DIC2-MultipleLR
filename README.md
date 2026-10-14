# DIC2 — Multiple Linear Regression Explorer

50 Startups / Boston Housing / 任意 CSV URL 的多元線性迴歸與特徵選擇分析 (CRISP-DM)。

- **Live (GitHub Pages)**: https://ollyye16.github.io/IoT-DIC2-MultipleLR/
- 下拉選單切換 problem：50 Startups (Profit)、Boston Housing (medv)、自訂 CSV URL
- 5 種特徵選擇：Lasso、Forward SFS、RFE、ElasticNet、Exhaustive Search，k = 1…p 比較 test RMSE / R²
- OLS 推論：係數、標準誤、t、p-value、95% CI、VIF、F-test；即時預測器；殘差診斷

## 架構

| 檔案 | 說明 |
|---|---|
| `app.py` | Flask 後端，`POST /api/analyze` 用 scikit-learn (lasso_path / enet_path / RFE / LinearRegression) + statsmodels (OLS, VIF) 計算全部模型 |
| `index.html` | 前端 (Chart.js)。後端連得到就用 Python 計算；連不到 (例如 GitHub Pages) 自動改用瀏覽器內 JavaScript，結果相同 |
| `api/index.py`, `vercel.json` | Vercel serverless 部署設定 |

## 本機執行

```bash
pip install -r requirements.txt
python app.py
```

開 http://localhost:5000/ ，頁面上會顯示 `Engine: Python (Flask + scikit-learn + statsmodels)`。

## 結果 (50 Startups, sklearn `train_test_split(test_size=0.2, random_state=0)`)

| k | 特徵 | Test RMSE | R² |
|---|---|---|---|
| 1 | R&D Spend | 8,274.87 | 0.9465 |
| **2** | **R&D Spend, Marketing Spend** | **8,198.80** | **0.9474** |
| 5 | 全部 | 9,137.99 | 0.9347 |

2 個特徵就是最佳點 (Occam's Razor)，全部放進去反而 overfitting。

Dataset: [50 Startups](https://github.com/Avik-Jain/100-Days-Of-ML-Code), [Boston Housing](https://github.com/selva86/datasets)
