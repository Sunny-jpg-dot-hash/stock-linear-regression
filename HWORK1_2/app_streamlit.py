import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 應用標題
st.title("股票價格預測應用 - 基於 Prophet 模型")

# 1. 數據上傳與處理
st.header("數據上傳與處理")
uploaded_file = st.file_uploader("上傳包含日期（Date）和股票價格數據的 CSV 文件", type="csv")

if uploaded_file is not None:
    # 讀取 CSV 文件
    data = pd.read_csv(uploaded_file)
    
    # 確保將數據中的千分位符號移除並轉換為浮點數格式
    for col in ['y', 'x1', 'x2', 'x3', 'x4', 'x5']:
        if col in data.columns:
            data[col] = data[col].replace({',': ''}, regex=True).astype(float)
    
    # 將日期列轉換為 datetime 格式
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # 檢查是否存在無效的日期或數據
    if data['Date'].isnull().sum() > 0 or data.isnull().sum().sum() > 0:
        st.error("數據中包含無效日期或價格，請檢查數據格式並重新上傳。")
    else:
        st.success("數據上傳成功並正確處理！")
        st.write(data.head())

        # 2. Prophet 模型初始化
        st.header("Prophet 模型預測設置")
        
        # 準備 Prophet 所需的數據格式
        df = data[['Date', 'y']].rename(columns={'Date': 'ds', 'y': 'y'})
        
        # 初始化 Prophet 模型，設置變化點靈敏度與不確定性區間
        model = Prophet(
            changepoint_prior_scale=0.5,  # 設置變化點靈敏度
            interval_width=0.95  # 設置 95% 不確定性區間
        )
        
        # 增加每月季節性，Fourier Order 設置為 5
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # 擬合數據
        model.fit(df)
        
        # 3. 預測未來 60 天的股票價格趨勢
        st.header("未來 60 天股票價格預測")
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        # 顯示預測結果
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # 4. 圖表繪製
        st.header("股票價格預測圖表")
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # 繪製實際數據（黑色線條）
        ax.plot(df['ds'], df['y'], color='black', label='實際數據')

        # 繪製預測結果（藍色線條，淺藍色陰影表示不確定性區間）
        ax.plot(forecast['ds'], forecast['yhat'], color='blue', label='預測數據')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)

        # 添加水平虛線顯示歷史平均值
        historical_avg = df['y'].mean()
        ax.axhline(historical_avg, color='gray', linestyle='--', label='Historical Average')

        # 添加紅色虛線標記預測初始化時間點
        forecast_init_date = df['ds'].max()
        ax.axvline(forecast_init_date, color='red', linestyle='--', label='Forecast Initialization')
        ax.text(forecast_init_date, max(df['y']), 'Forecast Initialization', color='red', ha='left', fontsize=10)

        # 添加綠色箭頭標記上升趨勢
        upward_trend_idx = forecast['yhat'].idxmax()
        ax.annotate('Upward Trend', 
                    xy=(forecast['ds'][upward_trend_idx], forecast['yhat'][upward_trend_idx]), 
                    xytext=(forecast['ds'][upward_trend_idx], forecast['yhat'][upward_trend_idx] + 200),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    color='green')

        # 設置圖表標題與坐標標籤
        plt.title('股票價格預測（含不確定性區間）', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('價格', fontsize=12)

        # 顯示圖表
        st.pyplot(fig)

        # 5. 組件圖展示
        st.header("Prophet 組件圖")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)














