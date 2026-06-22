# 同江特征化县域综合能源系统算例平台（V17.3.1）

## 平台内容

- 全中文界面；
- 所有经济指标统一按人民币展示；
- 内置V17.3.1正式15分钟结果；
- 展示S0—S4规划场景、R0—R4同容量运行场景、Q0—Q3可靠性场景；
- 展示碳上限和主网外送能力敏感性；
- 支持小时级在线快速试算；
- 支持汇总结果、场景数据和试算结果下载。

## 本地运行

```powershell
cd D:\StoreMore_Clone_V0\v17_3_1_streamlit_platform
pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开本地网址，通常为：

```text
http://localhost:8501
```

## 部署到Streamlit Community Cloud

1. 在GitHub新建一个仓库；
2. 将本文件夹内的所有文件上传到仓库根目录；
3. 打开 Streamlit Community Cloud 并点击 **Create app**；
4. 选择刚才的仓库、分支和入口文件 `app.py`；
5. 点击部署；
6. 部署完成后会生成 `https://你的应用名.streamlit.app/`。

## 计算说明

- 内置正式结果采用15分钟分辨率；
- 在线试算为保证云端响应速度，采用小时级六典型日；
- 金额统一采用 `1 EUR = 7.8 CNY` 换算；
- 当前同江内容属于特征化县域场景，不是同江真实10 kV线路、电压潮流或实测数据模型。
