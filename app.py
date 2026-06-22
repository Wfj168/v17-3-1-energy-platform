from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = APP_ROOT / "data" / "results"
FIGURE_ROOT = APP_ROOT / "data" / "figures"
MODEL_ROOT = APP_ROOT / "model"
CNY_PER_EUR = 7.8

sys.path.insert(0, str(MODEL_ROOT))

st.set_page_config(
    page_title="同江特征化县域综合能源系统算例平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --primary: #1677ff;
  --accent: #15a36d;
  --ink: #172033;
  --muted: #64748b;
  --surface: #f6f8fb;
}
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f7fbff 0%, #f5f8fc 100%);}
h1, h2, h3 {color: var(--ink); letter-spacing: 0.01em;}
.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #eef7ff 0%, #f4fff9 100%);
  border: 1px solid #dcecff;
  margin-bottom: 1rem;
}
.hero h1 {margin:0 0 .35rem 0; font-size: 2rem;}
.hero p {margin:0; color: var(--muted); font-size: 1rem;}
.note {
  padding: .9rem 1rem; border-radius: 12px; background: #fff8e8;
  border-left: 4px solid #f59e0b; color: #6b4f00; margin: .5rem 0 1rem 0;
}
.good {
  padding: .9rem 1rem; border-radius: 12px; background: #eefbf5;
  border-left: 4px solid #16a36a; color: #125b42; margin: .5rem 0 1rem 0;
}
.small-muted {color: var(--muted); font-size: .88rem;}
[data-testid="stMetric"] {
  background: white; border: 1px solid #e8edf4; padding: .8rem 1rem;
  border-radius: 14px; box-shadow: 0 3px 12px rgba(20, 35, 60, .04);
}
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.stTabs [data-baseweb="tab"] {height: 42px; border-radius: 10px; padding: 0 14px;}
</style>
""",
    unsafe_allow_html=True,
)

SCENARIO_NAME = {
    "S0": "S0 县域基准系统",
    "S1": "S1 低碳约束",
    "S2": "S2 低碳约束+储能",
    "S3": "S3 低碳约束+柔性负荷",
    "S4": "S4 低碳源荷储协同",
    "R0": "R0 同容量常规规则运行",
    "R1": "R1 同容量储能优化调度",
    "R2": "R2 规则储能+柔性负荷",
    "R3": "R3 源荷储经济协同调度",
    "R4": "R4 源荷储低碳协同调度",
    "Q0": "Q0 正常运行日",
    "Q1": "Q1 春灌高负荷—主网降额中等扰动",
    "Q2": "Q2 冬季采暖—低风光中等扰动",
    "Q3": "Q3 冬季高负荷—低风光—主网严重降额",
}

DAY_NAME = {
    "spring_irrigation": "春季灌溉日",
    "spring_pv_high": "春秋光伏大发日",
    "summer_commercial": "夏季商业制冷日",
    "autumn_processing": "秋季农产品加工日",
    "winter_heating": "冬季采暖日",
    "holiday_return": "春节返乡日",
}

COLOR = {
    "S0": "#64748b", "S1": "#3b82f6", "S2": "#10b981", "S3": "#f59e0b", "S4": "#ef4444",
    "R0": "#64748b", "R1": "#3b82f6", "R2": "#f59e0b", "R3": "#10b981", "R4": "#ef4444",
}

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_top(name: str) -> pd.DataFrame:
    return read_csv(str(RESULT_ROOT / name))


def load_dispatch(group: str, scenario: str) -> pd.DataFrame:
    return read_csv(str(RESULT_ROOT / group / scenario / "data" / "dispatch_results.csv"))


def load_capacity(group: str, scenario: str) -> pd.DataFrame:
    return read_csv(str(RESULT_ROOT / group / scenario / "data" / "capacity_results.csv"))


def cny_wan_from_million(value: float) -> float:
    return float(value) * 100.0


def fmt_wan(value: float) -> str:
    return f"{value:,.1f} 万元/年"


def metric_value(df: pd.DataFrame, scenario: str, col: str, default: float = 0.0) -> float:
    row = df[df["Scenario"] == scenario]
    if row.empty or col not in row.columns:
        return default
    return float(row[col].iloc[0])


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, y_title: str, color_col: str = "Scenario"):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color_col if color_col in df.columns else None,
        color_discrete_map=COLOR,
        text_auto=".2f",
    )
    fig.update_layout(
        title=title, xaxis_title="", yaxis_title=y_title,
        legend_title="", height=420, margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=14),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def show_hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def chinese_summary(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    out = df[available].copy().rename(columns={c: cols[c] for c in available})
    return out


def to_zip_bytes(files: list[Path]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if p.exists():
                zf.write(p, arcname=p.name)
    return bio.getvalue()


def page_home():
    show_hero(
        "同江特征化县域综合能源系统算例平台",
        "V17.3.1｜低碳规划—运行协同、源荷储优化、可靠性压力测试与敏感性分析",
    )
    st.markdown(
        '<div class="note"><b>适用场景：</b>几十至约200 MW规模的县域级或大型园区级农工商复合综合能源系统。当前“同江”部分属于特征化仿真场景，不代表真实10 kV线路潮流或实测15分钟数据。</div>',
        unsafe_allow_html=True,
    )

    planning = load_top("planning_summary_v17_3_1.csv")
    operation = load_top("operation_summary_v17_3_1.csv")
    s4 = planning[planning["Scenario"] == "S4"].iloc[0]
    r0 = operation[operation["Scenario"] == "R0"].iloc[0]
    r4 = operation[operation["Scenario"] == "R4"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S4年社会综合成本", fmt_wan(cny_wan_from_million(s4["Social total annual cost [million CNY/year]"])))
    c2.metric("S4年碳排放", f'{s4["Annual CO2 emissions [tCO2/year]"]:,.0f} tCO₂')
    c3.metric("S4新能源本地吸纳率", f'{s4["Renewable local absorption rate [%]"]:.2f}%')
    c4.metric("S4弃风弃光率", f'{s4["Renewable curtailment rate [%]"]:.2f}%')

    st.subheader("模型优化效果概览")
    k1, k2, k3, k4 = st.columns(4)
    cost_drop = 100 * (r0["Variable operating annual cost [million CNY/year]"] - r4["Variable operating annual cost [million CNY/year]"]) / r0["Variable operating annual cost [million CNY/year]"]
    co2_drop = 100 * (r0["Annual CO2 emissions [tCO2/year]"] - r4["Annual CO2 emissions [tCO2/year]"]) / r0["Annual CO2 emissions [tCO2/year]"]
    peak_drop = 100 * (r0["Peak grid import [MW]"] - r4["Peak grid import [MW]"]) / max(r0["Peak grid import [MW]"], 1e-9)
    curtail_drop = r0["Renewable curtailment rate [%]"] - r4["Renewable curtailment rate [%]"]
    k1.metric("年可变运行成本降低", f"{cost_drop:.1f}%")
    k2.metric("年碳排放降低", f"{co2_drop:.1f}%")
    k3.metric("主网峰值降低", f"{peak_drop:.1f}%")
    k4.metric("弃风弃光率降低", f"{curtail_drop:.2f} 个百分点")

    tabs = st.tabs(["模型逻辑", "同江特征", "结果图总览"])
    with tabs[0]:
        st.markdown(
            """
            **模型回答四类问题：**
            1. 光伏、风电、CHP、热泵、电制冷机和储能应该配置多大；
            2. 储能、柔性负荷与上级电网应该如何协同运行；
            3. 如何在经济成本、碳排放、新能源消纳与主网峰值之间取得平衡；
            4. 主网降额、低风光与高负荷叠加时，关键负荷能否得到保障。
            """
        )
    with tabs[1]:
        st.markdown(
            """
            平台以**春季灌溉、春秋光伏大发、夏季商业制冷、秋季农产品加工、冬季采暖、春节返乡**六类典型日描述同江县域用能特征，并设置主网降额、低风光和高负荷复合扰动场景。
            """
        )
    with tabs[2]:
        montage = [FIGURE_ROOT / "03_planning_scenario_performance.png", FIGURE_ROOT / "06_fixed_capacity_operation_effectiveness.png", FIGURE_ROOT / "17_reliability_stress_summary.png"]
        cols = st.columns(3)
        for col, p in zip(cols, montage):
            if p.exists(): col.image(str(p), use_container_width=True)


def page_inputs():
    show_hero("输入数据与典型日", "查看六类同江特征化代表日的电、热、冷负荷与风光资源")
    df = load_top("input_profiles_v17_3_1_tongjiang_characteristic.csv")
    days = df[["day_type", "day_name"]].drop_duplicates()
    options = days["day_type"].tolist()
    selected = st.selectbox("选择典型日", options, format_func=lambda x: DAY_NAME.get(x, x))
    sub = df[df["day_type"] == selected].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("电负荷峰值", f'{sub["electric_load"].max():.2f} MW')
    c2.metric("热负荷峰值", f'{sub["heat_load"].max():.2f} MW')
    c3.metric("冷负荷峰值", f'{sub["cooling_load"].max():.2f} MW')
    c4.metric("代表天数", f'{int(sub["day_weight"].iloc[0])} 天/年')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, subplot_titles=("电—热—冷负荷", "风光出力系数"))
    for col, name in [("electric_load", "电负荷"), ("heat_load", "热负荷"), ("cooling_load", "冷负荷")]:
        fig.add_trace(go.Scatter(x=sub["hour"], y=sub[col], name=name, mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["hour"], y=sub["pv_cf"], name="光伏出力系数", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub["hour"], y=sub["wt_cf"], name="风电出力系数", mode="lines"), row=2, col=1)
    fig.update_yaxes(title_text="功率 / MW", row=1, col=1)
    fig.update_yaxes(title_text="出力系数", row=2, col=1)
    fig.update_xaxes(title_text="时刻 / h", row=2, col=1)
    fig.update_layout(height=700, legend_orientation="h", margin=dict(l=20,r=20,t=70,b=20), font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    load_cols = ["residential_load", "commercial_load", "industrial_load", "cold_storage_load", "agro_processing_load", "irrigation_load"]
    load_name = {"residential_load":"居民", "commercial_load":"商业", "industrial_load":"一般工业", "cold_storage_load":"冷链", "agro_processing_load":"农产品加工", "irrigation_load":"灌溉"}
    melted = sub[["hour"] + load_cols].melt("hour", var_name="类型", value_name="功率")
    melted["类型"] = melted["类型"].map(load_name)
    fig2 = px.area(melted, x="hour", y="功率", color="类型", title="电负荷构成")
    fig2.update_layout(height=430, xaxis_title="时刻 / h", yaxis_title="功率 / MW", legend_title="", font=dict(size=14))
    st.plotly_chart(fig2, use_container_width=True)


def page_planning():
    show_hero("规划方案对比", "S0—S4共同优化设备容量与多典型日运行策略，全部成本以人民币展示")
    df = load_top("planning_summary_v17_3_1.csv").copy()
    df["场景"] = df["Scenario"].map(SCENARIO_NAME)
    df["年社会综合成本（万元）"] = df["Social total annual cost [million CNY/year]"] * 100
    df["年私人成本（万元）"] = df["Private total annual cost [million CNY/year]"] * 100

    a, b = st.columns(2)
    with a:
        st.plotly_chart(plot_bar(df, "Scenario", "年社会综合成本（万元）", "年社会综合成本", "万元/年"), use_container_width=True)
    with b:
        st.plotly_chart(plot_bar(df, "Scenario", "Annual CO2 emissions [tCO2/year]", "年度碳排放", "tCO₂/年"), use_container_width=True)

    st.subheader("设备容量配置")
    cap_cols = {
        "PV capacity [MW]":"光伏", "WT capacity [MW]":"风电", "CHP capacity [MW]":"CHP",
        "Heat pump capacity [MW]":"热泵", "Electric chiller capacity [MW]":"电制冷机",
        "Battery power capacity [MW]":"电储能功率", "Thermal storage power capacity [MW]":"热储能功率",
    }
    long = df[["Scenario"] + list(cap_cols)].melt("Scenario", var_name="设备", value_name="容量")
    long["设备"] = long["设备"].map(cap_cols)
    fig = px.bar(long, x="Scenario", y="容量", color="设备", barmode="group", title="功率型设备容量对比")
    fig.update_layout(height=500, xaxis_title="", yaxis_title="容量 / MW", legend_title="", font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    display = chinese_summary(df, {
        "Scenario":"场景", "Name":"场景名称",
        "Annualized capital cost [million CNY/year]":"年化资本成本（百万元/年）",
        "Fixed O&M cost [million CNY/year]":"固定运维成本（百万元/年）",
        "Variable operating annual cost [million CNY/year]":"年可变运行成本（百万元/年）",
        "Private total annual cost [million CNY/year]":"年私人成本总额（百万元/年）",
        "Carbon external cost [million CNY/year]":"碳外部成本（百万元/年）",
        "Social total annual cost [million CNY/year]":"年社会综合成本（百万元/年）",
        "Annual CO2 emissions [tCO2/year]":"年碳排放（tCO₂）",
        "Renewable local absorption rate [%]":"新能源本地吸纳率（%）",
        "Renewable curtailment rate [%]":"弃风弃光率（%）",
        "Annual unmet load [MWh/year]":"年缺供能量（MWh）",
    })
    st.dataframe(display.round(3), use_container_width=True, hide_index=True)


def page_operation():
    show_hero("同容量运行策略对比", "固定S4设备容量，仅改变规则运行、储能优化、柔性负荷与低碳协同策略")
    df = load_top("operation_summary_v17_3_1.csv").copy()
    df["年可变运行成本（万元）"] = df["Variable operating annual cost [million CNY/year]"] * 100

    metrics = [
        ("年可变运行成本（万元）", "运行成本", "万元/年"),
        ("Annual CO2 emissions [tCO2/year]", "年碳排放", "tCO₂/年"),
        ("Peak grid import [MW]", "主网最大购电功率", "MW"),
        ("Renewable curtailment rate [%]", "弃风弃光率", "%"),
        ("Renewable local absorption rate [%]", "新能源本地吸纳率", "%"),
        ("Annual grid import [MWh/year]", "年购电量", "MWh/年"),
    ]
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (field, title, unit) in zip(cols, metrics[i:i+3]):
            with col:
                st.plotly_chart(plot_bar(df, "Scenario", field, title, unit), use_container_width=True)

    st.subheader("关键典型日主网交换功率")
    day = st.selectbox("典型日", list(DAY_NAME), format_func=lambda x: DAY_NAME[x], key="op_day")
    d0 = load_dispatch("operation", "R0")
    d4 = load_dispatch("operation", "R4")
    fig = go.Figure()
    for data, name, color in [(d0, "R0 常规规则运行", "#64748b"), (d4, "R4 低碳协同优化", "#ef4444")]:
        sub = data[data["day_type"] == day]
        fig.add_trace(go.Scatter(x=sub["hour"], y=sub["P_grid_net"], name=name, mode="lines", line=dict(width=3, color=color)))
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    fig.update_layout(height=430, xaxis_title="时刻 / h", yaxis_title="主网净交换功率 / MW", legend_title="", font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("主网净交换功率大于0表示购电，小于0表示向上级电网送电。")


def page_lowcarbon():
    show_hero("低碳与新能源消纳", "区分本地吸纳、外送与弃风弃光，并展示成本—减排权衡")
    op = load_top("operation_summary_v17_3_1.csv").copy()
    disposition = op[["Scenario", "Renewable local absorption rate [%]", "Renewable export rate [%]", "Renewable curtailment rate [%]"]].rename(columns={
        "Renewable local absorption rate [%]":"本地吸纳", "Renewable export rate [%]":"外送", "Renewable curtailment rate [%]":"弃风弃光"
    })
    long = disposition.melt("Scenario", var_name="去向", value_name="比例")
    fig = px.bar(long, x="Scenario", y="比例", color="去向", barmode="stack", title="R0—R4新能源可用量去向")
    fig.update_layout(height=480, yaxis_title="比例 / %", xaxis_title="", legend_title="", font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    r3 = op[op["Scenario"]=="R3"].iloc[0]
    r4 = op[op["Scenario"]=="R4"].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R3年可变运行成本", fmt_wan(cny_wan_from_million(r3["Variable operating annual cost [million CNY/year]"])))
    c2.metric("R4年可变运行成本", fmt_wan(cny_wan_from_million(r4["Variable operating annual cost [million CNY/year]"])))
    c3.metric("R3年碳排放", f'{r3["Annual CO2 emissions [tCO2/year]"]:,.0f} tCO₂')
    c4.metric("R4年碳排放", f'{r4["Annual CO2 emissions [tCO2/year]"]:,.0f} tCO₂')

    tab1, tab2 = st.tabs(["碳上限敏感性", "外送能力敏感性"])
    with tab1:
        df = load_top("carbon_cap_sensitivity_v17_3_1.csv").copy()
        if "Carbon cap ratio to R3 [%]" in df.columns:
            x_values = df["Carbon cap ratio to R3 [%]"]
        else:
            x_values = pd.to_numeric(df["Scenario"].astype(str).str.replace("C", "", regex=False), errors="coerce")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x_values, y=df["Annual CO2 emissions [tCO2/year]"], name="年碳排放", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_values, y=df["Variable operating annual cost [million CNY/year]"]*100, name="年可变运行成本", mode="lines+markers"), secondary_y=True)
        fig.update_xaxes(title_text="碳排放上限相对R3自然排放 / %")
        fig.update_yaxes(title_text="年碳排放 / tCO₂", secondary_y=False)
        fig.update_yaxes(title_text="运行成本 / 万元·年⁻¹", secondary_y=True)
        fig.update_layout(height=480, legend_orientation="h", font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        df = load_top("export_capacity_sensitivity_v17_3_1.csv")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df["Export capacity limit [MW]"], y=df["Renewable curtailment rate [%]"], name="弃风弃光率", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df["Export capacity limit [MW]"], y=df["Variable operating annual cost [million CNY/year]"]*100, name="年可变运行成本", mode="lines+markers"), secondary_y=True)
        fig.update_xaxes(title_text="主网外送能力上限 / MW")
        fig.update_yaxes(title_text="弃风弃光率 / %", secondary_y=False)
        fig.update_yaxes(title_text="运行成本 / 万元·年⁻¹", secondary_y=True)
        fig.update_layout(height=480, legend_orientation="h", font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)


def page_reliability():
    show_hero("可靠性与压力测试", "关键负荷优先保障，检验主网降额、低风光和高负荷复合扰动下的供能能力")
    df = load_top("reliability_stress_summary_v17_3_1.csv")
    c1,c2,c3,c4 = st.columns(4)
    q3 = df[df["Scenario"]=="Q3"].iloc[0]
    c1.metric("Q3综合供能服务率", f'{q3["Total multi-energy service rate [%]"]:.2f}%')
    c2.metric("Q3关键负荷服务率", f'{q3["Critical-load service rate [%]"]:.2f}%')
    c3.metric("Q3缺供能量", f'{q3["Total unmet [MWh/day]"]:.2f} MWh/日')
    c4.metric("Q3缺供持续时间", f'{q3["Shortage duration [h/day]"]:.2f} h/日')

    left,right = st.columns(2)
    with left:
        st.plotly_chart(plot_bar(df, "Scenario", "Total multi-energy service rate [%]", "综合供能服务率", "%"), use_container_width=True)
    with right:
        st.plotly_chart(plot_bar(df, "Scenario", "Total unmet [MWh/day]", "事件日缺供能量", "MWh/日"), use_container_width=True)

    st.subheader("Q3严重扰动运行机理")
    q3d = load_dispatch("reliability", "Q3")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08, subplot_titles=("主网可用率与风光出力系数", "电负荷、关键电负荷与供电资源", "电池SOC与缺供"))
    fig.add_trace(go.Scatter(x=q3d["hour"], y=q3d["grid_import_availability"], name="主网可用率"), row=1,col=1)
    fig.add_trace(go.Scatter(x=q3d["hour"], y=q3d["pv_cf"], name="光伏出力系数"), row=1,col=1)
    fig.add_trace(go.Scatter(x=q3d["hour"], y=q3d["wt_cf"], name="风电出力系数"), row=1,col=1)
    for col,name in [("electric_load","总电负荷"),("critical_electric_load","关键电负荷"),("P_grid_buy","主网购电"),("P_PV","光伏"),("P_WT","风电"),("P_CHP","CHP发电"),("P_BAT_dis","电储能放电")]:
        fig.add_trace(go.Scatter(x=q3d["hour"], y=q3d[col], name=name, mode="lines"), row=2,col=1)
    fig.add_trace(go.Scatter(x=q3d["hour"], y=100*q3d["SOC_BAT"]/max(float(q3d["SOC_BAT"].max()),1e-9), name="电池SOC相对值/%"), row=3,col=1)
    fig.add_trace(go.Bar(x=q3d["hour"], y=q3d["P_unmet_e"], name="非关键电负荷缺供"), row=3,col=1)
    fig.update_yaxes(title_text="比例", row=1,col=1)
    fig.update_yaxes(title_text="功率 / MW", row=2,col=1)
    fig.update_yaxes(title_text="SOC / %；缺供 / MW", row=3,col=1)
    fig.update_xaxes(title_text="时刻 / h", row=3,col=1)
    fig.update_layout(height=820, legend_orientation="h", margin=dict(t=80), font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("缺供价值（VOLL）敏感性")
    vs = load_top("reliability_voll_sensitivity_v17_3_1.csv")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=vs["VOLL multiplier"], y=vs["Total unmet [MWh/day]"], name="缺供量", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=vs["VOLL multiplier"], y=vs["Event-day social operating cost [EUR/day]"]*CNY_PER_EUR/10000, name="事件日社会运行成本", mode="lines+markers"), secondary_y=True)
    fig.update_xaxes(title_text="VOLL倍数")
    fig.update_yaxes(title_text="缺供量 / MWh·日⁻¹", secondary_y=False)
    fig.update_yaxes(title_text="事件日社会运行成本 / 万元·日⁻¹", secondary_y=True)
    fig.update_layout(height=480, legend_orientation="h", font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)


def _result_value(df: pd.DataFrame, key: str, key_col: str, val_col: str = "Value", default=0.0):
    row = df[df[key_col] == key]
    return default if row.empty else float(row[val_col].iloc[0])


@st.cache_data(show_spinner=False)
def run_quick_case(scenario_key: str, carbon_ratio: float, export_cap: float, export_share: float, curtail_penalty: float, voll: float, seed: int):
    from data_profiles_v17_3_1_tongjiang import generate_tongjiang_characteristic_profiles_v17_3_1
    from scenarios_v17_3_1_county import get_planning_scenarios_v17_3_1
    from model_core_v17_3_1_county import solve_integrated_energy_system_v17_3_1

    profiles = generate_tongjiang_characteristic_profiles_v17_3_1(n_steps_per_hour=1, seed=int(seed))
    scenarios = get_planning_scenarios_v17_3_1()
    for sc in scenarios.values():
        sc["grid_export_capacity"] = float(export_cap)
        sc["max_annual_export_share_of_renewables"] = float(export_share)
        sc["renewable_curtailment_penalty"] = float(curtail_penalty)
        sc["electric_unserved_energy_penalty"] = float(voll)
        sc["co2_cap_ratio_to_s0"] = float(carbon_ratio)
        sc["milp_time_limit"] = 180.0
        sc["solver_display"] = False

    if scenario_key == "S0":
        return solve_integrated_energy_system_v17_3_1(profiles, scenarios["S0"]), profiles

    s0 = solve_integrated_energy_system_v17_3_1(profiles, scenarios["S0"])
    if not s0.get("success", False):
        return s0, profiles
    baseline = _result_value(s0["carbon"], "Daily average CO2 emissions", "Item")
    scenario = dict(scenarios[scenario_key])
    scenario["co2_cap"] = baseline * float(carbon_ratio)
    return solve_integrated_energy_system_v17_3_1(profiles, scenario), profiles


def page_online():
    show_hero("在线快速试算", "使用小时级六典型日快速求解；正式论文结果仍以15分钟离线计算为准")
    st.markdown('<div class="note">在线试算用于交互演示和参数方向判断。由于Streamlit云端资源有限，默认采用每小时1点；完整15分钟模型请在本地运行。</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        scenario = st.selectbox("规划场景", ["S0","S1","S2","S3","S4"], format_func=lambda x: SCENARIO_NAME[x])
        carbon_ratio = st.slider("碳上限相对S0比例", .65, 1.00, .80, .01)
    with c2:
        export_cap = st.slider("主网外送功率上限 / MW", 0.0, 20.0, 5.0, .5)
        export_share = st.slider("年新能源外送比例上限", 0.00, .30, .10, .01)
    with c3:
        curtail = st.slider("弃风弃光惩罚 / EUR·MWh⁻¹", 0.0, 100.0, 20.0, 5.0)
        voll = st.slider("电力缺供价值 / EUR·MWh⁻¹", 2000.0, 30000.0, 18000.0, 1000.0)
    seed = st.number_input("随机种子", min_value=1, max_value=9999, value=42)

    if st.button("开始快速求解", type="primary", use_container_width=True):
        with st.spinner("正在建立并求解小时级规划模型……"):
            try:
                result, profiles = run_quick_case(scenario, carbon_ratio, export_cap, export_share, curtail, voll, int(seed))
            except Exception as e:
                st.error(f"求解失败：{e}")
                return
        if not result.get("success", False):
            st.error(f"求解器未找到可行解：{result.get('message','未知原因')}")
            return
        st.success("求解成功")
        metrics = result["metrics"]
        cap = result["capacity"]
        dispatch = result["dispatch"]
        def mv(name): return _result_value(metrics, name, "Metric")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("年私人成本总额", fmt_wan(mv("Private total annual cost")*CNY_PER_EUR/10000))
        k2.metric("年社会综合成本", fmt_wan(mv("Social total annual cost")*CNY_PER_EUR/10000))
        k3.metric("年碳排放", f'{mv("Annual CO2 emissions"):,.0f} tCO₂')
        k4.metric("弃风弃光率", f'{mv("Renewable curtailment rate"):.2f}%')

        cap_cn = cap[["Technology","Capacity","Unit"]].copy()
        tech_map={"PV":"光伏","WT":"风电","CHP":"CHP","Heat pump":"热泵","Electric chiller":"电制冷机","Gas boiler":"燃气锅炉","Battery energy":"电储能能量","Battery power":"电储能功率","Thermal storage energy":"热储能能量","Thermal storage power":"热储能功率"}
        cap_cn["Technology"] = cap_cn["Technology"].map(tech_map).fillna(cap_cn["Technology"])
        cap_cn.columns=["设备","容量","单位"]
        st.dataframe(cap_cn.round(3), use_container_width=True, hide_index=True)

        day = "spring_pv_high"
        sub = dispatch[dispatch["day_type"]==day]
        fig = go.Figure()
        for col,name in [("electric_load","电负荷"),("P_PV","光伏"),("P_WT","风电"),("P_grid_net","主网净交换"),("P_BAT_dis","储能放电")]:
            fig.add_trace(go.Scatter(x=sub["hour"], y=sub[col], name=name, mode="lines+markers"))
        fig.update_layout(title="春秋光伏大发日快速试算结果", height=470, xaxis_title="时刻 / h", yaxis_title="功率 / MW", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        bio=BytesIO()
        with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,frame in [("capacity_results.csv",cap_cn),("dispatch_results.csv",dispatch),("metrics_results.csv",metrics),("cost_results.csv",result["cost"]),("carbon_results.csv",result["carbon"])]:
                zf.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
        st.download_button("下载本次试算结果ZIP", bio.getvalue(), file_name=f"{scenario}_quick_result.zip", mime="application/zip")


def page_gallery():
    show_hero("成果图集", "集中查看V17.3.1论文级结果图；图片均来自15分钟正式计算结果")
    files = sorted(FIGURE_ROOT.glob("*.png"))
    labels = {
        "01":"同江特征化多典型日输入", "02":"县域电负荷构成", "03":"规划场景综合性能",
        "04":"设备容量配置", "05":"规划成本构成", "06":"同容量运行效果",
        "07":"关键日主网交换", "08":"光伏大发日源荷储联动", "09":"关键日储能SOC",
        "10":"新能源去向", "11":"R3与R4成本—减排权衡", "12":"碳上限敏感性",
        "13":"外送能力敏感性", "14":"同江特征日改善效果", "15":"模型物理一致性诊断",
        "16":"优化效果汇总", "17":"可靠性压力测试", "18":"严重扰动可靠性机理", "19":"VOLL敏感性",
    }
    selected = st.multiselect("选择要展示的图片", [p.name for p in files], default=[p.name for p in files[:6]], format_func=lambda x: labels.get(x[:2], x))
    for name in selected:
        p=FIGURE_ROOT/name
        st.subheader(labels.get(name[:2], name))
        st.image(str(p), use_container_width=True)


def page_downloads():
    show_hero("数据与结果下载", "下载中文化结果表、原始调度数据和正式结果图")
    summary_files = sorted(RESULT_ROOT.glob("*.csv"))
    st.download_button(
        "下载全部汇总CSV",
        data=to_zip_bytes(summary_files),
        file_name="V17_3_1_汇总结果.zip",
        mime="application/zip",
        use_container_width=True,
    )
    c1,c2,c3 = st.columns(3)
    for col,(group,label,keys) in zip([c1,c2,c3],[
        ("planning","规划场景",["S0","S1","S2","S3","S4"]),
        ("operation","运行场景",["R0","R1","R2","R3","R4"]),
        ("reliability","可靠性场景",["Q0","Q1","Q2","Q3"]),
    ]):
        with col:
            st.subheader(label)
            key=st.selectbox("选择场景",keys,format_func=lambda x:SCENARIO_NAME[x],key=f"download_{group}")
            files=list((RESULT_ROOT/group/key/"data").glob("*.csv"))
            st.download_button("下载该场景全部CSV",to_zip_bytes(files),file_name=f"{key}_数据.zip",mime="application/zip",use_container_width=True)

    st.subheader("平台说明")
    st.markdown("- 金额统一按人民币展示，汇率采用 **1 EUR = 7.8 CNY**。\n- 平台内置结果来自V17.3.1默认15分钟正式算例。\n- 在线试算采用小时级快速模式。\n- 同江内容为特征化场景，不替代真实线路潮流、电压与实测数据。")


PAGES = {
    "首页总览": page_home,
    "输入与典型日": page_inputs,
    "规划方案": page_planning,
    "运行策略": page_operation,
    "低碳与新能源": page_lowcarbon,
    "可靠性分析": page_reliability,
    "在线快速试算": page_online,
    "成果图集": page_gallery,
    "下载中心": page_downloads,
}

with st.sidebar:
    st.markdown("## ⚡ 县域综合能源系统")
    st.caption("同江特征化算例平台 · V17.3.1")
    page_name = st.radio("功能导航", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown("**金额口径**")
    st.caption("人民币；1 EUR = 7.8 CNY")
    st.markdown("**计算口径**")
    st.caption("内置正式结果：15分钟；在线试算：小时级")

PAGES[page_name]()
