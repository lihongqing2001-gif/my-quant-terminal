# -*- coding: utf-8 -*-
import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go # 🔥 引入 Plotly 交互式图表库
from plotly.subplots import make_subplots

# --- 0. 全局配置 & 高级 UI 注入 ---
st.set_page_config(page_title="AlphaQuant Ultra", layout="wide", page_icon="⚡")

# ✨ 华尔街风格 CSS 注入
st.markdown("""
<style>
    /* 引入 Google Fonts: Roboto Mono (数字专用) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* 全局背景与字体优化 */
    .stApp {
        background-color: #f4f6f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 数字强制使用等宽字体，防止跳动 */
    .stMetric div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 26px;
    }
    
    /* 策略卡片样式升级 */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 24px 20px;
        border: 1px solid #eaedf0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-color: #d1d9e6;
    }
    
    /* 信号状态栏 */
    .signal-box {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 自定义按钮样式 */
    div.stButton > button {
        border-radius: 6px;
        height: 45px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* 侧边栏微调 */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #f0f2f6;
    }
    
    /* 说明书排版 */
    .manual-content h4 {
        color: #1a73e8;
        margin-top: 20px;
    }
    .manual-content li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'selected_params' not in st.session_state: st.session_state.selected_params = {'period': 20, 'dev': 2.0}
if 'target_symbol' not in st.session_state: st.session_state.target_symbol = 'rb'

# --- 1. 基础数据库 ---
BASIC_INFO = {
    "rb": {"name": "螺纹钢", "exch": "SHFE", "mode": "percent", "fee": 1.0, "mult": 10},
    "hc": {"name": "热卷", "exch": "SHFE", "mode": "percent", "fee": 1.0, "mult": 10},
    "i":  {"name": "铁矿石", "exch": "DCE",  "mode": "percent", "fee": 1.0, "mult": 100},
    "ma": {"name": "甲醇", "exch": "CZCE", "mode": "fixed",   "fee": 3.0, "mult": 10},
    "sa": {"name": "纯碱", "exch": "CZCE", "mode": "fixed",   "fee": 3.5, "mult": 20},
    "fg": {"name": "玻璃", "exch": "CZCE", "mode": "fixed",   "fee": 6.0, "mult": 20},
    "p":  {"name": "棕榈油", "exch": "DCE",  "mode": "percent", "fee": 2.5, "mult": 10},
    "ru": {"name": "橡胶", "exch": "SHFE", "mode": "fixed",   "fee": 3.0, "mult": 10},
    "ag": {"name": "白银", "exch": "SHFE", "mode": "percent", "fee": 0.5, "mult": 15},
    "au": {"name": "黄金", "exch": "SHFE", "mode": "fixed",   "fee": 10.0,"mult": 1000},
}

def get_symbol_info(symbol):
    default = {"name": symbol.upper(), "exch": "Unknown", "mode": "percent", "fee": 1.0, "mult": 10}
    return BASIC_INFO.get(symbol, default)

def on_select_change(): st.session_state.custom_input = ""
def on_input_change(): pass

# --- 2. 核心逻辑 (V8.0 极速引擎 - 保持不变) ---
@st.cache_data(ttl=3600*4)
def get_backtest_data(symbol_code):
    try:
        query = f"{symbol_code}0" if not any(c.isdigit() for c in symbol_code) else symbol_code
        df = ak.futures_zh_daily_sina(symbol=query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['openinterest'] = df['hold']
        df['volume'] = df['volume'].astype(float)
        start_date = datetime.datetime.now() - datetime.timedelta(days=365*3)
        df = df[df.index > start_date]
        return df, query
    except: return None, None

@st.cache_data(ttl=60)
def get_live_data(symbol_root):
    try:
        current_year = datetime.date.today().year % 100
        current_month = datetime.date.today().month
        contracts = []
        for i in range(6):
            m = (current_month + i - 1) % 12 + 1
            y = current_year + (current_month + i - 1) // 12
            contracts.append(f"{symbol_root}{y}{m:02d}")

        best_df, best_contract, max_oi = None, None, -1
        for code in contracts:
            try:
                df = ak.futures_zh_daily_sina(symbol=code)
                if not df.empty and df.iloc[-1]['hold'] > max_oi:
                    max_oi = df.iloc[-1]['hold']
                    best_contract = code
                    best_df = df
            except: pass
        if best_df is not None:
            best_df['date'] = pd.to_datetime(best_df['date'])
            best_df = best_df.set_index('date')
            return best_df, best_contract
        return None, None
    except: return None, None

def fast_optimize(df, period_range, dev_range, info):
    results = []
    price_change = df['close'].diff()
    fee_rate = info['fee'] / 10000.0 if info['mode'] == 'percent' else 0
    fixed_fee = info['fee'] if info['mode'] == 'fixed' else 0
    mult = info['mult']
    
    for p in period_range:
        ma = df['close'].rolling(window=p).mean()
        std = df['close'].rolling(window=p).std()
        for d in dev_range:
            upper = ma + d * std
            lower = ma - d * std
            long_entry = (df['close'] < lower)
            short_entry = (df['close'] > upper)
            
            pos = pd.Series(np.nan, index=df.index)
            pos[long_entry] = 1
            pos[short_entry] = -1
            pos[ (df['close'] >= ma) & (pos.shift(1)==1) ] = 0 
            pos[ (df['close'] <= ma) & (pos.shift(1)==-1) ] = 0 
            pos = pos.ffill().fillna(0)
            
            daily_pnl = pos.shift(1) * price_change * mult
            trades_count = pos.diff().abs().sum() / 2
            total_fee = trades_count * (df['close'].mean() * mult * fee_rate + fixed_fee) * 2
            total_pnl = daily_pnl.sum() - total_fee
            
            cum_pnl = daily_pnl.cumsum()
            peak = cum_pnl.cummax()
            drawdown = (cum_pnl - peak).min()
            dd_pct = abs(drawdown / 500000.0) * 100
            
            if daily_pnl.std() != 0:
                sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
            else:
                sharpe = -10
            
            if total_pnl > 0:
                results.append({'period': p, 'dev': d, 'pnl': total_pnl, 'sharpe': sharpe, 'drawdown': dd_pct, 'trades': trades_count})
    return pd.DataFrame(results)

def switch_to_live(period, dev, symbol):
    st.session_state.selected_params = {'period': period, 'dev': dev}
    st.session_state.target_symbol = symbol
    st.session_state.page = 'live'

def switch_to_home(): st.session_state.page = 'home'

# --- 3. UI 组件 ---
def render_manual():
    with st.expander("📖 系统白皮书与操作指南 (Docs)", expanded=False):
        st.markdown("""
        <div class="manual-content">
            <p><strong>AlphaQuant Ultra</strong> 是一款华尔街级别的量化决策系统。本版本引入了 Plotly 交互引擎，支持毫秒级回测与实时数据可视化。</p>
            <h4>🧠 核心原理</h4>
            <ul>
                <li><strong>均值回归 (Mean Reversion)：</strong> 价格像橡皮筋，拉得越紧（偏离均线越远），回弹概率越大。</li>
                <li><strong>布林通道 (Bollinger Bands)：</strong> 动态计算市场的“舒适区”。突破上轨即为超买，跌破下轨即为超卖。</li>
            </ul>
            <h4>🕹️ 操作流程</h4>
            <ol>
                <li><strong>Step 1:</strong> 选择左侧主力合约。</li>
                <li><strong>Step 2:</strong> 点击“启动极速扫描”，AI 将自动寻找过去 3 年最赚钱的参数。</li>
                <li><strong>Step 3:</strong> 部署推荐策略，进入实盘监控雷达。</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def render_card(col, title, row, key_suffix, desc, border_color):
    """渲染高级策略卡片"""
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {border_color};">
            <h3 style="color:#333;">{title}</h3>
            <p style="color:#888; font-size:12px; height:30px;">{desc}</p>
            <div style="margin: 20px 0;">
                <span style="font-family:'Roboto Mono'; font-size:28px; font-weight:700; color:#333;">¥{row['pnl']:,.0f}</span>
                <span style="font-size:12px; color:#2e7d32; background:#e8f5e9; padding:2px 6px; border-radius:4px;">+Alpha</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#555; border-top:1px solid #eee; padding-top:10px;">
                <span>⚡ 夏普: <b>{row['sharpe']:.2f}</b></span>
                <span>📉 回撤: <b>{row['drawdown']:.1f}%</b></span>
            </div>
            <div style="font-size:13px; color:#555; margin-top:5px;">
                <span>🔄 频次: <b>{int(row['trades'])}</b> 次 (3年)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button(f"立即部署", key=f"btn_{key_suffix}", use_container_width=True):
            switch_to_live(int(row['period']), float(row['dev']), st.session_state.current_train_symbol)
            st.rerun()

# ==========================================
# 📺 页面 1: 策略训练场
# ==========================================
def render_home():
    st.title("AlphaQuant Ultra ⚡")
    render_manual()
    st.divider()

    with st.sidebar:
        st.header("资产配置")
        commodity_map = {
            "螺纹钢 (RB)": "rb", "热卷 (HC)": "hc", "铁矿石 (I)": "i",
            "甲醇 (MA)": "ma", "纯碱 (SA)": "sa", "玻璃 (FG)": "fg",
            "棕榈油 (P)": "p", "橡胶 (RU)": "ru", "白银 (AG)": "ag", "黄金 (AU)": "au"
        }
        selected_key = st.selectbox("选择主力品种", list(commodity_map.keys()), on_change=on_select_change)
        custom_input = st.text_input("自定义代码", key="custom_input", on_change=on_input_change)
        symbol_code = custom_input.lower() if custom_input else commodity_map[selected_key]
        info = get_symbol_info(symbol_code)
        
        st.info(f"🏦 {info['exch']} | 💸 {info['fee']} ({info['mode']})")

    col_info, col_act = st.columns([3, 1])
    with col_info:
        st.markdown(f"### 正在分析: <span style='color:#2962ff'>{info['name']} ({symbol_code.upper()})</span>", unsafe_allow_html=True)
    with col_act:
        run_btn = st.button("启动极速扫描 (Instant Scan)", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.best_models = None
        with st.spinner("⚡️ 正在调用 V8.0 向量化引擎..."):
            df_train, _ = get_backtest_data(symbol_code)
            if df_train is not None:
                st.session_state.current_train_symbol = symbol_code
                p_range = range(10, 90, 5)
                d_range = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8] 
                res_df = fast_optimize(df_train, p_range, d_range, info)
                
                if not res_df.empty:
                    best_profit = res_df.sort_values(by='pnl', ascending=False).iloc[0]
                    valid_sharpe = res_df[res_df['trades'] >= 5]
                    if valid_sharpe.empty: valid_sharpe = res_df
                    best_sharpe = valid_sharpe.sort_values(by='sharpe', ascending=False).iloc[0]
                    res_df['score'] = res_df['pnl'] / (res_df['drawdown'] + 1)
                    best_balance = res_df.sort_values(by='score', ascending=False).iloc[0]
                    
                    st.session_state.best_models = {
                        'profit': best_profit, 'sharpe': best_sharpe, 'balance': best_balance,
                        'status': 'success'
                    }
                else:
                    st.session_state.best_models = {'status': 'failed_all_loss'}
            else:
                st.error("数据源异常")

    if 'best_models' in st.session_state:
        status = st.session_state.best_models.get('status')
        if status == 'failed_all_loss':
            st.error("策略失效：该品种在当前参数范围内无法实现盈利。")
        elif status == 'success':
            models = st.session_state.best_models
            st.success("运算完成。AI 优选出以下 3 组最佳参数：")
            
            c1, c2, c3 = st.columns(3)
            # 使用不同颜色的边框区分风格
            render_card(c1, "进取型 (Aggressive)", models['profit'], "p", "收益优先 | 适合激进资金", "#FF5252") # 珊瑚红
            render_card(c2, "稳健型 (Conservative)", models['sharpe'], "s", "稳健优先 | 适合保守资金", "#00C853") # 翡翠绿
            render_card(c3, "平衡型 (Balanced)", models['balance'], "b", "综合评分最高 | 推荐首选", "#2962FF") # 科技蓝

# ==========================================
# 📺 页面 2: 实盘指挥部 (Live)
# ==========================================
def render_live():
    # 顶部导航
    c1, c2 = st.columns([1, 8])
    with c1:
        st.button("← 返回", on_click=switch_to_home, use_container_width=True)
    
    params = st.session_state.selected_params
    target = st.session_state.target_symbol
    period = int(params['period'])
    dev = float(params['dev'])
    info = get_symbol_info(target)
    
    with st.spinner(f"正在接入 {target.upper()} 实时行情..."):
        df, contract_name = get_live_data(target)
        
    if df is not None:
        if len(df) > period + 20:
            df['MA'] = df['close'].rolling(window=period).mean()
            df['STD'] = df['close'].rolling(window=period).std()
            df['UP'] = df['MA'] + dev * df['STD']
            df['DOWN'] = df['MA'] - dev * df['STD']
            
            latest = df.iloc[-1]
            curr_price = latest['close']
            
            # 风控计算
            total_cash = 500000 
            risk_ratio = 0.2 
            margin_rate = 0.12 
            margin_per_lot = curr_price * info['mult'] * margin_rate
            max_lots = int((total_cash * risk_ratio) / margin_per_lot)
            if max_lots < 1: max_lots = 1
            
            buy_price = latest['DOWN']
            sell_price = latest['UP']
            
            # 信号判断
            # 使用更金融的配色: 涨/多=绿, 跌/空=红 (美股/加密货币习惯) 或 反之 (A股习惯)
            # 这里采用国际通用的：绿色=涨/做多(机会)，红色=跌/做空(警示) -> 稍微调整为 翡翠绿/珊瑚红
            
            signal_status = "观望 (WAIT)"
            bg_color = "#607d8b" # 灰色
            signal_reason = "价格位于通道内部，处于震荡区间。"
            
            if curr_price >= sell_price * 0.99:
                signal_status = "卖出信号 (SHORT)"
                bg_color = "#FF5252" # 珊瑚红
                signal_reason = "价格触及上轨压力位，回归概率大。"
            elif curr_price <= buy_price * 1.01:
                signal_status = "买入信号 (LONG)"
                bg_color = "#00C853" # 翡翠绿
                signal_reason = "价格触及下轨支撑位，反弹概率大。"

            # 仪表盘 UI
            st.markdown(f"""
            <div class="signal-box" style="background-color:{bg_color};">
                <h2 style="color:white; margin:0; font-size: 24px;">{signal_status}</h2>
                <p style="color:rgba(255,255,255,0.9); margin-top:5px; margin-bottom:0;">{signal_reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 核心数据
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("最新价", f"{curr_price:.0f}")
            k2.metric("做空压力位", f"{sell_price:.0f}")
            k3.metric("做多支撑位", f"{buy_price:.0f}")
            k4.metric("建议头寸", f"{max_lots} 手")

            st.divider()

            # 🔥 Plotly 交互式图表 🔥
            col_chart, col_data = st.columns([3, 1])
            with col_chart:
                st.subheader("价格通道监控 (Interactive)")
                plot_data = df.iloc[-150:] # 显示最近150天
                
                fig = go.Figure()

                # 1. 绘制通道区域 (Band Area)
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['UP'],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['DOWN'],
                    fill='tonexty', # 填充到上一条线
                    fillcolor='rgba(25, 118, 210, 0.08)', # 浅蓝色背景
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # 2. 绘制上下轨虚线
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['UP'],
                    mode='lines',
                    line=dict(color='rgba(25, 118, 210, 0.4)', width=1, dash='dash'),
                    name='上轨 (阻力)'
                ))
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['DOWN'],
                    mode='lines',
                    line=dict(color='rgba(25, 118, 210, 0.4)', width=1, dash='dash'),
                    name='下轨 (支撑)'
                ))

                # 3. 绘制中轨
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['MA'],
                    mode='lines',
                    line=dict(color='#FFA726', width=1.5),
                    name='价值中枢 (MA)'
                ))

                # 4. 绘制K线 (这里用收盘价连线简化，为了清晰展示通道关系)
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['close'],
                    mode='lines',
                    line=dict(color='#263238', width=2),
                    name='收盘价'
                ))

                # 5. 标记最新点
                fig.add_trace(go.Scatter(
                    x=[plot_data.index[-1]], y=[curr_price],
                    mode='markers',
                    marker=dict(size=12, color=bg_color, line=dict(width=2, color='white')),
                    name='最新价'
                ))

                # 图表布局优化
                fig.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)

            with col_data:
                st.markdown("""
                <div class="metric-card" style="padding:20px;">
                    <h4 style="margin-top:0; color:#333;">交易指令单</h4>
                """, unsafe_allow_html=True)
                
                action = "持有 (Hold)"
                if "SHORT" in signal_status: action = "卖出开仓 (Sell)"
                if "LONG" in signal_status: action = "买入开仓 (Buy)"
                
                st.markdown(f"""
                <ul style="padding-left:15px; font-size:14px; color:#444; line-height:2;">
                    <li><strong>合约:</strong> {contract_name}</li>
                    <li><strong>动作:</strong> <span style="font-weight:bold; color:{bg_color}">{action}</span></li>
                    <li><strong>挂单:</strong> <span style="font-family:'Roboto Mono'">{curr_price:.0f}</span></li>
                    <li><strong>止盈:</strong> <span style="font-family:'Roboto Mono'">{latest['MA']:.0f}</span></li>
                    <li><strong>止损:</strong> <span style="font-family:'Roboto Mono'">{latest['STD']*0.5:.0f}</span> pts</li>
                </ul>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("数据量不足，无法计算技术指标。")
    else:
        st.error("行情服务连接失败，请稍后重试。")

# 路由分发
if st.session_state.page == 'home':
    render_home()
else:
    render_live()
