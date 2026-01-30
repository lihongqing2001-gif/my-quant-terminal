# -*- coding: utf-8 -*-
import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib

# --- 0. 全局配置与高级 UI 注入 ---
st.set_page_config(page_title="AlphaQuant Pro", layout="wide", page_icon="⚡")
matplotlib.use("agg") 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ✨ CSS 美化魔法 (这是让 UI 变高级的关键)
st.markdown("""
<style>
    /* 全局背景色微调 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    h1 {
        color: #1a237e;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* 卡片式容器样式 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    
    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    
    /* 按钮美化 */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    /* 重点文字高亮 */
    .highlight {
        color: #2962ff;
        font-weight: bold;
    }
    
    /* 说明书样式 */
    .manual-text {
        font-size: 14px;
        color: #424242;
        line-height: 1.6;
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

# --- 2. 核心逻辑 (V8.0 极速引擎) ---
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
    """渲染内嵌说明书"""
    with st.expander("📖 产品白皮书 & 操作指南 (User Manual)", expanded=False):
        st.markdown("""
        <div class="manual-text">
        <h3>🚀 AlphaQuant 智能投研终端 (Pro)</h3>
        <p>本系统基于<strong>均值回归 (Mean Reversion)</strong> 原理，利用极速向量化引擎，为您寻找大宗商品的最佳交易机会。</p>
        
        <h4>🧠 核心原理</h4>
        <ul>
            <li><strong>价值中枢 (Middle)：</strong> 过去 N 天的均价，代表市场公允价值。</li>
            <li><strong>压力/支撑 (Bands)：</strong> 基于标准差 (σ) 构建的通道。突破上轨视为超买(做空)，跌破下轨视为超卖(做多)。</li>
            <li><strong>回归逻辑：</strong> 价格像橡皮筋，拉得越紧，回弹概率越大。</li>
        </ul>

        <h4>🕹️ 使用流程</h4>
        <ol>
            <li><strong>选品种：</strong> 在左侧选择主力合约（如螺纹钢、甲醇）。</li>
            <li><strong>跑测算：</strong> 点击“极速扫描”，系统会在1秒内回测过去3-5年的数据。</li>
            <li><strong>选策略：</strong> 系统会推荐三张卡片（进取型/防御型/平衡型），选择最适合你的一款。</li>
            <li><strong>看实盘：</strong> 进入“实盘指挥部”，获取具体的买卖点位和风控建议。</li>
        </ol>
        
        <p style="color:red; font-size:12px;">⚠️ 风险提示：历史回测不代表未来收益。量化模型仅作为决策辅助，请严格遵守风控纪律。</p>
        </div>
        """, unsafe_allow_html=True)

def render_card(col, title, row, key_suffix, desc, icon):
    """渲染高级美观的策略卡片"""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:#444;">{icon} {title}</h3>
            <p style="color:#888; font-size:12px;">{desc}</p>
            <div style="margin-top:15px; margin-bottom:15px;">
                <span style="font-size:24px; font-weight:bold; color:#2e7d32;">¥{row['pnl']:.0f}</span>
                <span style="font-size:12px; color:#666;"> 预期Alpha</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#555;">
                <span>⚡ 夏普: <b>{row['sharpe']:.2f}</b></span>
                <span>📉 回撤: <b>{row['drawdown']:.1f}%</b></span>
            </div>
            <div style="margin-top:5px; font-size:13px; color:#555;">
                <span>🔄 交易: <b>{int(row['trades'])}</b> 次</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Spacer
        if st.button(f"🚀 部署此策略", key=f"btn_{key_suffix}", use_container_width=True):
            switch_to_live(int(row['period']), float(row['dev']), st.session_state.current_train_symbol)
            st.rerun()

# ==========================================
# 📺 页面 1: 策略训练场
# ==========================================
def render_home():
    plt.close('all')
    
    # 顶部 Title 区域
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("AlphaQuant Pro ⚡")
        st.caption("基于向量化矩阵运算的智能大宗商品投研系统")
    with c2:
        st.image("https://img.icons8.com/color/96/bullish.png", width=80)

    # 插入说明书
    render_manual()
    
    st.divider()

    # 侧边栏
    with st.sidebar:
        st.header("🎯 资产配置 (Asset)")
        commodity_map = {
            "螺纹钢 (RB)": "rb", "热卷 (HC)": "hc", "铁矿石 (I)": "i",
            "甲醇 (MA)": "ma", "纯碱 (SA)": "sa", "玻璃 (FG)": "fg",
            "棕榈油 (P)": "p", "橡胶 (RU)": "ru", "白银 (AG)": "ag", "黄金 (AU)": "au"
        }
        selected_key = st.selectbox("选择主力品种", list(commodity_map.keys()), on_change=on_select_change)
        custom_input = st.text_input("自定义代码", key="custom_input", on_change=on_input_change)
        symbol_code = custom_input.lower() if custom_input else commodity_map[selected_key]
        info = get_symbol_info(symbol_code)
        
        st.info(f"🏦 {info['exch']}")
        st.caption(f"⚙️ 费率: {info['fee']} | 乘数: {info['mult']}")

    # 主操作区
    col_kpi, col_btn = st.columns([2, 1])
    with col_kpi:
        st.markdown(f"### 正在分析: <span class='highlight'>{info['name']} ({symbol_code.upper()})</span>", unsafe_allow_html=True)
    with col_btn:
        run_btn = st.button("🚀 启动极速扫描 (Instant Scan)", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.best_models = None
        with st.spinner("⚡️ 矩阵引擎正在运算 (Matrix Computing)..."):
            df_train, _ = get_backtest_data(symbol_code)
            if df_train is not None:
                st.session_state.current_train_symbol = symbol_code
                p_range = range(10, 90, 5)
                d_range = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8] 
                res_df = fast_optimize(df_train, p_range, d_range, info)
                
                if not res_df.empty:
                    # 排序逻辑
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

    # 结果展示区
    if 'best_models' in st.session_state:
        status = st.session_state.best_models.get('status')
        if status == 'failed_all_loss':
            st.error("⛔️ 策略失效：该品种在当前市场环境下无法获利。")
        elif status == 'success':
            models = st.session_state.best_models
            st.success(f"✅ 运算完成。为您挖掘出 3 组最佳参数：")
            
            c1, c2, c3 = st.columns(3)
            render_card(c1, "进取型 (Max PnL)", models['profit'], "p", "收益优先 | 适合激进资金", "🔥")
            render_card(c2, "防御型 (Max Sharpe)", models['sharpe'], "s", "稳健优先 | 适合保守资金", "🛡️")
            render_card(c3, "平衡型 (Balanced)", models['balance'], "b", "综合评分最高 | 推荐首选", "⚖️")

# ==========================================
# 📺 页面 2: 实盘 (Live)
# ==========================================
def render_live():
    plt.close('all')
    
    # 顶部导航
    c1, c2 = st.columns([1, 6])
    with c1:
        st.button("⬅️ 返回", on_click=switch_to_home, use_container_width=True)
    
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
            signal_status = "⚪ 观望 (WAIT)"
            signal_color = "#9e9e9e"
            bg_color = "#f5f5f5"
            signal_reason = "价格位于通道内，无偏离。"
            
            if curr_price >= sell_price * 0.99:
                signal_status = "🔴 卖出信号 (SHORT)"
                signal_color = "#d32f2f"
                bg_color = "#ffebee"
                signal_reason = f"价格触及上轨压力位，回归概率大。"
            elif curr_price <= buy_price * 1.01:
                signal_status = "🟢 买入信号 (LONG)"
                signal_color = "#2e7d32"
                bg_color = "#e8f5e9"
                signal_reason = f"价格触及下轨支撑位，反弹概率大。"

            # 仪表盘 UI
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:12px; border-left: 8px solid {signal_color}; margin-bottom:20px;">
                <h2 style="color:{signal_color}; margin:0;">{signal_status}</h2>
                <p style="color:#555; margin-top:5px;"><b>逻辑判定:</b> {signal_reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 核心数据卡片
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("最新价", f"{curr_price:.0f}")
            k2.metric("做空触发价", f"{sell_price:.0f}")
            k3.metric("做多触发价", f"{buy_price:.0f}")
            k4.metric("建议头寸", f"{max_lots} 手")

            st.divider()

            # 图表区
            col_chart, col_data = st.columns([3, 1])
            with col_chart:
                st.subheader("📉 价格通道监控")
                plot_data = df.iloc[-120:]
                fig, ax = plt.subplots(figsize=(10, 4.5))
                # 优化图表样式
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#f8f9fa')
                
                ax.plot(plot_data.index, plot_data['close'], 'k', lw=1.5, label='Price')
                ax.fill_between(plot_data.index, plot_data['UP'], plot_data['DOWN'], color='#1976d2', alpha=0.1)
                ax.plot(plot_data.index, plot_data['UP'], color='#1976d2', linestyle='--', alpha=0.5, lw=1)
                ax.plot(plot_data.index, plot_data['DOWN'], color='#1976d2', linestyle='--', alpha=0.5, lw=1)
                ax.plot(plot_data.index, plot_data['MA'], color='#ff9800', alpha=0.8, lw=1, label='MA')
                
                # 标记当前点
                point_color = 'red' if 'SHORT' in signal_status else ('green' if 'LONG' in signal_status else 'gray')
                ax.scatter(plot_data.index[-1], curr_price, s=120, color=point_color, zorder=5, edgecolors='white', linewidth=2)
                
                ax.legend(loc='upper left', frameon=False)
                ax.grid(True, linestyle=':', alpha=0.3)
                for spine in ax.spines.values(): spine.set_visible(False) # 去掉边框
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_data:
                st.markdown("""
                <div class="metric-card">
                    <h4>📋 执行指令单</h4>
                """, unsafe_allow_html=True)
                
                action = "Hold"
                if "SHORT" in signal_status: action = "Sell / Short"
                if "LONG" in signal_status: action = "Buy / Long"
                
                st.markdown(f"""
                - **合约:** `{contract_name}`
                - **动作:** **{action}**
                - **挂单:** `{curr_price:.0f}`
                - **止盈:** `{latest['MA']:.0f}`
                - **止损:** `{latest['STD']*0.5:.0f}` pts
                """)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("数据不足，无法计算指标。")
    else:
        st.error("行情连接失败，请稍后重试。")

# 路由
if st.session_state.page == 'home':
    render_home()
else:
    render_live()
