import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="股票随机散户模拟器", layout="wide")

st.title("📈 股票随机散户行为模拟器")
st.markdown("模拟在真实历史行情下，带有每日现金流和概率买卖行为的散户群体资产分化与演变过程（收益率以起始点0%计）。")

# Sidebar inputs
st.sidebar.header("⚙️ 模拟参数设置")

ticker = st.sidebar.text_input("股票 Ticker", value="QQQ").upper()

default_end = datetime.today().date()
default_start = default_end - timedelta(days=365)

start_date = st.sidebar.date_input("开始日期", value=default_start)
end_date = st.sidebar.date_input("结束日期", value=default_end)

num_agents = st.sidebar.slider("随机散户人数", min_value=10, max_value=500, value=100, step=10)
initial_cash = st.sidebar.number_input("每人初始现金 ($)", min_value=0.0, value=10000.0, step=1000.0)
daily_cash = st.sidebar.number_input("每日现金进账 ($)", min_value=0.0, value=100.0, step=10.0)

st.sidebar.subheader("交易行为概率设置")
prob_buy = st.sidebar.slider("每日买入概率 (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
prob_sell = st.sidebar.slider("每日卖出概率 (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

if prob_buy + prob_sell > 100.0:
    st.sidebar.error("错误：买入概率与卖出概率之和不能超过 100%！")
    st.stop()

prob_hold = 100.0 - prob_buy - prob_sell
st.sidebar.info(f"自动计算：什么都不做（Hold）概率为 **{prob_hold:.1f}%**")

trade_size_mean = st.sidebar.slider("交易资金/股票比例均值 (%) [正态分布]", min_value=10.0, max_value=100.0, value=50.0, step=5.0)
trade_size_std = st.sidebar.slider("交易比例标准差", min_value=1.0, max_value=30.0, value=15.0, step=1.0)

run_button = st.sidebar.button("🚀 开始模拟", type="primary")

@st.cache_data(ttl=3600)
def load_data(tk, start, end):
    df = yf.download(tk, start=start, end=end, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

if run_button:
    if start_date >= end_date:
        st.error("开始日期必须早于结束日期！")
        st.stop()
        
    with st.spinner(f"正在获取 {ticker} 的历史数据并进行群体模拟..."):
        df_stock = load_data(ticker, start_date, end_date)
        
        if df_stock is None or df_stock.empty:
            st.error(f"未能获取到 ticker [{ticker}] 的有效数据，请检查代号或日期范围。")
            st.stop()
            
        prices = df_stock['Close']
        dates = prices.index
        n_days = len(prices)
        
        if n_days < 2:
            st.error("数据天数太少，无法进行模拟。")
            st.stop()
            
        stock_values = prices.values
        benchmark_returns = (stock_values / stock_values[0] - 1.0) * 100.0
        
        # Simulation arrays
        cash = np.full(num_agents, initial_cash)
        shares = np.zeros(num_agents)
        
        portfolio_history = np.zeros((n_days, num_agents))
        net_invested_history = np.zeros((n_days, num_agents))
        
        np.random.seed(42) # Reproducible randomness per run configuration
        
        actions_rand = np.random.uniform(0, 100, size=(n_days, num_agents))
        buy_threshold = prob_buy
        sell_threshold = prob_buy + prob_sell
        
        trade_proportions = np.random.normal(trade_size_mean / 100.0, trade_size_std / 100.0, size=(n_days, num_agents))
        trade_proportions = np.clip(trade_proportions, 0.05, 1.0)
        
        for t in range(n_days):
            p = stock_values[t]
            
            # Daily cash inflow starting from t > 0
            if t > 0:
                cash += daily_cash
                
            net_invested = initial_cash + t * daily_cash
            net_invested_history[t] = net_invested
                
            # Execute actions
            act_t = actions_rand[t]
            prop_t = trade_proportions[t]
            
            # Buy actions
            buying_mask = act_t < buy_threshold
            if np.any(buying_mask):
                spend_cash = cash[buying_mask] * prop_t[buying_mask]
                can_buy_shares = spend_cash / p
                cash[buying_mask] -= spend_cash
                shares[buying_mask] += can_buy_shares
                
            # Sell actions
            selling_mask = (act_t >= buy_threshold) & (act_t < sell_threshold)
            if np.any(selling_mask):
                sell_shares = shares[selling_mask] * prop_t[selling_mask]
                shares[selling_mask] -= sell_shares
                cash[selling_mask] += sell_shares * p
                
            # Record total portfolio value
            portfolio_history[t] = cash + shares * p
            
        # Calculate true percentage return
        returns_history = ((portfolio_history - net_invested_history) / net_invested_history) * 100.0
        
        final_returns = returns_history[-1]
        best_idx = np.argmax(final_returns)
        worst_idx = np.argmin(final_returns)
        
        st.success(f"模拟完成！共模拟了 {num_agents} 位散户在 {n_days} 个交易日内的表现。")
        
        # --- Chart 1: Main Stock & Agent Performance ---
        st.subheader(f"📊 {ticker} 走势与散户群体收益率演变")
        
        fig1 = go.Figure()
        
        for i in range(num_agents):
            if i == best_idx or i == worst_idx:
                continue
            ret = final_returns[i]
            color = 'rgba(0, 230, 118, 0.2)' if ret >= 0 else 'rgba(255, 23, 68, 0.2)'
            fig1.add_trace(go.Scatter(
                x=dates,
                y=returns_history[:, i],
                mode='lines',
                line=dict(width=1, color=color),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        fig1.add_trace(go.Scatter(
            x=dates,
            y=returns_history[:, worst_idx],
            mode='lines',
            name=f'最差散户 (#{worst_idx+1}: {final_returns[worst_idx]:.2f}%)',
            line=dict(width=3.5, color='#FF1744')
        ))
        
        fig1.add_trace(go.Scatter(
            x=dates,
            y=returns_history[:, best_idx],
            mode='lines',
            name=f'最好散户 (#{best_idx+1}: +{final_returns[best_idx]:.2f}%)',
            line=dict(width=3.5, color='#00E676')
        ))
        
        fig1.add_trace(go.Scatter(
            x=dates,
            y=benchmark_returns,
            mode='lines',
            name=f'{ticker} 股票基准走势 ({benchmark_returns[-1]:.2f}%)',
            line=dict(width=2.5, color='dodgerblue', dash='dash')
        ))
        
        fig1.update_layout(
            title=f"散户收益率群像对比 vs {ticker} 涨跌幅",
            xaxis_title="日期",
            yaxis_title="收益率 (%)",
            hovermode="x unified",
            template="plotly_white",
            height=650,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=100)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # --- Chart 2: Histogram with Time Slider ---
        st.subheader("📉 散户收益率分布动态直方图 (时间 Slider)")
        st.markdown("拖动下方滑块，查看在不同交易日时，散户收益率分布的形态变化（默认展示最后一天）。")
        
        step_size = max(1, n_days // 60)
        frame_indices = list(range(0, n_days, step_size))
        if (n_days - 1) not in frame_indices:
            frame_indices.append(n_days - 1)
            
        fig2 = go.Figure()
        
        # Default display: Last day (n_days - 1)
        last_t_idx = n_days - 1
        last_rets = returns_history[last_t_idx]
        
        fig2.add_trace(go.Histogram(
            x=last_rets,
            xbins=dict(size=1.0), # 1% per bin
            marker_color='royalblue',
            opacity=0.75
        ))
        
        frames = []
        for t_idx in frame_indices:
            rets_t = returns_history[t_idx]
            current_date_str = dates[t_idx].strftime('%Y-%m-%d')
            frames.append(go.Frame(
                data=[go.Histogram(x=rets_t, xbins=dict(size=1.0))],
                name=str(t_idx),
                layout=dict(title_text=f"日期: {current_date_str} (交易日 {t_idx+1}/{n_days})")
            ))
            
        fig2.frames = frames
        
        # Find which step corresponds to the last day for active index
        active_step_idx = len(frame_indices) - 1
        
        sliders = [{
            "active": active_step_idx,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "当前进度日期: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 50, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(t_idx)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                    "label": dates[t_idx].strftime('%Y-%m-%d'),
                    "method": "animate"
                }
                for t_idx in frame_indices
            ]
        }]
        
        fig2.update_layout(
            title=f"日期: {dates[last_t_idx].strftime('%Y-%m-%d')} (交易日 {n_days}/{n_days})",
            xaxis_title="收益率 (%)",
            yaxis_title="散户人数",
            sliders=sliders,
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary metrics table
        st.subheader("📋 模拟结果摘要")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("股票总回报率", f"{benchmark_returns[-1]:.2f}%")
        col2.metric("最好散户回报率", f"+{final_returns[best_idx]:.2f}%")
        col3.metric("最差散户回报率", f"{final_returns[worst_idx]:.2f}%")
        col4.metric("散户盈利占比", f"{(np.sum(final_returns > 0) / num_agents * 100):.1f}%")
        
else:
    st.info("👈 请在左侧栏调整参数，然后点击 **“开始模拟”** 按钮生成网页模拟结果。")
