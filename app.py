import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="股票随机散户模拟器", layout="wide")

# --- Multi-language Dictionary ---
LANGS = {
    "中文": {
        "lang_label": "🌐 语言选择 (Language)",
        "title": "📈 股票随机散户行为模拟器",
        "desc": "模拟在真实历史行情下，带有每日现金流和概率买卖行为的散户群体资产分化与演变过程（收益率以起始点0%计）。",
        "sidebar_header": "⚙️ 模拟参数设置",
        "ticker": "股票 Ticker",
        "start_date": "开始日期",
        "end_date": "结束日期",
        "num_agents": "随机散户人数",
        "initial_cash": "每人初始现金 ($)",
        "daily_cash": "每日现金进账 ($)",
        "trade_prob_header": "交易行为概率设置",
        "prob_buy": "每日买入概率 (%)",
        "prob_sell": "每日卖出概率 (%)",
        "prob_err": "错误：买入概率与卖出概率之和不能超过 100%！",
        "prob_hold": "自动计算：什么都不做（Hold）概率为",
        "trade_size_mean": "交易资金/股票比例均值 (%) [正态分布]",
        "trade_size_std": "交易比例标准差",
        "run_btn": "🚀 开始模拟",
        "err_start_end": "开始日期必须早于结束日期！",
        "spinner": "正在获取 {ticker} 的历史数据并进行群体模拟...",
        "err_no_data": "未能获取到 ticker [{ticker}] 的有效数据，请检查代号或日期范围。",
        "err_too_short": "数据天数太少，无法进行模拟。",
        "success_sim": "模拟完成！共模拟了 {num_agents} 位散户在 {n_days} 个交易日内的表现。",
        "chart1_title": "📊 {ticker} 走势与散户群体收益率演变",
        "chart1_main_title": "散户收益率群像对比 vs {ticker} 涨跌幅",
        "xaxis_date": "日期",
        "yaxis_return": "收益率 (%)",
        "worst_agent": "最差散户",
        "best_agent": "最好散户",
        "benchmark": "股票基准走势",
        "chart2_title": "📉 散户收益率分布动态直方图 (时间 Slider)",
        "chart2_desc": "拖动下方滑块，查看在不同交易日时，散户收益率分布的形态变化（默认展示最后一天）。",
        "global_range_text": "全局收益率范围锁定：",
        "to_text": "至",
        "per_bin": "1% 每柱，默认展示最后一天",
        "date_prefix": "当前进度日期: ",
        "yaxis_agents": "散户人数",
        "summary_title": "📋 模拟结果摘要",
        "stock_return": "股票总回报率",
        "best_return": "最好散户回报率",
        "worst_return": "最差散户回报率",
        "profit_ratio": "散户盈利占比",
        "sidebar_tip": "👈 请在左侧栏调整参数，然后点击 **“开始模拟”** 按钮生成网页模拟结果。"
    },
    "日本語": {
        "lang_label": "🌐 言語選択 (Language)",
        "title": "📈 株式ランダム個人投資家シミュレーター",
        "desc": "実際の歴史的相場において、日々のキャッシュフローと確率的な売買行動を行う個人投資家群の資産の分化と推移をシミュレーションします（利回りは開始時点0%基準）。",
        "sidebar_header": "⚙️ シミュレーションパラメータ設定",
        "ticker": "銘柄 Ticker",
        "start_date": "開始日",
        "end_date": "終了日",
        "num_agents": "個人投資家人数",
        "initial_cash": "初期現金 ($)",
        "daily_cash": "毎日の現金流入 ($)",
        "trade_prob_header": "取引行動確率設定",
        "prob_buy": "毎日の買い確率 (%)",
        "prob_sell": "毎日の売り確率 (%)",
        "prob_err": "エラー：買い確率と売り確率の合計は 100% を超えることはできません！",
        "prob_hold": "自動計算：ホールド（何もしない）確率",
        "trade_size_mean": "取引資金/株式比率の平均 (%) [正規分布]",
        "trade_size_std": "取引比率の標準偏差",
        "run_btn": "🚀 シミュレーション開始",
        "err_start_end": "開始日は終了日より前の日付を指定してください！",
        "spinner": "{ticker} の履歴データを取得し、群集シミュレーションを実行中...",
        "err_no_data": "ticker [{ticker}] の有効なデータを取得できませんでした。ティッカーまたは日付範囲を確認してください。",
        "err_too_short": "データの日数が少なすぎるため、シミュレーションを実行できません。",
        "success_sim": "シミュレーション完了！{n_days} 営業日における {num_agents} 人の個人投資家のパフォーマンスをシミュレーションしました。",
        "chart1_title": "📊 {ticker} の値動きと個人投資家群の利回り推移",
        "chart1_main_title": "個人投資家の利回り比較 vs {ticker} 騰落率",
        "xaxis_date": "日付",
        "yaxis_return": "利回り (%)",
        "worst_agent": "最悪の投資家",
        "best_agent": "最高の投資家",
        "benchmark": "株式ベンチマーク",
        "chart2_title": "📉 個人投資家利回り分布の動的ヒストグラム (タイムスライダー)",
        "chart2_desc": "下のスライダーをドラッグして、営業日ごとの利回り分布の変化を確認します（デフォルトでは最終日を表示）。",
        "global_range_text": "全体利回り範囲固定：",
        "to_text": "〜",
        "per_bin": "1%刻み、デフォルトは最終日",
        "date_prefix": "現在の日付: ",
        "yaxis_agents": "投資家人数",
        "summary_title": "📋 シミュレーション結果サマリー",
        "stock_return": "株式総リターン",
        "best_return": "最高投資家リターン",
        "worst_return": "最悪投資家リターン",
        "profit_ratio": "利益を出した投資家の割合",
        "sidebar_tip": "👈 左側サイドバーでパラメータを調整し、**「シミュレーション開始」** ボタンをクリックしてください。"
    }
}

# --- Sidebar Language Selector at the very top ---
selected_lang = st.sidebar.selectbox("🌐 语言选择 / Language", list(LANGS.keys()), index=0)
t = LANGS[selected_lang]

st.title(t["title"])
st.markdown(t["desc"])

# Sidebar inputs
st.sidebar.header(t["sidebar_header"])

ticker = st.sidebar.text_input(t["ticker"], value="QQQ").upper()

default_end = datetime.today().date()
default_start = default_end - timedelta(days=365)

start_date = st.sidebar.date_input(t["start_date"], value=default_start)
end_date = st.sidebar.date_input(t["end_date"], value=default_end)

num_agents = st.sidebar.slider(t["num_agents"], min_value=10, max_value=500, value=100, step=10)
initial_cash = st.sidebar.number_input(t["initial_cash"], min_value=0.0, value=10000.0, step=1000.0)
daily_cash = st.sidebar.number_input(t["daily_cash"], min_value=0.0, value=0.0, step=10.0)

st.sidebar.subheader(t["trade_prob_header"])
prob_buy = st.sidebar.slider(t["prob_buy"], min_value=0.0, max_value=100.0, value=25.0, step=1.0)
prob_sell = st.sidebar.slider(t["prob_sell"], min_value=0.0, max_value=100.0, value=25.0, step=1.0)

if prob_buy + prob_sell > 100.0:
    st.sidebar.error(t["prob_err"])
    st.stop()

prob_hold = 100.0 - prob_buy - prob_sell
st.sidebar.info(f"{t['prob_hold']} **{prob_hold:.1f}%**")

trade_size_mean = st.sidebar.slider(t["trade_size_mean"], min_value=10.0, max_value=100.0, value=50.0, step=5.0)
trade_size_std = st.sidebar.slider(t["trade_size_std"], min_value=1.0, max_value=30.0, value=15.0, step=1.0)

run_button = st.sidebar.button(t["run_btn"], type="primary")

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
        st.error(t["err_start_end"])
        st.stop()
        
    with st.spinner(t["spinner"].format(ticker=ticker)):
        df_stock = load_data(ticker, start_date, end_date)
        
        if df_stock is None or df_stock.empty:
            st.error(t["err_no_data"].format(ticker=ticker))
            st.stop()
            
        prices = df_stock['Close']
        dates = prices.index
        n_days = len(prices)
        
        if n_days < 2:
            st.error(t["err_too_short"])
            st.stop()
            
        stock_values = prices.values
        benchmark_returns = (stock_values / stock_values[0] - 1.0) * 100.0
        
        # Simulation arrays
        cash = np.full(num_agents, initial_cash)
        shares = np.zeros(num_agents)
        
        portfolio_history = np.zeros((n_days, num_agents))
        net_invested_history = np.zeros((n_days, num_agents))
        
        np.random.seed(42)
        
        actions_rand = np.random.uniform(0, 100, size=(n_days, num_agents))
        buy_threshold = prob_buy
        sell_threshold = prob_buy + prob_sell
        
        trade_proportions = np.random.normal(trade_size_mean / 100.0, trade_size_std / 100.0, size=(n_days, num_agents))
        trade_proportions = np.clip(trade_proportions, 0.05, 1.0)
        
        for t_idx_loop in range(n_days):
            p = stock_values[t_idx_loop]
            
            if t_idx_loop > 0:
                cash += daily_cash
                
            net_invested = initial_cash + t_idx_loop * daily_cash
            net_invested_history[t_idx_loop] = net_invested
                
            act_t = actions_rand[t_idx_loop]
            prop_t = trade_proportions[t_idx_loop]
            
            buying_mask = act_t < buy_threshold
            if np.any(buying_mask):
                spend_cash = cash[buying_mask] * prop_t[buying_mask]
                can_buy_shares = spend_cash / p
                cash[buying_mask] -= spend_cash
                shares[buying_mask] += can_buy_shares
                
            selling_mask = (act_t >= buy_threshold) & (act_t < sell_threshold)
            if np.any(selling_mask):
                sell_shares = shares[selling_mask] * prop_t[selling_mask]
                shares[selling_mask] -= sell_shares
                cash[selling_mask] += sell_shares * p
                
            portfolio_history[t_idx_loop] = cash + shares * p
            
        returns_history = ((portfolio_history - net_invested_history) / net_invested_history) * 100.0
        
        final_returns = returns_history[-1]
        best_idx = np.argmax(final_returns)
        worst_idx = np.argmin(final_returns)
        
        global_min_ret = float(np.min(returns_history))
        global_max_ret = float(np.max(returns_history))
        
        st.success(t["success_sim"].format(num_agents=num_agents, n_days=n_days))
        
        # --- Chart 1: Main Stock & Agent Performance ---
        st.subheader(t["chart1_title"].format(ticker=ticker))
        
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
            name=f"{t['worst_agent']} (#{worst_idx+1}: {final_returns[worst_idx]:.2f}%)",
            line=dict(width=3.5, color='#FF1744')
        ))
        
        fig1.add_trace(go.Scatter(
            x=dates,
            y=returns_history[:, best_idx],
            mode='lines',
            name=f"{t['best_agent']} (#{best_idx+1}: +{final_returns[best_idx]:.2f}%)",
            line=dict(width=3.5, color='#00E676')
        ))
        
        fig1.add_trace(go.Scatter(
            x=dates,
            y=benchmark_returns,
            mode='lines',
            name=f"{t['benchmark']} ({benchmark_returns[-1]:.2f}%)",
            line=dict(width=2.5, color='dodgerblue', dash='dash')
        ))
        
        fig1.update_layout(
            title=t["chart1_main_title"].format(ticker=ticker),
            xaxis_title=t["xaxis_date"],
            yaxis_title=t["yaxis_return"],
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
        
        # --- Chart 2: Histogram with Time Slider & Fixed Global X-Axis ---
        st.subheader(t["chart2_title"])
        st.markdown(f"**{t['global_range_text']}** `{global_min_ret:.1f}%` {t['to_text']} `+{global_max_ret:.1f}%`（{t['per_bin']}）。")
        st.markdown(t["chart2_desc"])
        
        step_size = max(1, n_days // 60)
        frame_indices = list(range(0, n_days, step_size))
        if (n_days - 1) not in frame_indices:
            frame_indices.append(n_days - 1)
            
        fig2 = go.Figure()
        
        last_t_idx = n_days - 1
        last_rets = returns_history[last_t_idx]
        
        fig2.add_trace(go.Histogram(
            x=last_rets,
            xbins=dict(start=global_min_ret - 2, end=global_max_ret + 2, size=1.0),
            marker_color='royalblue',
            opacity=0.75
        ))
        
        frames = []
        for t_idx in frame_indices:
            rets_t = returns_history[t_idx]
            current_date_str = dates[t_idx].strftime('%Y-%m-%d')
            frames.append(go.Frame(
                data=[go.Histogram(x=rets_t, xbins=dict(start=global_min_ret - 2, end=global_max_ret + 2, size=1.0))],
                name=str(t_idx),
                layout=dict(
                    title_text=f"日期 / Date: {current_date_str} (交易日 {t_idx+1}/{n_days})",
                    xaxis=dict(range=[global_min_ret - 2, global_max_ret + 2])
                )
            ))
            
        fig2.frames = frames
        
        active_step_idx = len(frame_indices) - 1
        
        sliders = [{
            "active": active_step_idx,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": t["date_prefix"],
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
            title=f"日期 / Date: {dates[last_t_idx].strftime('%Y-%m-%d')} (交易日 {n_days}/{n_days})",
            xaxis_title=t["yaxis_return"],
            yaxis_title=t["yaxis_agents"],
            xaxis=dict(range=[global_min_ret - 2, global_max_ret + 2]),
            sliders=sliders,
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary metrics table
        st.subheader(t["summary_title"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t["stock_return"], f"{benchmark_returns[-1]:.2f}%")
        col2.metric(t["best_return"], f"+{final_returns[best_idx]:.2f}%")
        col3.metric(t["worst_return"], f"{final_returns[worst_idx]:.2f}%")
        col4.metric(t["profit_ratio"], f"{(np.sum(final_returns > 0) / num_agents * 100):.1f}%")
        
else:
    st.info(t["sidebar_tip"])
