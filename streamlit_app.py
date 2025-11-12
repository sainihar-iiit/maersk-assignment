import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import pyttsx3, speech_recognition as sr, re
from prophet import Prophet
from tqdm import tqdm


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="E-Commerce AI Dashboard", page_icon="🛒", layout="wide")
# -------------------------------------------------
# 🔝 NAVIGATION BAR (Smooth Scroll)
# -------------------------------------------------
st.markdown("""
<style>
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(135deg, #e8f0ff, #d6e6ff);
    padding: 0.6rem 1rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 1.5rem;
}
.navbar button {
    background-color: #0a66c2;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1.2rem;
    margin: 0 0.4rem;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}
.navbar button:hover {
    background-color: #084a8f;
    transform: scale(1.05);
}
</style>

<div class="navbar">
    <button onclick="window.location.href='#about-section'">About</button>
    <button onclick="window.location.href='#kpi-section'">KPIs</button>
    <button onclick="window.location.href='#quick-section'">Quick Insights</button>
    <button onclick="window.location.href='#query-section'">Ask About Your Data</button>
    <button onclick="window.location.href='#preview-section'">Data Preview</button>
</div>
""", unsafe_allow_html=True)


# 🌈 --- Custom CSS for KPI & Quick Insights ---

st.markdown("""
<style>
body {background-color:#f8f9fb;}

/* KPI STYLING */
.metric-container {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    margin-bottom: 25px;
}
.metric-card {
    background: linear-gradient(135deg, #f3f6fc, #dbe9ff);
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    padding: 1.2rem;
    width: 23%;
    text-align: center;
    transition: all 0.4s ease;
    transform: scale(1);
}
.metric-card:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    background: linear-gradient(135deg, #e2f0ff, #c2e1ff);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0a66c2;
    margin-top: 5px;
    animation: fadeIn 1s ease-in-out;
}
.metric-label {
    font-size: 1rem;
    color: #444;
    font-weight: 500;
}

/* QUICK INSIGHTS STYLING */
.quick-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
    margin-top: 10px;
}
.quick-card {
    background: linear-gradient(135deg, #e8f0ff, #d7e7ff);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    text-align: center;
    color: #0a66c2;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}
.quick-card:hover {
    background: linear-gradient(135deg, #cfe3ff, #b9d7ff);
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    border-color: #0a66c2;
}
.selected {
    border-color: #0056d6 !important;
    background: linear-gradient(135deg, #b9d7ff, #cfe3ff);
    box-shadow: 0 0 15px rgba(0,86,214,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MAIN TITLE SECTION (Enhanced with Styled Banner)
# -------------------------------------------------
st.markdown("""
<style>
.title-banner {
    background: linear-gradient(120deg, #e8f0ff, #c9ddff, #e8f0ff);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    transition: all 0.4s ease;
}
.title-banner:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(0, 86, 214, 0.25);
}
.title-text {
    font-size: 2rem;
    font-weight: 800;
    color: #0a66c2;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}
.subtitle-text {
    font-size: 1.05rem;
    color: #555;
    font-weight: 500;
}
</style>
<div class="title-banner">
    <div class="title-text">🛍 AI-Powered E-Commerce Analytics Dashboard</div>
    <div class="subtitle-text">Ask questions via text or voice — get instant charts, KPIs, and intelligent insights powered by Gemini AI.</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/pc/Desktop/assignment/merged.csv", low_memory=False)
    for col in df.columns:
        if "date" in col.lower() or "timestamp" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(df.columns):
        df["delivery_time_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.days
    if "order_purchase_timestamp" in df.columns:
        df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df

merged = load_data()

# -------------------------------------------------
# GEMINI CONFIG
# -------------------------------------------------
genai.configure(api_key="AIzaSyArWsznqkiFQzinwHvyzY6viihtGW1MOlY")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -------------------------------------------------
# TEXT-TO-SPEECH
# -------------------------------------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text[:200])
        engine.runAndWait()
        engine.stop()
    except:
        pass

# -------------------------------------------------
# SPEECH RECOGNITION
# -------------------------------------------------
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Listening... Speak your question now.")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        return query.lower().strip()
    except:
        st.warning("⚠ Could not recognize your voice.")
        return None

# -------------------------------------------------
# PARSE QUERY
# -------------------------------------------------
def parse_query(query: str):
    q = query.lower()
    agg, chart, n, order, rank, rank_pos = "sum", "bar", 5, "desc", None, None
    if any(word in q for word in ["average", "mean", "avg"]): agg = "mean"
    elif any(word in q for word in ["count", "number", "orders", "customers"]): agg = "count"
    elif any(word in q for word in ["total", "sum", "revenue", "sales"]): agg = "sum"

    if "pie" in q: chart = "pie"
    elif "line" in q or "trend" in q: chart = "line"
    elif "bar" in q or "chart" in q: chart = "bar"

    if "bottom" in q or "lowest" in q: order = "asc"

    rank_match = re.search(r"(\d+)(st|nd|rd|th)\s*(highest|lowest)", q)
    if rank_match:
        rank_pos = int(rank_match.group(1))
        rank = "highest" if "highest" in q else "lowest"

    match = re.search(r"(top|bottom)\s*(\d+)", q)
    if match:
        n = int(match.group(2))
    return agg, chart, n, order, rank, rank_pos

# -------------------------------------------------
# DETECT AXES
# -------------------------------------------------
def detect_axes(query):
    q = query.lower()
    if "city" in q: return ("customer_city", "price")
    if "state" in q: return ("customer_state", "price")
    if "category" in q or "type" in q: return ("product_category_name_english", "price")
    if "product" in q: return ("product_id", "price")
    if "seller" in q: return ("seller_id", "price")
    if "payment" in q: return ("payment_type", "payment_value")
    if "delivery" in q or "shipping" in q: return ("customer_state", "delivery_time_days")
    if "review" in q or "rating" in q: return ("product_category_name_english", "review_score")
    if "month" in q or "year" in q or "trend" in q: return ("order_month", "price")
    return ("product_category_name_english", "price")

# -------------------------------------------------
# PRETTY LABEL
# -------------------------------------------------
def prettify_label(name):
    return {
        "customer_city": "Cities",
        "customer_state": "States",
        "product_category_name_english": "Product Categories",
        "seller_id": "Sellers",
        "product_id": "Products",
        "payment_type": "Payment Methods",
        "delivery_time_days": "Delivery Time (Days)",
        "review_score": "Review Score",
        "order_month": "Monthly Trend"
    }.get(name, name.replace("_", " ").title())

# -------------------------------------------------
# CHART FUNCTION (Updated with data labels for all bars)
# -------------------------------------------------
def show_chart(df, x, y, agg, chart, topn, order, rank, rank_pos):
    df = df.dropna(subset=[x, y])
    grouped = df.groupby(x)[y].agg(agg).sort_values(ascending=(order == "asc"))

    # Handle rank queries (like 3rd highest)
    if rank and rank_pos:
        sorted_df = grouped.sort_values(ascending=(rank == "lowest")).reset_index()
        if rank_pos <= len(sorted_df):
            target_row = sorted_df.iloc[rank_pos - 1]
            key, val = target_row[x], target_row[y]
            st.success(f"🏅 {rank_pos}ᵗʰ {rank.title()} {prettify_label(x)}: {key} — ₹{val:,.2f}")
        else:
            st.warning("Not enough data for that rank.")
        return

    # Prepare top/bottom n
    grouped = grouped.head(topn).reset_index()

    # 🥧 PIE CHART for payment methods
    if "payment_type" in x:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(grouped[y], labels=grouped[x], autopct='%1.1f%%', startangle=90)
        ax.set_title("Payment Method Distribution", fontsize=14, color="#0a66c2")
        ax.axis('equal')
        st.pyplot(fig)
        return

    # 📊 BAR CHART with labels for all bars
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=grouped[y], y=grouped[x],
                palette="Blues_r" if order == "desc" else "Reds_r", ax=ax)

    # ✅ Add labels for all bars instead of only top one
    for container in ax.containers:
        ax.bar_label(container, fmt='₹%.0f', label_type='edge', padding=3)

    ax.set_xlabel(prettify_label(y))
    ax.set_ylabel(prettify_label(x))
    ax.set_title(f"Top {topn} {prettify_label(x)} by {prettify_label(y)}", fontsize=12, color="#0a66c2")
    st.pyplot(fig)

# -------------------------------------------------
# ENHANCED YEAR, QUARTER & GROUP COMPARISON FUNCTION
# -------------------------------------------------
def show_comparison_chart(df, year1, year2, group_col=None, quarter=None):
    """Compare sales trends or group performance across two years or quarters with better colors."""
    if "order_purchase_timestamp" not in df.columns:
        st.warning("Timestamp column not found for comparison.")
        return

    df["year"] = df["order_purchase_timestamp"].dt.year
    df["month"] = df["order_purchase_timestamp"].dt.month_name().str[:3]
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter

    # Filter relevant data
    df_filtered = df[df["year"].isin([year1, year2])]
    if quarter:
        df_filtered = df_filtered[df_filtered["quarter"] == quarter]

    if df_filtered.empty:
        st.warning("No data found for the selected period.")
        return

    # -----------------------
    # 1️⃣ MONTHLY OR QUARTERLY TREND COMPARISON
    # -----------------------
    if group_col is None:
        st.subheader(
            f"📈 Sales Trend Comparison: {year1} vs {year2}"
            + (f" (Quarter {quarter})" if quarter else "")
        )

        monthly = (
            df_filtered.groupby(["year", "month"])["price"].sum().reset_index()
        )
        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        monthly["month"] = pd.Categorical(monthly["month"], categories=month_order, ordered=True)
        monthly = monthly.sort_values(["month", "year"])

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=monthly,
            x="month",
            y="price",
            hue="year",
            marker="o",
            palette={year1: "#1f77b4", year2: "#ff7f0e"},
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(f"Monthly Sales Comparison ({year1} vs {year2})", color="#0a66c2", fontsize=13)
        ax.set_ylabel("Total Sales (₹)")
        ax.set_xlabel("Month")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        return

    # -----------------------
    # 2️⃣ GROUPED COMPARISON (STATE, CITY, CATEGORY, ETC.)
    # -----------------------
    grouped = (
        df_filtered.groupby(["year", group_col])["price"].sum().reset_index()
    )

    # Top 10 groups based on most recent year
    latest_year = grouped[grouped["year"] == year2].sort_values("price", ascending=False)
    top_groups = latest_year[group_col].head(10).tolist()
    grouped = grouped[grouped[group_col].isin(top_groups)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 💪 Use strong, distinct colors for both years
    color_map = {year1: "#1f77b4", year2: "#ff7f0e"}  # Dark Blue & Orange
    sns.barplot(
        data=grouped,
        x=group_col,
        y="price",
        hue="year",
        palette=color_map,
        ax=ax,
    )

    # ✍️ Add data labels for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt='₹%.0f', label_type='edge', fontsize=8, padding=3)

    title = f"{prettify_label(group_col)} Sales Comparison ({year1} vs {year2})"
    if quarter:
        title += f" - Quarter {quarter}"

    ax.set_title(title, color="#0a66c2", fontsize=13, pad=12)
    ax.set_ylabel("Total Sales (₹)")
    ax.set_xlabel(prettify_label(group_col))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Year", title_fontsize=10, fontsize=9, loc="upper right", frameon=False)
    st.pyplot(fig)

from prophet import Prophet
from tqdm import tqdm

# -------------------------------------------------
# 🔮 FORECASTING FUNCTIONS
# -------------------------------------------------
def show_forecast(df, periods=6):
    """Forecast total sales for the next N months."""
    if "order_purchase_timestamp" not in df.columns:
        st.warning("No date column found for forecasting.")
        return

    # Aggregate daily sales
    df_time = df.groupby("order_purchase_timestamp")["price"].sum().reset_index()
    df_time = df_time.rename(columns={"order_purchase_timestamp": "ds", "price": "y"})

    if len(df_time) < 30:
        st.warning("Not enough data to produce a meaningful forecast.")
        return

    # Fit Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_time)

    # Forecast next 'periods' months
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title(f"🔮 Total Sales Forecast (Next {periods} Months)", fontsize=13, color="#0a66c2")
    st.pyplot(fig1)

    # Show last and forecasted summary
    latest = df_time["y"].iloc[-1]
    predicted = forecast["yhat"].iloc[-periods:].mean()
    st.success(f"💡 Predicted average monthly sales for next {periods} months: ₹{predicted:,.0f}")
    st.info(f"Last recorded monthly sales: ₹{latest:,.0f}")

def show_group_forecast(df, group_col, periods=6):
    """Forecast top 5 groups' sales trends (state, city, category, etc.)"""
    if "order_purchase_timestamp" not in df.columns:
        st.warning("No date column found for forecasting.")
        return

    top_groups = df.groupby(group_col)["price"].sum().nlargest(5).index.tolist()
    st.info(f"🔮 Forecasting next {periods} months for top 5 {prettify_label(group_col)}...")

    fig, ax = plt.subplots(figsize=(10, 6))
    for grp in tqdm(top_groups, desc="Forecasting"):
        sub = df[df[group_col] == grp].groupby("order_purchase_timestamp")["price"].sum().reset_index()
        sub = sub.rename(columns={"order_purchase_timestamp": "ds", "price": "y"})
        if len(sub) < 20:
            continue

        m = Prophet(yearly_seasonality=True)
        m.fit(sub)
        future = m.make_future_dataframe(periods=periods, freq="M")
        fc = m.predict(future)
        ax.plot(fc["ds"], fc["yhat"], label=grp)

    ax.set_title(f"📈 Forecast (Next {periods} Months) by {prettify_label(group_col)}", fontsize=13, color="#0a66c2")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales (₹)")
    ax.legend(title=prettify_label(group_col), fontsize=8)
    st.pyplot(fig)


# -------------------------------------------------
# PROCESS QUERY (Now handles quarter & month filters)
# -------------------------------------------------
def process_query(query):
    if not query or len(query.strip()) == 0:
        st.warning("Please enter a query.")
        return

    q = query.lower().strip()
    year_match = re.findall(r"(20\d{2})", q)
    quarter_match = re.search(r"(q[1-4]|quarter\s*[1-4])", q)
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)", q
    )

    # ✅ Detect group columns
    group_col = None
    if "state" in q:
        group_col = "customer_state"
    elif "city" in q:
        group_col = "customer_city"
    elif "category" in q or "product category" in q:
        group_col = "product_category_name_english"
    elif "seller" in q:
        group_col = "seller_id"

    # ✅ Handle Forecast Queries (moved OUTSIDE seller block)
    if any(word in q for word in ["forecast", "predict", "future", "next"]):
        match_period = re.search(r"(\d+)\s*(month|months|year|years)", q)
        periods = 6  # default
        if match_period:
            num = int(match_period.group(1))
            if "year" in match_period.group(2):
                periods = num * 12
            else:
                periods = num

        # Detect specific group forecast
        if "state" in q:
            show_group_forecast(merged, "customer_state", periods)
            return
        elif "city" in q:
            show_group_forecast(merged, "customer_city", periods)
            return
        elif "category" in q:
            show_group_forecast(merged, "product_category_name_english", periods)
            return
        elif "product" in q:
            show_group_forecast(merged, "product_id", periods)
            return

        # Otherwise, overall forecast
        show_forecast(merged, periods)
        return

    # ✅ Handle Quarterly Comparison
    if "compare" in q and len(year_match) >= 2:
        year1, year2 = map(int, year_match[:2])
        quarter = None
        if quarter_match:
            quarter = int(re.search(r"[1-4]", quarter_match.group()).group())
        show_comparison_chart(merged, year1, year2, group_col, quarter)
        return

    # ✅ Handle Month and Year Specific Queries (e.g., "April 2018")
    if month_match and year_match:
        month_name = month_match.group().capitalize()
        year = int(year_match[0])
        month_num = pd.to_datetime(month_name, format="%B").month

        df_filtered = merged[
            (merged["order_purchase_timestamp"].dt.year == year)
            & (merged["order_purchase_timestamp"].dt.month == month_num)
        ]

        if df_filtered.empty:
            st.warning(f"No data for {month_name} {year}.")
            return

        st.info(f"📊 Showing data for {month_name} {year}")
        agg, chart, n, order, rank, rank_pos = parse_query(q)
        x, y = detect_axes(q)
        show_chart(df_filtered, x, y, agg, chart, n, order, rank, rank_pos)
        return

    # ✅ Default (existing logic)
    agg, chart, n, order, rank, rank_pos = parse_query(q)
    x, y = detect_axes(q)
    show_chart(merged, x, y, agg, chart, n, order, rank, rank_pos)

# -------------------------------------------------
# ABOUT SECTION (Enhanced Design)
# -------------------------------------------------
# -------------------------------------------------
# ABOUT SECTION (Enhanced Design)
# -------------------------------------------------
st.markdown('<a id="about-section"></a>', unsafe_allow_html=True)
st.markdown("""
<style>
.about-card {
    background: linear-gradient(135deg, #f9fbff, #e3edff);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: all 0.4s ease;
}
.about-card:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 22px rgba(0, 86, 214, 0.25);
}
.about-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0a66c2;
    margin-bottom: 0.6rem;
}
.about-text {
    font-size: 1rem;
    color: #333;
    line-height: 1.6;
}
.about-highlight {
    color: #0a66c2;
    font-weight: 600;
}
</style>

<div class="about-card">
    <div class="about-title">🧠 About This Dashboard</div>
    <div class="about-text">
        This <span class="about-highlight">AI-Powered E-Commerce Analytics Dashboard</span> helps businesses 
        unlock actionable insights from their data. It analyzes sales, customers, products, and delivery metrics 
        to uncover trends and patterns that matter most.
        <br><br>
        💡 <b>What You Can Do:</b><br>
        • Instantly explore <span class="about-highlight">key business KPIs</span> like revenue, orders, and customer reach.<br>
        • Ask questions using <span class="about-highlight">text or voice</span> — get instant visual answers powered by AI.<br>
        • View smart visualizations — bar, pie, and trend charts that update dynamically.<br>
        • Get <span class="about-highlight">quick insights</span> or deep analysis of cities, states, categories, or time periods.<br><br>
        Designed for <b>managers, analysts, and business owners</b> who want to make 
        <span class="about-highlight">data-driven decisions</span> — without needing to code.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# KPI METRICS (Enhanced UI)
# -------------------------------------------------
st.markdown('<a id="kpi-section"></a>', unsafe_allow_html=True)
st.markdown("### 📈 Key Performance Indicators")

kpi_html = f"""
<div class="metric-container">
    <div class="metric-card"><div class="metric-label">🧾 Total Orders</div><div class="metric-value">{merged['order_id'].nunique():,}</div></div>
    <div class="metric-card"><div class="metric-label">👥 Unique Customers</div><div class="metric-value">{merged['customer_unique_id'].nunique() if 'customer_unique_id' in merged.columns else merged['customer_id'].nunique():,}</div></div>
    <div class="metric-card"><div class="metric-label">🏙 Total Cities</div><div class="metric-value">{merged['customer_city'].nunique():,}</div></div>
    <div class="metric-card"><div class="metric-label">🌎 Total States</div><div class="metric-value">{merged['customer_state'].nunique():,}</div></div>
</div>
<div class="metric-container">
    <div class="metric-card"><div class="metric-label">💰 Total Revenue</div><div class="metric-value">₹{merged['price'].sum():,.0f}</div></div>
    <div class="metric-card"><div class="metric-label">🏪 Total Sellers</div><div class="metric-value">{merged['seller_id'].nunique():,}</div></div>
    <div class="metric-card"><div class="metric-label">⭐ Avg Review Score</div><div class="metric-value">{merged['review_score'].mean():.2f}</div></div>
    <div class="metric-card"><div class="metric-label">🚚 Avg Delivery Time</div><div class="metric-value">{merged['delivery_time_days'].mean():.1f} days</div></div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# -------------------------------------------------
# BEAUTIFULLY STYLED QUICK INSIGHTS SECTION (ALL-IN-ONE)
# -------------------------------------------------
st.markdown('<a id="quick-section"></a>', unsafe_allow_html=True)
st.markdown("""
<style>
.quick-section {
    background: linear-gradient(135deg, #f8fbff, #e4edff);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: all 0.4s ease;
}
.quick-section:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(0, 86, 214, 0.25);
}
.quick-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0a66c2;
    margin-bottom: 0.6rem;
    text-align: center;
}
.quick-desc {
    font-size: 1rem;
    color: #444;
    text-align: center;
    margin-bottom: 1.5rem;
}
.quick-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 15px;
    margin-top: 10px;
}
.quick-card {
    background: linear-gradient(135deg, #e8f0ff, #d7e7ff);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    text-align: center;
    color: #0a66c2;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}
.quick-card:hover {
    background: linear-gradient(135deg, #cfe3ff, #b9d7ff);
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    border-color: #0a66c2;
}
.selected-card {
    border-color: #0056d6 !important;
    background: linear-gradient(135deg, #b9d7ff, #cfe3ff);
    box-shadow: 0 0 15px rgba(0,86,214,0.4);
}
</style>
""", unsafe_allow_html=True)

# 💡 Title + Description + Buttons (inside one bordered section)
st.markdown("""
<div class="quick-section">
    <div class="quick-title">⚡ Quick Insights</div>
    <div class="quick-desc">
        Explore popular analytics instantly — click a card below to generate real-time visual insights 
        powered by AI and your dataset.
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Buttons inside section
# ----------------------------
quick_insights = [
    ("Top 5 Product Categories by Sales", "top 5 customer_product category by sales"),
    ("Bottom 5 Product Categories", "bottom 5 customer_product category by sales"),
    ("Top 5 States by Sales", "top 5 states by sales"),
    ("Bottom 5 States by Sales", "bottom 5 states by sales"),
    ("Top 5 Products by Sales", "top 5 products by sales"),
    ("Bottom 5 Products by Sales", "bottom 5 products by sales"),
    ("Top 5 Cities by Sales", "top 5 customer_city by sales"),
    ("Payment Insights (Pie Chart)", "top 5 payment methods in pie chart"),
]

if "selected_card" not in st.session_state:
    st.session_state.selected_card = None

# Create grid layout for buttons
cols = st.columns(4)
st.markdown("<div class='quick-grid'>", unsafe_allow_html=True)
for i, (title, query) in enumerate(quick_insights):
    col = cols[i % 4]
    if col.button(title, key=f"card_{i}"):
        st.session_state.selected_card = query
st.markdown("</div></div>", unsafe_allow_html=True)

# Display selected insight chart
if st.session_state.selected_card:
    st.info(f"📊 Showing results for: {st.session_state.selected_card.title()}")
    process_query(st.session_state.selected_card)



# -------------------------------------------------
# 💬 ASK ABOUT YOUR DATA (Styled Section)
# -------------------------------------------------
st.markdown('<a id="query-section"></a>', unsafe_allow_html=True)
st.markdown("""
<style>
.query-section {
    background: linear-gradient(135deg, #f9fbff, #e4edff);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: all 0.4s ease;
}
.query-section:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(0, 86, 214, 0.25);
}
.query-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0a66c2;
    text-align: center;
    margin-bottom: 0.6rem;
}
.query-desc {
    font-size: 1rem;
    color: #444;
    text-align: center;
    margin-bottom: 1.5rem;
}
.query-input input {
    border-radius: 12px !important;
    font-size: 1rem !important;
    padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="query-section">
    <div class="query-title">💬 Ask About Your Data</div>
    <div class="query-desc">
        Type or speak your question — for example:
        <b>"Top 10 cities by sales in March 2018"</b> or
        <b>"Average delivery time by state"</b>.<br>
        Our AI will interpret it and show you the correct chart.
    </div>
</div>
""", unsafe_allow_html=True)

# Text input & voice buttons (inside card)
user_query = st.text_input("🔎 Type your question (e.g., 'Top 5 states by sales')", key="query_input")
colq1, colq2 = st.columns(2)
with colq1:
    if st.button("🚀 Analyze Query"):
        process_query(user_query)
with colq2:
    if st.button("🎙 Voice Input"):
        q = listen()
        if q:
            st.success(f"You said: {q}")
            process_query(q)


# -------------------------------------------------
# 📂 DATA PREVIEW (Styled Section)
# -------------------------------------------------
st.markdown('<a id="preview-section"></a>', unsafe_allow_html=True)
st.markdown("""
<style>
.preview-section {
    background: linear-gradient(135deg, #f8fbff, #e9f1ff);
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: all 0.4s ease;
}
.preview-section:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(0, 86, 214, 0.25);
}
.preview-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0a66c2;
    margin-bottom: 0.6rem;
    text-align: center;
}
.preview-desc {
    font-size: 1rem;
    color: #444;
    text-align: center;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="preview-section">
    <div class="preview-title">📂 Data Preview</div>
    <div class="preview-desc">
        Here's a glimpse of your dataset. Use it to understand available columns and data structure.<br>
        Scroll to explore or filter insights above using voice or typed queries.
    </div>
</div>
""", unsafe_allow_html=True)

st.dataframe(merged.head(10))