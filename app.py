import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Real Estate Analytics & Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #020817 0%, #0f172a 40%, #111827 100%);
        color: #e2e8f0;
    }
    .main-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-size: 34px;
        font-weight: 800;
        text-align: center;
        color: #f8fafc;
        letter-spacing: 0.03em;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .main-title .title-text {
        background: linear-gradient(90deg, #7dd3fc, #c084fc, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-title .title-icon {
        display: inline-block;
        font-size: 1.1em;
        filter: drop-shadow(0 3px 12px rgba(125, 211, 252, 0.35));
    }
    .section-header {
        color: #f8fafc;
        font-weight: 700;
        font-size: 24px;
        margin-top: 10px;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 14px !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 18px 16px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }
    .stTabs [role="tablist"] {
        gap: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 18px;
    }
    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px 12px 0 0;
        color: #cbd5e1;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, rgba(59,130,246,0.25), rgba(168,85,247,0.3));
        border-color: rgba(96,165,250,0.7);
        color: #f8fafc;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 36px rgba(124, 58, 237, 0.4);
    }
    .stSuccess {
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(22,163,74,0.18), rgba(34,197,94,0.22));
        border: 1px solid rgba(74,222,128,0.35);
        color: #dcfce7;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title"><span class="title-icon">🏠</span><span class="title-text">Real Estate Analytics & House Price Prediction Dashboard</span></div>',
    unsafe_allow_html=True,
)

# Load Cleaned Data & Trained Model
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_redfin_housing.csv')

@st.cache_data
def load_city_prices():
    raw_df = pd.read_csv('data/redfin_housing.csv')
    return raw_df.groupby('City')['Price (USD)'].mean().sort_index().to_dict()

@st.cache_resource
def load_model():
    return joblib.load('models/linear_regression_model.pkl')

try:
    df = load_data()
    city_price_map = load_city_prices()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files. Ensure `data/cleaned_redfin_housing.csv`, `data/redfin_housing.csv`, and `models/linear_regression_model.pkl` exist! Details: {e}")
    st.stop()

# Create Tabs for Dashboard Sections
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🔍 Data Insights & Trends", "🤖 Price Predictor"])

# ==================== TAB 1: MARKET OVERVIEW ====================
with tab1:
    st.markdown('<div class="section-header">Executive Summary & Key Metrics</div>', unsafe_allow_html=True)
    
    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Average Price", f"${df['Price (USD)'].mean():,.2f}")
    col3.metric("Median Home Area", f"{df['Area (SQFT)'].median():,.0f} SQFT")
    col4.metric("Avg Price / SQFT", f"${(df['Price (USD)'] / df['Area (SQFT)']).mean():,.2f}")

    st.markdown("---")

    # Interactive Distribution Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("House Price Distribution")
        fig_price = px.histogram(
            df,
            x="Price (USD)",
            nbins=30,
            color_discrete_sequence=['#60a5fa'],
            title="Distribution of Property Prices"
        )
        fig_price.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font=dict(size=22, color='#f8fafc'),
            xaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)'),
            yaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)'),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.subheader("Price vs. Living Area (SQFT)")
        fig_scatter = px.scatter(
            df,
            x="Area (SQFT)",
            y="Price (USD)",
            color="Beds",
            title="Living Area vs Price",
            hover_data=['Baths'],
            render_mode='svg'
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.75))
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font=dict(size=22, color='#f8fafc'),
            xaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)'),
            yaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

# ==================== TAB 2: DATA INSIGHTS ====================
with tab2:
    st.markdown('<div class="section-header">Deep-Dive Insights</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Average Price by Bedrooms")
        bed_avg = df.groupby('Beds')['Price (USD)'].mean().reset_index()
        fig_beds = px.bar(
            bed_avg,
            x='Beds',
            y='Price (USD)',
            color='Price (USD)',
            color_continuous_scale='Blues',
            title="Mean Price by Bedroom Count"
        )
        fig_beds.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font=dict(size=20, color='#f8fafc'),
            xaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)'),
            yaxis=dict(color='#cbd5e1', gridcolor='rgba(148,163,184,0.18)')
        )
        st.plotly_chart(fig_beds, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.subheader("Feature Correlation Matrix")
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font=dict(size=20, color='#f8fafc')
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

# ==================== TAB 3: PRICE PREDICTOR ====================
with tab3:
    st.markdown('<div class="section-header">Real-Time Price Estimator</div>', unsafe_allow_html=True)
    st.write("Adjust property parameters to estimate market valuation.")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        beds_input = st.number_input("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
        baths_input = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        area_input = st.number_input("Square Footage (SQFT)", min_value=200, max_value=20000, value=1500, step=50)

    with form_col2:
        year_input = st.selectbox("Sale Year", [2023, 2024, 2025, 2026], index=0)
        month_input = st.slider("Sale Month", 1, 12, 6)
        
        city_options = sorted(city_price_map.keys())
        selected_city = st.selectbox("City", city_options)
        city_avg_input = city_price_map[selected_city]
        st.caption(f"Selected city average price: ${city_avg_input:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Predicted Price 🚀", use_container_width=True):
        input_df = pd.DataFrame([{
            'Beds': beds_input,
            'Baths': baths_input,
            'Area (SQFT)': area_input,
            'Sale_Year': year_input,
            'Sale_Month': month_input,
            'City_Avg_Price': city_avg_input
        }])

        pred_price = model.predict(input_df)[0]
        st.success(f"### Estimated Market Price: **${pred_price:,.2f}**")