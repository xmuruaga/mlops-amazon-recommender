
import os
import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import random

# ------------------------------------------------------------------
# Point Streamlit at FastAPI
#  • In Docker, compose will inject API_URL=http://fastapi:8000
#  • When you run locally, it silently falls back to localhost
# ------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")


# Page configuration and theme
st.set_page_config(
    page_title="Amazon Fine Food Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar controls

with st.sidebar:
    st.header("Controls")
    # Fetch user IDs from FastAPI endpoint (try /api/users for common setups)
    USERS_ENDPOINT = f"{API_URL}/api/users"
    if 'all_user_ids' not in st.session_state:
        try:
            resp = requests.get(USERS_ENDPOINT, timeout=10)
            resp.raise_for_status()
            st.session_state['all_user_ids'] = resp.json().get('user_ids', [])
        except Exception as e:
            st.session_state['all_user_ids'] = []
            st.error(f"Error fetching user IDs: {e}")
    st.markdown("**Try a random user:**")
    if 'random_user_ids' not in st.session_state:
        if st.session_state['all_user_ids']:
            st.session_state['random_user_ids'] = random.sample(st.session_state['all_user_ids'], min(3, len(st.session_state['all_user_ids'])))
        else:
            st.session_state['random_user_ids'] = []
    # Display only 3 random user buttons in 3 rows
    for uid in st.session_state.get('random_user_ids', [])[:3]:
        if st.button(uid, key=f"uid_{uid}"):
            st.session_state['user_id_input'] = uid
            st.session_state['run_recs'] = True
    # Refresh button below randoms
    if st.button("🔄 Refresh", key="refresh_user_ids", help="Refresh random user IDs"):
        if st.session_state['all_user_ids']:
            st.session_state['random_user_ids'] = random.sample(st.session_state['all_user_ids'], min(3, len(st.session_state['all_user_ids'])))
        else:
            st.session_state['random_user_ids'] = []
    user_id = st.text_input("User ID", value=st.session_state.get('user_id_input', ''), help="Amazon user ID")
    k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)
    run_recs = st.button("Show Results", key="run_recs_btn")
    # If a random user was clicked, auto-trigger results
    if st.session_state.get('run_recs'):
        run_recs = True
        st.session_state['run_recs'] = False


# Custom header with Amazon logo and smaller subtitle inline
header_cols = st.columns([0.18, 0.82])
with header_cols[0]:
    LOGO_PATH = Path(__file__).parent / "static" / "Amazon_logo.svg"
    st.image(str(LOGO_PATH), width=140)
with header_cols[1]:
    st.markdown("<span style='font-size:1.5em;font-weight:500;color:#222;vertical-align:middle;'>Food Recommender</span>", unsafe_allow_html=True)

# Load reviews once

def load_reviews():
    path = os.path.join("data", "raw", "Reviews.csv")
    df = pd.read_csv(path, low_memory=False)
    if "Time" in df.columns:
        df["ReviewTime"] = pd.to_datetime(df["Time"], unit="s")
    return df


@st.cache_data
def reviews_df():
    return load_reviews()

df = reviews_df()


if user_id and run_recs:
    user_reviews = df[df['UserId'] == user_id]
    if user_reviews.empty:
        st.warning("No reviews found for this user.")
    else:
        # User Profile & Metrics
        num = len(user_reviews)
        avg = user_reviews['Score'].mean()
        rating_counts = user_reviews['Score'].value_counts().sort_index()
        for r in range(1, 6):
            if r not in rating_counts:
                rating_counts.loc[r] = 0
        rating_counts = rating_counts.sort_index()
        with st.container():
            # All metrics and charts in one row
            row_cols = st.columns([1, 1, 1])
            with row_cols[0]:
                st.subheader(f"User {user_id}", anchor=False)
                st.metric("Total Reviews", num)
                st.metric("Average Rating", f"{avg:.2f}")
            with row_cols[1]:
                st.markdown("**Ratings Distribution**")
                st.bar_chart(rating_counts, use_container_width=True, height=230)
            with row_cols[2]:
                st.markdown("**Rating Trend**")
                import altair as alt
                trend = user_reviews.groupby(user_reviews['ReviewTime'].dt.to_period('M')).Score.mean()
                trend_df = trend.reset_index()
                trend_df['ReviewTime'] = trend_df['ReviewTime'].astype(str)
                chart = alt.Chart(trend_df).mark_line(point=True).encode(
                    x=alt.X('ReviewTime', title='Month'),
                    y=alt.Y('Score', title='Avg Rating', scale=alt.Scale(domain=[0, 5]))
                ).properties(height=230)
                st.altair_chart(chart, use_container_width=True)

        st.divider()

        # Latest Purchases Gallery
        st.subheader("Latest 10 Purchases")
        latest = user_reviews.sort_values('ReviewTime', ascending=False).head(10)
        items = []
        for idx, row in latest.iterrows():
            pid = row['ProductId']
            title = row.get('Summary', 'No title')
            user_score = row['Score']
            avg_score = df[df['ProductId'] == pid]['Score'].mean()
            if avg_score is None or pd.isna(avg_score):
                avg_score = "N/A"
            thumb = f"https://images-na.ssl-images-amazon.com/images/P/{pid}.jpg"
            url = f"https://www.amazon.com/dp/{pid}"
            items.append((pid, title, user_score, avg_score, thumb, url))

        # Always render 2 rows of 5 columns for alignment
        for row_idx in range(2):
            cols_purch = st.columns(5, gap="large")
            for col_idx in range(5):
                idx = row_idx * 5 + col_idx
                if idx < len(items):
                    pid, title, user_score, avg_score, thumb, url = items[idx]
                    with cols_purch[col_idx]:
                        st.markdown(
                            f"<div style='background:#fafbfc;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);border:1px solid #e3e6ea;padding:12px 8px 8px 8px;margin-top:25px;display:flex;flex-direction:column;align-items:center;min-width:150px;min-height:160px;'>"
                            f"<a href='{url}' target='_blank'>"
                            f"<img src='{thumb}' style='width:100px;height:100px;object-fit:cover;border-radius:8px;'/></a>"
                            f"<div style='margin-top:2px;font-size:0.93em;color:#555;'>User Rating: {user_score:.1f}</div>"
                            f"<div style='margin-top:2px;font-size:0.93em;color:#555;'>Avg Rating: {avg_score if avg_score == 'N/A' else f'{avg_score:.2f}'}</div>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(title)
                else:
                    with cols_purch[col_idx]:
                        st.markdown(
                            "<div style='visibility:hidden;min-width:150px;min-height:160px;'></div>",
                            unsafe_allow_html=True
                        )

                    
                    

        st.divider()
        # Fetch recommendations with spinner
        with st.spinner("Preparing your recommendations…"):
            try:
                resp = requests.get(f"{API_URL}/recommend", params={'user_id': user_id, 'k': k}, timeout=10)
                resp.raise_for_status()
                recs = resp.json().get('recommendations', [])
            except Exception as e:
                st.error(f"Error fetching recommendations: {e}")
                recs = []

        if recs:
            st.subheader(f"Top {len(recs)} Recommendations")
            rec_items = []
            for rec in recs:
                asin = rec.get('asin') or rec.get('product_id')
                title = rec.get('title', asin)
                avg_r = rec.get('avg_rating')
                if avg_r is None or pd.isna(avg_r):
                    prod_scores = df[df['ProductId'] == asin]['Score']
                    avg_r = prod_scores.mean() if len(prod_scores) > 0 else "N/A"
                thumb = rec.get('thumbnail_url') or f"https://images-na.ssl-images-amazon.com/images/P/{asin}.jpg"
                url = rec.get('product_url') or f"https://www.amazon.com/dp/{asin}"
                rec_items.append((asin, title, avg_r, thumb, url))
            cols2 = st.columns(5, gap="large")
            for i, (asin, title, avg_r, thumb, url) in enumerate(rec_items):
                with cols2[i % 5]:
                    st.markdown(
                        f"<div style='background:#fafbfc;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);border:1px solid #e3e6ea;padding:12px 8px 8px 8px;margin-top:25px;display:flex;flex-direction:column;align-items:center;min-width:150px;min-height:160px;'>"
                        f"<a href='{url}' target='_blank'>"
                        f"<img src='{thumb}' style='width:100px;height:100px;object-fit:cover;border-radius:8px;'/></a>"
                        f"<div style='margin-top:2px;font-size:0.93em;color:#555;'>Avg Rating: {avg_r if avg_r == 'N/A' else f'{avg_r:.2f}'}</div>"
                        f"<div style='margin-top:4px;font-size:0.92em;color:#555;text-align:center;'>{title}</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No recommendations available.")

