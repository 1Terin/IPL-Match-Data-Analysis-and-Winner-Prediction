import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="IPL Dashboard", layout="wide")

# -------------------------------
# SIDEBAR STYLE
# -------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] * {
    font-size: 18px !important;
}

/* CARD STYLE */
.card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.card-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("IPL DATA DASHBOARD")

# -------------------------------
# LOAD DATA
# -------------------------------
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

matches = matches.dropna(subset=['winner'])

teams = sorted(matches['team1'].dropna().unique())
seasons = sorted(matches['season'].dropna().unique())

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Team Analysis", "Player Analysis", "Prediction"])

# -------------------------------
# FILTERS
# -------------------------------
st.subheader("FILTERS")

col1, col2 = st.columns(2)

with col1:
    selected_team = st.selectbox("Team", ["All"] + teams)

with col2:
    selected_season = st.selectbox("Season", ["All"] + list(seasons))

filtered = matches.copy()

if selected_team != "All":
    filtered = filtered[
        (filtered['team1'] == selected_team) |
        (filtered['team2'] == selected_team)
    ]

if selected_season != "All":
    filtered = filtered[filtered['season'] == selected_season]

# -------------------------------
# 🔥 ADDED FIX: FILTER DELIVERIES
# -------------------------------
filtered_deliveries = deliveries[
    deliveries['match_id'].isin(filtered['id'])
]

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Matches", len(filtered))
col2.metric("Teams", filtered['team1'].nunique())
col3.metric("Avg Runs", round(deliveries['total_runs'].mean(), 2))

# ===============================
# OVERVIEW
# ===============================
if page == "Overview":

    col1, col2 = st.columns(2)

    # Matches Won
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Matches Won by Team</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x='winner', data=filtered, ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Toss Impact
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Toss Impact</div>', unsafe_allow_html=True)

        toss_win = filtered[filtered['toss_winner'] == filtered['winner']]
        sizes = [len(toss_win), len(filtered) - len(toss_win)]

        fig, ax = plt.subplots(figsize=(3,2))
        ax.pie(sizes, labels=["Win Toss+Match", "Lose After Toss"], autopct='%1.1f%%')
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TEAM ANALYSIS
# ===============================
elif page == "Team Analysis":

    col1, col2 = st.columns(2)

    # Matches per City
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Matches per City</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x='city', data=filtered, ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Correlation Heatmap</div>', unsafe_allow_html=True)

        # 🔥 FIXED (uses filtered)
        corr = filtered.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PLAYER ANALYSIS
# ===============================
elif page == "Player Analysis":

    # 🔥 FIXED (uses filtered_deliveries)
    player_stats = filtered_deliveries.groupby('batter').agg({
        'batsman_runs': 'sum',
        'ball': 'count'
    })

    col1, col2 = st.columns(2)

    # Top batsmen
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top Batsmen</div>', unsafe_allow_html=True)

        # 🔥 FIXED
        top_batsmen = filtered_deliveries.groupby('batter')['batsman_runs'] \
            .sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(5,3))
        top_batsmen.plot(kind='bar', ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Scatter
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Runs vs Balls</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(x='ball', y='batsman_runs', data=player_stats, ax=ax)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    col1, col2 = st.columns(2)

    # Box plot
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Runs Distribution</div>', unsafe_allow_html=True)

        # 🔥 FIXED
        total_runs = filtered_deliveries.groupby('match_id')['total_runs'].sum()

        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(x=total_runs, ax=ax)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Clustering
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Player Clustering</div>', unsafe_allow_html=True)

        kmeans = KMeans(n_clusters=3)
        player_stats['cluster'] = kmeans.fit_predict(player_stats)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(
            x='ball',
            y='batsman_runs',
            hue='cluster',
            data=player_stats,
            ax=ax
        )
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
elif page == "Prediction":

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Team 1", teams)
        team2 = st.selectbox("Team 2", teams)

    with col2:
        toss_winner = st.selectbox("Toss Winner", teams)

    data = matches[['team1','team2','toss_winner','winner']].dropna()

    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    X = data[['team1','team2','toss_winner']]
    y = data['winner']

    model = RandomForestClassifier()
    model.fit(X, y)

    if st.button("Predict Winner"):
        input_data = pd.DataFrame([[team1, team2, toss_winner]],
                                 columns=['team1','team2','toss_winner'])

        for col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

        pred = model.predict(input_data)
        result = le.inverse_transform(pred)

        st.success(f"🏆 Predicted Winner: {result[0]}")