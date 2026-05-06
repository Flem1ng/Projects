import pandas as pd
import streamlit as st
import plotly.express as px
from nba_api.stats.endpoints import leaguegamefinder
import google.generativeai as genai


# =========================
# CONFIG
# =========================
GEMINI_API_KEY = "AIzaSyBrAUtZUSXMYF56v7QYcgaBLW-v1jDyQPU"
SEASON_START_DATE = "2025-10-01"

st.set_page_config(
    page_title="NBA Analytics Dashboard",
    page_icon="🏀",
    layout="wide"
)


# =========================
# DATA
# =========================
@st.cache_data
def collect_nba_games():
    gamefinder = leaguegamefinder.LeagueGameFinder()
    games = gamefinder.get_data_frames()[0]

    df = games[["GAME_DATE", "TEAM_NAME", "MATCHUP", "WL", "PTS"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[df["GAME_DATE"] >= SEASON_START_DATE]

    df["LOCATION"] = df["MATCHUP"].apply(lambda x: "HOME" if "vs." in x else "AWAY")
    df["OPPONENT"] = df["MATCHUP"].apply(
        lambda x: x.split("vs. ")[1] if "vs." in x else x.split("@ ")[1]
    )

    return df


def create_team_summary(df):
    summary = df.groupby("TEAM_NAME").agg(
        GAMES=("TEAM_NAME", "count"),
        WINS=("WL", lambda x: (x == "W").sum()),
        LOSSES=("WL", lambda x: (x == "L").sum()),
        AVG_PTS=("PTS", "mean"),
        MAX_PTS=("PTS", "max")
    ).reset_index()

    summary["AVG_PTS"] = summary["AVG_PTS"].round(1)
    return summary.sort_values(by="AVG_PTS", ascending=False)


def gemini_analysis(summary):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        sample = summary.head(10).to_string(index=False)

        prompt = f"""
        You are a sports data analyst.

        Analyze this NBA team summary.

        Return:
        1. Main scoring trends
        2. Strongest offensive teams
        3. Win/loss observations
        4. One recommendation for deeper analysis

        Data:
        {sample}
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception:
        top_team = summary.iloc[0]["TEAM_NAME"]
        top_avg = summary.iloc[0]["AVG_PTS"]
        best_team = summary.sort_values(by="WINS", ascending=False).iloc[0]["TEAM_NAME"]

        return (
            f"Gemini quota was unavailable, so this local analysis was generated.\n\n"
            f"The highest scoring team is {top_team}, averaging {top_avg} points per game.\n\n"
            f"{best_team} has one of the strongest win totals in the dataset.\n\n"
            "Overall, teams with higher scoring averages tend to show stronger performance. "
            "A deeper analysis could compare home vs away performance and include player statistics."
        )


# =========================
# UI STYLE
# =========================
st.markdown("""
<style>
.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: #f5c542;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #333;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

.big-title {
    font-size: 48px;
    font-weight: 800;
    color: #f5c542;
}

.subtitle {
    font-size: 18px;
    color: #d1d5db;
}
</style>
""", unsafe_allow_html=True)


# =========================
# APP
# =========================
st.markdown('<div class="big-title">🏀 NBA Interactive Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Game outcomes, team performance, charts, and AI-powered insights</div>', unsafe_allow_html=True)

st.divider()

with st.spinner("Loading NBA data..."):
    df = collect_nba_games()
    summary = create_team_summary(df)

# Sidebar filters
st.sidebar.title("Filters")

teams = sorted(df["TEAM_NAME"].unique())
selected_team = st.sidebar.selectbox("Select Team", ["All Teams"] + teams)

location_filter = st.sidebar.selectbox("Location", ["All", "HOME", "AWAY"])
result_filter = st.sidebar.selectbox("Result", ["All", "W", "L"])

filtered_df = df.copy()

if selected_team != "All Teams":
    filtered_df = filtered_df[filtered_df["TEAM_NAME"] == selected_team]

if location_filter != "All":
    filtered_df = filtered_df[filtered_df["LOCATION"] == location_filter]

if result_filter != "All":
    filtered_df = filtered_df[filtered_df["WL"] == result_filter]


# Metrics
total_games = len(filtered_df)
wins = len(filtered_df[filtered_df["WL"] == "W"])
losses = len(filtered_df[filtered_df["WL"] == "L"])
avg_pts = round(filtered_df["PTS"].mean(), 1) if total_games > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Games", total_games)
col2.metric("Wins", wins)
col3.metric("Losses", losses)
col4.metric("Average Points", avg_pts)

st.divider()


# Charts
left, right = st.columns(2)

with left:
    st.subheader("Top 10 Teams by Average Points")
    top10 = summary.head(10)

    fig = px.bar(
        top10,
        x="TEAM_NAME",
        y="AVG_PTS",
        title="Average Points per Game",
        text="AVG_PTS"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Wins by Team")
    wins_chart = summary.sort_values(by="WINS", ascending=False).head(10)

    fig2 = px.bar(
        wins_chart,
        x="TEAM_NAME",
        y="WINS",
        title="Top 10 Teams by Wins",
        text="WINS"
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)


st.divider()


# Team-specific chart
if selected_team != "All Teams":
    st.subheader(f"Game Points Trend: {selected_team}")

    team_df = filtered_df.sort_values(by="GAME_DATE")

    fig3 = px.line(
        team_df,
        x="GAME_DATE",
        y="PTS",
        markers=True,
        title=f"{selected_team} Points Over Time"
    )
    st.plotly_chart(fig3, use_container_width=True)


# Tables
st.subheader("Filtered Game Results")
display_df = filtered_df[["GAME_DATE", "TEAM_NAME", "OPPONENT", "LOCATION", "WL", "PTS"]].copy()
display_df["GAME_DATE"] = display_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
st.dataframe(display_df, use_container_width=True)

st.subheader("Team Summary")
st.dataframe(summary, use_container_width=True)


st.divider()


# Gemini AI
st.subheader("🤖 Gemini AI Insights")

if st.button("Generate AI Analysis"):
    with st.spinner("Generating analysis..."):
        analysis = gemini_analysis(summary)
        st.info(analysis)


# Download buttons
st.divider()
st.subheader("Download Data")

csv_games = df.to_csv(index=False).encode("utf-8")
csv_summary = summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Game Results CSV",
    data=csv_games,
    file_name="nba_current_season_games.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Team Summary CSV",
    data=csv_summary,
    file_name="nba_team_summary.csv",
    mime="text/csv"
)