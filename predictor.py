import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="2027 ODI WC Predictor | WSA Engine",
    page_icon="🏏",
    layout="wide"
)

# Custom Styling for Dark Dashboard Theme
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    h1 { color: #38bdf8; }
    .stMetric { background-color: #1e293b; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# PHASE 1: MACRO DATA & RATING SYSTEM
# ==============================================================================
def build_wsa_team_profiles():
    return {
        "India": 122.5,
        "Australia": 120.0,
        "South Africa": 116.0,
        "England": 112.5,
        "New Zealand": 108.0,
        "Pakistan": 104.5,
        "Afghanistan": 97.0,
        "Sri Lanka": 96.5,
        "Bangladesh": 93.0,
        "Zimbabwe": 89.0,
        "Ireland": 88.5,
        "Netherlands": 86.0,
        "Scotland": 84.0,
        "Namibia": 80.0
    }

# ==============================================================================
# PHASE 2: SIMULATOR CLASS
# ==============================================================================
class WSATournamentSimulator2027:
    def __init__(self, team_ratings, seed=42):
        self.team_ratings = team_ratings
        np.random.seed(seed)
        sorted_teams = sorted(team_ratings.keys(), key=lambda t: team_ratings[t], reverse=True)
        self.direct_qualifiers = sorted_teams[:11]
        self.super_series_teams = sorted_teams[11:]

    def simulate_match(self, team1, team2):
        r1 = self.team_ratings.get(team1, 90.0)
        r2 = self.team_ratings.get(team2, 90.0)
        prob_team1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
        return team1 if np.random.random() < prob_team1 else team2

    def run_round_robin(self, teams):
        points = {team: 0 for team in teams}
        nrc = {team: 0.0 for team in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                winner = self.simulate_match(t1, t2)
                loser = t1 if winner == t2 else t2
                points[winner] += 2
                nrc[winner] += np.random.uniform(0.1, 1.5)
                nrc[loser] -= np.random.uniform(0.1, 1.5)
        return sorted(teams, key=lambda x: (points[x], nrc[x]), reverse=True), points, nrc

    def run_tournament_cycle(self):
        super_series_winner = self.run_round_robin(self.super_series_teams)[0][0]
        round_2_teams = self.direct_qualifiers + [super_series_winner]
        
        group_a = round_2_teams[0::2]
        group_b = round_2_teams[1::2]
        
        sorted_a, pts_a, nrc_a = self.run_round_robin(group_a)
        sorted_b, pts_b, nrc_b = self.run_round_robin(group_b)
        
        top_a, top_b = sorted_a[:3], sorted_b[:3]
        fourth_a, fourth_b = sorted_a[3], sorted_b[3]
        best_fourth = fourth_a if (pts_a[fourth_a], nrc_a[fourth_a]) > (pts_b[fourth_b], nrc_b[fourth_b]) else fourth_b
        
        super_7_teams = top_a + top_b + [best_fourth]
        sorted_s7, _, _ = self.run_round_robin(super_7_teams)
        final_four = sorted_s7[:4]
        
        sf1_winner = self.simulate_match(final_four[0], final_four[3])
        sf2_winner = self.simulate_match(final_four[1], final_four[2])
        champion = self.simulate_match(sf1_winner, sf2_winner)
        
        return {"champion": champion, "qualifiers": final_four}

    def execute_monte_carlo(self, iterations=10000):
        champion_distribution = {team: 0 for team in self.team_ratings}
        qualification_distribution = {team: 0 for team in self.team_ratings}
        
        for _ in range(iterations):
            result = self.run_tournament_cycle()
            champion_distribution[result["champion"]] += 1
            for qualified_team in result["qualifiers"]:
                qualification_distribution[qualified_team] += 1
                
        final_rankings = []
        for team in self.team_ratings:
            final_rankings.append({
                "Team": team,
                "Base Rating": self.team_ratings[team],
                "Semi-Final Chance (%)": round((qualification_distribution[team] / iterations) * 100, 2),
                "Champion Win Chance (%)": round((champion_distribution[team] / iterations) * 100, 2),
                "_raw_win": champion_distribution[team]
            })
            
        df_rankings = (
            pd.DataFrame(final_rankings)
            .sort_values(by="_raw_win", ascending=False)
            .drop(columns=["_raw_win"])
        )
        df_rankings.index = range(1, len(self.team_ratings) + 1)
        return df_rankings

# ==============================================================================
# PHASE 3: INTERACTIVE DASHBOARD FRONTEND
# ==============================================================================
st.title("🏏 2027 ODI World Cup Predictor")
st.caption("Powered by Wolverine Sports Analytics (WSA) Monte Carlo Simulation Engine")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")
iterations = st.sidebar.slider("Monte Carlo Iterations", min_value=1000, max_value=50000, value=10000, step=1000)
seed = st.sidebar.number_input("Random Seed (for Determinism)", value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("Tournament Rules")
st.sidebar.text("• 14 Teams\n• Round 1: Super Series\n• Round 2: 2 Groups of 6\n• Round 3: Super 7 Stage\n• Final: Knockouts")

# Execution Button
if st.button("🚀 Run Live Monte Carlo Simulation", type="primary"):
    with st.spinner(f"Running {iterations:,} Monte Carlo tournament loops..."):
        ratings_matrix = build_wsa_team_profiles()
        engine = WSATournamentSimulator2027(ratings_matrix, seed=seed)
        df_results = engine.execute_monte_carlo(iterations=iterations)

        # Highlight Top Winner Metrics
        top_team = df_results.iloc[0]["Team"]
        top_chance = df_results.iloc[0]["Champion Win Chance (%)"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Favorite", top_team)
        col2.metric("Championship Win Probability", f"{top_chance}%")
        col3.metric("Simulations Completed", f"{iterations:,}")

        st.markdown("---")
        st.subheader("Predictive Leaderboard")
        
        # Display Interactive Data Table
        st.dataframe(
            df_results.style.background_gradient(subset=["Champion Win Chance (%)"], cmap="Greens")
                            .background_gradient(subset=["Semi-Final Chance (%)"], cmap="Blues"),
            use_container_width=True
        )
