import pandas as pd
import plotly.express as px


def date_based(day, month, year, state, energy_df):

    required_row = energy_df[
        (energy_df["Day"] == day) & 
        (energy_df["Month"] == month) & 
        (energy_df["Year"] == year) & 
        (energy_df["State"] == state) 
        ]
    
    if len(required_row) == 0: return 0, {}
    data = [
        [float(required_row["Thermal"]), "Thermal"],
        [float(required_row["Hydro"]), "Hydro"],
        [float(required_row["Gas/Diesel/Naptha"]), "Gas/Diesel/Naptha"],
        [float(required_row["Wind"]), "Wind"] 
    ]

    day_df = pd.DataFrame(data=data, columns=["Value", "Energy"])
    fig = px.pie(day_df, values='Value', names='Energy', title='Different Type of Energy')
    return 1, fig

def month_based(month, year, state, energy_df):

    # df = energy_df[energy_df["State"] == state].groupby(['Month', 'Year']).sum()
    required_row = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['Month'] == month) &
                    (energy_df['State'] == state)
                    ]

    if len(required_row) == 0: return 0, {}
    data = [
        [required_row["Thermal"].sum(), "Thermal"],
        [required_row["Hydro"].sum(), "Hydro"],
        [required_row["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [required_row["Wind"].sum(), "Wind"] 
    ]

    day_df = pd.DataFrame(data=data, columns=["Value", "Energy"])
    fig = px.pie(day_df, values='Value', names='Energy', title='Different Type of Energy')
    return 1, fig

def year_based(year, state, energy_df):
    required_row = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['State'] == state)
                    ]
    if len(required_row) == 0: return 0, {}
    data = [
        [required_row["Thermal"].sum(), "Thermal"],
        [required_row["Hydro"].sum(), "Hydro"],
        [required_row["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [required_row["Wind"].sum(), "Wind"] 
    ]

    day_df = pd.DataFrame(data=data, columns=["Value", "Energy"])
    fig = px.pie(day_df, values='Value', names='Energy', title='Different Type of Energy')
    return 1, fig