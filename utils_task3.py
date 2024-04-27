import pandas as pd
import plotly.express as px

# this contains the data manipulation and graph plotting functions

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

    day_df = pd.DataFrame(data=data, columns=["Energy", "Energy Type"])
    fig = px.pie(day_df, values='Energy', names='Energy Type', title=f'Energy Classification - {state} - {day}/{month}/{year}')
    return 1, fig

def month_based(month, year, state, energy_df):

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
    fig = px.pie(day_df, values='Value', names='Energy', title=f'Energy Classification - {state} - {month}/{year}')
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
    fig = px.pie(day_df, values='Value', names='Energy', title=f'Energy Classification - {state} - {year}')
    return 1, fig

def month2_based(state, year, energy_df):
    df2 = energy_df[(energy_df["Year"] == year) & (energy_df["State"] == state)].groupby("Month", as_index=False).sum()
    if year == 2023: df2 = df2.assign(Month_Name=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"])
    else: df2 = df2.assign(Month_Name=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    row = []
    for i in range(df2.shape[0]):
        
        row.append([df2.loc[i]["Month_Name"], df2.loc[i]["Thermal"], "Thermal"])
        row.append([df2.loc[i]["Month_Name"], df2.loc[i]["Wind"], "Wind"])
        row.append([df2.loc[i]["Month_Name"], df2.loc[i]["Hydro"], "Hydro"])
        row.append([df2.loc[i]["Month_Name"], df2.loc[i]["Gas/Diesel/Naptha"], "Gas/Diesel/Naptha"])

    yearly_df = pd.DataFrame(data=row, columns=["Month", "Energy", "Energy Type"])
    fig = px.bar(yearly_df, x="Month", y="Energy",color='Energy Type')
    return fig

def state_based(year, energy_df):
    df2 = energy_df[energy_df["Year"] == year].groupby("State", as_index=False).sum()
    row = []
    for i in range(df2.shape[0]):
        
        row.append([df2.loc[i]["State"], df2.loc[i]["Thermal"], "Thermal"])
        row.append([df2.loc[i]["State"], df2.loc[i]["Wind"], "Wind"])
        row.append([df2.loc[i]["State"], df2.loc[i]["Hydro"], "Hydro"])
        row.append([df2.loc[i]["State"], df2.loc[i]["Gas/Diesel/Naptha"], "Gas/Diesel/Naptha"])
    
    yearly_df = pd.DataFrame(data=row, columns=["State", "Energy", "Energy Type"])
    fig = px.bar(yearly_df, x="State", y="Energy",color='Energy Type')
    return fig

def compare_states_date(state1, state2, energy_df, day, month, year):

    required_row1 = energy_df[
        (energy_df["Day"] == day) & 
        (energy_df["Month"] == month) & 
        (energy_df["Year"] == year) & 
        (energy_df["State"] == state1) 
        ]
    
    if len(required_row1) == 0: return 0, {}
    data1 = [
        [state1, float(required_row1["Thermal"]), "Thermal"],
        [state1, float(required_row1["Hydro"]), "Hydro"],
        [state1, float(required_row1["Gas/Diesel/Naptha"]), "Gas/Diesel/Naptha"],
        [state1, float(required_row1["Wind"]), "Wind"] 
    ]

    required_row2 = energy_df[
        (energy_df["Day"] == day) & 
        (energy_df["Month"] == month) & 
        (energy_df["Year"] == year) & 
        (energy_df["State"] == state2) 
        ]
    
    if len(required_row2) == 0: return 0, {}
    data2 = [
        [state2, float(required_row2["Thermal"]), "Thermal"],
        [state2, float(required_row2["Hydro"]), "Hydro"],
        [state2, float(required_row2["Gas/Diesel/Naptha"]), "Gas/Diesel/Naptha"],
        [state2, float(required_row2["Wind"]), "Wind"] 
    ]

    data = data1 + data2
    yearly_df = pd.DataFrame(data=data, columns=["State", "Value", "Energy Type"])
    fig = px.bar(yearly_df, x="Energy Type", y="Value",color='State', barmode="group")
    return 1, fig

def compare_states_month(state1, state2, energy_df, month, year):
    
    required_row1 = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['Month'] == month) &
                    (energy_df['State'] == state1)
                    ]

    if len(required_row1) == 0: return 0, {}
    data1 = [
        [state1, required_row1["Thermal"].sum(), "Thermal"],
        [state1, required_row1["Hydro"].sum(), "Hydro"],
        [state1, required_row1["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [state1, required_row1["Wind"].sum(), "Wind"] 
    ]

    required_row2 = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['Month'] == month) &
                    (energy_df['State'] == state2)
                    ]

    if len(required_row2) == 0: return 0, {}
    data2 = [
        [state2, required_row2["Thermal"].sum(), "Thermal"],
        [state2, required_row2["Hydro"].sum(), "Hydro"],
        [state2, required_row2["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [state2, required_row2["Wind"].sum(), "Wind"] 
    ]
    
    data = data1 + data2 

    yearly_df = pd.DataFrame(data=data, columns=["State", "Energy", "Energy Type"])
    fig = px.bar(yearly_df, x="Energy Type", y="Energy",color='State', barmode="group")
    return 1, fig

def compare_states_year(state1, state2, energy_df, year):
    required_row1 = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['State'] == state1)
                    ]
    if len(required_row1) == 0: return 0, {}
    data1 = [
        [state1, required_row1["Thermal"].sum(), "Thermal"],
        [state1, required_row1["Hydro"].sum(), "Hydro"],
        [state1, required_row1["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [state1, required_row1["Wind"].sum(), "Wind"] 
    ]

    required_row2 = energy_df[
                    (energy_df['Year'] == year) &
                    (energy_df['State'] == state2)
                    ]
    if len(required_row2) == 0: return 0, {}

    data2 = [
        [state2, required_row2["Thermal"].sum(), "Thermal"],
        [state2, required_row2["Hydro"].sum(), "Hydro"],
        [state2, required_row2["Gas/Diesel/Naptha"].sum(), "Gas/Diesel/Naptha"],
        [state2, required_row2["Wind"].sum(), "Wind"] 
    ]

    data = data1 + data2

    yearly_df = pd.DataFrame(data=data, columns=["State", "Energy", "Energy Type"])
    fig = px.bar(yearly_df, x="Energy Type", y="Energy",color='State', barmode="group")
    return 1, fig
