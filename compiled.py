import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import geopandas as gpd
from task3 import task3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

with st.sidebar:

    choose = option_menu(" ", ["About", "Task1","Task2", "Task3", "Task4", "Task5"],
                         icons=['house', 'pin-map','bar-chart-steps','pin-map-fill','star','calculator'],
                         menu_icon="list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#FFF3E2"},
        "icon": {"color": "#7C9070", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee", "color": "#7C9070", "font-weight": "bold"},
        "nav-link-selected": {"background-color": "#FEE8B0", 'color': '#7C9070', 'border-radius': '5px'},
    }
    )

logo = Image.open(r'Data/logo.jpeg')
if choose == "About":
    st.write("# India's Energy Story: Data Insights and Analysis")
    st.write("## Data Sources")
    st.write("""- The dataset compiled from the daily power supply position reports published by the Regional Load 
    Dispatch Centers (RLDCs) across India. Spanning from January 1, 2020, to July 1, 2023, the dataset offers a comprehensive
     view of the power supply situation for each state within the respective regions. The RLDCs involved include the Northern
      Regional Load Dispatch Center (NRLDC), Southern Regional Load Dispatch Center (SRLDC), Western Regional Load Dispatch 
      Center (WRLDC), Eastern Regional Load Dispatch Center (ERLDC), and North Eastern Regional Load Dispatch Center (NERLDC).""")
    st.write("""- Each entry in the dataset encapsulates information pertinent to the power supply dynamics within the region. 
    This includes data on Renewable Energy generation—categorized into Solar, Wind, Hydro, and Bio-Gas—alongside figures pertaining
     to Thermal Generation, power Demand, and prevailing Weather conditions. Such a dataset not only serves as a critical 
     resource for understanding the energy landscape of the regions covered but also aids in the assessment of renewable versus 
     traditional energy generation capacities over the specified period.""")
    st.write("- NRLDC: https://nrldc.in/reports/daily-reports/daily-regional-power-supply-position/")
    st.write("- SRLDC: https://srldc.in/Daily-Reports")
    st.write("- WRLDC: https://www.wrldc.in/content/168_1_DailyReports.aspx")
    st.write("- ERLDC: https://erldc.in/en/mis-reports/daily-psp-report/")
    st.write("- NERLDC: https://www.nerldc.in/power-supply-position-psp-report/")
    st.write("""- Indian states geoJson data for rendering state production of different types of energy and it's consumption has been taken 
    from kaggle. Any mistake found in the geography of data is the reult of the dataset('we searched for better one but to fail').""")
    st.write("- GeoJason dataset: https://www.kaggle.com/datasets/sauravmishra1710/indian-state-geojson-data")

    st.write("## Tasks")
    st.write("### 1. Production of renewable energy in Indian states")
    st.write("""The data of the amount of different forms of renewable energy produced in various states of India has been 
    processed. This task aims to visualize the production of renewable energy in different  states so that the production values 
    of states can be known and compared easily with the help of interactive visuals. The visuals will be such that information 
    about all the states can be viewed simultaneously at intervals of each year. """)
    st.write("### 2. Generation and consumption of electric energy - 2 maps")
    st.write("""This task will try to establish a comparison between the production and consumption of electric energy in 
    different states of India. The visualization of this task is planned in a way that the comparisons can be made directly on 
    the political map of India. """)
    st.write("### 3. Analysis of energy produced in each state")
    st.write("""There are different types of energy produced namely wind energy, solar energy, thermal energy and hydro energy.
     This task will allow us to view the data for each kind of energy produced in each state in a hierarchical format so that the
      visualization could become easily understood to users.""")
    st.write("### 4. Effects of temperature on consumption of electricity")
    st.write("""As the title of the task suggests, we will plot the data of average temperature of different states against the 
    consumption of electricity over a time period of a year so that it becomes handy to analyze the electricity consumption trends.""")
    st.write("### 5. Factors affecting production of renewable energy")
    st.write("""There exist various climate variables which impact the production of electric energy such as humidity, 
    temperature, wind speed, wind direction, etc. So, the objective of this task is to obtain a relationship between 
    environmental factors and production of different forms of renewable energy.""")
    

    st.write("## Team Members -Group 25")
    st.write("- Amit Kumar :smirk:")    
    st.write("- Praveen Singh :unamused:")
    st.write("- Abhishek Piwal :penguin:")
    st.write("- Harshit Kumar Tiwari :confused:")
    st.write("- Sameer Khan :sunglasses:")
    st.write("- Chinmay Agarwal :sunglasses:")
    st.write("- Harsit Sinha :sunglasses:")
    st.write("- Sujal Singh :sunglasses:")


elif choose=='Task1':
    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Comic Sans'; color: #cca300} 
        </style> """, unsafe_allow_html=True)
    er_list = ["Renewable", "Non-Renewable", "Others"]
    st.markdown('<p class="font">Quarterly Energy Production Map</p>', unsafe_allow_html=True)  

    energydata = pd.read_csv('Data/FinalTask1_Data.csv')
    geojson_path = "Data/Indian_States.json"
    geodata = gpd.read_file(geojson_path)
    geodata['NAME_1'] = geodata['NAME_1'].str.upper()
    energydata['State'] = energydata['State'].str.upper()
    # energydata = energydata.groupby(['state', 'Year']).sum().reset_index()


    with st.form(key='energy_form'):
        text_style = '<p style="font-family:sans-serif; color:red; font-size: 15px;">***These input fields are required***</p>'
        st.markdown(text_style, unsafe_allow_html=True)
        column0, column1 = st.columns([1,1])
        # column11, column11 = st.columns([1,1])
        with column0:
            energy_type=st.selectbox('Energy Type',er_list, index=0, help='Choose the type of energy whose map you desire')
        with column1:
            year = st.selectbox("Select Year", sorted(energydata['Year'].unique()))
        with column0:
            quarters_for_year = sorted(energydata[energydata['Year'] == year]['Quarter'].unique())
            quarter = st.selectbox("Select Quarter", quarters_for_year)
        with column1:
            st.write("")
            st.write("")
            submitted_energy_type = st.form_submit_button('Submit')

    # Plot Choropleth map
    if submitted_energy_type:
        filtered_energydata = energydata[(energydata['Year'] == year) & (energydata['Quarter'] == quarter)]
        merged_data = pd.merge(geodata, filtered_energydata, how='left', left_on='NAME_1', right_on='State')
        merged_data['Renewable'] = merged_data['Renewable'].fillna(-1000)
        merged_data['Non-Renewable'] = merged_data['Non-Renewable'].fillna(-1000)
        merged_data['Others'] = merged_data['Others'].fillna(-1000)
        # merged_data = merged_data.groupby(['Year']).sum().reset_index()
        # st.write(merged_data.shape)
        if(energy_type == "Renewable"):
            st.subheader("Renewable Energy")
            fig = px.choropleth(
                merged_data, 
                geojson=merged_data.geometry, 
                locations=merged_data.index, 
                color='Renewable', 
                hover_name='NAME_1', 
                # hover_data={'Your_Column1': True, 'Your_Column2': True},
                projection='mercator',
                color_continuous_scale='Greens'  # Set color scale to green
            )
            fig.update_layout(
                autosize=False,
                width=800,  # Set width
                height=600,  # Set height
                margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig)



        elif(energy_type=="Non-Renewable"):
            st.subheader("Non-Renewable Energy")
            fig = px.choropleth(
                merged_data, 
                geojson=merged_data.geometry, 
                locations=merged_data.index, 
                color='Non-Renewable', 
                hover_name='NAME_1', 
                projection='mercator',
                color_continuous_scale='YlOrRd'  # Set color scale to green
            )
            fig.update_layout(
                autosize=False,
                width=800,  # Set width
                height=600,  # Set height
                margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig)
        elif(energy_type=="Others"):
            st.subheader("Other Form of Energy")
            fig = px.choropleth(
                merged_data, 
                geojson=merged_data.geometry, 
                locations=merged_data.index, 
                color='Others', 
                hover_name='NAME_1', 
                projection='mercator',
                color_continuous_scale='Blues'  # Set color scale to green
            )
            fig.update_layout(
                autosize=False,
                width=800,  # Set width
                height=600,  # Set height
                margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig)
    
    
elif choose=='Task2':
    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Comic Sans'; color: #cca300} 
        </style> """, unsafe_allow_html=True) 
    lis = ["Consumed", "Produced",]
    st.markdown('<p class="font">Quarterly Energy Production Map</p>', unsafe_allow_html=True)  

    energydata = pd.read_csv('Data/FinalTask2_Data.csv')
    geojson_path = "Data/Indian_States.json"
    geodata = gpd.read_file(geojson_path)
    geodata['NAME_1'] = geodata['NAME_1'].str.upper()
    energydata['State'] = energydata['State'].str.upper()
    energydata.rename(columns={'Demand Met': 'Consumed'}, inplace=True)


    with st.form(key='energy_form'):
        text_style = '<p style="font-family:sans-serif; color:red; font-size: 15px;">***These input fields are required***</p>'
        st.markdown(text_style, unsafe_allow_html=True)
        column0, column1 = st.columns([1,1])
        with column0:
            selection_type=st.selectbox('Selection Type',lis, index=0, help='Choose wether you want to see production or consuption data')
        with column1:
            year = st.selectbox("Select Year", sorted(energydata['Year'].unique()))
        with column0:
            quarters_for_year = sorted(energydata[energydata['Year'] == year]['Quarter'].unique())
            quarter = st.selectbox("Select Quarter", quarters_for_year)
        with column1:
            st.write("")
            st.write("")
            submitted_energy_type = st.form_submit_button('Submit')


    # Plot Choropleth map
    if submitted_energy_type:
        filtered_energydata = energydata[(energydata['Year'] == year) & (energydata['Quarter'] == quarter)]
        merged_data = pd.merge(geodata, filtered_energydata, how='left', left_on='NAME_1', right_on='State')
        merged_data['Produced'] = merged_data['Produced'].fillna(-1000)
        merged_data['Consumed'] = merged_data['Consumed'].fillna(-1000)
        # st.write(merged_data.columns)
        if(selection_type == "Produced"):
            st.subheader("Produced Energy")
            fig = px.choropleth(
                merged_data, 
                geojson=merged_data.geometry, 
                locations=merged_data.index, 
                color='Produced', 
                hover_name='NAME_1', 
                projection='mercator',
                color_continuous_scale='Greens',
                # animation_frame='Quarter'
            )
            # fig.update_geos(
            #     showcountries=True,  # Show country boundaries
            #     countrycolor='lightgrey',  # Set the color of the country boundaries
            #     countrywidth=1  # Set the width of the country boundaries
            # )
            fig.update_layout(
                autosize=False,
                width=800,  # Set width
                height=600,  # Set height
                margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig)
        elif(selection_type=="Consumed"):
            st.subheader("Consumed Energy")
            fig = px.choropleth(
                merged_data, 
                geojson=merged_data.geometry, 
                locations=merged_data.index, 
                color='Consumed', 
                hover_name='NAME_1', 
                projection='mercator',
                color_continuous_scale='Blues'  # Set color scale to green
            )
            fig.update_layout(
                autosize=False,
                width=800,  # Set width
                height=600,  # Set height
                margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig)
    

elif choose == 'Task3':
    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Comic Sans'; color: #cca300} 
        </style> """, unsafe_allow_html=True)
    energy_df = pd.read_csv("Data/energy.csv")
    task3(energy_df)

elif choose == 'Task4':
    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Comic Sans'; color: #cca300} 
        </style> """, unsafe_allow_html=True)
    df = pd.read_csv("Data/mergedData.csv")
    def plot_state_data(df, state):
        state_df = df[df['State'] == state]
        
        state_df['temp_c'] = (state_df['temp_c1'] + state_df['temp_c2'] + state_df['temp_c3'] + state_df['temp_c4']) / 4
        state_df['Demand Met'] = state_df['Demand Met'].astype(float)
        state_df = state_df[['Date', 'temp_c', 'Demand Met']]

            # Check if temp_c is 0 for every month
        if (state_df['temp_c'] == 0).all():
            st.text("Sufficient data isn't available for this state")
            return

        scaler = MinMaxScaler()
        state_df[['temp_c', 'Demand Met']] = scaler.fit_transform(state_df[['temp_c', 'Demand Met']])
        state_df['Date'] = pd.to_datetime(state_df['Date'])
        state_df['Month'] = state_df['Date'].dt.month

        monthly_avg = state_df.groupby('Month').mean().reset_index()
        
        # Set style to classic to have white background
        plt.style.use('classic')

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=monthly_avg, x='Month', y='Demand Met', marker='o', ax=ax1, label='Energy Consumed (KWh/day)')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Energy Consumed (KWh/day)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        sns.lineplot(data=monthly_avg, x='Month', y='temp_c', marker='o', ax=ax2, color='tab:red', label='Temperature (°C)')
        ax2.set_ylabel('Temperature (°C)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        def inverse_transform_y(y, i):
            if i == 1:
                return scaler.inverse_transform([[0, y]])[0]
            else:
                return scaler.inverse_transform([[y, 0]])[0]

        yticks = ax2.get_yticks()
        ax2.set_yticklabels([f'{inverse_transform_y(y, 0)[0]:.1f}' for y in yticks])

        yticks = ax1.get_yticks()
        ax1.set_yticklabels([f'{inverse_transform_y(y, 1)[1]:.0f}' for y in yticks])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(1.1, 1), borderaxespad=0.1)

        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


        plt.title(f'Monthly Average Energy Consumed and Temperature - {state}')

        ax1.grid(True)

        st.pyplot(fig)

    st.title('State-wise Monthly Average Energy Consumed and Temperature')

    

    # Create a dropdown to select state
    state_selection = st.selectbox('Select State', df['State'].unique())

    # Plot data for selected state
    plot_state_data(df, state_selection)


elif choose == 'Task5':
    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Comic Sans'; color: #cca300} 
        </style> """, unsafe_allow_html=True)
    # Load data
    df = pd.read_csv("Data/mergedData.csv")
    # Get unique states in the DataFrame
    states = df['State'].unique()

    # Streamlit app
    st.title("Factors affecting production of renewable energy")

    selected_state = st.selectbox("Select State", states)

    # Filter the DataFrame for the selected state
    state_df = df[df['State'] == selected_state]

    # Preprocessing steps
    state_df['solarradiation_c'] = (state_df['solarradiation_c4'] + state_df['solarradiation_c4'] + state_df['solarradiation_c4'] + state_df['solarradiation_c4']) / 4
    state_df['solarenergy'] = (state_df['solarenergy_c1'] + state_df['solarenergy_c2'] + state_df['solarenergy_c3'] + state_df['solarenergy_c4']) / 4
    state_df = state_df[['Date', 'solarradiation_c', 'solarenergy']]

    scaler = MinMaxScaler()
    state_df[['solarradiation_c', 'solarenergy']] = scaler.fit_transform(state_df[['solarradiation_c', 'solarenergy']])
    state_df['Date'] = pd.to_datetime(state_df['Date'])
    state_df['Month'] = state_df['Date'].dt.month

    # Group by month and calculate the mean for both 'solarenergy' and 'cloudcover_c'
    monthly_avg = state_df.groupby('Month').mean().reset_index()

    # Create figure and plot for the selected state
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot solarenergy with shaded area
    sns.lineplot(data=monthly_avg, x='Month', y='solarenergy', ax=ax1, label='solarenergy(kWh/m^2)', linewidth=2, color='cyan')  # Change color to cyan
    ax1.fill_between(monthly_avg['Month'], monthly_avg['solarenergy'], color='skyblue', alpha=0.3)  # Adding translucent shade
    ax1.set_xlabel('Month')
    ax1.set_ylabel('solarenergy(kWh/m^2)', color='cyan')  # Change color to cyan
    ax1.tick_params(axis='y', labelcolor='cyan')  # Change color to cyan

    # Create another y-axis for cloudcover
    ax2 = ax1.twinx()
    sns.lineplot(data=monthly_avg, x='Month', y='solarradiation_c', ax=ax2, label='Solar Radiation (kW/m2)', linewidth=2, color='orange')  # Change color to magenta
    ax2.fill_between(monthly_avg['Month'], monthly_avg['solarradiation_c'], color='orange', alpha=0.3)  # Adding translucent shade, change color to purple
    ax2.set_ylabel('Solar Radiation (kW/m2)', color='orange')  # Change color to magenta
    ax2.tick_params(axis='y', labelcolor='orange')  # Change color to magenta

    # Inverse-transform y-axis ticks to original values
    def inverse_transform_y(y, i):
        if i == 1:
            return scaler.inverse_transform([[0, y]])[0]
        else:
            return scaler.inverse_transform([[y, 0]])[0]

    # Apply inverse transform to y-axis ticks for cloudcover
    yticks = ax2.get_yticks()
    ax2.set_yticklabels([f'{inverse_transform_y(y, 0)[0]:.1f}' for y in yticks])

    # Apply inverse transform to y-axis ticks
    yticks = ax1.get_yticks()
    ax1.set_yticklabels([f'{inverse_transform_y(y, 1)[1]:.0f}' for y in yticks])

    # Customize legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(1.1, 1), borderaxespad=0.1)

    # Set x-axis ticks and labels
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Add title
    plt.title(f'Monthly Average solarenergy and solar radiation_c4 - {selected_state}')

    # Show grid
    ax1.grid(False)

    # Show plot in Streamlit app
    st.pyplot(fig)
    
    # Heat map
    # Disable the PyplotGlobalUseWarning
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Load data
    df = pd.read_csv("Data/mergedData.csv")

    # Calculate average values for the columns
    df['cloudcover'] = (df['cloudcover_c1'] + df['cloudcover_c2'] + df['cloudcover_c3'] + df['cloudcover_c4']) / 4
    df['solarenergy'] = (df['solarenergy_c1'] + df['solarenergy_c2'] + df['solarenergy_c3'] + df['solarenergy_c4']) / 4
    df['humidity'] = (df['humidity_c1'] + df['humidity_c2'] + df['humidity_c3'] + df['humidity_c4']) / 4
    df['precip'] = (df['precip_c1'] + df['precip_c2'] + df['precip_c3'] + df['precip_c4']) / 4
    df['windspeed'] = (df['windspeed_c1'] + df['windspeed_c2'] + df['windspeed_c3'] + df['windspeed_c4']) / 4
    df['solarradiation'] = (df['solarradiation_c1'] + df['solarradiation_c2'] + df['solarradiation_c3'] + df['solarradiation_c4']) / 4
    df['temp'] = (df['temp_c1'] + df['temp_c2'] + df['temp_c3'] + df['temp_c4']) / 4

    selected_columns = ['Hydro', 'Wind', 'solarenergy', 'cloudcover', 'humidity', 'precip', 'windspeed', 'solarradiation', 'temp']
    corr_df = df[selected_columns]
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing values
    corr_df.dropna(inplace=True)
    # Compute the correlation matrix
    corr_matrix = corr_df.corr()

    # # Display the correlation plots with styling
    # st.subheader("Correlation Heatmap")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    # ax.set_title('Correlation Heatmap')
    # st.pyplot(fig)



    # st.subheader("Pearson Correlation Plot")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
    # ax.set_title('Pearson Correlation Plot')
    # st.pyplot(fig)

    st.subheader("Triangular Correlation Plot")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, square=True, ax=ax)
    ax.set_title('Triangular Correlation Plot')
    st.pyplot(fig)


    # Plot selected scatter plot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # # Scatter plot between 'solarradiation' and 'solarenergy'
    # ax1.scatter(df['solarradiation'], df['solarenergy'], color='blue')
    # ax1.set_xlabel('solarradiation')
    # ax1.set_ylabel('solarenergy')
    # ax1.set_title('Scatter Plot: solarradiation vs solarenergy')

    # # Scatter plot between 'temp' and 'solarenergy'
    # ax2.scatter(df['temp'], df['solarenergy'], color='red')
    # ax2.set_xlabel('temp')
    # ax2.set_ylabel('solarenergy')
    # ax2.set_title('Scatter Plot: temp vs solarenergy')



    # # Display the plots in Streamlit
    # st.pyplot(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot between 'solarradiation' and 'solarenergy' with best-fit line
    axs[0, 0].scatter(df['solarradiation'], df['solarenergy'], color='blue')
    axs[0, 0].set_xlabel('solarradiation')
    axs[0, 0].set_ylabel('solarenergy')
    axs[0, 0].set_title('Scatter Plot: solarradiation vs solarenergy')



    # Scatter plot between 'temp' and 'solarenergy' with best-fit line
    axs[0, 1].scatter(df['temp'], df['solarenergy'], color='red')
    axs[0, 1].set_xlabel('temp')
    axs[0, 1].set_ylabel('solarenergy')
    axs[0, 1].set_title('Scatter Plot: temp vs solarenergy')



    # Scatter plot between 'solarenergy' and 'windspeed' with best-fit line
    axs[1, 0].scatter(df['solarenergy'], df['windspeed'], color='green')
    axs[1, 0].set_xlabel('solarenergy')
    axs[1, 0].set_ylabel('windspeed')
    axs[1, 0].set_title('Scatter Plot: solarenergy vs windspeed')



    # Scatter plot between 'solarenergy' and 'humidity' with best-fit line
    axs[1, 1].scatter(df['solarenergy'], df['humidity'], color='purple')
    axs[1, 1].set_xlabel('solarenergy')
    axs[1, 1].set_ylabel('humidity')
    axs[1, 1].set_title('Scatter Plot: solarenergy vs humidity')



    # Display the plots in Streamlit
    st.pyplot(fig)
