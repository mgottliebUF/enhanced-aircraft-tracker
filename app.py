import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# Contiguous US Bounding Box Coordinates
US_BBOX = {
    'min_lon': -125.0,   # West Coast (California)
    'max_lon': -66.0,    # East Coast (Maine)
    'min_lat': 24.0,     # Southern Border (Florida)
    'max_lat': 49.0,     # Northern Border (Washington/Minnesota)
}

def fetch_real_flight_data():
    """
    Fetch real-time flight data from OpenSky Network API
    Filters flights within the continental US
    """
    try:
        # Parameters for API request
        params = {
            'lamin': US_BBOX['min_lat'],
            'lomin': US_BBOX['min_lon'],
            'lamax': US_BBOX['max_lat'],
            'lomax': US_BBOX['max_lon']
        }
        
        # Make API request
        response = requests.get(
            "https://opensky-network.org/api/states/all", 
            params=params
        )
        
        if response.status_code != 200:
            st.error(f"Failed to fetch flight data: {response.status_code}")
            return pd.DataFrame()
        
        # Parse response
        data = response.json()
        
        flights_data = []
        for flight in data['states']:
            # Validate and clean data
            if (flight[5] is not None and flight[6] is not None and 
                US_BBOX['min_lon'] <= flight[5] <= US_BBOX['max_lon'] and 
                US_BBOX['min_lat'] <= flight[6] <= US_BBOX['max_lat']):
                
                flights_data.append({
                    'flight_id': flight[1].strip() or 'Unknown',
                    'longitude': flight[5],
                    'latitude': flight[6],
                    'altitude': (flight[7] or 0) * 3.28084,  # Convert meters to feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # Convert m/s to knots
                    'heading': flight[10],
                    'vertical_rate': (flight[11] or 0) * 196.85,  # Convert m/s to ft/min
                    'country': flight[2]
                })
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

# Page setup
st.set_page_config(layout="wide", page_title="Real-Time US Flight Tracker")

# Title
st.title("Real-Time Flight Tracking over Continental US")

# Data loading with caching
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_data():
    return fetch_real_flight_data()

# Fetch data
df = load_data()

# Check if we have data
if df.empty:
    st.warning("No flight data available. Check network connection.")
else:
    # Flight selection
    available_flights = sorted(df['flight_id'].unique())
    selected_flights = st.sidebar.multiselect(
        "Select Flights", 
        available_flights, 
        default=available_flights[:min(3, len(available_flights))]
    )

    # Filter data
    flight_data = df[df['flight_id'].isin(selected_flights)]

    # Main display
    col1, col2 = st.columns(2)

    with col1:
        # Create US map with flight trajectories
        fig = go.Figure()

        # Add US map background
        fig.add_trace(go.Scattergeo(
            lon=[-98.5795],
            lat=[39.8283],
            mode='markers',
            marker=dict(
                size=1,
                color='rgba(0,0,0,0)',
                opacity=0
            ),
            showlegend=False
        ))

        # Plot flights
        for flight in selected_flights:
            flight_subset = flight_data[flight_data['flight_id'] == flight]
            fig.add_trace(go.Scattergeo(
                lon=flight_subset['longitude'],
                lat=flight_subset['latitude'],
                mode='lines+markers',
                name=flight,
                line=dict(width=2),
                marker=dict(
                    size=7,
                    color='red'
                )
            ))

        # Update layout for US map
        fig.update_geos(
            visible=True,
            resolution=50,
            scope='usa',
            showland=True, 
            landcolor="rgb(217, 217, 217)",
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)",
            showlakes=True, 
            lakecolor="rgb(255, 255, 255)",
            showsubunits=True,
            showcountries=True,
            showocean=True,
            oceancolor="rgb(230, 255, 255)"
        )

        fig.update_layout(
            title='Real-Time Flight Paths Across the United States',
            height=600,
            margin={"r":0,"t":30,"l":0,"b":0}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Flight Details Table
        st.dataframe(
            flight_data[['flight_id', 'country', 'altitude', 'ground_speed', 'heading']], 
            hide_index=True
        )

    # Detailed Flight Information
    st.header("Selected Flight Details")
    for flight in selected_flights:
        flight_subset = flight_data[flight_data['flight_id'] == flight]
        
        st.subheader(f"Flight {flight}")
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Country", flight_subset['country'].iloc[0])
        with cols[1]:
            st.metric("Max Altitude", f"{flight_subset['altitude'].max():.0f} ft")
        with cols[2]:
            st.metric("Avg Ground Speed", f"{flight_subset['ground_speed'].mean():.0f} knots")
        with cols[3]:
            st.metric("Heading", f"{flight_subset['heading'].mean():.0f}°")

    # Additional context
    st.markdown("""
    ### About This Visualization
    - Real-time flight data from OpenSky Network
    - Flights tracked over Continental US
    - Data refreshes every minute
    - [Learn more about OpenSky Network](https://opensky-network.org/)
    """)
