import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import math
import time

# Global variable to track last API call time
LAST_API_CALL_TIME = 0
API_CALL_INTERVAL = 60  # Minimum seconds between API calls

# Predefined major US airports
MAJOR_AIRPORTS = {
    'JFK': {'name': 'John F. Kennedy International', 'lat': 40.6413, 'lon': -73.7781},
    'LAX': {'name': 'Los Angeles International', 'lat': 33.9416, 'lon': -118.4085},
    'ORD': {'name': 'O\'Hare International', 'lat': 41.9742, 'lon': -87.9073},
    'ATL': {'name': 'Hartsfield-Jackson Atlanta', 'lat': 33.6367, 'lon': -84.4281},
    'DFW': {'name': 'Dallas/Fort Worth International', 'lat': 32.8998, 'lon': -97.0403},
    'DEN': {'name': 'Denver International', 'lat': 39.8561, 'lon': -104.6737},
    'SFO': {'name': 'San Francisco International', 'lat': 37.6213, 'lon': -122.3790},
    'SEA': {'name': 'Seattle-Tacoma International', 'lat': 47.4502, 'lon': -122.3088},
    'MIA': {'name': 'Miami International', 'lat': 25.7617, 'lon': -80.1918},
    'LAS': {'name': 'McCarran International', 'lat': 36.0840, 'lon': -115.1537}
}

# Contiguous US Bounding Box Coordinates
US_BBOX = {
    'min_lon': -125.0,
    'max_lon': -66.0,
    'min_lat': 24.0,
    'max_lat': 49.0,
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

def match_airport_code(flight_id):
    """
    Attempt to extract airport codes from flight ID
    """
    # Common flight ID patterns like AA1234, UA567
    possible_airports = [code for code in MAJOR_AIRPORTS.keys() 
                         if code in flight_id]
    return possible_airports[0] if possible_airports else None

def fetch_real_flight_data():
    """
    Fetch real-time flight data from OpenSky Network API
    Filters flights within the continental US
    """
    global LAST_API_CALL_TIME
    
    # Rate limiting
    current_time = time.time()
    if current_time - LAST_API_CALL_TIME < API_CALL_INTERVAL:
        st.warning(f"Waiting {API_CALL_INTERVAL} seconds between API calls to avoid rate limits.")
        time.sleep(API_CALL_INTERVAL - (current_time - LAST_API_CALL_TIME))
    
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
        
        # Update last API call time
        LAST_API_CALL_TIME = time.time()
        
        if response.status_code != 200:
            st.error(f"Failed to fetch flight data: {response.status_code}")
            return pd.DataFrame()
        
        # Parse response
        data = response.json()
        
        # Limit number of flights to prevent overwhelming the app
        max_flights = 50
        flights_data = []
        for flight in data['states'][:max_flights]:
            # Validate and clean data
            if (flight[5] is not None and flight[6] is not None and 
                US_BBOX['min_lon'] <= flight[5] <= US_BBOX['max_lon'] and 
                US_BBOX['min_lat'] <= flight[6] <= US_BBOX['max_lat']):
                
                # Try to match airport code
                airport_code = match_airport_code(flight[1].strip() or 'Unknown')
                origin = MAJOR_AIRPORTS.get(airport_code, None)
                
                flight_entry = {
                    'flight_id': flight[1].strip() or 'Unknown',
                    'longitude': flight[5],
                    'latitude': flight[6],
                    'altitude': (flight[7] or 0) * 3.28084,  # Convert meters to feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # Convert m/s to knots
                    'heading': flight[10],
                    'vertical_rate': (flight[11] or 0) * 196.85,  # Convert m/s to ft/min
                    'country': flight[2]
                }
                
                # Add origin information if available
                if origin:
                    flight_entry.update({
                        'origin_airport': airport_code,
                        'origin_name': origin['name'],
                        'origin_lat': origin['lat'],
                        'origin_lon': origin['lon']
                    })
                
                flights_data.append(flight_entry)
        
        # Convert to DataFrame
        df = pd.DataFrame(flights_data)
        
        # Calculate additional metrics
        if not df.empty:
            # Calculate estimated flight distance and time
            distances = []
            est_times = []
            for _, row in df.iterrows():
                if 'origin_lat' in row and 'origin_lon' in row:
                    distance = haversine_distance(
                        row['origin_lat'], row['origin_lon'], 
                        row['latitude'], row['longitude']
                    )
                    distances.append(distance)
                    # Estimate time based on ground speed (if available)
                    est_time = distance / row['ground_speed'] if row['ground_speed'] > 0 else np.nan
                    est_times.append(est_time)
                else:
                    distances.append(np.nan)
                    est_times.append(np.nan)
            
            df['estimated_distance'] = distances
            df['estimated_time'] = est_times
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

# Page setup
st.set_page_config(layout="wide", page_title="Advanced US Flight Tracker")

# Title and refresh
st.title("Real-Time Flight Tracking & Analytics")
refresh_button = st.button("Refresh Flight Data")

# Data loading with caching
if 'last_refresh' not in st.session_state or refresh_button:
    st.session_state['last_refresh'] = datetime.now()
    st.session_state['flight_data'] = fetch_real_flight_data()

# Fetch data
df = st.session_state.get('flight_data', pd.DataFrame())

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

    # Main display - First row of visualizations
    col1, col2, col3 = st.columns(3)

    with col1:
        # US Map with Flight Paths
        fig_map = go.Figure()

        # Add US map background
        fig_map.add_trace(go.Scattergeo(
            lon=[-98.5795],
            lat=[39.8283],
            mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            showlegend=False
        ))

        # Plot flights and origin points
        for flight in selected_flights:
            flight_subset = flight_data[flight_data['flight_id'] == flight]
            
            # Flight path
            fig_map.add_trace(go.Scattergeo(
                lon=flight_subset['longitude'],
                lat=flight_subset['latitude'],
                mode='lines+markers',
                name=flight,
                line=dict(width=2),
                marker=dict(size=7, color='red')
            ))
            
            # Origin point if available
            if 'origin_lon' in flight_subset.columns and 'origin_lat' in flight_subset.columns:
                fig_map.add_trace(go.Scattergeo(
                    lon=[flight_subset['origin_lon'].iloc[0]],
                    lat=[flight_subset['origin_lat'].iloc[0]],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='star'),
                    name=f"{flight} Origin"
                ))

        # Update layout for US map
        fig_map.update_geos(
            visible=True, resolution=50, scope='usa',
            showland=True, landcolor="rgb(217, 217, 217)",
            showlakes=True, lakecolor="rgb(255, 255, 255)",
        )
        fig_map.update_layout(title='Flight Paths', height=400)
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        # Altitude Distribution
        fig_altitude = px.histogram(
            flight_data, 
            x='altitude', 
            color='flight_id', 
            title='Altitude Distribution',
            labels={'altitude': 'Altitude (feet)'}
        )
        fig_altitude.update_layout(height=400)
        st.plotly_chart(fig_altitude, use_container_width=True)

    with col3:
        # Ground Speed Distribution
        fig_speed = px.box(
            flight_data, 
            x='flight_id', 
            y='ground_speed', 
            title='Ground Speed by Flight',
            labels={'ground_speed': 'Speed (knots)'}
        )
        fig_speed.update_layout(height=400)
        st.plotly_chart(fig_speed, use_container_width=True)

    # Second row of visualizations
    col4, col5, col6 = st.columns(3)

    with col4:
        # Vertical Rate Analysis
        fig_vertical = px.scatter(
            flight_data, 
            x='flight_id', 
            y='vertical_rate', 
            title='Vertical Speed Variation',
            labels={'vertical_rate': 'Vertical Speed (ft/min)'}
        )
        fig_vertical.update_layout(height=400)
        st.plotly_chart(fig_vertical, use_container_width=True)

    with col5:
        # Flight Details Table
        # Use .get() to safely access columns with defaults
        details_df = flight_data.copy()
        
        # Prepare columns with safe fallback
        display_columns = ['flight_id', 'altitude', 'ground_speed']
        optional_columns = ['origin_airport', 'estimated_distance', 'estimated_time']
        
        # Add optional columns if they exist
        for col in optional_columns:
            if col in details_df.columns:
                display_columns.append(col)
        
        st.dataframe(
            details_df[display_columns], 
            column_config={
                'estimated_distance': st.column_config.NumberColumn(
                    "Est. Distance (miles)",
                    format="%.1f mi" if 'estimated_distance' in details_df.columns else None
                ),
                'estimated_time': st.column_config.NumberColumn(
                    "Est. Flight Time (hrs)",
                    format="%.2f hrs" if 'estimated_time' in details_df.columns else None
                )
            },
            hide_index=True,
            use_container_width=True
        )

    with col6:
        # Heading Distribution
        fig_heading = px.pie(
            flight_data, 
            names='flight_id', 
            values='heading', 
            title='Flight Heading Distribution'
        )
        fig_heading.update_layout(height=400)
        st.plotly_chart(fig_heading, use_container_width=True)

    # Last Updated Timestamp
    st.markdown(f"**Last Refreshed:** {st.session_state.get('last_refresh', 'Never')}")

    # Additional context
    st.markdown("""
    ### About This Visualization
    - Real-time flight data from OpenSky Network
    - Limited to 50 flights to prevent API overload
    - Multiple analytics and visualizations
    - Estimated flight metrics
    - Click 'Refresh Flight Data' to get latest information
    """)
