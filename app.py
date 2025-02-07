import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import math
import time

# Global configuration
st.set_page_config(layout="wide", page_title="Advanced Flight Tracker", page_icon="✈️")

# Predefined major US airports with additional information
MAJOR_AIRPORTS = {
    'JFK': {'name': 'John F. Kennedy International', 'lat': 40.6413, 'lon': -73.7781, 'city': 'New York'},
    'LAX': {'name': 'Los Angeles International', 'lat': 33.9416, 'lon': -118.4085, 'city': 'Los Angeles'},
    'ORD': {'name': 'O\'Hare International', 'lat': 41.9742, 'lon': -87.9073, 'city': 'Chicago'},
    'ATL': {'name': 'Hartsfield-Jackson Atlanta', 'lat': 33.6367, 'lon': -84.4281, 'city': 'Atlanta'},
    'DFW': {'name': 'Dallas/Fort Worth International', 'lat': 32.8998, 'lon': -97.0403, 'city': 'Dallas'},
    'DEN': {'name': 'Denver International', 'lat': 39.8561, 'lon': -104.6737, 'city': 'Denver'},
    'SFO': {'name': 'San Francisco International', 'lat': 37.6213, 'lon': -122.3790, 'city': 'San Francisco'},
    'SEA': {'name': 'Seattle-Tacoma International', 'lat': 47.4502, 'lon': -122.3088, 'city': 'Seattle'},
    'MIA': {'name': 'Miami International', 'lat': 25.7617, 'lon': -80.1918, 'city': 'Miami'},
    'LAS': {'name': 'McCarran International', 'lat': 36.0840, 'lon': -115.1537, 'city': 'Las Vegas'}
}

# Contiguous US Bounding Box Coordinates
US_BBOX = {
    'min_lon': -125.0,   # West Coast (California)
    'max_lon': -66.0,    # East Coast (Maine)
    'min_lat': 24.0,     # Southern Border (Florida)
    'max_lat': 49.0,     # Northern Border (Washington/Minnesota)
}

def predict_future_path(current_lat, current_lon, heading, distance=100):
    """
    Predict future flight path based on current position and heading
    
    :param current_lat: Current latitude
    :param current_lon: Current longitude
    :param heading: Current flight heading in degrees
    :param distance: Distance to predict (in miles)
    :return: Tuple of (predicted_lat, predicted_lon)
    """
    # Convert heading to radians
    heading_rad = math.radians(heading)
    
    # Earth's radius in miles
    R = 3958.8  # miles
    
    # Convert current position to radians
    lat1 = math.radians(current_lat)
    lon1 = math.radians(current_lon)
    
    # Calculate new position
    lat2 = math.asin(math.sin(lat1) * math.cos(distance/R) + 
                     math.cos(lat1) * math.sin(distance/R) * math.cos(heading_rad))
    
    lon2 = lon1 + math.atan2(math.sin(heading_rad) * math.sin(distance/R) * math.cos(lat1),
                              math.cos(distance/R) - math.sin(lat1) * math.sin(lat2))
    
    # Convert back to degrees
    return math.degrees(lat2), math.degrees(lon2)

def generate_flight_path(current_lat, current_lon, heading, points=20):
    """
    Generate a series of points for past and future flight path
    
    :param current_lat: Current latitude
    :param current_lon: Current longitude
    :param heading: Current flight heading
    :param points: Number of points to generate
    :return: Tuple of (past_lats, past_lons, future_lats, future_lons)
    """
    # Past points (simulating historical path)
    past_lats = [current_lat + np.random.normal(0, 0.1) for _ in range(points)]
    past_lons = [current_lon + np.random.normal(0, 0.1) for _ in range(points)]
    
    # Future points prediction
    future_lats = [current_lat]
    future_lons = [current_lon]
    
    for _ in range(points):
        # Predict next point based on current position and heading
        next_lat, next_lon = predict_future_path(future_lats[-1], future_lons[-1], heading)
        future_lats.append(next_lat)
        future_lons.append(next_lon)
    
    return past_lats, past_lons, future_lats[1:], future_lons[1:]

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
        
        # Limit number of flights
        max_flights = 50
        flights_data = []
        
        for flight in data['states'][:max_flights]:
            # Validate and clean data
            if (flight[5] is not None and flight[6] is not None and 
                US_BBOX['min_lon'] <= flight[5] <= US_BBOX['max_lon'] and 
                US_BBOX['min_lat'] <= flight[6] <= US_BBOX['max_lat']):
                
                # Flight details
                flight_id = flight[1].strip() or 'Unknown'
                current_lat = flight[6]
                current_lon = flight[5]
                heading = flight[10] or 0
                
                # Generate past and future paths
                past_lats, past_lons, future_lats, future_lons = generate_flight_path(
                    current_lat, current_lon, heading
                )
                
                flight_entry = {
                    'flight_id': flight_id,
                    'longitude': current_lon,
                    'latitude': current_lat,
                    'altitude': (flight[7] or 0) * 3.28084,  # Convert meters to feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # Convert m/s to knots
                    'heading': heading,
                    'past_path_lats': past_lats,
                    'past_path_lons': past_lons,
                    'future_path_lats': future_lats,
                    'future_path_lons': future_lons
                }
                
                flights_data.append(flight_entry)
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

# Main Streamlit App
def main():
    # Custom CSS for blinking effect and enhanced UI
    st.markdown("""
    <style>
    @keyframes blink {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    .blink-flight {
        animation: blink 1s infinite;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stTitle {
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 class='stTitle'>🛫 Advanced Flight Tracker</h1>", unsafe_allow_html=True)

    # Refresh button
    col_refresh, col_info = st.columns([3, 1])
    with col_refresh:
        refresh_button = st.button("🔄 Refresh Flight Data", use_container_width=True)
    with col_info:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

    # Data loading
    if 'flight_data' not in st.session_state or refresh_button:
        st.session_state['flight_data'] = fetch_real_flight_data()

    # Check if we have data
    if st.session_state['flight_data'].empty:
        st.warning("No flight data available. Check network connection.")
        return

    # Flight selection
    df = st.session_state['flight_data']
    available_flights = sorted(df['flight_id'].unique())
    selected_flights = st.multiselect(
        "Select Flights", 
        available_flights, 
        default=available_flights[:min(5, len(available_flights))],
        help="Choose flights to visualize on the map"
    )

    # Filter data
    flight_data = df[df['flight_id'].isin(selected_flights)]

    # Full-width map visualization
    st.markdown("### 🗺️ Real-Time Flight Paths")
    
    # Create map figure
    fig_map = go.Figure()

    # Add US map background with more detailed styling
    fig_map.add_trace(go.Scattergeo(
        lon=[-98.5795],
        lat=[39.8283],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        showlegend=False
    ))

    # Plot flights with past and future paths
    for _, flight in flight_data.iterrows():
        # Past path (lighter, dashed)
        fig_map.add_trace(go.Scattergeo(
            lon=flight['past_path_lons'],
            lat=flight['past_path_lats'],
            mode='lines',
            name=f"{flight['flight_id']} - Past Path",
            line=dict(width=2, color='gray', dash='dot')
        ))
        
        # Future path (lighter, dashed)
        fig_map.add_trace(go.Scattergeo(
            lon=flight['future_path_lons'],
            lat=flight['future_path_lats'],
            mode='lines',
            name=f"{flight['flight_id']} - Future Path",
            line=dict(width=2, color='lightblue', dash='dot')
        ))
        
        # Current position (blinking marker)
        fig_map.add_trace(go.Scattergeo(
            lon=[flight['longitude']],
            lat=[flight['latitude']],
            mode='markers',
            name=flight['flight_id'],
            marker=dict(
                size=15, 
                color='red',
                symbol='circle',
                opacity=0.7,
                line=dict(width=2, color='darkred')
            )
        ))

    # Update layout for a more comprehensive US map
    fig_map.update_geos(
        visible=True, 
        resolution=50, 
        scope='usa',
        showland=True, 
        landcolor="rgb(229, 229, 229)",
        countrycolor="white",
        coastlinecolor="white",
        showocean=True, 
        oceancolor="rgb(204, 229, 255)",
        showlakes=True, 
        lakecolor="rgb(204, 229, 255)",
        showrivers=True,
        rivercolor="rgb(128, 191, 255)"
    )

    fig_map.update_layout(
        title='Flight Paths Across the United States',
        height=800,  # Significantly larger map
        margin={"r":0,"t":30,"l":0,"b":0},
        geo=dict(
            center=dict(
                lon=-98.5795,
                lat=39.8283
            ),
            projection_scale=1.5  # Zoom level
        )
    )

    # Display the map
    st.plotly_chart(fig_map, use_container_width=True)

    # Additional information sections
    st.markdown("### 📊 Flight Details")
    
    # Create columns for flight details
    cols = st.columns(len(selected_flights))
    
    for i, (_, flight) in enumerate(flight_data.iterrows()):
        with cols[i]:
            st.markdown(f"#### {flight['flight_id']}")
            st.metric("Altitude", f"{flight['altitude']:.0f} ft")
            st.metric("Ground Speed", f"{flight['ground_speed']:.0f} knots")
            st.metric("Heading", f"{flight['heading']:.0f}°")

# Run the main application
main()
