import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Aircraft patterns
AIRCRAFT_PATTERNS = {
    'commercial_large': {
        'cruise_alt': (32000, 38000),
        'speed': (450, 550),
        'climb_rate': (1000, 2500),
        'rcs_range': (70, 100),
    },
    'commercial_medium': {
        'cruise_alt': (28000, 34000),
        'speed': (400, 480),
        'climb_rate': (800, 2000),
        'rcs_range': (50, 80),
    },
    'private_jet': {
        'cruise_alt': (15000, 25000),
        'speed': (300, 450),
        'climb_rate': (500, 1500),
        'rcs_range': (30, 50),
    }
}

def generate_weather_data(flight_path_df):
    """Generate weather data along flight path"""
    weather_data = pd.DataFrame()
    weather_data['timestamp'] = flight_path_df['timestamp']
    weather_data['wind_speed'] = 10 + 5 * np.sin(flight_path_df['time']/1000) + np.random.normal(0, 2, len(flight_path_df))
    weather_data['wind_direction'] = 180 + 45 * np.sin(flight_path_df['time']/2000) + np.random.normal(0, 5, len(flight_path_df))
    weather_data['temperature'] = 15 - (flight_path_df['altitude']/1000) * 2 + np.random.normal(0, 1, len(flight_path_df))
    weather_data['turbulence'] = np.random.choice(['None', 'Light', 'Moderate', 'Severe'], p=[0.5, 0.3, 0.15, 0.05], size=len(flight_path_df))
    return weather_data

def generate_flight_data(n_flights=10, points_per_flight=200):
    flights = []
    for i in range(n_flights):
        aircraft_type = np.random.choice(list(AIRCRAFT_PATTERNS.keys()), p=[0.4, 0.3, 0.3])
        pattern = AIRCRAFT_PATTERNS[aircraft_type]
        t = np.linspace(0, points_per_flight * 5, points_per_flight)
        cruise_alt = np.random.uniform(*pattern['cruise_alt'])
        z = np.zeros(points_per_flight)
        climb_idx = int(points_per_flight * 0.15)
        cruise_idx = int(points_per_flight * 0.8)
        z[:climb_idx] = np.cumsum(np.random.uniform(pattern['climb_rate'][0]/720, pattern['climb_rate'][1]/720, climb_idx))
        z[:climb_idx] = z[:climb_idx] * (cruise_alt / z[climb_idx-1])
        z[climb_idx:cruise_idx] = cruise_alt + np.random.normal(0, 200, cruise_idx-climb_idx)
        z[cruise_idx:] = np.linspace(cruise_alt, 0, len(z[cruise_idx:]))
        speed = np.random.uniform(*pattern['speed'])
        distance = speed * t/3600
        base_theta = np.linspace(0, np.pi/4, points_per_flight)
        course_corrections = np.sin(t/500) * 0.1
        holding_pattern = 0  # Added this missing variable
        aspect_effect = 0  # Added this missing variable
        theta = base_theta + course_corrections + holding_pattern
        x = distance * np.cos(theta)
        y = distance * np.sin(theta)
        x += np.random.normal(0, 0.01, points_per_flight)
        y += np.random.normal(0, 0.01, points_per_flight)
        ground_speed = np.sqrt(np.gradient(x)**2 + np.gradient(y)**2) * 3600
        vertical_speed = np.gradient(z) * 60
        heading = np.degrees(np.arctan2(np.gradient(y), np.gradient(x))) % 360
        base_rcs = np.random.uniform(*pattern['rcs_range'], points_per_flight)
        altitude_effect = -z/50000
        rcs = base_rcs + aspect_effect + altitude_effect
        start_time = datetime.now() - timedelta(hours=np.random.uniform(0, 24))
        timestamps = [start_time + timedelta(seconds=int(t_)) for t_ in t]
        flight_df = pd.DataFrame({
            'flight_id': f'FL{i:03d}',
            'aircraft_type': aircraft_type,
            'timestamp': timestamps,
            'time': t,
            'longitude': x,
            'latitude': y,
            'altitude': z,
            'ground_speed': ground_speed,
            'vertical_speed': vertical_speed,
            'heading': heading,
            'radar_cross_section': rcs
        })
        weather_df = generate_weather_data(flight_df)
        flight_df = pd.concat([flight_df, weather_df], axis=1)
        flights.append(flight_df)
    return pd.concat(flights, ignore_index=True)

# Page setup
st.set_page_config(layout="wide", page_title="Aircraft Tracking System")

# Title
st.title("Aircraft Tracking System")

# Sidebar controls
st.sidebar.header("Control Panel")

# Data loading
@st.cache_data
def load_data():
    return generate_flight_data()

df = load_data()

# Flight selection
available_flights = sorted(pd.unique(df['flight_id']))
selected_flights = st.sidebar.multiselect("Select Flights", available_flights, default=[available_flights[0]])

# Display options
show_weather = st.sidebar.checkbox("Show Weather", True)
show_trajectory = st.sidebar.checkbox("Show Trajectory", True)
rcs_threshold = st.sidebar.slider("RCS Threshold", 1.0, 5.0, 2.0)

# Filter data
flight_data = df[df['flight_id'].isin(selected_flights)]

# Main display
col1, col2 = st.columns(2)

with col1:
    # 3D Flight Path
    fig_3d = px.scatter_3d(flight_data, x='longitude', y='latitude', z='altitude', color='flight_id', title="3D Flight Paths")
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    if show_weather:
        fig_weather = go.Figure()
        for flight_id in selected_flights:
            flight_subset = flight_data[flight_data['flight_id'] == flight_id]
            fig_weather.add_trace(go.Scatter(x=flight_subset['time'], y=flight_subset['wind_speed'], name=f"{flight_id} - Wind Speed", mode='lines'))
        fig_weather.update_layout(title="Weather Conditions")
        st.plotly_chart(fig_weather, use_container_width=True)

# Flight Details
st.header("Flight Details")
for flight_id in selected_flights:
    flight_subset = flight_data[flight_data['flight_id'] == flight_id]
    st.subheader(f"Flight {flight_id}")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Aircraft Type", flight_subset['aircraft_type'].iloc[0].replace('_', ' ').title())
    with cols[1]:
        st.metric("Max Ground Speed", f"{flight_subset['ground_speed'].max():.0f} knots")
    with cols[2]:
        st.metric("Max Altitude", f"{flight_subset['altitude'].max():.0f} feet")
    with cols[3]:
        st.metric("Avg Vertical Speed", f"{flight_subset['vertical_speed'].mean():.0f} ft/min")
