import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import math
import time

class KalmanFilter:
    """
    Advanced Kalman Filter for Aerospace Tracking
    Implements a multi-dimensional state estimation model
    """
    def __init__(self, initial_state, process_noise_cov, measurement_noise_cov):
        """
        Initialize Kalman Filter
        
        :param initial_state: Initial state vector [lat, lon, alt, velocity, heading]
        :param process_noise_cov: Process noise covariance matrix
        :param measurement_noise_cov: Measurement noise covariance matrix
        """
        # State vector: [latitude, longitude, altitude, velocity, heading]
        self.state = np.array(initial_state, dtype=float)
        
        # State transition matrix (how state evolves)
        self.F = np.eye(5)  # Identity matrix initially
        
        # Measurement matrix (how measurements relate to state)
        self.H = np.eye(5)
        
        # Process noise covariance (uncertainty in system dynamics)
        self.Q = process_noise_cov
        
        # Measurement noise covariance (uncertainty in measurements)
        self.R = measurement_noise_cov
        
        # Estimation error covariance
        self.P = np.eye(5)
    
    def predict(self, dt=1.0):
        """
        Prediction step of Kalman Filter
        
        :param dt: Time step
        :return: Predicted state
        """
        # Update state transition matrix with time step
        self.F = np.array([
            [1, 0, 0, dt, 0],    # Latitude update
            [0, 1, 0, 0, dt],    # Longitude update
            [0, 0, 1, 0, 0],     # Altitude update
            [0, 0, 0, 1, 0],     # Velocity update
            [0, 0, 0, 0, 1]      # Heading update
        ])
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement):
        """
        Update step of Kalman Filter
        
        :param measurement: New measurement vector
        :return: Updated state
        """
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimation
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        
        # Update error covariance
        self.P = (np.eye(5) - K @ self.H) @ self.P
        
        return self.state
    
    def predict_trajectory(self, steps=20):
        """
        Predict future trajectory using Kalman Filter
        
        :param steps: Number of prediction steps
        :return: Predicted trajectory
        """
        predicted_states = [self.state]
        
        for _ in range(steps):
            # Predict next state
            predicted_state = self.predict()
            predicted_states.append(predicted_state)
        
        return np.array(predicted_states)

def create_process_noise_covariance():
    """
    Create a realistic process noise covariance matrix
    
    :return: Process noise covariance matrix
    """
    # Different noise levels for different state variables
    return np.diag([
        0.0001,   # Latitude noise
        0.0001,   # Longitude noise
        0.01,     # Altitude noise
        0.1,      # Velocity noise
        0.01      # Heading noise
    ])

def create_measurement_noise_covariance():
    """
    Create a realistic measurement noise covariance matrix
    
    :return: Measurement noise covariance matrix
    """
    # Diagonal matrix representing measurement uncertainties
    return np.diag([
        0.001,    # Latitude measurement noise
        0.001,    # Longitude measurement noise
        0.1,      # Altitude measurement noise
        0.5,      # Velocity measurement noise
        0.1       # Heading measurement noise
    ])

def fetch_real_flight_data():
    """
    Fetch real-time flight data from OpenSky Network API
    Filters flights within the continental US
    """
    try:
        # Contiguous US Bounding Box Coordinates
        US_BBOX = {
            'min_lon': -125.0,
            'max_lon': -66.0,
            'min_lat': 24.0,
            'max_lat': 49.0,
        }
        
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
            if (flight[5] is not None and flight[6] is not None):
                # Flight details
                flight_id = flight[1].strip() or 'Unknown'
                current_lat = flight[6]
                current_lon = flight[5]
                
                # Initial state for Kalman Filter
                initial_state = [
                    current_lat,      # Latitude
                    current_lon,      # Longitude
                    (flight[7] or 0) * 3.28084,  # Altitude (feet)
                    (flight[9] or 0) * 1.94384,  # Ground speed (knots)
                    flight[10] or 0   # Heading
                ]
                
                # Create Kalman Filter
                kf = KalmanFilter(
                    initial_state,
                    create_process_noise_covariance(),
                    create_measurement_noise_covariance()
                )
                
                # Predict trajectory
                predicted_trajectory = kf.predict_trajectory()
                
                flight_entry = {
                    'flight_id': flight_id,
                    'longitude': current_lon,
                    'latitude': current_lat,
                    'altitude': initial_state[2],
                    'ground_speed': initial_state[3],
                    'heading': initial_state[4],
                    'predicted_lats': predicted_trajectory[:, 0],
                    'predicted_lons': predicted_trajectory[:, 1],
                    'predicted_alts': predicted_trajectory[:, 2]
                }
                
                flights_data.append(flight_entry)
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="Kalman Filter Flight Tracker", page_icon="🛩️")
    
    st.title("🛩️ Kalman Filter Enhanced Flight Tracking")
    
    # Sidebar for advanced controls
    st.sidebar.header("Kalman Filter Tracking Parameters")
    noise_level = st.sidebar.slider(
        "Measurement Noise Level", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        help="Adjust the uncertainty in flight measurements"
    )
    
    # Refresh mechanism
    if 'flight_data' not in st.session_state:
        st.session_state['flight_data'] = fetch_real_flight_data()
    
    # Refresh button
    if st.button("🔄 Refresh Flight Data"):
        st.session_state['flight_data'] = fetch_real_flight_data()
    
    # Data processing
    df = st.session_state['flight_data']
    
    if df.empty:
        st.warning("No flight data available.")
        return
    
    # Flight selection
    available_flights = sorted(df['flight_id'].unique())
    selected_flights = st.multiselect(
        "Select Flights", 
        available_flights, 
        default=available_flights[:min(5, len(available_flights))],
        help="Advanced Kalman Filter trajectory analysis"
    )
    
    # Filtered flight data
    flight_data = df[df['flight_id'].isin(selected_flights)]
    
    # Visualization
    st.markdown("### 🗺️ Kalman Filter Trajectory Prediction")
    
    # Create map figure
    fig_map = go.Figure()
    
    # Plot flights with predicted trajectories
    for _, flight in flight_data.iterrows():
        # Predicted trajectory
        fig_map.add_trace(go.Scattergeo(
            lon=flight['predicted_lons'],
            lat=flight['predicted_lats'],
            mode='lines+markers',
            name=f"{flight['flight_id']} Predicted Path",
            line=dict(width=2, color='blue', dash='dot'),
            marker=dict(size=5, color='lightblue')
        ))
        
        # Current position marker
        fig_map.add_trace(go.Scattergeo(
            lon=[flight['longitude']],
            lat=[flight['latitude']],
            mode='markers',
            name=flight['flight_id'],
            marker=dict(
                size=15, 
                color='red',
                symbol='circle',
                line=dict(width=2, color='darkred')
            )
        ))
    
    # Enhanced map styling
    fig_map.update_geos(
        visible=True, 
        resolution=50, 
        scope='usa',
        showland=True, 
        landcolor="rgb(229, 229, 229)",
        oceancolor="rgb(204, 229, 255)"
    )
    
    fig_map.update_layout(
        title='Kalman Filter Enhanced Flight Trajectories',
        height=800,
        geo=dict(
            center=dict(lon=-98.5795, lat=39.8283),
            projection_scale=1.5
        )
    )
    
    # Display map
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Detailed Flight Analytics
    st.markdown("### 📊 Kalman Filter Performance Metrics")
    
    # Create columns for detailed analytics
    cols = st.columns(len(selected_flights))
    
    for i, (_, flight) in enumerate(flight_data.iterrows()):
        with cols[i]:
            st.markdown(f"#### {flight['flight_id']} Analysis")
            
            # Performance metrics
            st.metric("Current Altitude", f"{flight['altitude']:.0f} ft")
            st.metric("Ground Speed", f"{flight['ground_speed']:.0f} knots")
            st.metric("Heading", f"{flight['heading']:.0f}°")
            
            # Trajectory prediction details
            st.markdown("#### Trajectory Prediction")
            st.metric("Predicted Latitude Range", 
                      f"{flight['predicted_lats'].min():.4f} to {flight['predicted_lats'].max():.4f}")
            st.metric("Predicted Longitude Range", 
                      f"{flight['predicted_lons'].min():.4f} to {flight['predicted_lons'].max():.4f}")
            st.metric("Predicted Altitude Range", 
                      f"{flight['predicted_alts'].min():.0f} to {flight['predicted_alts'].max():.0f} ft")

# Run the main application
main()
