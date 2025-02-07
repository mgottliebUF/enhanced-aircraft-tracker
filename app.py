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
        self.initial_state = np.array(initial_state, dtype=float)
        
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
        
        # Store historical states for visualization
        self.state_history = [self.state]
        self.innovation_history = []
    
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
        
        # Calculate innovation (difference between measurement and prediction)
        innovation = measurement - self.H @ self.state
        self.innovation_history.append(innovation)
        
        # Update state estimation
        self.state = self.state + K @ innovation
        
        # Store state for visualization
        self.state_history.append(self.state)
        
        # Update error covariance
        self.P = (np.eye(5) - K @ self.H) @ self.P
        
        return self.state
    
    def get_state_history(self):
        """
        Get historical states for visualization
        
        :return: Array of historical states
        """
        return np.array(self.state_history)
    
    def get_innovation_history(self):
        """
        Get innovation history for analysis
        
        :return: Array of innovations
        """
        return np.array(self.innovation_history)
    
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
                
                # Get state history for additional analysis
                state_history = kf.get_state_history()
                innovation_history = kf.get_innovation_history()
                
                flight_entry = {
                    'flight_id': flight_id,
                    'longitude': current_lon,
                    'latitude': current_lat,
                    'altitude': initial_state[2],
                    'ground_speed': initial_state[3],
                    'heading': initial_state[4],
                    'predicted_lats': predicted_trajectory[:, 0],
                    'predicted_lons': predicted_trajectory[:, 1],
                    'predicted_alts': predicted_trajectory[:, 2],
                    'state_history': state_history,
                    'innovation_history': innovation_history
                }
                
                flights_data.append(flight_entry)
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="Advanced Kalman Filter Flight Tracker", page_icon="🛩️")
    
    st.title("🛩️ Advanced Kalman Filter Flight Tracking System")
    
    # Sidebar for advanced controls
    st.sidebar.header("Kalman Filter Analysis")
    analysis_type = st.sidebar.selectbox(
        "Select Visualization Type",
        [
            "Trajectory Prediction",
            "State Estimation Detailed",
            "Innovation Analysis",
            "Error Covariance Visualization"
        ]
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
        "Select Flights for Analysis", 
        available_flights, 
        default=available_flights[:min(3, len(available_flights))],
        help="Choose flights for detailed Kalman Filter analysis"
    )
    
    # Filtered flight data
    flight_data = df[df['flight_id'].isin(selected_flights)]
    
    # Visualization based on selected analysis type
    if analysis_type == "Trajectory Prediction":
        # Visualization of predicted trajectories
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
    
    elif analysis_type == "State Estimation Detailed":
        # Detailed state estimation visualization
        st.markdown("### 📊 Kalman Filter State Estimation Analysis")
        
        # Create multiple plots for state variables
        state_vars = ['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Heading']
        
        for var_idx, var_name in enumerate(state_vars):
            fig = go.Figure()
            
            for _, flight in flight_data.iterrows():
                # Get state history for the specific flight
                state_history = flight['state_history']
                
                # Plot the state variable over time
                fig.add_trace(go.Scatter(
                    y=state_history[:, var_idx],
                    mode='lines+markers',
                    name=f"{flight['flight_id']} - {var_name}",
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f'Kalman Filter {var_name} State Estimation',
                xaxis_title='Time Steps',
                yaxis_title=var_name
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Innovation Analysis":
        # Innovation analysis visualization
        st.markdown("### 🔍 Kalman Filter Innovation Analysis")
        
        # Create multiple plots for innovation variables
        state_vars = ['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Heading']
        
        for var_idx, var_name in enumerate(state_vars):
            fig = go.Figure()
            
            for _, flight in flight_data.iterrows():
                # Get innovation history for the specific flight
                innovation_history = flight['innovation_history']
                
                # Plot the innovation for the specific state variable
                fig.add_trace(go.Scatter(
                    y=innovation_history[:, var_idx],
                    mode='lines+markers',
                    name=f"{flight['flight_id']} - {var_name}",
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f'Kalman Filter {var_name} Innovation',
                xaxis_title='Measurement Updates',
                yaxis_title=f'{var_name} Innovation'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Error Covariance Visualization
        # Error covariance analysis
        st.markdown("### 📈 Kalman Filter Error Covariance Analysis")
        
        # Display detailed flight information
        st.dataframe(
            flight_data[['flight_id', 'latitude', 'longitude', 'altitude', 'ground_speed', 'heading']],
            use_container_width=True
        )
        
        # Additional statistical summary
        st.markdown("#### Statistical Summary")
        stats_summary = flight_data[['altitude', 'ground_speed', 'heading']].describe()
        st.dataframe(stats_summary, use_container_width=True)

# Run the main application
main()
