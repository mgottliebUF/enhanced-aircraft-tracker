import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import math

class KalmanFilter:
    """
    Advanced Kalman Filter for Flight State Estimation
    """
    def __init__(self, initial_state):
        """
        Initialize Kalman Filter
        
        :param initial_state: Initial state vector [latitude, longitude, altitude, velocity, acceleration]
        """
        # State vector
        self.state = np.array(initial_state, dtype=float)
        
        # State transition matrix
        self.F = np.eye(5)
        self.F[0, 1] = 0.1  # Slight coupling between state variables
        self.F[1, 2] = 0.1
        
        # Measurement matrix
        self.H = np.eye(5)
        
        # Process noise covariance
        self.Q = np.diag([0.001, 0.001, 0.01, 0.1, 0.01])
        
        # Measurement noise covariance
        self.R = np.diag([0.01, 0.01, 0.1, 0.5, 0.1])
        
        # Estimation error covariance
        self.P = np.eye(5)
        
        # Tracking histories
        self.state_history = [self.state.copy()]
        self.innovation_history = []
    
    def predict(self):
        """Prediction step of Kalman Filter"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state
    
    def update(self, measurement):
        """Update step of Kalman Filter"""
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Calculate innovation
        innovation = measurement - self.H @ self.state
        self.innovation_history.append(innovation)
        
        # Update state estimation
        self.state = self.state + K @ innovation
        self.state_history.append(self.state.copy())
        
        # Update error covariance
        self.P = (np.eye(5) - K @ self.H) @ self.P
        
        return self.state

def fetch_real_flight_data():
    """
    Fetch real-time flight data from OpenSky Network API
    """
    try:
        # Contiguous US Bounding Box Coordinates
        US_BBOX = {
            'min_lon': -125.0, 'max_lon': -66.0,
            'min_lat': 24.0, 'max_lat': 49.0,
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
        
        # Process flights
        flights_data = []
        for flight in data['states'][:50]:  # Limit to 50 flights
            if flight[5] is not None and flight[6] is not None:
                flights_data.append({
                    'flight_id': flight[1].strip() or 'Unknown',
                    'latitude': flight[6],
                    'longitude': flight[5],
                    'altitude': (flight[7] or 0) * 3.28084,  # feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # knots
                    'heading': flight[10] or 0,
                    'vertical_rate': (flight[11] or 0) * 196.85  # ft/min
                })
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

def run_kalman_filter_simulation(flight_data):
    """
    Run Kalman Filter simulation for a flight
    
    :param flight_data: DataFrame with flight data
    :return: Kalman Filter object with simulation results
    """
    # Initial state
    initial_state = [
        flight_data['latitude'].iloc[0],
        flight_data['longitude'].iloc[0],
        flight_data['altitude'].iloc[0],
        flight_data['ground_speed'].iloc[0],
        0  # Initial acceleration
    ]
    
    # Create Kalman Filter
    kf = KalmanFilter(initial_state)
    
    # Simulate measurements with some noise
    for i in range(1, len(flight_data)):
        # Add some measurement noise
        noisy_measurement = [
            flight_data['latitude'].iloc[i] + np.random.normal(0, 0.01),
            flight_data['longitude'].iloc[i] + np.random.normal(0, 0.01),
            flight_data['altitude'].iloc[i] + np.random.normal(0, 1),
            flight_data['ground_speed'].iloc[i] + np.random.normal(0, 0.1),
            0  # Acceleration measurement
        ]
        
        # Predict and update
        kf.predict()
        kf.update(noisy_measurement)
    
    return kf

def main():
    st.set_page_config(layout="wide", page_title="Real-Time Flight Tracker", page_icon="🛩️")
    
    # Title and introduction
    st.title("🛩️ Real-Time Flight Tracking & Analysis")
    
    # Fetch flight data
    if 'flight_data' not in st.session_state:
        st.session_state['flight_data'] = fetch_real_flight_data()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Flight Data"):
        st.session_state['flight_data'] = fetch_real_flight_data()
    
    # Get flight data
    df = st.session_state['flight_data']
    
    if df.empty:
        st.warning("No flight data available.")
        return
    
    # Main visualization section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Flight Trajectories Map
        st.subheader("🗺️ Flight Trajectories")
        fig_map = go.Figure(go.Scattergeo(
            lon=df['longitude'],
            lat=df['latitude'],
            text=df['flight_id'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['altitude'],
                colorscale='Viridis',
                colorbar=dict(title='Altitude (ft)'),
                showscale=True
            )
        ))
        fig_map.update_geos(
            visible=True, 
            resolution=50, 
            scope='usa',
            showland=True, 
            landcolor="rgb(229, 229, 229)",
            oceancolor="rgb(204, 229, 255)"
        )
        fig_map.update_layout(height=400)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        # Altitude Distribution
        st.subheader("📊 Altitude Distribution")
        fig_altitude = px.box(
            df, 
            y='altitude', 
            title='Flight Altitude Distribution',
            labels={'altitude': 'Altitude (feet)'}
        )
        fig_altitude.update_layout(height=400)
        st.plotly_chart(fig_altitude, use_container_width=True)
    
    with col3:
        # Ground Speed Distribution
        st.subheader("🚀 Ground Speed Distribution")
        fig_speed = px.box(
            df, 
            y='ground_speed', 
            title='Flight Ground Speed Distribution',
            labels={'ground_speed': 'Ground Speed (knots)'}
        )
        fig_speed.update_layout(height=400)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    # Kalman Filter Analysis Section
    st.header("🔍 Kalman Filter Flight State Estimation")
    
    # Select flights for Kalman Filter analysis
    selected_flights = st.multiselect(
        "Select Flights for Kalman Filter Analysis", 
        df['flight_id'].unique(), 
        default=df['flight_id'].unique()[:3]
    )
    
    # Kalman Filter Visualization for selected flights
    for flight in selected_flights:
        st.subheader(f"Kalman Filter Analysis for {flight}")
        
        # Get flight data for specific flight
        flight_data = df[df['flight_id'] == flight]
        
        # Run Kalman Filter simulation
        kf = run_kalman_filter_simulation(flight_data)
        
        # Visualize Kalman Filter results
        col_state, col_innovation = st.columns(2)
        
        with col_state:
            # State Estimation Visualization
            st.markdown("#### State Estimation")
            fig_state = go.Figure()
            
            # Convert state history to DataFrame for easier plotting
            state_history = pd.DataFrame(
                kf.state_history, 
                columns=['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Acceleration']
            )
            
            for column in state_history.columns:
                fig_state.add_trace(go.Scatter(
                    y=state_history[column],
                    mode='lines',
                    name=column
                ))
            
            fig_state.update_layout(
                title='Kalman Filter State Estimation',
                xaxis_title='Time Steps',
                yaxis_title='State Value'
            )
            st.plotly_chart(fig_state, use_container_width=True)
        
        with col_innovation:
            # Innovation Analysis
            st.markdown("#### Innovation Analysis")
            if kf.innovation_history:
                fig_innovation = go.Figure()
                
                # Convert innovation history to DataFrame
                innovation_history = pd.DataFrame(
                    kf.innovation_history, 
                    columns=['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Acceleration']
                )
                
                for column in innovation_history.columns:
                    fig_innovation.add_trace(go.Scatter(
                        y=innovation_history[column],
                        mode='lines',
                        name=column
                    ))
                
                fig_innovation.update_layout(
                    title='Kalman Filter Innovations',
                    xaxis_title='Measurement Steps',
                    yaxis_title='Innovation'
                )
                st.plotly_chart(fig_innovation, use_container_width=True)
            else:
                st.write("No innovation data available")

# Run the main application
if __name__ == "__main__":
    main()
