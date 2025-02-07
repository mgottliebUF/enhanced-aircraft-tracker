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
    """
    def __init__(self, initial_state, process_noise_cov, measurement_noise_cov):
        # State vector: [latitude, longitude, altitude, velocity, heading]
        self.state = np.array(initial_state, dtype=float)
        
        # State transition and measurement matrices
        self.F = np.eye(5)
        self.H = np.eye(5)
        
        # Noise covariance matrices
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        
        # Error covariance
        self.P = np.eye(5)
        
        # Tracking histories
        self.state_history = [self.state]
        self.innovation_history = []
        self.kalman_gain_history = []
    
    def predict(self, dt=1.0):
        """Prediction step of Kalman Filter"""
        self.F = np.array([
            [1, 0, 0, dt, 0],    # Latitude update
            [0, 1, 0, 0, dt],    # Longitude update
            [0, 0, 1, 0, 0],     # Altitude update
            [0, 0, 0, 1, 0],     # Velocity update
            [0, 0, 0, 0, 1]      # Heading update
        ])
        
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement):
        """Update step of Kalman Filter"""
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Store Kalman Gain
        self.kalman_gain_history.append(K)
        
        # Calculate innovation
        innovation = measurement - self.H @ self.state
        self.innovation_history.append(innovation)
        
        # Update state estimation
        self.state = self.state + K @ innovation
        self.state_history.append(self.state)
        
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
        for flight in data['states'][:30]:  # Limit to 30 flights
            if flight[5] is not None and flight[6] is not None:
                flights_data.append({
                    'flight_id': flight[1].strip() or 'Unknown',
                    'latitude': flight[6],
                    'longitude': flight[5],
                    'altitude': (flight[7] or 0) * 3.28084,  # feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # knots
                    'heading': flight[10] or 0
                })
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="Advanced Flight Tracking", page_icon="✈️")
    
    # Title and introduction
    st.title("🛩️ Advanced Flight Tracking & Kalman Filter Analysis")
    st.markdown("""
    This application provides comprehensive flight tracking and analysis 
    using advanced Kalman Filter algorithms.
    """)
    
    # Sidebar for navigation
    page = st.sidebar.radio("Select Analysis View", [
        "Flight Map",
        "Flight Details",
        "Performance Metrics",
        "Kalman Filter Visualization",
        "Statistical Analysis"
    ])
    
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
    
    # Select flights
    selected_flights = st.multiselect(
        "Select Flights", 
        df['flight_id'].unique(), 
        default=df['flight_id'].unique()[:5]
    )
    filtered_df = df[df['flight_id'].isin(selected_flights)]
    
    # Render selected page
    if page == "Flight Map":
        # Trajectory Map Visualization
        st.header("🗺️ Flight Trajectories")
        
        # Create map figure
        fig_map = go.Figure()
        
        for flight in selected_flights:
            flight_data = filtered_df[filtered_df['flight_id'] == flight]
            
            # Current position
            fig_map.add_trace(go.Scattergeo(
                lon=[flight_data['longitude'].iloc[0]],
                lat=[flight_data['latitude'].iloc[0]],
                mode='markers',
                name=flight,
                marker=dict(
                    size=10, 
                    color='red',
                    symbol='aircraft'
                )
            ))
        
        # Update map layout
        fig_map.update_geos(
            visible=True, 
            resolution=50, 
            scope='usa',
            showland=True, 
            landcolor="rgb(229, 229, 229)",
            oceancolor="rgb(204, 229, 255)"
        )
        
        fig_map.update_layout(
            title='Current Flight Positions',
            height=600
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    
    elif page == "Flight Details":
        # Detailed Flight Information
        st.header("📋 Detailed Flight Information")
        
        # Create detailed dataframe view
        details_df = filtered_df.copy()
        details_df['altitude'] = details_df['altitude'].round(0)
        details_df['ground_speed'] = details_df['ground_speed'].round(1)
        details_df['heading'] = details_df['heading'].round(1)
        
        st.dataframe(
            details_df, 
            use_container_width=True,
            column_config={
                "altitude": st.column_config.NumberColumn("Altitude (ft)"),
                "ground_speed": st.column_config.NumberColumn("Ground Speed (knots)"),
                "heading": st.column_config.NumberColumn("Heading (degrees)")
            }
        )
    
    elif page == "Performance Metrics":
        # Performance Metrics Visualization
        st.header("📊 Flight Performance Metrics")
        
        # Altitude distribution
        fig_altitude = px.box(
            filtered_df, 
            x='flight_id', 
            y='altitude', 
            title='Altitude Distribution by Flight',
            labels={'altitude': 'Altitude (feet)'}
        )
        st.plotly_chart(fig_altitude, use_container_width=True)
        
        # Ground speed distribution
        fig_speed = px.box(
            filtered_df, 
            x='flight_id', 
            y='ground_speed', 
            title='Ground Speed Distribution by Flight',
            labels={'ground_speed': 'Ground Speed (knots)'}
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    elif page == "Kalman Filter Visualization":
        # Kalman Filter Simulation
        st.header("🔍 Kalman Filter State Estimation")
        
        # Simulate Kalman Filter for selected flights
        for flight in selected_flights:
            flight_data = filtered_df[filtered_df['flight_id'] == flight]
            
            st.subheader(f"Kalman Filter Analysis for {flight}")
            
            # Create sample Kalman Filter
            initial_state = [
                flight_data['latitude'].iloc[0],
                flight_data['longitude'].iloc[0],
                flight_data['altitude'].iloc[0],
                flight_data['ground_speed'].iloc[0],
                flight_data['heading'].iloc[0]
            ]
            
            # Noise covariance matrices
            Q = np.diag([0.0001, 0.0001, 0.01, 0.1, 0.01])
            R = np.diag([0.001, 0.001, 0.1, 0.5, 0.1])
            
            kf = KalmanFilter(initial_state, Q, R)
            
            # Simulate some measurements
            measurements = np.array([initial_state])
            for _ in range(10):
                # Simulate noisy measurement
                noisy_measurement = initial_state + np.random.normal(0, 0.1, 5)
                measurements = np.vstack([measurements, noisy_measurement])
                kf.update(noisy_measurement)
            
            # Visualization of state estimation
            fig = go.Figure()
            state_labels = ['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Heading']
            
            for i, label in enumerate(state_labels):
                fig.add_trace(go.Scatter(
                    y=[state[i] for state in kf.state_history],
                    mode='lines+markers',
                    name=label
                ))
            
            fig.update_layout(
                title=f'Kalman Filter State Estimation for {flight}',
                xaxis_title='Measurement Steps',
                yaxis_title='State Value'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Statistical Analysis":
        # Statistical Analysis
        st.header("📈 Statistical Flight Analysis")
        
        # Compute statistical summary
        stats_summary = filtered_df[['altitude', 'ground_speed', 'heading']].describe()
        st.dataframe(stats_summary, use_container_width=True)
        
        # Correlation heatmap
        fig_corr = px.imshow(
            filtered_df[['altitude', 'ground_speed', 'heading']].corr(), 
            title='Correlation Heatmap of Flight Parameters',
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# Run the main application
main()
