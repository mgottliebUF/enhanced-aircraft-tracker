import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import math

class KalmanFilter:
    """
    Advanced Kalman Filter for Aerospace Tracking
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
        self.F[0, 1] = 1  # Latitude depends on longitude
        self.F[1, 2] = 1  # Longitude depends on altitude
        
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
        """
        Prediction step of Kalman Filter
        
        :return: Predicted state
        """
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
        
        # Calculate innovation (measurement residual)
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
    st.set_page_config(layout="wide", page_title="Advanced Kalman Filter Flight Tracker", page_icon="🛩️")
    
    # Title and introduction
    st.title("🛩️ Advanced Kalman Filter Flight Tracking")
    st.markdown("""
    Comprehensive flight tracking with advanced Kalman Filter state estimation.
    """)
    
    # Sidebar for navigation
    page = st.sidebar.radio("Select Analysis View", [
        "Flight Map",
        "Kalman Filter Analysis",
        "Flight Details",
        "Performance Metrics",
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
                    symbol='circle'
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
    
    elif page == "Kalman Filter Analysis":
        # Kalman Filter Visualization
        st.header("🔍 Kalman Filter State Estimation")
        
        # Run Kalman Filter simulation for selected flights
        for flight in selected_flights:
            st.subheader(f"Kalman Filter Analysis for {flight}")
            
            # Get flight data
            flight_data = filtered_df[filtered_df['flight_id'] == flight]
            
            # Run Kalman Filter simulation
            kf = run_kalman_filter_simulation(flight_data)
            
            # Visualize state history
            state_labels = ['Latitude', 'Longitude', 'Altitude', 'Velocity', 'Acceleration']
            fig_state = go.Figure()
            
            # State history plot
            state_history = np.array(kf.state_history)
            for i, label in enumerate(state_labels):
                fig_state.add_trace(go.Scatter(
                    y=state_history[:, i],
                    mode='lines+markers',
                    name=label
                ))
            
            fig_state.update_layout(
                title=f'Kalman Filter State Estimation for {flight}',
                xaxis_title='Time Steps',
                yaxis_title='State Value'
            )
            
            st.plotly_chart(fig_state, use_container_width=True)
            
            # Innovation history plot
            if kf.innovation_history:
                fig_innovation = go.Figure()
                innovation_history = np.array(kf.innovation_history)
                
                for i, label in enumerate(state_labels):
                    fig_innovation.add_trace(go.Scatter(
                        y=innovation_history[:, i],
                        mode='lines+markers',
                        name=label
                    ))
                
                fig_innovation.update_layout(
                    title=f'Kalman Filter Innovation for {flight}',
                    xaxis_title='Measurement Steps',
                    yaxis_title='Innovation'
                )
                
                st.plotly_chart(fig_innovation, use_container_width=True)
    
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
if __name__ == "__main__":
    main()
