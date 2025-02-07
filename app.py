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
        
        :param initial_state: Initial state vector [position, velocity, acceleration]
        """
        # State vector (3D: position, velocity, acceleration)
        self.state = np.array(initial_state, dtype=float)
        
        # State transition matrix
        self.F = np.array([
            [1, 1, 0.5],  # Position depends on velocity and partial acceleration
            [0, 1, 1],    # Velocity depends on acceleration
            [0, 0, 1]     # Acceleration is relatively constant
        ])
        
        # Measurement matrix (we can measure position directly)
        self.H = np.array([
            [1, 0, 0],    # Position measurement
            [0, 1, 0],    # Velocity measurement
            [0, 0, 1]     # Acceleration measurement
        ])
        
        # Process noise covariance
        self.Q = np.diag([0.1, 0.1, 0.01])  # Noise in position, velocity, acceleration
        
        # Measurement noise covariance
        self.R = np.diag([1.0, 0.1, 0.01])  # Measurement uncertainties
        
        # Estimation error covariance
        self.P = np.eye(3)
        
        # Tracking histories
        self.state_history = [self.state.copy()]
        self.innovation_history = []
        self.kalman_gain_history = []
    
    def predict(self):
        """Prediction step of Kalman Filter"""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement):
        """Update step of Kalman Filter"""
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Store Kalman Gain
        self.kalman_gain_history.append(K.flatten())
        
        # Calculate innovation (measurement residual)
        innovation = measurement - self.H @ self.state
        self.innovation_history.append(innovation)
        
        # Update state estimation
        self.state = self.state + K @ innovation
        self.state_history.append(self.state.copy())
        
        # Update error covariance
        self.P = (np.eye(3) - K @ self.H) @ self.P
        
        return self.state

def generate_flight_trajectory(flight_id):
    """
    Generate a synthetic flight trajectory for Kalman Filter analysis
    
    :param flight_id: Flight identifier
    :return: DataFrame with flight trajectory data
    """
    # Seed for reproducibility
    np.random.seed(hash(flight_id) % 2**32)
    
    # Generate trajectory parameters
    num_steps = 50
    
    # Initial conditions with some randomness
    initial_lat = np.random.uniform(30, 45)
    initial_lon = np.random.uniform(-120, -70)
    initial_altitude = np.random.uniform(20000, 40000)
    
    # Generate trajectory with smooth variations
    lats = initial_lat + np.cumsum(np.random.normal(0, 0.01, num_steps))
    lons = initial_lon + np.cumsum(np.random.normal(0, 0.01, num_steps))
    altitudes = initial_altitude + np.cumsum(np.random.normal(0, 10, num_steps))
    ground_speeds = 400 + np.cumsum(np.random.normal(0, 5, num_steps))
    
    # Create DataFrame
    trajectory_df = pd.DataFrame({
        'flight_id': flight_id,
        'latitude': lats,
        'longitude': lons,
        'altitude': altitudes,
        'ground_speed': ground_speeds,
        'time_step': range(num_steps)
    })
    
    return trajectory_df

def run_kalman_filter_simulation(flight_data):
    """
    Run Kalman Filter simulation for a flight
    
    :param flight_data: DataFrame with flight data
    :return: Kalman Filter object with simulation results
    """
    # Initial state for Kalman Filter
    # Use first row of data for initial conditions
    initial_state = [
        flight_data['latitude'].iloc[0],
        flight_data['ground_speed'].iloc[0],
        0  # Initial acceleration
    ]
    
    # Create Kalman Filter
    kf = KalmanFilter(initial_state)
    
    # Simulate measurements with noise
    for i in range(1, len(flight_data)):
        # Prepare noisy measurement
        measurement = [
            flight_data['latitude'].iloc[i] + np.random.normal(0, 0.01),
            flight_data['ground_speed'].iloc[i] + np.random.normal(0, 0.1),
            0  # Assume zero acceleration initially
        ]
        
        # Predict and update
        kf.predict()
        kf.update(measurement)
    
    return kf

def main():
    st.set_page_config(layout="wide", page_title="Flight Kalman Filter Analysis", page_icon="✈️")
    
    # Title
    st.title("🛩️ Advanced Flight Tracking with Kalman Filter")
    
    # Generate sample flight data if not existing
    if 'flight_data' not in st.session_state:
        # Generate multiple flight trajectories
        flight_ids = ['UAL263', 'DAL1456', 'AAL789']
        st.session_state['flight_data'] = pd.concat([
            generate_flight_trajectory(flight_id) for flight_id in flight_ids
        ])
    
    # Refresh button
    if st.sidebar.button("🔄 Generate New Flight Data"):
        flight_ids = ['UAL263', 'DAL1456', 'AAL789']
        st.session_state['flight_data'] = pd.concat([
            generate_flight_trajectory(flight_id) for flight_id in flight_ids
        ])
    
    # Get flight data
    df = st.session_state['flight_data']
    
    # Select flights for Kalman Filter analysis
    selected_flights = st.multiselect(
        "Select Flights for Kalman Filter Analysis", 
        df['flight_id'].unique(), 
        default=df['flight_id'].unique()
    )
    
    # Kalman Filter Analysis for selected flights
    for flight in selected_flights:
        st.header(f"Kalman Filter Analysis for {flight}")
        
        # Get flight data for specific flight
        flight_data = df[df['flight_id'] == flight]
        
        # Run Kalman Filter simulation
        kf = run_kalman_filter_simulation(flight_data)
        
        # State History Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("State History")
            state_history_df = pd.DataFrame(
                kf.state_history, 
                columns=['Position', 'Velocity', 'Acceleration']
            )
            st.dataframe(state_history_df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Innovation Analysis")
            if kf.innovation_history:
                innovation_df = pd.DataFrame(
                    kf.innovation_history, 
                    columns=['Position Innovation', 'Velocity Innovation', 'Acceleration Innovation']
                )
                st.dataframe(innovation_df.describe(), use_container_width=True)
            else:
                st.write("No innovation data available")
        
        # Kalman Gain Analysis
        st.subheader("Kalman Gain Analysis")
        if kf.kalman_gain_history:
            kalman_gain_df = pd.DataFrame(
                kf.kalman_gain_history, 
                columns=['Gain_1', 'Gain_2', 'Gain_3']
            )
            st.dataframe(kalman_gain_df.describe(), use_container_width=True)
        else:
            st.write("No Kalman Gain data available")
        
        # Visualize Flight Trajectory
        st.subheader("Flight Trajectory")
        fig_traj = go.Figure()
        
        # Actual trajectory
        fig_traj.add_trace(go.Scatter(
            x=flight_data['longitude'], 
            y=flight_data['latitude'], 
            mode='lines+markers',
            name='Actual Path'
        ))
        
        # Kalman Filter estimated path
        estimated_path = np.array(kf.state_history)
        fig_traj.add_trace(go.Scatter(
            x=estimated_path[:, 0], 
            y=estimated_path[:, 1], 
            mode='lines+markers',
            name='Kalman Filter Estimate'
        ))
        
        fig_traj.update_layout(
            title=f'Flight Path Estimation for {flight}',
            xaxis_title='Longitude',
            yaxis_title='Latitude'
        )
        
        st.plotly_chart(fig_traj, use_container_width=True)

# Run the main application
if __name__ == "__main__":
    main()
