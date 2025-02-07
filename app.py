import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import math
from typing import List, Dict, Tuple

class AdvancedKalmanFilter:
    """
    Enhanced Kalman Filter for Flight Tracking and Prediction
    Incorporates advanced aerospace tracking concepts
    """
    def __init__(self, initial_state: np.ndarray):
        """
        Initialize Advanced Kalman Filter
        
        :param initial_state: Initial state vector 
        """
        # State vector: [latitude, longitude, altitude, ground_speed, vertical_rate, heading]
        self.state = initial_state
        
        # State transition matrix (non-linear dynamics)
        self.F = np.eye(6)
        self.F[0, 1] = 0.1  # Latitude depends on longitude
        self.F[1, 2] = 0.1  # Longitude depends on altitude
        
        # Measurement matrix
        self.H = np.eye(6)
        
        # Process noise covariance (model uncertainty)
        self.Q = np.diag([0.001, 0.001, 0.01, 0.1, 0.1, 0.01])
        
        # Measurement noise covariance
        self.R = np.diag([0.01, 0.01, 0.1, 0.5, 0.5, 0.1])
        
        # Estimation error covariance
        self.P = np.eye(6)
        
        # Historical tracking
        self.state_history = [self.state.copy()]
        self.innovation_history = []
        self.kalman_gain_history = []
    
    def predict(self) -> np.ndarray:
        """Prediction step of Kalman Filter"""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update step of Kalman Filter"""
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Store Kalman Gain
        self.kalman_gain_history.append(K.flatten())
        
        # Calculate innovation
        innovation = measurement - self.H @ self.state
        self.innovation_history.append(innovation)
        
        # Update state estimation
        self.state = self.state + K @ innovation
        self.state_history.append(self.state.copy())
        
        # Update error covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.state

class RealTimeFlightDataFetcher:
    """
    Advanced flight data retrieval from OpenSky Network API
    """
    BASE_URL = "https://opensky-network.org/api/states/all"
    
    @staticmethod
    def fetch_flight_data(bounding_box: Dict[str, float] = None) -> pd.DataFrame:
        """
        Fetch real-time flight data from OpenSky Network API
        
        :param bounding_box: Dictionary with min/max lat/lon coordinates
        :return: DataFrame with flight data
        """
        # Default to Continental US if no bounding box provided
        if bounding_box is None:
            bounding_box = {
                'min_lon': -125.0, 'max_lon': -66.0,
                'min_lat': 24.0, 'max_lat': 49.0,
            }
        
        try:
            # Parameters for API request
            params = {
                'lamin': bounding_box['min_lat'],
                'lomin': bounding_box['min_lon'],
                'lamax': bounding_box['max_lat'],
                'lomax': bounding_box['max_lon']
            }
            
            # Make API request
            response = requests.get(
                RealTimeFlightDataFetcher.BASE_URL, 
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
                        'vertical_rate': (flight[11] or 0) * 196.85,  # ft/min
                        'timestamp': pd.Timestamp.now()
                    })
            
            return pd.DataFrame(flights_data)
        
        except Exception as e:
            st.error(f"Error fetching flight data: {e}")
            return pd.DataFrame()

def generate_synthetic_flight_data(n_flights=10):
    """
    Generate synthetic flight trajectory data with realistic characteristics
    
    :param n_flights: Number of flights to generate
    :return: DataFrame with flight trajectory parameters
    """
    flights = []
    
    # Major airport coordinates (lat, lon)
    airports = {
        'JFK': (40.6413, -73.7781),
        'LAX': (33.9416, -118.4085),
        'ORD': (41.9742, -87.9073),
        'ATL': (33.6367, -84.4281),
        'DFW': (32.8998, -97.0403),
        'DEN': (39.8561, -104.6737),
        'SFO': (37.6213, -122.3790),
        'SEA': (47.4502, -122.3088),
        'MIA': (25.7617, -80.1918),
        'LAS': (36.0840, -115.1537)
    }
    
    for i in range(n_flights):
        # Randomly select origin and destination
        origin_airport = np.random.choice(list(airports.keys()))
        dest_airport = np.random.choice([a for a in airports.keys() if a != origin_airport])
        
        origin_lat, origin_lon = airports[origin_airport]
        dest_lat, dest_lon = airports[dest_airport]
        
        # Generate trajectory
        num_points = 100
        lats = np.linspace(origin_lat, dest_lat, num_points)
        lons = np.linspace(origin_lon, dest_lon, num_points)
        
        # Add some randomness to simulate real flight paths
        lats += np.random.normal(0, 0.05, num_points)
        lons += np.random.normal(0, 0.05, num_points)
        
        # Altitude profile
        initial_altitude = np.random.uniform(25000, 35000)
        altitudes = np.linspace(initial_altitude, initial_altitude, num_points)
        altitudes += np.random.normal(0, 200, num_points)
        
        # Ground speed variations
        ground_speeds = np.linspace(400, 500, num_points)
        ground_speeds += np.random.normal(0, 20, num_points)
        
        # Vertical rate
        vertical_rates = np.zeros(num_points)
        vertical_rates[0:20] = np.linspace(0, 2000, 20)  # Climbing
        vertical_rates[-20:] = np.linspace(-2000, 0, 20)  # Descending
        
        # Heading (direction of flight)
        heading = np.arctan2(dest_lat - origin_lat, dest_lon - origin_lon)
        headings = np.full(num_points, math.degrees(heading))
        headings += np.random.normal(0, 2, num_points)
        
        # Create DataFrame for this flight
        flight_df = pd.DataFrame({
            'flight_id': f'FL{i+1}',
            'origin': origin_airport,
            'destination': dest_airport,
            'latitude': lats,
            'longitude': lons,
            'altitude': altitudes,
            'ground_speed': ground_speeds,
            'vertical_rate': vertical_rates,
            'heading': headings,
            'timestamp': pd.date_range(start='2025-01-01', periods=num_points, freq='T')
        })
        
        flights.append(flight_df)
    
    return pd.concat(flights, ignore_index=True)

def run_kalman_filter_simulation(trajectory_data: pd.DataFrame) -> Dict:
    """
    Run Kalman Filter simulation on trajectory data
    
    :param trajectory_data: DataFrame with trajectory information
    :return: Dictionary of Kalman Filter analysis results
    """
    # Select specific flights
    flight_ids = trajectory_data['flight_id'].unique()
    results = {}
    
    for flight_id in flight_ids:
        # Filter data for specific flight
        flight_data = trajectory_data[trajectory_data['flight_id'] == flight_id]
        
        # Initial state for Kalman Filter
        initial_state = np.array([
            flight_data['latitude'].iloc[0],
            flight_data['longitude'].iloc[0],
            flight_data['altitude'].iloc[0],
            flight_data['ground_speed'].iloc[0],
            flight_data['vertical_rate'].iloc[0],
            flight_data['heading'].iloc[0]
        ])
        
        # Create Kalman Filter
        kf = AdvancedKalmanFilter(initial_state)
        
        # Simulate measurements and updates
        for _, row in flight_data.iterrows():
            # Prepare noisy measurement
            measurement = np.array([
                row['latitude'] + np.random.normal(0, 0.01),
                row['longitude'] + np.random.normal(0, 0.01),
                row['altitude'] + np.random.normal(0, 50),
                row['ground_speed'] + np.random.normal(0, 5),
                row['vertical_rate'] + np.random.normal(0, 20),
                row['heading'] + np.random.normal(0, 1)
            ])
            
            # Predict and update
            kf.predict()
            kf.update(measurement)
        
        # Store results
        results[flight_id] = {
            'state_history': np.array(kf.state_history),
            'innovation_history': np.array(kf.innovation_history),
            'kalman_gain_history': np.array(kf.kalman_gain_history)
        }
    
    return results

def main():
    st.set_page_config(layout="wide", page_title="Advanced Flight Tracking", page_icon="✈️")
    
    # Title and introduction
    st.title("✈️ Advanced Commercial Flight Tracking System")
    st.markdown("""
    ## Comprehensive Aerospace Flight Analysis
    
    ### Key Features:
    - Real-time and Simulated Flight Tracking
    - Advanced Kalman Filter Implementation
    - Comprehensive Visualization Techniques
    """)
    
    # Sidebar for analysis options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Mode",
        [
            "Flight Trajectories",
            "Kalman Filter Analysis",
            "Performance Metrics",
            "Statistical Insights",
            "Flight Path Comparison"
        ]
    )
    
    # Sidebar for geographic bounds configuration
    st.sidebar.header("Geographic Bounds")
    min_lon = st.sidebar.number_input("Minimum Longitude", value=-125.0, min_value=-180.0, max_value=180.0)
    max_lon = st.sidebar.number_input("Maximum Longitude", value=-66.0, min_value=-180.0, max_value=180.0)
    min_lat = st.sidebar.number_input("Minimum Latitude", value=24.0, min_value=-90.0, max_value=90.0)
    max_lat = st.sidebar.number_input("Maximum Latitude", value=49.0, min_value=-90.0, max_value=90.0)
    
    # Geographic bounds for API
    bounding_box = {
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat
    }
    
    # Generate or fetch data
    if 'synthetic_flight_data' not in st.session_state:
        # Generate synthetic flight trajectory data
        st.session_state['synthetic_flight_data'] = generate_synthetic_flight_data()
    
    # Fetch real flight data
    if st.sidebar.button("🔄 Fetch Real Flight Data"):
        with st.spinner("Fetching real-time flight data..."):
            st.session_state['real_flight_data'] = RealTimeFlightDataFetcher.fetch_flight_data(bounding_box)
    
    # Ensure real flight data exists
    if 'real_flight_data' not in st.session_state:
        st.session_state['real_flight_data'] = pd.DataFrame()
    
    # Get trajectory data
    synthetic_data = st.session_state['synthetic_flight_data']
    real_data = st.session_state['real_flight_data']
    
    # Rest of the previous implementation remains the same...
    # (Copy the entire existing main() implementation from the previous code)
    # Specifically from line 394 to the end of the previous implementation

    if analysis_type == "Flight Trajectories":
        # Flight Trajectory Visualization
        st.header("🗺️ Flight Path Mapping")
        
        # Create map figure
        fig_map = go.Figure()
        
        # Add US map background
        fig_map.add_trace(go.Scattergeo(
            lon=[-98.5795],
            lat=[39.8283],
            mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        # Plot synthetic flights
        for flight_id in synthetic_data['flight_id'].unique():
            flight_subset = synthetic_data[synthetic_data['flight_id'] == flight_id]
            
            fig_map.add_trace(go.Scattergeo(
                lon=flight_subset['longitude'],
                lat=flight_subset['latitude'],
                mode='lines+markers',
                name=f"{flight_id} (Synthetic)",
                line=dict(width=2),
                marker=dict(size=7, color='blue')
            ))
        
        # Plot real flights
        for flight_id in real_data['flight_id'].unique():
            flight_subset = real_data[real_data['flight_id'] == flight_id]
            
            fig_map.add_trace(go.Scattergeo(
                lon=flight_subset['longitude'],
                lat=flight_subset['latitude'],
                mode='markers',
                name=f"{flight_id} (Real)",
                marker=dict(size=7, color='red')
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
            title='Flight Trajectories Across the United States',
            height=800
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    
    elif analysis_type == "Kalman Filter Analysis":
        # Kalman Filter Analysis
        st.header("🧮 Flight Trajectory Estimation")
        
        # Run Kalman Filter simulation on synthetic data
        kalman_results = run_kalman_filter_simulation(synthetic_data)
        
        # Visualization of Kalman Filter results
        for flight_id, results in kalman_results.items():
            st.subheader(f"Kalman Filter Analysis for {flight_id}")
            
            # State History Visualization
            fig_state = go.Figure()
            state_labels = ['Latitude', 'Longitude', 'Altitude', 'Ground Speed', 'Vertical Rate', 'Heading']
            
            for i, label in enumerate(state_labels):
                fig_state.add_trace(go.Scatter(
                    y=results['state_history'][:, i],
                    mode='lines',
                    name=label
                ))
            
            fig_state.update_layout(
                title=f'Kalman Filter State Estimation for {flight_id}',
                xaxis_title='Time Steps',
                yaxis_title='State Value'
            )
            
            st.plotly_chart(fig_state, use_container_width=True)
    
    elif analysis_type == "Performance Metrics":
        # Performance Metrics
        st.header("📊 Flight Performance Analysis")
        
        # Combine synthetic and real flight data
        combined_data = pd.concat([synthetic_data, real_data])
        
        # Compute performance metrics
        performance_metrics = []
        
        for flight_id in combined_data['flight_id'].unique():
            flight_subset = combined_data[combined_data['flight_id'] == flight_id]
            
            performance_metrics.append({
                'Flight ID': flight_id,
                'Origin': flight_subset['origin'].iloc[0] if 'origin' in flight_subset.columns else 'N/A',
                'Destination': flight_subset['destination'].iloc[0] if 'destination' in flight_subset.columns else 'N/A',
                'Max Altitude (ft)': flight_subset['altitude'].max(),
                'Min Altitude (ft)': flight_subset['altitude'].min(),
                'Avg Ground Speed (knots)': flight_subset['ground_speed'].mean(),
                'Max Ground Speed (knots)': flight_subset['ground_speed'].max(),
                'Vertical Rate Range (ft/min)': flight_subset['vertical_rate'].max() - flight_subset['vertical_rate'].min()
            })
        
        # Create performance metrics DataFrame
        performance_df = pd.DataFrame(performance_metrics)
        
        # Display performance metrics
        st.dataframe(performance_df, use_container_width=True)
    
    elif analysis_type == "Statistical Insights":
        # Statistical Analysis
        st.header("📈 Flight Data Statistical Insights")
        
        # Combine synthetic and real flight data
        combined_data = pd.concat([synthetic_data, real_data])
        
        # Select numeric columns for correlation
        numeric_cols = ['latitude', 'longitude', 'altitude', 'ground_speed', 'vertical_rate', 'heading']
        
        # Compute correlation matrix
        correlation_matrix = combined_data[numeric_cols].corr()
        
        # Correlation Heatmap
        fig_corr = px.imshow(
            correlation_matrix, 
            title='Correlation Heatmap of Flight Parameters',
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(combined_data[numeric_cols].describe(), use_container_width=True)
    
    elif analysis_type == "Flight Path Comparison":
        # Flight Path Comparison
        st.header("🔍 Flight Path Detailed Comparison")
        
        # Select flights for comparison
        synthetic_flights = synthetic_data['flight_id'].unique()
        real_flights = real_data['flight_id'].unique()
        
        # Multi-select flights
        compared_synthetic_flights = st.multiselect(
            "Select Synthetic Flights", 
            synthetic_flights, 
            default=synthetic_flights[:3]
        )
        
        compared_real_flights = st.multiselect(
            "Select Real Flights", 
            real_flights, 
            default=real_flights[:3]
        )
        
        # Create comparison figure
        fig_comparison = go.Figure()
        
        # Add synthetic flight paths
        for flight_id in compared_synthetic_flights:
            flight_subset = synthetic_data[synthetic_data['flight_id'] == flight_id]
            fig_comparison.add_trace(go.Scatter(
                x=flight_subset['longitude'],
                y=flight_subset['latitude'],
                mode='lines+markers',
                name=f"{flight_id} (Synthetic)",
                line=dict(dash='dot')
            ))
        
        # Add real flight paths
        for flight_id in compared_real_flights:
            flight_subset = real_data[real_data['flight_id'] == flight_id]
            fig_comparison.add_trace(go.Scatter(
                x=flight_subset['longitude'],
                y=flight_subset['latitude'],
                mode='markers',
                name=f"{flight_id} (Real)"
            ))
        
        fig_comparison.update_layout(
            title='Comparative Flight Path Analysis',
            xaxis_title='Longitude',
            yaxis_title='Latitude'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)

# Run the main application
if __name__ == "__main__":
    main()
