import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import math
import time

# Advanced Aerospace Tracking Algorithms
class AerospaceTrackingModel:
    """
    Advanced trajectory prediction and tracking model 
    incorporating physics-based principles
    """
    def __init__(self, initial_state):
        """
        Initialize tracking model with initial flight state
        
        :param initial_state: Dictionary containing initial flight parameters
        """
        # Fundamental flight parameters
        self.latitude = initial_state.get('latitude', 0)
        self.longitude = initial_state.get('longitude', 0)
        self.altitude = initial_state.get('altitude', 0)
        self.velocity = initial_state.get('ground_speed', 0)
        self.heading = initial_state.get('heading', 0)
        self.vertical_rate = initial_state.get('vertical_rate', 0)
        
        # Advanced tracking parameters
        self.acceleration = 0
        self.turn_rate = 0
        self.bank_angle = 0
        
        # Physical constants
        self.EARTH_RADIUS = 6371000  # meters
        self.GRAVITY = 9.81  # m/s^2
        
        # Tracking and prediction parameters
        self.prediction_interval = 300  # 5 minutes of prediction
        self.time_step = 30  # 30-second intervals
    
    def calculate_great_circle_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points
        
        :return: Distance in kilometers
        """
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.EARTH_RADIUS * c / 1000  # Convert to kilometers
    
    def predict_trajectory(self):
        """
        Advanced trajectory prediction using physics-based models
        
        :return: Predicted trajectory points
        """
        # Initialize prediction arrays
        pred_lats = [self.latitude]
        pred_lons = [self.longitude]
        pred_alts = [self.altitude]
        
        # Current state variables
        current_lat = self.latitude
        current_lon = self.longitude
        current_alt = self.altitude
        current_heading = self.heading
        current_velocity = self.velocity
        
        # Incorporate more advanced prediction factors
        for _ in range(int(self.prediction_interval / self.time_step)):
            # Wind and atmospheric effects simulation
            wind_effect_lat = np.random.normal(0, 0.01)
            wind_effect_lon = np.random.normal(0, 0.01)
            
            # Advanced heading and navigation calculations
            # Simulate slight course corrections and navigation adjustments
            heading_variation = np.random.normal(0, 2)  # Degrees of variation
            new_heading = (current_heading + heading_variation) % 360
            
            # Calculate new position using great circle navigation
            R = self.EARTH_RADIUS / 1000  # Radius in kilometers
            d = current_velocity * (self.time_step / 3600)  # Distance traveled
            
            # Convert to radians
            lat1 = math.radians(current_lat)
            lon1 = math.radians(current_lon)
            heading_rad = math.radians(new_heading)
            
            # Calculate new latitude
            lat2 = math.asin(
                math.sin(lat1) * math.cos(d/R) + 
                math.cos(lat1) * math.sin(d/R) * math.cos(heading_rad)
            )
            
            # Calculate new longitude
            lon2 = lon1 + math.atan2(
                math.sin(heading_rad) * math.sin(d/R) * math.cos(lat1),
                math.cos(d/R) - math.sin(lat1) * math.sin(lat2)
            )
            
            # Convert back to degrees
            new_lat = math.degrees(lat2)
            new_lon = math.degrees(lon2)
            
            # Altitude calculation with more realistic vertical dynamics
            vertical_change = (self.vertical_rate * self.time_step / 60)
            new_alt = current_alt + vertical_change
            
            # Add some randomness to simulate real-world variations
            new_alt += np.random.normal(0, 50)
            
            # Update current state
            current_lat = new_lat
            current_lon = new_lon
            current_alt = new_alt
            current_heading = new_heading
            
            # Store predicted points
            pred_lats.append(current_lat)
            pred_lons.append(current_lon)
            pred_alts.append(current_alt)
        
        return pred_lats, pred_lons, pred_alts
    
    def calculate_intercept_trajectory(self, target_lat, target_lon):
        """
        Calculate optimal intercept trajectory
        
        :param target_lat: Latitude of target
        :param target_lon: Longitude of target
        :return: Intercept trajectory details
        """
        # Calculate initial bearing and distance
        distance = self.calculate_great_circle_distance(
            self.latitude, self.longitude, 
            target_lat, target_lon
        )
        
        # Calculate initial bearing
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        lat2 = math.radians(target_lat)
        lon2 = math.radians(target_lon)
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        initial_bearing = math.degrees(math.atan2(y, x))
        
        return {
            'distance': distance,
            'bearing': initial_bearing,
            'estimated_time': distance / self.velocity if self.velocity > 0 else float('inf')
        }

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
                
                # Create initial state for tracking model
                initial_state = {
                    'latitude': current_lat,
                    'longitude': current_lon,
                    'altitude': (flight[7] or 0) * 3.28084,  # Convert meters to feet
                    'ground_speed': (flight[9] or 0) * 1.94384,  # Convert m/s to knots
                    'heading': flight[10] or 0,
                    'vertical_rate': (flight[11] or 0) * 196.85  # Convert m/s to ft/min
                }
                
                # Initialize aerospace tracking model
                tracking_model = AerospaceTrackingModel(initial_state)
                
                # Predict trajectory
                pred_lats, pred_lons, pred_alts = tracking_model.predict_trajectory()
                
                flight_entry = {
                    'flight_id': flight_id,
                    'longitude': current_lon,
                    'latitude': current_lat,
                    'altitude': initial_state['altitude'],
                    'ground_speed': initial_state['ground_speed'],
                    'heading': initial_state['heading'],
                    'vertical_rate': initial_state['vertical_rate'],
                    'predicted_lats': pred_lats,
                    'predicted_lons': pred_lons,
                    'predicted_alts': pred_alts
                }
                
                flights_data.append(flight_entry)
        
        return pd.DataFrame(flights_data)
    
    except Exception as e:
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="Advanced Aerospace Flight Tracker", page_icon="🛩️")
    
    st.title("🛩️ Advanced Aerospace Flight Tracking System")
    
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
    
    # Flight selection with additional context
    available_flights = sorted(df['flight_id'].unique())
    selected_flights = st.multiselect(
        "Select Flights", 
        available_flights, 
        default=available_flights[:min(5, len(available_flights))],
        help="Advanced trajectory analysis for selected flights"
    )
    
    # Filtered flight data
    flight_data = df[df['flight_id'].isin(selected_flights)]
    
    # Visualization
    st.markdown("### 🗺️ Advanced Trajectory Prediction")
    
    # Create map figure with enhanced styling
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
        title='Advanced Flight Trajectory Prediction',
        height=800,
        geo=dict(
            center=dict(lon=-98.5795, lat=39.8283),
            projection_scale=1.5
        )
    )
    
    # Display map
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Detailed Flight Analytics
    st.markdown("### 📊 Flight Performance Metrics")
    
    # Create columns for detailed analytics
    cols = st.columns(len(selected_flights))
    
    for i, (_, flight) in enumerate(flight_data.iterrows()):
        with cols[i]:
            st.markdown(f"#### {flight['flight_id']} Analysis")
            
            # Performance metrics
            st.metric("Current Altitude", f"{flight['altitude']:.0f} ft")
            st.metric("Ground Speed", f"{flight['ground_speed']:.0f} knots")
            st.metric("Vertical Rate", f"{flight['vertical_rate']:.0f} ft/min")
            
            # Advanced tracking information
            tracking_model = AerospaceTrackingModel({
                'latitude': flight['latitude'],
                'longitude': flight['longitude'],
                'altitude': flight['altitude'],
                'ground_speed': flight['ground_speed'],
                'heading': flight['heading'],
                'vertical_rate': flight['vertical_rate']
            })
            
            # Random target for intercept demonstration
            target_lat = flight['latitude'] + np.random.uniform(-5, 5)
            target_lon = flight['longitude'] + np.random.uniform(-5, 5)
            
            # Calculate intercept trajectory
            intercept_info = tracking_model.calculate_intercept_trajectory(
                target_lat, target_lon
            )
            
            st.markdown("#### Intercept Analysis")
            st.metric("Intercept Distance", f"{intercept_info['distance']:.2f} km")
            st.metric("Intercept Bearing", f"{intercept_info['bearing']:.2f}°")
            st.metric("Estimated Intercept Time", 
                      f"{intercept_info['estimated_time']:.2f} hours" 
                      if intercept_info['estimated_time'] != float('inf') 
                      else "N/A")

# Run the main application
main()
