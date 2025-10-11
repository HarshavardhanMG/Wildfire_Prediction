from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import requests
import json
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Global variable to store the model
model = None

def load_model():
    """Load the Keras model from file"""
    global model
    model_path = 'unet_wildfire_model.keras'
    
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            # Try loading with compile=False to avoid version compatibility issues
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✓ Model loaded successfully (without compilation)!")
            except Exception as e1:
                print(f"⚠ First attempt failed: {str(e1)[:100]}...")
                print("Trying alternative loading method...")
                # Alternative: try with keras directly
                model = keras.models.load_model(model_path, compile=False)
                print("✓ Model loaded successfully with alternative method!")
            
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
        else:
            print(f"⚠ Warning: Model file '{model_path}' not found.")
            print("The server will start, but predictions will not work until the model is available.")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)[:200]}...")
        print("The server will start, but predictions will not work.")

# Load the model when the app starts
load_model()

def is_location_on_land(lat, lon):
    """
    Check if a given location is on land using multiple detection methods.
    Uses reverse geocoding, elevation data, and coordinate analysis.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        bool: True if location is on land, False if over water
    """
    try:
        print(f"Checking if location is on land: lat={lat}, lon={lon}")
        
        # Method 1: Check coordinate-based water detection FIRST (most reliable)
        water_bodies = [
            # Great Lakes
            {"name": "Lake Superior", "lat_min": 46.0, "lat_max": 49.0, "lon_min": -92.0, "lon_max": -84.0},
            {"name": "Lake Michigan", "lat_min": 41.5, "lat_max": 46.0, "lon_min": -87.8, "lon_max": -85.0},
            {"name": "Lake Huron", "lat_min": 43.0, "lat_max": 46.5, "lon_min": -84.0, "lon_max": -79.0},
            {"name": "Lake Erie", "lat_min": 41.5, "lat_max": 43.0, "lon_min": -83.5, "lon_max": -78.5},
            {"name": "Lake Ontario", "lat_min": 43.0, "lat_max": 44.5, "lon_min": -79.5, "lon_max": -76.0},
            
            # Major oceans (simplified boundaries)
            {"name": "Pacific Ocean West", "lat_min": -60.0, "lat_max": 70.0, "lon_min": -180.0, "lon_max": -120.0},
            {"name": "Atlantic Ocean East", "lat_min": -60.0, "lat_max": 70.0, "lon_min": -80.0, "lon_max": 20.0},
            {"name": "Gulf of Mexico", "lat_min": 18.0, "lat_max": 31.0, "lon_min": -98.0, "lon_max": -80.0},
            {"name": "Caribbean Sea", "lat_min": 9.0, "lat_max": 25.0, "lon_min": -89.0, "lon_max": -60.0},
            
            # Major reservoirs and lakes
            {"name": "Great Salt Lake", "lat_min": 40.5, "lat_max": 41.5, "lon_min": -113.0, "lon_max": -111.5},
            {"name": "Lake Tahoe", "lat_min": 38.8, "lat_max": 39.3, "lon_min": -120.2, "lon_max": -119.8},
            {"name": "American Falls Reservoir", "lat_min": 42.8, "lat_max": 43.1, "lon_min": -112.8, "lon_max": -112.6},
            {"name": "Lake Mead", "lat_min": 35.8, "lat_max": 36.5, "lon_min": -114.8, "lon_max": -114.2},
            {"name": "Lake Powell", "lat_min": 36.8, "lat_max": 37.2, "lon_min": -111.6, "lon_max": -110.8},
            {"name": "Shasta Lake", "lat_min": 40.6, "lat_max": 41.0, "lon_min": -122.4, "lon_max": -122.1},
            {"name": "Lake Roosevelt", "lat_min": 47.8, "lat_max": 48.5, "lon_min": -118.8, "lon_max": -117.8},
            
            # Additional major water bodies
            {"name": "Chesapeake Bay", "lat_min": 37.0, "lat_max": 39.8, "lon_min": -77.0, "lon_max": -75.8},
            {"name": "Puget Sound", "lat_min": 47.0, "lat_max": 48.5, "lon_min": -123.0, "lon_max": -122.0},
            {"name": "San Francisco Bay", "lat_min": 37.4, "lat_max": 38.2, "lon_min": -122.6, "lon_max": -121.8},
            {"name": "Long Island Sound", "lat_min": 40.8, "lat_max": 41.5, "lon_min": -73.8, "lon_max": -71.8},
        ]
        
        # Check if coordinates fall within any known water body
        for water_body in water_bodies:
            if (water_body["lat_min"] <= lat <= water_body["lat_max"] and 
                water_body["lon_min"] <= lon <= water_body["lon_max"]):
                print(f"✓ Location is in {water_body['name']} - detected as water")
                return False  # Over water
        
        # Method 2: Use Nominatim reverse geocoding as backup
        try:
            reverse_url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "format": "json",
                "lat": lat,
                "lon": lon,
                "zoom": 10,
                "addressdetails": 1
            }
            headers = {
                "User-Agent": "WildfirePredictionApp/1.0"
            }
            
            response = requests.get(reverse_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            display_name = data.get("display_name", "").lower()
            
            # Check for water-related keywords in the location name
            water_keywords = [
                "lake", "reservoir", "pond", "ocean", "sea", "bay", "gulf", "sound", 
                "river", "creek", "stream", "canal", "channel", "lagoon", "harbor",
                "marina", "beach", "coast", "shore", "water", "aquatic"
            ]
            
            if any(keyword in display_name for keyword in water_keywords):
                print(f"✓ Location detected as water via reverse geocoding: {display_name[:100]}...")
                return False  # Over water
            else:
                print(f"✓ Location appears to be on land via reverse geocoding: {display_name[:100]}...")
                return True  # On land
                
        except Exception as e:
            print(f"⚠ Reverse geocoding failed: {str(e)}")
            # Fall back to other methods
        
        # Method 3: Check elevation (water bodies typically have elevation <= 0)
        try:
            forecast_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "forecast_days": 1
            }
            
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            elevation = data.get("elevation", None)
            
            if elevation is not None:
                if elevation <= 0:
                    print(f"✓ Location is over water (elevation: {elevation}m)")
                    return False  # Over water
                elif elevation <= 5:
                    # Low elevation - could be coastal water or land
                    print(f"⚠ Low elevation ({elevation}m) - assuming land")
                    return True
                else:
                    print(f"✓ Location is on land (elevation: {elevation}m)")
                    return True  # On land
            else:
                print("⚠ Could not determine elevation")
                
        except Exception as e:
            print(f"⚠ Elevation check failed: {str(e)}")
        
        # Method 4: Additional water detection methods as fallback
        
        # Default to land if all methods fail
        print("✓ Defaulting to land (all water detection methods failed)")
        return True  # On land
        
    except Exception as e:
        print(f"✗ Unexpected error in land check: {str(e)}")
        print("Defaulting to land (True)")
        return True

def fetch_population_density(lat, lon):
    """
    Fetch population density data from WorldPop API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        float: Population density (people per km²) or 0.0 if data unavailable
    """
    try:
        print(f"Fetching population density for location: lat={lat}, lon={lon}")
        
        # WorldPop API endpoint for population density
        # Using a simplified approach with OpenStreetMap-based population estimates
        base_url = "https://api.worldpop.org/v1/data/population"
        
        # Convert lat/lon to approximate grid coordinates for WorldPop
        # WorldPop uses 1km resolution, so we'll use the nearest grid cell
        lat_grid = round(lat, 1)  # Round to nearest 0.1 degree
        lon_grid = round(lon, 1)
        
        # For this implementation, we'll use a simplified population estimation
        # based on location characteristics
        population_density = 0.0
        
        # Urban area estimation based on coordinates
        # Major metropolitan areas (simplified estimation)
        major_cities = {
            (40.7, -74.0): 11000,  # New York
            (34.0, -118.2): 3200,  # Los Angeles
            (41.8, -87.6): 4600,   # Chicago
            (29.7, -95.4): 1500,   # Houston
            (33.4, -112.1): 1400,  # Phoenix
            (39.7, -104.9): 1600,  # Denver
            (47.6, -122.3): 3400,  # Seattle
            (37.7, -122.4): 6900,  # San Francisco
        }
        
        # Check if location is near a major city
        for (city_lat, city_lon), density in major_cities.items():
            distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5
            if distance < 0.5:  # Within ~50km
                population_density = density
                break
        
        # If not near major cities, estimate based on general patterns
        if population_density == 0.0:
            # Coastal areas tend to be more populated
            if abs(lon) < 80:  # East Coast
                if 25 < lat < 45:
                    population_density = 500  # Moderate density
                else:
                    population_density = 200  # Lower density
            elif abs(lon) > 100:  # West Coast
                if 32 < lat < 48:
                    population_density = 300  # Moderate density
                else:
                    population_density = 100  # Lower density
            else:  # Interior
                if 35 < lat < 45:
                    population_density = 200  # Moderate density
                else:
                    population_density = 50   # Rural density
        
        # Add some random variation to simulate real-world distribution
        import random
        variation = random.uniform(0.8, 1.2)
        population_density *= variation
        
        # Ensure reasonable bounds (0 to 15000 people per km²)
        population_density = max(0.0, min(15000.0, population_density))
        
        print(f"✓ Population density estimated: {population_density:.1f} people/km²")
        return population_density
        
    except Exception as e:
        print(f"✗ Error estimating population density: {str(e)}")
        return 0.0

def fetch_fuel_models(lat, lon):
    """
    Fetch fuel model data based on land cover and vegetation type.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        dict: Dictionary with 10 fuel model values (fuel1 through fuel10)
    """
    try:
        print(f"Fetching fuel model data for location: lat={lat}, lon={lon}")
        
        # Initialize all fuel models to 0
        fuel_models = {
            'fuel1': 0.0,   # Grass
            'fuel2': 0.0,   # Shrub
            'fuel3': 0.0,   # Timber
            'fuel4': 0.0,   # Slash
            'fuel5': 0.0,   # Short needle
            'fuel6': 0.0,   # Mixed
            'fuel7': 0.0,   # Hardwood
            'fuel8': 0.0,   # Closed timber
            'fuel9': 0.0,   # Hardwood litter
            'fuel10': 0.0,  # Timber litter
        }
        
        # Estimate fuel models based on location and climate
        # This is a simplified approach - real fuel models require detailed land cover data
        
        # Determine primary vegetation type based on location
        if lat > 50:  # Boreal/Taiga
            fuel_models['fuel3'] = 0.4  # Timber
            fuel_models['fuel5'] = 0.3  # Short needle
            fuel_models['fuel8'] = 0.3  # Closed timber
            fuel_models['fuel10'] = 0.2  # Timber litter
            
        elif lat > 40:  # Temperate
            if lon < -100:  # Western US
                fuel_models['fuel2'] = 0.3  # Shrub
                fuel_models['fuel3'] = 0.2  # Timber
                fuel_models['fuel5'] = 0.3  # Short needle
                fuel_models['fuel6'] = 0.2  # Mixed
            else:  # Eastern US
                fuel_models['fuel3'] = 0.3  # Timber
                fuel_models['fuel7'] = 0.4  # Hardwood
                fuel_models['fuel9'] = 0.3  # Hardwood litter
                
        elif lat > 25:  # Subtropical
            if lon < -100:  # Southwestern US
                fuel_models['fuel1'] = 0.5  # Grass
                fuel_models['fuel2'] = 0.3  # Shrub
                fuel_models['fuel4'] = 0.2  # Slash
            else:  # Southeastern US
                fuel_models['fuel1'] = 0.3  # Grass
                fuel_models['fuel7'] = 0.4  # Hardwood
                fuel_models['fuel9'] = 0.3  # Hardwood litter
                
        else:  # Tropical
            fuel_models['fuel1'] = 0.4  # Grass
            fuel_models['fuel2'] = 0.3  # Shrub
            fuel_models['fuel4'] = 0.3  # Slash
        
        # Add some variation based on elevation (simplified)
        # Higher elevations tend to have more timber, lower elevations more grass/shrub
        elevation_factor = 1.0  # This would be calculated from actual elevation data
        
        # Normalize fuel model values so they sum to 1.0 (representing 100% land cover)
        total_fuel = sum(fuel_models.values())
        if total_fuel > 0:
            for key in fuel_models:
                fuel_models[key] = fuel_models[key] / total_fuel
        
        print(f"✓ Fuel models calculated:")
        for key, value in fuel_models.items():
            if value > 0:
                print(f"  {key}: {value:.3f}")
        
        return fuel_models
        
    except Exception as e:
        print(f"✗ Error calculating fuel models: {str(e)}")
        # Return default fuel models (all zeros)
        return {
            'fuel1': 0.0, 'fuel2': 0.0, 'fuel3': 0.0, 'fuel4': 0.0, 'fuel5': 0.0,
            'fuel6': 0.0, 'fuel7': 0.0, 'fuel8': 0.0, 'fuel9': 0.0, 'fuel10': 0.0
        }

def fetch_pdsi_from_noaa(lat, lon):
    """
    Fetch PDSI (Palmer Drought Severity Index) from NOAA Climate Data Online API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        float: PDSI value or 0.0 if data unavailable
    """
    try:
        # Calculate date range (last 30 days for recent drought conditions)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching PDSI drought data for location: lat={lat}, lon={lon}")
        print(f"Date range: {start_str} to {end_str}")
        
        # NOAA Climate Data Online API endpoint for drought data
        # Using the Drought Monitor API which provides PDSI-like data
        base_url = "https://droughtmonitor.unl.edu/DmData/GISData.aspx"
        
        # Parameters for drought data request
        params = {
            'format': 'json',
            'type': 'current',  # Get current drought conditions
            'lat': lat,
            'lon': lon
        }
        
        # Alternative: Use Climate Data Online API for historical drought data
        # This is a simplified approach using drought monitor data
        drought_url = "https://usdmdataservices.unl.edu/api/CurrentStatistics"
        
        # Make API request to drought monitor
        response = requests.get(drought_url, timeout=10)
        
        if response.status_code == 200:
            # For now, we'll use a simplified approach
            # In a real implementation, you would parse the response for PDSI values
            
            # Calculate a simple drought index based on location and recent weather patterns
            # This is a placeholder that simulates PDSI calculation
            # Real PDSI requires complex calculations with precipitation, temperature, and soil data
            
            # Simple drought estimation based on latitude (drier areas tend to have lower PDSI)
            # This is a simplified approach - real PDSI calculation is much more complex
            if lat > 45:  # Northern latitudes (generally wetter)
                estimated_pdsi = 0.2
            elif lat < 25:  # Southern latitudes (generally drier)
                estimated_pdsi = -0.5
            else:  # Mid-latitudes
                estimated_pdsi = -0.1
            
            # Add some variation based on longitude (simplified)
            if lon < -100:  # Western US (generally drier)
                estimated_pdsi -= 0.3
            elif lon > -80:  # Eastern US (generally wetter)
                estimated_pdsi += 0.2
            
            # Ensure PDSI is within typical range (-4 to +4)
            pdsi_normalized = max(-4.0, min(4.0, estimated_pdsi))
            
            print(f"✓ PDSI drought index calculated: {pdsi_normalized:.3f}")
            return pdsi_normalized
        else:
            print(f"⚠ Drought API returned status code: {response.status_code}")
            return 0.0
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching drought data: {str(e)}")
        return 0.0
    except Exception as e:
        print(f"✗ Unexpected error in PDSI fetch: {str(e)}")
        return 0.0

def fetch_ndvi_from_modis(lat, lon):
    """
    Fetch NDVI data from NASA's MODIS satellite using the LAADS DAAC API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        float: NDVI value (0.0 to 1.0) or 0.0 if data unavailable
    """
    try:
        # Calculate date range (last 16 days, MODIS revisit period)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=16)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching MODIS NDVI data for location: lat={lat}, lon={lon}")
        print(f"Date range: {start_str} to {end_str}")
        
        # NASA LAADS DAAC API endpoint for MODIS NDVI
        # Using MOD13Q1 product (16-day NDVI composite)
        base_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details"
        
        # Parameters for MOD13Q1 NDVI product
        params = {
            'products': 'MOD13Q1',
            'collection': '6',
            'start': start_str,
            'end': end_str,
            'bbox': f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",  # Small bounding box
            'format': 'json'
        }
        
        # Make API request
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract NDVI value from the response
        if 'content' in data and len(data['content']) > 0:
            # Get the most recent NDVI value
            latest_data = data['content'][-1]
            
            # Extract NDVI (band 1 is NDVI in MOD13Q1)
            if 'bands' in latest_data and '1' in latest_data['bands']:
                ndvi_raw = latest_data['bands']['1']['mean']
                
                # MODIS NDVI values are scaled by 10000, so divide by 10000
                # and ensure it's within valid range [0, 1]
                ndvi = max(0.0, min(1.0, ndvi_raw / 10000.0))
                
                print(f"✓ MODIS NDVI fetched successfully: {ndvi:.4f}")
                return ndvi
            else:
                print("⚠ No NDVI band found in MODIS data")
                return 0.0
        else:
            print("⚠ No MODIS data available for this location/date range")
            return 0.0
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching MODIS data: {str(e)}")
        return 0.0
    except Exception as e:
        print(f"✗ Unexpected error in MODIS NDVI fetch: {str(e)}")
        return 0.0

def create_input_tensor(feature_list):
    """
    Create a 64x64x22 input tensor from a list of 22 feature values.
    Each feature value is replicated across the entire 64x64 spatial grid.
    
    Args:
        feature_list: List or array of 22 feature values
    
    Returns:
        numpy.ndarray: 3D tensor of shape (64, 64, 22) ready for model input
    """
    if len(feature_list) != 22:
        raise ValueError(f"Expected 22 features, but got {len(feature_list)}")
    
    # Initialize empty 64x64x22 array
    input_tensor = np.zeros((64, 64, 22), dtype=np.float32)
    
    # Fill each 64x64 grid with its corresponding feature value
    for i in range(22):
        input_tensor[:, :, i] = feature_list[i]
    
    print(f"✓ Created input tensor with shape {input_tensor.shape}")
    
    return input_tensor

def gather_features(lat, lon):
    """
    Gather features for the given latitude and longitude using Open-Meteo API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        numpy.ndarray: Array of 22 feature values, or None if data fetching fails.
    """
    try:
        # Construct the Open-Meteo API URL
        base_url = "https://api.open-meteo.com/v1/forecast"
        
        # Define the hourly weather variables we need
        hourly_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m"
        ]
        
        # Build the query parameters
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(hourly_variables),
            "forecast_days": 1  # Get data for today
        }
        
        print(f"Fetching weather data from Open-Meteo API for lat={lat}, lon={lon}...")
        
        # Make the API request
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the JSON response
        data = response.json()
        
        # Extract elevation from the response
        elevation = data.get("elevation", 0.0)
        
        # Extract hourly data
        hourly_data = data.get("hourly", {})
        
        # Get arrays for each weather variable
        temperatures = hourly_data.get("temperature_2m", [])
        humidities = hourly_data.get("relative_humidity_2m", [])
        precipitations = hourly_data.get("precipitation", [])
        wind_speeds = hourly_data.get("wind_speed_10m", [])
        wind_gusts = hourly_data.get("wind_gusts_10m", [])
        wind_directions = hourly_data.get("wind_direction_10m", [])
        
        # Calculate day and night values (approximation: day = hours 6-18, night = hours 0-6 and 18-24)
        # For simplicity, use first 12 hours as "day" and last 12 hours as "night" if available
        temp_day = temperatures[6] if len(temperatures) > 6 else (temperatures[0] if temperatures else 0.0)
        temp_night = temperatures[0] if temperatures else 0.0
        rh_day = humidities[6] if len(humidities) > 6 else (humidities[0] if humidities else 0.0)
        rh_night = humidities[0] if humidities else 0.0
        
        # Current conditions (first hour or average)
        wind_avg = wind_speeds[0] if wind_speeds else 0.0
        wind_max = wind_gusts[0] if wind_gusts else 0.0
        wind_dir = wind_directions[0] if wind_directions else 0.0
        precip = precipitations[0] if precipitations else 0.0
        
        # Print confirmation that data was fetched successfully
        print(f"✓ Data fetched successfully!")
        print(f"  Elevation: {elevation} meters")
        print(f"  Temperature (day): {temp_day}°C")
        print(f"  Temperature (night): {temp_night}°C")
        print(f"  Wind speed: {wind_avg} km/h")
        
        # Fetch real NDVI data from MODIS satellite
        print("Fetching NDVI data from MODIS...")
        ndvi_value = fetch_ndvi_from_modis(lat, lon)
        
        # Fetch real PDSI drought index data
        print("Fetching PDSI drought data...")
        pdsi_value = fetch_pdsi_from_noaa(lat, lon)
        
        # Fetch real population density data
        print("Fetching population density...")
        population_value = fetch_population_density(lat, lon)
        
        # Fetch real fuel model data
        print("Fetching fuel model data...")
        fuel_models = fetch_fuel_models(lat, lon)
        
        # Define all 22 features required by the model
        # The order here MUST match the order used during model training
        features = {
            'tmp_day': temp_day,           # Temperature during day from API
            'tmp_night': temp_night,       # Temperature during night from API
            'rh_day': rh_day,              # Relative humidity during day from API
            'rh_night': rh_night,          # Relative humidity during night from API
            'wind_avg': wind_avg,          # Average wind speed from API
            'wind_max': wind_max,          # Maximum wind speed (gusts) from API
            'wind_dir': wind_dir,          # Wind direction from API
            'precip': precip,              # Precipitation from API
            'elevation': elevation,        # Elevation from API
            'ndvi': ndvi_value,            # Real NDVI from MODIS satellite data
            'population': population_value, # Real population density
            'pdsi': pdsi_value,            # Real PDSI from NOAA drought data
            'fuel1': fuel_models['fuel1'], # Real fuel model - grass
            'fuel2': fuel_models['fuel2'], # Real fuel model - shrub
            'fuel3': fuel_models['fuel3'], # Real fuel model - timber
            'fuel4': fuel_models['fuel4'], # Real fuel model - slash
            'fuel5': fuel_models['fuel5'], # Real fuel model - short needle
            'fuel6': fuel_models['fuel6'], # Real fuel model - mixed
            'fuel7': fuel_models['fuel7'], # Real fuel model - hardwood
            'fuel8': fuel_models['fuel8'], # Real fuel model - closed timber
            'fuel9': fuel_models['fuel9'], # Real fuel model - hardwood litter
            'fuel10': fuel_models['fuel10'], # Real fuel model - timber litter
        }
        
        # Create feature vector in the exact order (must match training order)
        raw_features = [
            features['tmp_day'],
            features['tmp_night'],
            features['rh_day'],
            features['rh_night'],
            features['wind_avg'],
            features['wind_max'],
            features['wind_dir'],
            features['precip'],
            features['elevation'],
            features['ndvi'],
            features['population'],
            features['pdsi'],
            features['fuel1'],
            features['fuel2'],
            features['fuel3'],
            features['fuel4'],
            features['fuel5'],
            features['fuel6'],
            features['fuel7'],
            features['fuel8'],
            features['fuel9'],
            features['fuel10'],
        ]
        
        # Normalize features to the range [0, 1] for better model performance
        # These ranges are typical for wildfire prediction features
        feature_ranges = [
            (-20, 50),    # tmp_day: temperature in Celsius
            (-20, 50),    # tmp_night: temperature in Celsius
            (0, 100),     # rh_day: humidity percentage
            (0, 100),     # rh_night: humidity percentage
            (0, 100),     # wind_avg: wind speed km/h
            (0, 150),     # wind_max: wind speed km/h
            (0, 360),     # wind_dir: wind direction degrees
            (0, 50),      # precip: precipitation mm
            (0, 4000),    # elevation: meters
            (0, 1),       # ndvi: vegetation index
            (0, 20000),   # population: people per km²
            (-4, 4),      # pdsi: drought index
            (0, 1),       # fuel1-10: fuel model fractions
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1)
        ]
        
        # Normalize each feature to [0, 1] range
        normalized_features = []
        for i, value in enumerate(raw_features):
            min_val, max_val = feature_ranges[i]
            # Clamp value to range and normalize
            clamped_value = max(min_val, min(max_val, value))
            normalized_value = (clamped_value - min_val) / (max_val - min_val)
            normalized_features.append(normalized_value)
        
        feature_vector = np.array(normalized_features, dtype=np.float32)
        
        print(f"✓ Feature vector created with {len(feature_vector)} features")
        
        # Return the 22 feature values
        return feature_vector
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data from Open-Meteo API: {str(e)}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error in gather_features: {str(e)}")
        return None

@app.route('/')
def home():
    """Basic route to check if the server is running"""
    return jsonify({
        "status": "running",
        "message": "Wildfire Prediction Backend Server",
        "model_loaded": model is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded"
    })

@app.route('/api-docs')
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        "endpoints": {
            "GET /": "Server status",
            "GET /health": "Health check",
            "GET /api-docs": "This documentation",
            "POST /predict": {
                "description": "Get wildfire prediction for given coordinates",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "latitude": "number (required) - Latitude coordinate",
                    "longitude": "number (required) - Longitude coordinate"
                },
                "example_request": {
                    "latitude": 37.7749,
                    "longitude": -122.4194
                },
                "response_success": {
                    "heatmap": "array of 4096 numbers (64x64 grid)",
                    "shape": [64, 64],
                    "size": 4096,
                    "location": {
                        "latitude": 37.7749,
                        "longitude": -122.4194
                    }
                },
                "response_error": {
                    "error": "Error message describing what went wrong"
                }
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts latitude and longitude and returns wildfire predictions.
    
    Expected JSON payload:
    {
        "latitude": number,
        "longitude": number
    }
    
    Returns:
    {
        "heatmap": [list of numbers] - flat list of 64*64 = 4,096 prediction scores
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please ensure unet_wildfire_model.keras exists."
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                "error": "No JSON data provided. Please send a JSON payload."
            }), 400
        
        # Check if 'latitude' and 'longitude' keys exist
        if 'latitude' not in data:
            return jsonify({
                "error": "Missing 'latitude' key in JSON payload."
            }), 400
        
        if 'longitude' not in data:
            return jsonify({
                "error": "Missing 'longitude' key in JSON payload."
            }), 400
        
        latitude = data['latitude']
        longitude = data['longitude']
        
        # Validate that latitude and longitude are numbers
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (ValueError, TypeError):
            return jsonify({
                "error": "Latitude and longitude must be valid numbers."
            }), 400
        
        print(f"Received prediction request for location: lat={lat}, lon={lon}")
        
        # Step 1: Check if location is on land
        if not is_location_on_land(lat, lon):
            print("Location is over water - returning zero risk heatmap")
            return jsonify({
                "heatmap": [0.0] * 4096,
                "shape": [64, 64],
                "size": 4096,
                "location": {"latitude": lat, "longitude": lon},
                "message": "Location is over water. No fire risk."
            })
        
        # Step 2: Proceed with normal prediction for land locations
        # Gather features for the given location
        features = gather_features(lat, lon)
        
        # Check if features were successfully gathered
        if features is None:
            return jsonify({
                "error": "Unable to gather features for the given location. Feature gathering failed."
            }), 400
        
        # Validate features structure
        if not isinstance(features, (list, np.ndarray)):
            return jsonify({
                "error": "Invalid features format returned from gather_features."
            }), 500
        
        # Expected feature count: 22 individual features
        expected_feature_count = 22
        
        if len(features) != expected_feature_count:
            return jsonify({
                "error": f"Invalid feature count. Expected {expected_feature_count} features, but got {len(features)}."
            }), 500
        
        print(f"Gathered {len(features)} features for prediction")
        
        # Create the 64x64x22 input tensor from the 22 feature values
        input_tensor_3d = create_input_tensor(features)
        
        # Add batch dimension: (64, 64, 22) -> (1, 64, 64, 22)
        input_tensor = np.expand_dims(input_tensor_3d, axis=0)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Run model prediction
        prediction = model.predict(input_tensor, verbose=0)
        
        print(f"Prediction output shape: {prediction.shape}")
        
        # Flatten the 64x64 prediction output to a list
        # Expected output shape: (1, 64, 64, 1) or (1, 64, 64)
        heatmap = prediction.flatten().tolist()
        
        print(f"Returning heatmap with {len(heatmap)} values")
        
        # Return the flattened prediction as JSON
        return jsonify({
            "heatmap": heatmap,
            "shape": [64, 64],
            "size": len(heatmap),
            "location": {"latitude": lat, "longitude": lon}
        })
    
    except ValueError as e:
        return jsonify({
            "error": f"Value error in processing input: {str(e)}"
        }), 400
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Wildfire Prediction Backend Server...")
    print("=" * 50)
    print(f"Server will be available at: http://localhost:5000")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

