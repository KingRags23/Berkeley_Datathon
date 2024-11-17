import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import MarkerCluster, MiniMap
from scipy.spatial import cKDTree
import math

# Load the data
df = pd.read_csv('/Users/raghavansrinivas/Downloads/2024-11-16.csv', sep=',')

# Convert event_date to datetime and create basic features
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df['year'] = df['event_date'].dt.year
df['month'] = df['event_date'].dt.month
df['day'] = df['event_date'].dt.day

# Fill missing latitude and longitude values
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(df['latitude'].median())
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(df['longitude'].median())

# Create KDTree for efficient spatial queries
coords = np.deg2rad(df[['latitude', 'longitude']].values)
tree = cKDTree(coords)

def get_safe_directions_optimized(current_lat, current_lon, tree, coords, threshold_km=50):
    # Convert threshold to radians
    threshold_rad = threshold_km / 6371.0  # Earth's radius in km
    
    # Convert current point to radians
    current_point = np.array([[np.deg2rad(current_lat), np.deg2rad(current_lon)]])
    
    # Find all points within threshold radius
    nearby_indices = tree.query_ball_point(current_point[0], threshold_rad)
    
    if len(nearby_indices) <= 1:  # If only the point itself is found
        return ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    
    # Calculate bearings to nearby points
    nearby_coords = coords[nearby_indices]
    d_lon = nearby_coords[:, 1] - current_point[0, 1]
    y = np.sin(d_lon) * np.cos(nearby_coords[:, 0])
    x = (np.cos(current_point[0, 0]) * np.sin(nearby_coords[:, 0]) -
         np.sin(current_point[0, 0]) * np.cos(nearby_coords[:, 0]) * np.cos(d_lon))
    bearings = (np.degrees(np.arctan2(y, x)) + 360) % 360
    
    # Define direction ranges
    directions = {
        'north': (337.5, 22.5),
        'northeast': (22.5, 67.5),
        'east': (67.5, 112.5),
        'southeast': (112.5, 157.5),
        'south': (157.5, 202.5),
        'southwest': (202.5, 247.5),
        'west': (247.5, 292.5),
        'northwest': (292.5, 337.5)
    }
    
    safe_directions = []
    for direction, (start_angle, end_angle) in directions.items():
        if start_angle > end_angle:
            has_conflict = np.any((bearings >= start_angle) | (bearings <= end_angle))
        else:
            has_conflict = np.any((bearings >= start_angle) & (bearings <= end_angle))
        
        if not has_conflict:
            safe_directions.append(direction)
    
    return safe_directions

# Encode categorical variables (only those needed for the model)
categorical_columns = ['disorder_type', 'event_type', 'sub_event_type', 'country', 'region', 'admin1', 'location']
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

# Train model
features = ['year', 'month', 'day', 'disorder_type_encoded', 'event_type_encoded', 'sub_event_type_encoded', 
           'country_encoded', 'region_encoded', 'admin1_encoded', 'location_encoded', 'latitude', 'longitude']
target = 'fatalities'

df[features] = df[features].fillna(0)
X = df[features]
y = df[target].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create map
map_center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=5, tiles='OpenStreetMap')
minimap = MiniMap()
m.add_child(minimap)
marker_cluster = MarkerCluster().add_to(m)

# Batch process safety directions
print("Calculating safe directions...")
safe_directions_cache = {}
batch_size = 100  # Process in batches to show progress
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    for _, row in batch.iterrows():
        safe_directions_cache[(row['latitude'], row['longitude'])] = get_safe_directions_optimized(
            row['latitude'], row['longitude'], tree, coords)
    print(f"Processed {min(i + batch_size, len(df))}/{len(df)} locations")

# Add markers with cached safe directions
print("Adding markers to map...")
for _, row in df.iterrows():
    safe_directions = safe_directions_cache.get((row['latitude'], row['longitude']), [])
    safe_directions_text = ", ".join(safe_directions).upper() if safe_directions else "No clear safe directions identified"
    
    popup_html = f"""
    <div style='max-width: 300px'>
        <h4>{row['country']}</h4>
        <p><strong>Location:</strong> {row['location']}</p>
        <p><strong>Date:</strong> {row['event_date'].strftime('%Y-%m-%d')}</p>
        <p><strong>Notes:</strong> {str(row.get('notes', 'No additional notes'))[:100]}...</p>
        <p><strong>Safe Directions:</strong> {safe_directions_text}</p>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html)
    ).add_to(marker_cluster)

# Save the map
print("Saving map...")
m.save('optimized_conflict_zones.html')
print("Map saved as optimized_conflict_zones.html. Open it in a browser to view.")
