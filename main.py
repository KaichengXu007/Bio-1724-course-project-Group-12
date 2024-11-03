import folium
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from esda.moran import Moran
from libpysal.weights import Queen, KNN
from esda.getisord import G
from folium.plugins import HeatMap
import requests
import rasterio
import osmnx as ox
import numpy as np
from shapely.geometry import Point, LineString
from pyproj import Proj, transform
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: Data Collection and Preparation
def collect_and_prepare_data(location="Toronto, Canada", filter_options={}):
    try:
        # Collect walkable paths with accessibility criteria
        G = ox.graph_from_place(location, network_type='walk')
        gdf = ox.graph_to_gdfs(G, nodes=False)

        # Check available columns to see which accessibility data is present
        print("Available columns in OSM data:", gdf.columns)

        # Broad filtering for accessible paths
        filters = []
        if 'wheelchair' in gdf.columns:
            filters.append(gdf['wheelchair'] == 'yes')
        if 'surface' in gdf.columns:
            filters.append(gdf['surface'].isin(['paved', 'asphalt']))
        if 'incline' in gdf.columns:
            filters.append(gdf['incline'].fillna('flat') == 'flat')

        if filters:
            accessible_paths = gdf.loc[filters[0]]
            for condition in filters[1:]:
                accessible_paths = accessible_paths.loc[condition]
        else:
            # If accessibility data is missing, use general walkable paths
            accessible_paths = gdf[gdf['highway'].isin(['footway', 'path', 'pedestrian'])]

        # Collect accessible public transit stops (bus and subway stations)
        transit_stops = ox.geometries_from_place(
            location,
            tags={'public_transport': 'platform', 'wheelchair': 'yes'}
        )
        accessible_bus_stops = transit_stops[(transit_stops['bus'] == 'yes') & (transit_stops['wheelchair'] == 'yes')]
        accessible_subway_stations = transit_stops[
            (transit_stops['railway'] == 'station') & (transit_stops['wheelchair'] == 'yes')]

        # Collect accessible transit routes (e.g., bus lines)
        transit_routes = ox.geometries_from_place(
            location,
            tags={'route': 'bus', 'wheelchair': 'yes'}
        )
        accessible_transit_routes = transit_routes[(transit_routes['wheelchair'] == 'yes')]

        # Display results
        if accessible_paths.empty and accessible_bus_stops.empty and accessible_subway_stations.empty:
            print("No accessible paths or transit routes found with the specified criteria.")
        else:
            print("Accessible walking paths found:", accessible_paths)
            print("Accessible bus stops found:", accessible_bus_stops)
            print("Accessible subway stations found:", accessible_subway_stations)
            print("Accessible transit routes found:", accessible_transit_routes)

        # Spatial indexing for faster operations
        accessible_paths.sindex
        accessible_bus_stops.sindex
        accessible_bus_stops = accessible_bus_stops[
            accessible_bus_stops['geometry'].apply(lambda geom: isinstance(geom, Point))
        ]
        accessible_subway_stations.sindex
        accessible_subway_stations = accessible_subway_stations[
            accessible_subway_stations['geometry'].apply(lambda geom: isinstance(geom, Point))
        ]

        # Return accessible data for visualization and analysis
        return accessible_paths, accessible_bus_stops, accessible_subway_stations, accessible_transit_routes
    except Exception as e:
        print(f"Error during data collection: {e}")
        return None, None, None, None


# Step 2: Data Visualization and Mapping
def visualize_data(accessible_paths, accessible_bus_stops, accessible_subway_stations, accessible_transit_routes):
    try:
        if accessible_paths.empty and accessible_bus_stops.empty and accessible_subway_stations.empty:
            print("No accessible paths or transit stops to visualize.")
            return

        # Static Map Visualization using Matplotlib and Geopandas
        fig, ax = plt.subplots(figsize=(10, 10))
        accessible_paths.plot(ax=ax, color='blue', linewidth=1, label='Accessible Paths')
        accessible_bus_stops.plot(ax=ax, color='green', marker='o', label='Accessible Bus Stops')
        accessible_subway_stations.plot(ax=ax, color='red', marker='^', label='Accessible Subway Stations')
        plt.title('Accessible Paths and Transit Stops in Toronto')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Custom legend with Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Accessible Paths'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Accessible Bus Stops'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
                   label='Accessible Subway Stations')
        ]
        plt.legend(handles=legend_elements)
        plt.show()

        # Interactive map with Folium
        m = folium.Map(location=[43.7, -79.4], zoom_start=12)
        folium.GeoJson(accessible_paths.to_json(), name="Accessible Paths").add_to(m)
        folium.GeoJson(accessible_bus_stops.to_json(), name="Accessible Bus Stops").add_to(m)
        folium.GeoJson(accessible_subway_stations.to_json(), name="Accessible Subway Stations").add_to(m)
        folium.GeoJson(accessible_transit_routes.to_json(), name="Accessible Transit Routes").add_to(m)

        # Add tooltips to bus stops and subway stations
        for _, row in accessible_bus_stops.iterrows():
            folium.Marker(
                [row.geometry.y, row.geometry.x],
                popup=f"Bus Stop: {row.get('name', 'Unknown')}",
                icon=folium.Icon(color="green", icon="bus", prefix="fa")
            ).add_to(m)

        for _, row in accessible_subway_stations.iterrows():
            folium.Marker(
                [row.geometry.y, row.geometry.x],
                popup=f"Subway Station: {row.get('name', 'Unknown')}",
                icon=folium.Icon(color="red", icon="subway", prefix="fa")
            ).add_to(m)

        # Add heatmap for accessible paths
        heat_data = [[point.xy[1][0], point.xy[0][0]] for point in accessible_paths.geometry]
        HeatMap(heat_data).add_to(m)

        folium.LayerControl().add_to(m)
        m.save("accessible_paths_map.html")
        print("Interactive map saved to accessible_paths_map.html")
    except Exception as e:
        print(f"Error during visualization: {e}")


# Step 3: Spatial Descriptive Statistics
def spatial_statistics(accessible_paths):
    try:
        if accessible_paths.empty:
            print("No accessible paths available for spatial statistics.")
            return

        # Spatial Descriptive Statistics
        total_paths = len(accessible_paths)
        print(f"Total accessible paths: {total_paths}")

        path_types = accessible_paths['highway'].value_counts()
        print("Path types distribution:", path_types)

        # Spatial Autocorrelation (Moran’s I)
        W = Queen.from_dataframe(accessible_paths, use_index=True)  # Fix for FutureWarning
        highway_values = accessible_paths['highway'].apply(lambda x: 1 if x in ['footway', 'path'] else 0).values

        moran = Moran(highway_values, W)
        print("Spatial Autocorrelation (Moran's I):", moran.I)
        print("P-value:", moran.p_sim)

        # Handling Disconnected Components with KNN if needed
        if len(W.islands) > 0:
            print(f"Warning: {len(W.islands)} disconnected components found. Switching to KNN.")
            W = KNN.from_dataframe(accessible_paths, k=5)  # Use KNN for connectedness
            moran_knn = Moran(highway_values, W)
            print("Spatial Autocorrelation (Moran's I) with KNN:", moran_knn.I)
            print("P-value with KNN:", moran_knn.p_sim)

        # Sample a smaller subset for Getis-Ord G calculation
        sample_size = min(len(accessible_paths), 5000)
        accessible_paths_sample = accessible_paths.sample(n=sample_size)
        W_sample = Queen.from_dataframe(accessible_paths_sample, use_index=True)

        highway_values_sample = accessible_paths_sample['highway'].apply(
            lambda x: 1 if x in ['footway', 'path'] else 0
        ).values

        # Calculating the Getis-Ord G statistic
        g_stat_sample = G(highway_values_sample, W_sample)
        g_value = g_stat_sample.G if hasattr(g_stat_sample, 'G') else None
        print("Getis-Ord G statistic (sample):", g_value)

        # Access the statistic or print directly if `Zs` is not available
        save_spatial_statistics_report(total_paths, path_types, moran.I, moran.p_sim, g_value)

    except Exception as e:
        print(f"Error during spatial statistics: {e}")

# Step 4: Save Spatial Statistics to a File
def save_spatial_statistics_report(total_paths, path_types, moran_value, moran_p_value, g_stat_sample):
    try:
        with open('spatial_statistics_report.txt', 'w') as f:
            f.write(f"Total accessible paths: {total_paths}\n")
            f.write(f"Path types distribution:\n{path_types}\n")
            f.write(f"Spatial Autocorrelation (Moran's I): {moran_value}\n")
            f.write(f"P-value: {moran_p_value}\n")
            f.write(f"Getis-Ord G statistic (sample): {g_stat_sample}\n")
        print("Spatial statistics report saved to 'spatial_statistics_report.txt'")
    except Exception as e:
        print(f"Error saving report: {e}")

# Step 6: Routing Algorithm (Stub Example using Dijkstra’s)
def route_planning(G, orig_node, dest_node):
    try:
        # Find the shortest path using Dijkstra's algorithm
        route = ox.shortest_path(G, orig_node, dest_node, weight='length')
        return route
    except Exception as e:
        print(f"Error in route planning: {e}")
        return None

# Step 7: Calculate Accessibility Scoring Based on Slope
wgs84 = Proj("epsg:4326")  # WGS84
utm17n = Proj("epsg:32617")  # UTM Zone 17N
def calculate_slope_from_hgt(hgt_file, accessible_paths):
    try:
        # Open HGT file to read elevation data
        with rasterio.open(hgt_file) as src:
            slope_scores = []  # List to store slope values

            # Iterate over each accessible path to calculate slope
            for _, path in accessible_paths.iterrows():
                if isinstance(path.geometry, LineString):
                    coords = list(path.geometry.coords)
                    elevation_diff = []

                    for i in range(1, len(coords)):
                        lon1, lat1 = coords[i - 1]
                        lon2, lat2 = coords[i]

                        # Convert coordinates to UTM for precise distance calculation
                        x1, y1 = transform(wgs84, utm17n, lon1, lat1)
                        x2, y2 = transform(wgs84, utm17n, lon2, lat2)

                        # Get elevation at two points
                        row1, col1 = src.index(lon1, lat1)
                        row2, col2 = src.index(lon2, lat2)
                        elevation1 = src.read(1)[row1, col1]
                        elevation2 = src.read(1)[row2, col2]

                        if np.isnan(elevation1) or np.isnan(elevation2):
                            continue  # Skip if elevation data is missing

                        # Calculate horizontal distance and elevation change
                        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Distance in meters
                        elevation_change = elevation2 - elevation1
                        slope = (elevation_change / distance) * 100  # Slope in percentage

                        elevation_diff.append(slope)

                    # Average slope for the path
                    average_slope = np.mean(elevation_diff) if elevation_diff else 0
                    slope_scores.append(average_slope)
                else:
                    slope_scores.append(0)

            # Add slope scores to accessible_paths DataFrame
            accessible_paths['slope_score'] = slope_scores
            return accessible_paths

    except Exception as e:
        print(f"Error during slope calculation: {e}")
        return accessible_paths

# Accessibility Scoring Function
def accessibility_scoring(accessible_paths):
    # Example scoring based on slope
    accessible_paths['accessibility_score'] = accessible_paths['slope_score'].apply(
        lambda x: 5 if x <= 12 else 1
    )
    return accessible_paths

# Barrier and Obstruction Analysis
def analyze_barriers_and_obstructions(accessible_paths):
    # Identify paths with high slope as potential barriers
    barriers = accessible_paths[accessible_paths['slope_score'] > 12]  # Threshold for "steep" paths
    print(f"Found {len(barriers)} paths with steep slopes.")
    return barriers

# Visualization Function
def visualize_accessibility_scoring_and_barriers(accessible_paths, barriers):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define a custom colormap to ensure that high slope areas (barriers) are red
    cmap = mcolors.ListedColormap(['yellow', 'orange', 'red'])
    bounds = [1, 4, 12, 20]  # Define boundaries for the color map
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot accessibility score with the custom color map
    accessible_paths.plot(ax=ax, column='slope_score', legend=True, cmap=cmap, norm=norm)

    # Plot barriers in red color explicitly
    barriers.plot(ax=ax, color='red', marker='x', label='Barriers (Steep Slope)', linewidth=0.5)

    # Adding a legend for barriers
    plt.legend()
    plt.title("Accessibility Scoring and Barriers")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical')

    plt.show()

# Fetch Incident Data from TomTom API
def fetch_incident_data(api_key):
    bbox = "-79.639319,43.581024,-79.115102,43.855457"
    fields = "{incidents{type,geometry{type,coordinates},properties{iconCategory}}}"
    language = "en-US"
    category_filter = "0,1,2,3,4,5,6,7,8,9,10,11,14"
    time_validity_filter = "present"

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        "bbox": bbox,
        "fields": fields,
        "language": language,
        "categoryFilter": category_filter,
        "timeValidityFilter": time_validity_filter,
        "key": api_key
    }

    response = requests.get(url, params=params)
    incidents = []
    if response.status_code == 200:
        data = response.json()
        incidents = data.get("incidents", [])
    else:
        print("Failed to fetch incident data:", response.status_code, response.text)

    return incidents


# Integrate Incident Data into Visualization
def visualize_incidents_on_map(map_obj, incidents):
    category_mapping = {
        0: "Unknown",
        1: "Accident",
        2: "Fog",
        3: "Dangerous Conditions",
        4: "Rain",
        5: "Ice",
        6: "Jam",
        7: "Lane Closed",
        8: "Road Closed",
        9: "Road Works",
        10: "Wind",
        11: "Flooding",
        14: "Broken Down Vehicle"
    }

    for incident in incidents:
        icon_category = incident["properties"].get("iconCategory", 0)
        event_type = category_mapping.get(icon_category, "Unknown")
        coordinates = incident["geometry"].get("coordinates", [None, None])

        # Handle nested list scenario and ensure coordinates are two floats
        if isinstance(coordinates[0], list):
            # Extract the first element if coordinates are nested as [[lon, lat]]
            coordinates = coordinates[0]

        try:
            lon, lat = float(coordinates[0]), float(coordinates[1])
            folium.Marker(
                location=[lat, lon],  # latitude, longitude as floats
                popup=f"Event: {event_type}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(map_obj)
        except (TypeError, ValueError, IndexError) as e:
            print(f"Skipping invalid coordinates for incident: {coordinates}, Error: {e}")


# Step: Safety Score Calculation
def calculate_safety_scores(accessible_paths, incidents, slope_weight=0.3, barrier_weight=0.5, incident_weight=0.2):
    # Check if weights sum up to 1.0
    if slope_weight + barrier_weight + incident_weight != 1.0:
        raise ValueError("The weights must sum to 1.0.")

    # Define severity scores for each incident type
    incident_severity = {
        "Unknown": 1,
        "Accident": 5,
        "Fog": 3,
        "Dangerous Conditions": 4,
        "Rain": 2,
        "Ice": 4,
        "Jam": 2,
        "Lane Closed": 3,
        "Road Closed": 5,
        "Road Works": 3,
        "Wind": 2,
        "Flooding": 4,
        "Broken Down Vehicle": 2
    }

    # Iterate through paths and calculate a safety score for each path
    safety_scores = []
    for index, path in accessible_paths.iterrows():
        slope_score = 5 if path['slope_score'] < 5 else (3 if path['slope_score'] < 10 else 1)
        barrier_score = 1 if path['slope_score'] > 12 else 5  # Assume a simple barrier score based on slope threshold

        # Incident impact score
        incident_score = 5  # Default safe if no incident
        path_centroid = path.geometry.centroid

        for incident in incidents:
            coordinates = incident.get('geometry', {}).get('coordinates', [None, None])

            # Handle cases where coordinates are nested lists (e.g., LineString or MultiPoint)
            if isinstance(coordinates[0], list):
                for coord_pair in coordinates:
                    if len(coord_pair) == 2:
                        try:
                            lon, lat = float(coord_pair[0]), float(coord_pair[1])
                            incident_location = Point(lon, lat)
                            distance_to_incident = path_centroid.distance(incident_location)

                            if distance_to_incident < 0.005:  # Threshold distance for incidents (e.g., 500 meters)
                                incident_type = incident.get('properties', {}).get('iconCategory', "Unknown")
                                severity = incident_severity.get(incident_type, 1)  # Default to 1 for unknown types
                                incident_score = min(incident_score, max(5 - severity, 1))  # Higher severity lowers the score
                        except (TypeError, ValueError):
                            # Skip if coordinates are invalid
                            continue
            else:
                if len(coordinates) == 2:
                    try:
                        lon, lat = float(coordinates[0]), float(coordinates[1])
                        incident_location = Point(lon, lat)
                        distance_to_incident = path_centroid.distance(incident_location)

                        if distance_to_incident < 0.005:  # Threshold distance for incidents (e.g., 500 meters)
                            incident_type = incident.get('properties', {}).get('iconCategory', "Unknown")
                            severity = incident_severity.get(incident_type, 1)  # Default to 1 for unknown types
                            incident_score = min(incident_score, max(5 - severity, 1))  # Higher severity lowers the score
                    except (TypeError, ValueError):
                        # Skip if coordinates are invalid
                        continue

        # Calculate weighted safety score
        safety_score = (slope_weight * slope_score) + (barrier_weight * barrier_score) + (incident_weight * incident_score)
        safety_scores.append(safety_score)

    # Add safety scores to accessible_paths DataFrame
    accessible_paths['safety_score'] = safety_scores
    return accessible_paths

# Visualization Function
def visualize_safety_scores(accessible_paths):
    fig, ax = plt.subplots(figsize=(10, 10))
    accessible_paths.plot(ax=ax, column='safety_score', legend=True, cmap='coolwarm')
    plt.title("Safety Scores of Accessible Paths")
    plt.show()

# Step 5: Real-Time Data Integration (Stub Example)
# def get_real_time_transit_data(api_url):
#     try:
#         response = requests.get(api_url)
#         return response.json()  # Placeholder for real-time transit data
#     except Exception as e:
#         print(f"Error fetching real-time transit data: {e}")
#         return None

# Main block to execute functions
if __name__ == '__main__':
    location = "Toronto, Canada"  # Define the location to analyze
    #api_url = "https://myttc.ca/finch_station.json"  # Replace with actual API URL for real-time data
    hgt_file = "C:/Users/xukai/PycharmProjects/Bio_1724_A2/N43W080.hgt"
    api_key = "ZeoUu9MSUbclmIura6m79Z1pq4pk9bNd"

    # Step 1: Collect and prepare data
    accessible_paths, accessible_bus_stops, accessible_subway_stations, accessible_transit_routes = collect_and_prepare_data(location)
    #
    if accessible_paths.empty and accessible_bus_stops.empty and accessible_subway_stations.empty:
        print("No accessible paths or transit stops were found after data collection.")
    else:
        # Step 2: Perform spatial statistics
        spatial_statistics(accessible_paths)
        # Step 3: Visualize accessible paths
        visualize_data(accessible_paths, accessible_bus_stops, accessible_subway_stations, accessible_transit_routes)

    accessible_paths = calculate_slope_from_hgt(hgt_file, accessible_paths)
    # Step 4: Perform accessibility scoring
    accessible_paths = accessibility_scoring(accessible_paths)
    # Step 5: Analyze barriers and obstructions
    barriers = analyze_barriers_and_obstructions(accessible_paths)
    print("Barrier detection output:", barriers)
    # Step 6: Visualize accessibility scoring and barriers
    visualize_accessibility_scoring_and_barriers(accessible_paths, barriers)

    # Step 8: Route planning (Example usage)
    # G = ox.graph_from_place(location, network_type='walk')  # Create graph for routing
    # orig_node = ox.distance.nearest_nodes(G, X=-79.3832, Y=43.6532)  # Replace with actual origin coordinates
    # dest_node = ox.distance.nearest_nodes(G, X=-79.3932, Y=43.6532)  # Replace with actual destination coordinates
    # route = route_planning(G, orig_node, dest_node)
    # if route:
    #     print(f"Optimal route: {route}")
    #
    # # Example visualization of the route
    # ox.plot_graph_route(G, route, route_linewidth=2, node_size=0)

    toronto_map = folium.Map(location=[43.7, -79.4], zoom_start=12)

    # Step 8: Fetch Incident Data and Visualize
    incidents = fetch_incident_data(api_key)
    visualize_incidents_on_map(toronto_map, incidents)

    # Step 9: Calculate safety scores
    accessible_paths = calculate_safety_scores(accessible_paths, incidents, slope_weight=0.3, barrier_weight=0.5, incident_weight=0.2)

    #  Step 10: Visualize safety scores**
    visualize_safety_scores(accessible_paths)

    # Save or display the map
    toronto_map.save("Toronto_Accessibility_Map_with_Incidents.html")