import folium
import requests
import random
import heapq
from itertools import count
from shapely.geometry import Point, LineString
import logging
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define severity scores for incidents
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


def yen_k_shortest_paths(start_node, end_node, edges_by_node, k=5):
    """
    Find the K shortest paths between the start and end nodes using Yen's algorithm.
    Args:
        start_node (int): The starting node.
        end_node (int): The ending node.
        edges_by_node (dict): Dictionary containing edges connected to each node.
        k (int): The number of shortest paths to find.
    Returns:
        list: A list of the K shortest distinct paths found.
    """
    logging.info("Starting Yen's algorithm to find candidate paths.")
    # Step 1: Find the first shortest path using Dijkstra's algorithm
    first_path = dijkstra_path(start_node, end_node, edges_by_node)
    if not first_path:
        logging.info("No path found between start and end nodes.")
        return []  # No path exists

    shortest_paths = [first_path]
    potential_paths = []
    visited_paths = set()
    visited_paths.add(tuple((edge['properties']['u'], edge['properties']['v']) for edge in first_path))

    for k_i in range(1, k):
        logging.info(f"Finding path {k_i + 1}...")
        # The previous shortest path to base deviations on
        prev_shortest = shortest_paths[-1]

        for i in range(len(prev_shortest)):
            # Step 2: Generate a potential path by deviating from the previous path
            spur_node = prev_shortest[i]['properties']['u']
            root_path = prev_shortest[:i]

            # Temporarily remove edges in the root path to create more deviation
            removed_edges = []
            for path in shortest_paths:
                if path[:i] == root_path and len(path) > i:
                    removed_edge = path[i]
                    if removed_edge in edges_by_node.get(removed_edge['properties']['u'], []):
                        edges_by_node[removed_edge['properties']['u']].remove(removed_edge)
                        removed_edges.append(removed_edge)

            # Find the spur path from spur_node to end_node
            spur_path = dijkstra_path(spur_node, end_node, edges_by_node)

            # If a spur path exists, concatenate it with the root path
            if spur_path:
                potential_path = root_path + spur_path
                path_tuple = tuple((edge['properties']['u'], edge['properties']['v']) for edge in potential_path)
                if path_tuple not in visited_paths:
                    path_cost = sum(edge['properties']['length'] for edge in potential_path)
                    potential_paths.append((path_cost, potential_path))
                    visited_paths.add(path_tuple)
                else:
                    logging.info(f"Duplicate path found and skipped: {path_tuple}")

            # Restore removed edges
            for edge in removed_edges:
                edges_by_node[edge['properties']['u']].append(edge)

        # If no more potential paths exist, break
        if not potential_paths:
            logging.info("No more potential paths available. Breaking out of the loop.")
            break

        # Sort potential paths by cost and select the best one
        potential_paths.sort(key=lambda x: x[0])  # Sort by cost
        next_path = potential_paths.pop(0)[1]
        shortest_paths.append(next_path)

    if len(shortest_paths) < k:
        logging.info(f"Only {len(shortest_paths)} distinct paths found out of requested {k}.")

    logging.info(f"Yen's algorithm completed. Found {len(shortest_paths)} distinct paths.")
    output_paths_to_file(shortest_paths, start_node, end_node)
    return shortest_paths

def output_paths_to_file(paths, start_node, end_node):
    """
    Output the found paths to a text file.
    Args:
        paths (list): List of paths found.
        start_node (int): The starting node.
        end_node (int): The ending node.
    """
    with open('k_shortest_paths_output.txt', 'w') as file:
        for idx, path in enumerate(paths):
            file.write(f"Path {idx + 1}:\n")
            description = []
            for edge in path:
                u = edge['properties'].get('u', 'unknown')
                v = edge['properties'].get('v', 'unknown')
                highway = edge['properties'].get('highway', '').lower()
                mode = "walking" if highway in ['pedestrian', 'footway', 'path'] else "public transit" if highway in ['tram', 'subway'] else "other"
                description.append(f"{u} -> {v} (using {mode})")
            path_description = f"{' -> '.join(description)}"
            file.write(f"{path_description}\n")
            file.write("\n")
    logging.info("Paths have been written to k_shortest_paths_output.txt")


def dijkstra_path(start_node, end_node, edges_by_node):
    """
    Find the shortest path between the start and end nodes using Dijkstra's algorithm.
    Args:
        start_node (int): The starting node.
        end_node (int): The ending node.
        edges_by_node (dict): Dictionary containing edges connected to each node.
    Returns:
        list: The shortest path found as a list of edges.
    """
    counter = count()  # Unique tie-breaker
    queue = []
    heapq.heappush(queue, (0, next(counter), start_node, []))  # (cost, tie-breaker, current_node, path_so_far)
    visited = set()

    while queue:
        cost, _, node, path = heapq.heappop(queue)

        # If we reach the end node, return the path
        if node == end_node:
            return path

        # Skip already visited nodes
        if node in visited:
            continue

        visited.add(node)

        # Explore neighbors
        for edge in edges_by_node.get(node, []):
            try:
                next_node = edge['properties']['v']
                edge_cost = edge['properties']['length']
                if next_node not in visited:
                    new_cost = cost + edge_cost
                    heapq.heappush(queue, (new_cost, next(counter), next_node, path + [edge]))
            except KeyError as ke:
                continue

    return []  # No path found

def calculate_segment_score(segment, real_time_data, weight_slope=0.5, weight_incident=0.5):
    """
    Calculate the safety score for a given segment based on slope and real-time incident data.
    Args:
        segment (dict): The edge segment to calculate the score for.
        real_time_data (list): Real-time incident data.
        weight_slope (float): Weight assigned to the slope score.
        weight_incident (float): Weight assigned to the incident score.
    Returns:
        float: The safety score for the segment.
    """
    if segment['geometry'] and isinstance(segment['geometry'], LineString):
        slope_score = segment['properties'].get('safety_score', 5)  # Default to 5 if not found
        incident_score = fetch_incident_score(segment, real_time_data)
        return (weight_slope * slope_score) + (weight_incident * incident_score)
    return 0


def calculate_safety_score(path, real_time_data, weight_slope=0.5, weight_incident=0.5):
    """
    Calculate the safety score for a given path based on slope and real-time incident data.
    Args:
        path (list): List of edges representing the path.
        real_time_data (list): Real-time incident data.
        weight_slope (float): Weight assigned to the slope score.
        weight_incident (float): Weight assigned to the incident score.
    Returns:
        float: The average safety score for the path.
    """
    safety_scores = []
    logging.info("Calculating safety scores in parallel...")

    # Calculate scores for all segments in parallel, only for walking segments
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_segment_score, segment, real_time_data, weight_slope, weight_incident)
                   for segment in path if segment['properties'].get('highway', '').lower() in ['pedestrian', 'footway', 'path']]
        for future in as_completed(futures):
            safety_scores.append(future.result())

    return sum(safety_scores) / len(safety_scores) if safety_scores else 0  # Average safety score


def fetch_incident_data(api_key):
    """
    Fetch real-time incident data using the provided API key.
    Args:
        api_key (str): The API key to access the incident data.
    Returns:
        list: A list of incidents with relevant details.
    """
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


def fetch_incident_score(segment, real_time_data):
    """
    Calculate the incident score for a segment based on real-time data.
    Args:
        segment (dict): A dictionary representing an edge with geometry and properties.
        real_time_data (list): List of real-time incidents.
    Returns:
        float: The incident score for the segment.
    """
    path_centroid = segment['geometry'].centroid
    cumulative_incident_severity = 0
    for incident in real_time_data:
        try:
            coordinates = incident.get('geometry', {}).get('coordinates', [None, None])
            if isinstance(coordinates[0], list):
                coordinates = coordinates[0]
            lon, lat = float(coordinates[0]), float(coordinates[1])
            incident_location = Point(lon, lat)
            distance_to_incident = path_centroid.distance(incident_location)
            if distance_to_incident < 0.005:  # Within 500 meters
                severity = incident_severity.get(incident.get('properties', {}).get('iconCategory', "Unknown"), 1)
                cumulative_incident_severity += severity
        except (TypeError, ValueError, IndexError):
            continue
    max_severity = len(real_time_data) * max(incident_severity.values())
    return max(5 - (5 * (cumulative_incident_severity / max_severity)), 1)


def genetic_algorithm(candidate_paths, generations=100):
    """
    Apply a genetic algorithm to optimize the candidate paths.
    Args:
        candidate_paths (list): List of candidate paths to start the optimization.
        generations (int): Number of generations to run the genetic algorithm.
    Returns:
        list: The optimal path found after the given number of generations.
    """
    logging.info("Starting genetic algorithm for path optimization...")
    if not candidate_paths:
        logging.error("No candidate paths provided for genetic algorithm.")
        return []

    population = random.choices(candidate_paths, k=len(candidate_paths))
    if not population:
        logging.error("Initial population is empty. Returning an empty path.")
        return []

    for generation in range(generations):
        logging.info(f"Generation {generation + 1} of {generations}...")
        with ThreadPoolExecutor() as executor:
            fitness_scores = list(executor.map(fitness_function, population))
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda item: item[0], reverse=True)]
        population = sorted_population[:max(len(candidate_paths) // 2, 1)]  # Ensure at least one path remains in the population
        new_population = []
        for _ in range(len(candidate_paths) // 2):
            if len(population) < 2:
                break
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))
        population.extend(new_population)

    logging.info("Genetic algorithm completed.")
    if population:
        return max(population, key=fitness_function)
    else:
        logging.error("Population is empty after genetic algorithm. Returning an empty path.")
        return []


# Update the fitness function to consider different travel methods
def fitness_function(path):
    """
    Calculate the fitness score of a path based on travel time, safety, and mode transitions.
    Args:
        path (list): List of edges representing the path.
    Returns:
        float: The fitness score for the path.
    """
    time_cost = calculate_time_cost(path)
    # safety_score = calculate_safety_score(path, fetch_incident_data(api_key))
    safety_score = 1
    transitions = count_mode_transitions(path)
    return (0.6 * -time_cost) + (0.3 * safety_score) + (0.1 * -transitions)


def calculate_time_cost(path):
    """
    Calculate the total time cost for a given path.
    Args:
        path (list): List of edges representing the path.
    Returns:
        float: The total time cost for the path.
    """
    total_time_cost = 0
    for edge in path:
        length = edge['properties']['length']
        mode = edge['properties'].get('highway', '').lower()

        if mode in ['tram', 'subway']:
            speed = 30  # Assume an average speed of 30 km/h for tram/subway
        elif mode in ['pedestrian', 'footway', 'path']:
            speed = 5  # Assume an average speed of 5 km/h for walking
        else:
            speed = 10  # Default to 10 km/h for other unspecified travel modes

        time_cost = length / (speed * 1000 / 60)  # Convert speed to meters per minute
        total_time_cost += time_cost
    return total_time_cost


def count_mode_transitions(path):
    """
    Count the number of mode transitions in a given path.
    Args:
        path (list): List of edges representing the path.
    Returns:
        int: The number of mode transitions in the path.
    """
    mode_transitions = 0
    for i in range(1, len(path)):
        prev_mode = path[i - 1]['properties'].get('highway', '').lower()
        curr_mode = path[i]['properties'].get('highway', '').lower()
        if (prev_mode in ['pedestrian', 'footway', 'path'] and curr_mode in ['tram', 'subway']) or \
                (prev_mode in ['tram', 'subway'] and curr_mode in ['pedestrian', 'footway', 'path']):
            mode_transitions += 1
    return mode_transitions


def crossover(parent1, parent2):
    """
    Perform crossover between two parent paths to create a child path.
    Args:
        parent1 (list): The first parent path.
        parent2 (list): The second parent path.
    Returns:
        list: The child path resulting from the crossover.
    """
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    return parent1[:split_point] + parent2[split_point:]


def mutate(path, mutation_rate=0.1):
    """
    Apply mutation to a given path to introduce diversity.
    Args:
        path (list): The path to mutate.
        mutation_rate (float): Probability of mutation occurring.
    Returns:
        list: The mutated path.
    """
    if random.random() < mutation_rate:
        idx = random.randint(0, len(path) - 1)
        path[idx]['properties']['length'] *= random.uniform(0.9, 1.1)
    return path


def load_edges_from_geojson(file_path):
    """
    Load edges from a GeoJSON file into a graph-like dictionary structure.
    Args:
        file_path (str): Path to the GeoJSON file.
    Returns:
        dict: A dictionary where keys are node IDs and values are lists of edges.
    """
    logging.info("Loading edges from GeoJSON file...")
    gdf = gpd.read_file(file_path)
    edges_by_node = {}
    for _, row in gdf.iterrows():
        u, v, length = row['u'], row['v'], row['length']
        geometry = row['geometry']
        edge = {'properties': {'u': u, 'v': v, 'length': length, 'highway': row.get('highway', 'other')}, 'geometry': geometry}
        edges_by_node.setdefault(u, []).append(edge)
    logging.info("Edges loaded successfully.")
    return edges_by_node

def visualize_candidate_paths(candidate_paths, map_obj):
    """
    Visualize candidate paths on the given map object.
    Args:
        candidate_paths (list): List of candidate paths to visualize.
        map_obj (folium.Map): The map object to add paths to.
    """
    logging.info("Visualizing candidate paths...")
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Define different colors for candidate paths
    for idx, path in enumerate(candidate_paths[:20]):  # Visualize up to 20 paths
        color = colors[idx % len(colors)]
        path_coordinates = []

        for edge in path:
            if edge['geometry'] is None or not hasattr(edge['geometry'], 'coords'):
                logging.debug(f"Edge from node {edge['properties']['u']} to node {edge['properties']['v']} has no valid geometry. Skipping this edge.")
                continue
            path_coordinates.extend(edge['geometry'].coords)

        if path_coordinates:
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in path_coordinates],
                color=color,
                weight=3,
                opacity=0.7
            ).add_to(map_obj)

    logging.info("Candidate paths visualization completed.")



def visualize_incidents_on_map(map_obj, incidents):
    """
    Visualize incidents on the given map object.
    Args:
        map_obj (folium.Map): The map object to add incidents to.
        incidents (list): List of incidents to be visualized.
    """
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

def visualize_optimal_path(optimal_path, map_obj):
    """
    Visualize the optimal path on the given map object and mark the nodes.
    Args:
        optimal_path (list): List of edges representing the optimal path.
        map_obj (folium.Map): The map object to add the optimal path to.
    """
    logging.info("Visualizing optimal path...")
    for edge in optimal_path:
        if edge['geometry'] is None or not hasattr(edge['geometry'], 'coords'):
            logging.debug(f"Edge from node {edge['properties']['u']} to node {edge['properties']['v']} has no valid geometry. Skipping this edge.")
            continue

        folium.PolyLine(
            locations=[(coord[1], coord[0]) for coord in edge['geometry'].coords],  # Note: coord[1] for latitude, coord[0] for longitude
            color='blue',
            weight=4,
            opacity=0.8
        ).add_to(map_obj)


    logging.info("Optimal path visualization completed.")

def output_optimal_path_to_file(optimal_path):
    """
    Output the optimal path found by the genetic algorithm to a text file.
    Args:
        optimal_path (list): The optimal path found as a list of edges.
    """
    with open('optimal_path_output.txt', 'w') as file:
        description = []
        for edge in optimal_path:
            u = edge['properties'].get('u', 'unknown')
            v = edge['properties'].get('v', 'unknown')
            highway = edge['properties'].get('highway', '').lower()
            mode = "walking" if highway in ['pedestrian', 'footway', 'path'] else "public transit" if highway in ['tram', 'subway'] else "other"
            description.append(f"{u} -> {v} (using {mode})")
        path_description = f"{' -> '.join(description)}"
        file.write(f"{path_description}")
    logging.info("Optimal path has been written to optimal_path_output.txt")


location = "Toronto, Canada"
api_key = "ZeoUu9MSUbclmIura6m79Z1pq4pk9bNd"
file_path = "connected_network_cleaned.geojson"
edges_by_node = load_edges_from_geojson(file_path)

start_node = 1490121868
end_node = 540591304

toronto_map = folium.Map(location=[43.7, -79.42], zoom_start=12)

# Yen
candidate_paths = yen_k_shortest_paths(start_node, end_node, edges_by_node, k=5)

visualize_candidate_paths(candidate_paths, toronto_map)

toronto_map.save("candidate_paths.html")
# toronto_map.save("best_paths_ACO.ver.11.27.html")

optimal_path = genetic_algorithm(candidate_paths)

output_optimal_path_to_file(optimal_path)

# visualize_incidents_on_map(toronto_map, fetch_incident_data(api_key))
# toronto_map_optimal = folium.Map(location=[43.7, -79.42], zoom_start=12)

# visualize_optimal_path(optimal_path, toronto_map)

toronto_map.save("optimal_path_GA.html")