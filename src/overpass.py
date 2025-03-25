from shapely.geometry import shape
from openai import OpenAI

import geopandas as gpd
import numpy as np
import google.generativeai as genai

import math
import requests
import geojson
import json
import copy
import os


# CONSTANTS
overpass_url = "http://overpass-api.de/api/interpreter"
data_dir = "data/"
features_dir = data_dir + "features/"
config_dir = data_dir + "config/"
prompts_dir = data_dir + "prompts/"
output_prefix = "features"

# GEMINI API KEY AND MODEL NAME
MODEL_NAME = "gemini-2.0-flash"
with open("mykeys/gemini_api_key.txt", "r") as f:
    GEMINI_API_KEY = f.read()

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# SYSTEM PROMPTS
with open(prompts_dir + "name_features.txt", "r") as f:
    SYSTEM_PROMPT_NAMES = f.read()
with open(prompts_dir + "assign_colors_to_names.txt", "r") as f:
    SYSTEM_PROMPT_COLORS = f.read()

# FUNCTIONS
def get_features_descriptive_string(feature_collection_no_coords):
    """Generates a descriptive string for the features in the GeoJSON."""
    feature_descriptions = []
    for feature in feature_collection_no_coords["features"]:
        feature_str = ""
        feature_id = feature.get("id", "Unknown")
        feature_str = f"{feature_id}: {{"
        properties = feature.get("properties", {})
        for key, value in properties.items():
            feature_str += f"{key}: {value}, "
        feature_str = feature_str.rstrip(", ") + "}"
        feature_descriptions.append(feature_str)
    return "\n".join(feature_descriptions)

def name_features(geojson_str_no_coords, feature_collection):
    """Names features in GeoJSON using Google Gemini API."""

    # Generation config 
    # Set the model configuration
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "application/json",
    }

    # Create the names model
    model_names = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT_NAMES
    )

    # Generate autotest completion using Gemini API
    completion = model_names.generate_content(geojson_str_no_coords)

    try:
        response_content = completion.text
        names_mapped_to_feature_ids = json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response text: {completion.text}") #print the raw response for debugging.
        return feature_collection #Return the original feature collection.

    # Flip the dictionary to map feature IDs to names
    feature_ids_mapped_to_names = {str(v): k for k, v in names_mapped_to_feature_ids.items()}

    # For each entry in the names_mapped_to_feature_ids dictionary, add the unique name to the feature properties
    for feature in feature_collection["features"]:
        if str(feature["id"]) in feature_ids_mapped_to_names:
            feature["properties"]["unique_name"] = feature_ids_mapped_to_names[str(feature["id"])]

    # Get list of unique names
    unique_names_list = list(names_mapped_to_feature_ids.keys())
    # Iterate and keep list of different words before the first "_"
    unique_name_types_list = [name.split("_")[0] for name in unique_names_list]
    # Remove duplicates
    unique_name_types_list = list(set(unique_name_types_list))
    unique_name_types_list_str = ", ".join(unique_name_types_list)

    # Create the color model
    model_colors = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT_COLORS
    )

    # Generate colors for unique names using Gemini API
    completion = model_colors.generate_content(unique_name_types_list_str)
    try:
        response_content = completion.text
        colors_mapped_to_names = json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response text: {completion.text}") #print the raw response for debugging.
        return feature_collection #Return the original feature collection.
    
    # Add colors to features
    for feature in feature_collection["features"]:
        if "unique_name" in feature["properties"]:
            unique_name_type = feature["properties"]["unique_name"].split("_")[0]
            # Check if the unique name is in the colors mapping
            if unique_name_type in colors_mapped_to_names:
                feature["properties"]["color"] = colors_mapped_to_names[unique_name_type]

    # Return the updated feature collection
    return feature_collection

def overpass_to_geojson_featurecollection(overpass_json):
    """Converts Overpass JSON to GeoJSON FeatureCollection, creating polygons when possible."""
    features = []
    for element in overpass_json['elements']:
        if 'geometry' in element:
            coords = [(coord['lon'], coord['lat']) for coord in element['geometry']]
            if len(coords) > 2 and coords[0] == coords[-1]:  # Check for polygon
                geometry = geojson.Polygon([coords])
            else:
                geometry = geojson.LineString(coords)
            properties = element.get('tags', {})
            feature = geojson.Feature(geometry=geometry, properties=properties, id=element['id'])
            features.append(feature)
    
    # Re-order features based on centroids, using y coordinate as 1st key (descending),
    # and x coordinate as 2nd key (ascending)
    features.sort(key=lambda f: (
        -shape(f['geometry']).centroid.y,  # y coordinate descending
        shape(f['geometry']).centroid.x    # x coordinate ascending
    ))

    return geojson.FeatureCollection(features)

def simplify_geojson(feature_collection):
    """Replaces coordinates with count and centroid, simplifying the GeoJSON."""
    feature_collection_no_coords = copy.deepcopy(feature_collection)
    for feature in feature_collection_no_coords["features"]:
        if "geometry" in feature and "coordinates" in feature["geometry"]:
            coordinates = feature["geometry"]["coordinates"]
            
            # Count coordinates (handles both Point, LineString, Polygon, etc.)
            if isinstance(coordinates, list):
                num_coords = sum(len(part) for part in coordinates) if isinstance(coordinates[0], list) else 1
            else:
                num_coords = 1  # Fallback in case of unexpected data

            # Remove type key
            if "type" in feature:
                del feature["type"]
            
            # Remove geometry key and replace it by geometry_type and geometry_n_coords
            if "geometry" in feature:
                geometry = feature["geometry"]
                feature["geometry_type"] = geometry["type"]
                feature["geometry_n_coords"] = num_coords
                feature["geometry_centroid"] = shape(geometry).centroid.coords[0]
                del feature["geometry"]

    return feature_collection_no_coords

def get_geojson_features(bbox, config_name=""):
    """
    Fetches Overpass features and saves them as GeoJSON. 
    If config_name is provided, it will be used to fetch hardcoded features.
    """

    # Overpass Query (read from data/prompts/overpass_query.txt)
    file_path_overpass_query = prompts_dir + "overpass_query.txt"
    with open(file_path_overpass_query,"r") as f:
        overpass_query = f.read()
    south_west_north_east = [bbox[1], bbox[0], bbox[3], bbox[2]]
    south_west_north_east_str = ",".join(map(str, south_west_north_east))
    formatted_overpass_query = overpass_query.format(bbox=south_west_north_east_str)

    # Run the Overpass Turbo Query
    # print(f"Running Overpass Turbo Query:\n {formatted_overpass_query}")
    response = requests.get(overpass_url, params={"data": formatted_overpass_query})
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    overpass_data = response.json()

    # Convert to GeoJSON string (if features_mapped_to_names.json exists, add unique names as properties)
    feature_collection = overpass_to_geojson_featurecollection(overpass_data)
    geojson_str = geojson.dumps(feature_collection, indent=2)

    # Save overpass GeoJSON (create dir if it doesn't exist)
    file_path_geojson = features_dir + config_name + "/overpass.geojson"
    os.makedirs(os.path.dirname(file_path_geojson), exist_ok=True)
    with open(file_path_geojson, "w") as f:
        f.write(geojson_str)

    # Mix features with hardcoded features
    if config_name:
        # Load hardcoded features if file exists
        file_path_config_features = features_dir + config_name + "/hardcoded.geojson"
        if not os.path.exists(file_path_config_features):
            print(f"Warning: Hardcoded features file '{file_path_config_features}' does not exist.")
        else:
            with open(file_path_config_features, "r") as f:
                hardcoded_features = json.load(f)
            # Add an id from 1 to n to each feature
            for i, feature in enumerate(hardcoded_features["features"]):
                feature["id"] = i + 1
            # Add hardcoded features to the feature collection
            feature_collection["features"].extend(hardcoded_features["features"])

    # Modify features to replace coordinates with count and other simplifications
    feature_collection_no_coords = simplify_geojson(feature_collection)

     # Get descriptive string for features
    features_descriptive_string = get_features_descriptive_string(feature_collection_no_coords)
    
    # Save descriptive string
    file_path_feature_descriptive_string = features_dir + config_name + "/descriptive.txt"
    with open(file_path_feature_descriptive_string, "w") as f:
        f.write(features_descriptive_string)

    # Name and assign colors to features using an LLM model
    feature_collection_processed = name_features(features_descriptive_string, feature_collection)

    # Add a feature with id 0 with the center coords of the bbox
    center_coords = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    feature_collection_processed["features"].append({
        "type": "Feature",
        "id": 0,
        "geometry": {
            "type": "Point",
            "coordinates": center_coords
        },
        "properties": {
            "category": "Static Locations",
            "name": "Bbox Center",
            "unique_name": "bbox_center",
            "color": "#000000"  # Black color
        }
    })

    # Convert to GeoJSON string
    geojson_str_processed = geojson.dumps(feature_collection_processed, indent=2)

    # Save named GeoJSON
    file_path_geojson_processed = features_dir + config_name + "/processed.geojson"
    with open(file_path_geojson_processed, "w") as f:
        f.write(geojson_str_processed)

    return feature_collection_processed

# Transform lat/lon to tile coordinates
def lonLatToTile(lon, lat, zoom):
    sin = np.sin(lat * np.pi / 180)
    z2 = 2 ** zoom
    x = np.floor(z2 * (lon / 360 + 0.5))
    y = np.floor(z2 * (0.5 - np.arctanh(sin) / (2 * np.pi)))

    return int(x), int(y)

# Transform tile coordinates to lat/lon of the tile's upper left corner
def tileToLonLat(x, y, zoom):
    n = 2 ** zoom
    lon_deg = (x / n) * 360 - 180  # Longitude calculation
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))  # Latitude calculation using inverse Gudermannian function
    lat_deg = lat_rad * (180 / np.pi)
    return lon_deg, lat_deg

# Function to get real bounding box given a bbox
def get_tiles_adapted_bbox(bbox, zoom):
    min_lon, min_lat, max_lon, max_lat = bbox
    x1, y1 = lonLatToTile(min_lon, max_lat, zoom) # min_lat and max_lat order correction
    x2, y2 = lonLatToTile(max_lon, min_lat, zoom) # min_lat and max_lat order correction

    adapted_min_lon, adapted_max_lat = tileToLonLat(x1, y1, zoom)
    adapted_max_lon, adapted_min_lat = tileToLonLat(x2 + 1, y2 + 1, zoom)

    top_left_tile = [x1, y1] # corrected y value
    bottom_right_tile = [x2, y2] # corrected y value
    tiles_adapted_bbox = [ adapted_min_lon, adapted_min_lat, adapted_max_lon, adapted_max_lat]
    return top_left_tile, bottom_right_tile, tiles_adapted_bbox

# Compute square bbox
def compute_square_bbox(center, side_length_m, zoom):
    """Compute a square bounding box centered at (lon, lat) with a given side length in meters."""
    # Unpack center coordinates
    lon, lat = center

    # Earth radius in meters
    EARTH_RADIUS = 6378137  # WGS 84
    
    # Convert meters to degrees
    delta_lat = (side_length_m / 2) / 110574  # 1 degree ≈ 110.574 km
    delta_lon = (side_length_m / 2) / (111320 * math.cos(math.radians(lat)))  # Adjust for latitude

    # Compute bounding box
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    bbox = (min_lon, min_lat, max_lon, max_lat)

    # Get adapted bbox
    top_left_tile, bottom_right_tile, tiles_adapted_bbox = get_tiles_adapted_bbox(bbox, zoom)

    # Save to visualization config file
    VIS_CONFIG = {}
    VIS_CONFIG["bbox"] = bbox
    VIS_CONFIG["top_left_tile"] = top_left_tile
    VIS_CONFIG["bottom_right_tile"] = bottom_right_tile
    VIS_CONFIG["tiles_adapted_bbox"] = tiles_adapted_bbox

    # Save the updated config file
    file_path_vis_config = config_dir + 'visualization.json'
    with open(file_path_vis_config, "w") as file:
        json.dump(VIS_CONFIG, file, indent=2)

    return bbox, tiles_adapted_bbox

def lonLatToArena(lon, lat, bbox, arena_width, arena_height):
    min_lon = bbox[0] # min lon
    min_lat = bbox[1] # max lon
    max_lon = bbox[2] # min lat
    max_lat = bbox[3] # max lat

    x = ((lon - min_lon) / (max_lon - min_lon)) * arena_width
    y = ((max_lat - lat) / (max_lat - min_lat)) * arena_height
    return x, y

# Given a bounding box and the arena size, map the coordinates of all features to the arena
# This is: long must be between [0, arena_width] and lat must be between [0, arena_height]
def map_features_to_arena(features_geojson, bbox, arena_width, arena_height):
    features = features_geojson["features"]

    # Map coordinates to arena
    for feature in features:
        if "geometry" in feature:
            if feature["geometry"]["type"] == "MultiPolygon":
                for i, part in enumerate(feature["geometry"]["coordinates"][0]):
                    for j, coord in enumerate(part):
                        x, y = lonLatToArena(coord[0], coord[1], bbox, arena_width, arena_height) 
                        feature["geometry"]["coordinates"][0][i][j] = [x, y]
            if feature["geometry"]["type"] == "Polygon":
                for i, part in enumerate(feature["geometry"]["coordinates"]):
                    for j, coord in enumerate(part):
                        x, y = lonLatToArena(coord[0], coord[1], bbox, arena_width, arena_height) 
                        feature["geometry"]["coordinates"][i][j] = [x, y]
            elif feature["geometry"]["type"] == "LineString":
                for i, coord in enumerate(feature["geometry"]["coordinates"]):
                    x, y = lonLatToArena(coord[0], coord[1], bbox, arena_width, arena_height)
                    feature["geometry"]["coordinates"][i] = [x, y]
            elif feature["geometry"]["type"] == "Point":
                x, y = lonLatToArena(feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][1], bbox, arena_width, arena_height)
                feature["geometry"]["coordinates"] = [x, y]

    # Remove outer coordinates from linestrings
    for feature in features:
        if "geometry" in feature:
            if feature["geometry"]["type"] == "LineString":
                feature["geometry"]["coordinates"] = [coord for coord in feature["geometry"]["coordinates"] if coord[0] >= 0 and coord[0] <= arena_width and coord[1] >= 0 and coord[1] <= arena_height]
                if len(feature["geometry"]["coordinates"]) == 0:
                    features.remove(feature)

    return features_geojson