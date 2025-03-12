import geopandas as gpd
import requests
import geojson
import json
import os
from shapely.geometry import shape

# VARIABLES
overpass_url = "http://overpass-api.de/api/interpreter"
data_dir = "data/"
output_prefix = "features"

# FUNCTIONS
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

    # Check if features_mapped_to_names.json exists
    file_path = data_dir + 'features_mapped_to_names.json'
    if not os.path.exists(file_path):
        print(f"{file_path} not found, skipping feature names")
        return geojson.FeatureCollection(features)
    
    # Load features_mapped_to_names.json
    with open(file_path, 'r') as f:
        features_mapped_to_names = json.load(f)
    
    # Flip keys and values in features_mapped_to_names dict
    features_mapped_to_names = {v: k for k, v in features_mapped_to_names.items()}

    # Add names to features
    for feature in features:
        if feature['id'] in features_mapped_to_names:
            # print(features_mapped_to_names[feature['id']])
            feature['properties']['unique_name'] = features_mapped_to_names[feature['id']]

    return geojson.FeatureCollection(features)

def replace_coordinates_with_count(feature_collection):
    """Replaces coordinates with count and centroid, simplifying the GeoJSON."""
    for feature in feature_collection["features"]:
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

    return feature_collection

def save_overpass_features_as_geojson(overpass_query):
    """Plots features from an Overpass Turbo query on a Folium map."""

    # Run the Overpass Turbo Query
    formatted_overpass_query = overpass_query.format(bbox=bbox)
    # print(f"Running Overpass Turbo Query:\n {formatted_overpass_query}")
    response = requests.get(overpass_url, params={"data": formatted_overpass_query})
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    overpass_data = response.json()

    # Convert to GeoJSON string (if features_mapped_to_names.json exists, add unique names as properties)
    feature_collection = overpass_to_geojson_featurecollection(overpass_data)
    geojson_str = geojson.dumps(feature_collection, indent=2)

    # Save original GeoJSON
    file_path_geojson = data_dir + output_prefix + ".geojson"
    with open(file_path_geojson, "w") as f:
        f.write(geojson_str)

    # Modify features to replace coordinates with count and other simplifications
    feature_collection_no_coords = replace_coordinates_with_count(feature_collection)

    # Convert to GeoJSON string
    geojson_str_no_coords = geojson.dumps(feature_collection_no_coords, indent=2)

    # Save simplified GeoJSON
    file_path_geojson_no_coords = data_dir + output_prefix + "_no_coords.geojson"
    with open(file_path_geojson_no_coords, "w") as f:
        f.write(geojson_str_no_coords)


###############################################
#################### MAIN #####################
###############################################

# Bounding Box
bbox = "55.39671, 10.544325, 55.411684, 10.567929"

# Overpass Query
overpass_query = f"""
[out:json];
(
  // Lakes, Reservoirs, Ponds, Pools
  way[natural~"water|reservoir"][water~"lake|reservoir|pond|pool"]({bbox});
  relation[natural~"water|reservoir"][water~"lake|reservoir|pond|pool"]({bbox});
  node[natural="water"][water~"lake|reservoir|pond|pool"]({bbox});
  way[natural="water"]({bbox});
  relation[natural="water"]({bbox});
  way[landuse="reservoir"]({bbox});
  relation[landuse="reservoir"]({bbox});

  // Rivers, Streams, Canals, Ditches
  way[waterway~"river|stream|canal|ditch"]({bbox});
  relation[waterway~"river|stream|canal|ditch"]({bbox});

  // Coastlines
  way[natural="coastline"]({bbox});

  // Wetlands
  way[natural="wetland"]({bbox});
  relation[natural="wetland"]({bbox});

  // Buildings
  way[building]({bbox});
  relation[building]({bbox});

  // Forests and Woodlands
  way[natural~"wood|forest"]({bbox});
  relation[natural~"wood|forest"]({bbox});
  way[landuse="forest"]({bbox});
  relation[landuse="forest"]({bbox});

  // Parks and Recreation Areas
  way[leisure="park"]({bbox});
  relation[leisure="park"]({bbox});
  way[landuse~"recreation_ground|grass"]({bbox});
  relation[landuse~"recreation_ground|grass"]({bbox});
  node[leisure="park"]({bbox});

  // Roads
  way[highway]({bbox});
  relation[highway]({bbox});

  // Fields (Comprehensive)
  way[landuse~"farmland|meadow|greenfield|grassland|pasture|allotments|village_green"]({bbox});
  relation[landuse~"farmland|meadow|greenfield|grassland|pasture|allotments|village_green"]({bbox});
  way[natural~"grass|scrub|heath"]({bbox});
  relation[natural~"grass|scrub|heath"]({bbox});
  way[area="yes"][!landuse][!natural][!highway][!building][!waterway][!leisure]({bbox});
  relation[area="yes"][!landuse][!natural][!highway][!building][!waterway][!leisure]({bbox});
);
out geom;
"""

# Save Overpass Features as GeoJSON
save_overpass_features_as_geojson(overpass_query)