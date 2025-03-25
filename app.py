import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Thread, Lock
import os
import json
import random
import time
import numpy as np

# Import Swarm and Robot classes
from src.swarm import SwarmAgent, Swarm, Robot
from src.overpass import compute_square_bbox, get_geojson_features, map_features_to_arena

# Create the Flask application
app = Flask(__name__)
CORS(app)

# Logging configuration
# Get the werkzeug logger
log = logging.getLogger('werkzeug')

class NoSuccessFilter(logging.Filter):
    def filter(self, record):
        return "200" not in record.getMessage()  # Filters out 200 OK logs

log.addFilter(NoSuccessFilter())

# CONFIGURATION FILES (within /data/config folder)
SIM_CONFIGS_FILE = os.path.join(app.root_path, "data", "config", "sim_configs.json")
SIM_CONFIG_FILE = os.path.join(app.root_path, "data", "config", "simulation.json")
VIS_CONFIG_FILE = os.path.join(app.root_path, "data", "config", "visualization.json")

# Load simulation configurations
with open(SIM_CONFIGS_FILE, "r") as file:
    SIM_CONFIGS = json.load(file)

# Select the simulation configuration
REGENERATE_FEATURES = False
SIM_CONFIG_NAMES = ["uma_sar_scenario", "geelsa_greenhouses", "valsain_plaza"]
SIM_CONFIG_NAME = SIM_CONFIG_NAMES[0]
SIM_CONFIG = SIM_CONFIGS[SIM_CONFIG_NAME]

# Save configuration to JSON file
with open(SIM_CONFIG_FILE, "w") as file:
    json.dump(SIM_CONFIG, file, indent=2)

# Assign values to Python variables
ENV = SIM_CONFIG["env"]
SETTINGS = SIM_CONFIG["settings"]

# Compute the bounding box
bbox, tiles_adapted_bbox = compute_square_bbox(ENV["center"], ENV["side_length_meters"], ENV["zoom"])

features_processed_geojson_file = os.path.join(app.root_path, "data", "features", SIM_CONFIG_NAME, "processed.geojson")
if REGENERATE_FEATURES or not os.path.exists(features_processed_geojson_file):
    # Get Overpass features as GeoJSON
    features_geojson = get_geojson_features(tiles_adapted_bbox, SIM_CONFIG_NAME)
    # Create dir if it doesn't exist
    os.makedirs(os.path.dirname(features_processed_geojson_file), exist_ok=True)
    # Save features to the GeoJSON file
    with open(features_processed_geojson_file, "w") as file:
        json.dump(features_geojson, file, indent=2)
else: 
    # Load features from the existing GeoJSON file
    with open(features_processed_geojson_file, "r") as file:
        features_geojson = json.load(file)


# Calculate lon and lat per meter
def calculate_lon_lat_per_meter(bbox):
    """
    Calculate the approximate conversion factors for longitude and latitude per meter.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    # Convert meters to degrees (approximate)
    lat_per_meter = 1 / 111320  # degrees per meter
    lon_per_meter = 1 / (111320 * np.cos(np.radians(y_center)))  # degrees per meter

    return lon_per_meter, lat_per_meter

# Simulation functions
def initialize_robot_positions(n, bbox, distance_from_edge=50):
    """
    Initialize robot positions within the given bounding box.
    Distance from edge is given in meters.
    """
    x_min, y_min, x_max, y_max = bbox
    lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter(bbox)

    distance_from_edge_lat = distance_from_edge * lat_per_meter
    distance_from_edge_lon = distance_from_edge * lon_per_meter
    x = np.random.uniform(x_min + distance_from_edge_lon, x_max - distance_from_edge_lon, n)
    y = np.random.uniform(y_min + distance_from_edge_lat, y_max - distance_from_edge_lat, n)
    return x, y

# Initialize the swarm
def initialize_swarm(n=SETTINGS["number_of_robots"], bbox=bbox, max_speed=SETTINGS["max_speed"]):
    x_min, y_min, x_max, y_max = bbox
    lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter(bbox)
    METERS_FROM_EDGE, WIDTH, HEIGHT = 50, 20, 20
    x_i, x_f = x_min + METERS_FROM_EDGE * lon_per_meter, x_min + (METERS_FROM_EDGE + WIDTH) * lon_per_meter
    y_i, y_f = y_max - (METERS_FROM_EDGE + HEIGHT) * lat_per_meter, y_max - METERS_FROM_EDGE * lat_per_meter

    start_bbox = (x_i, y_i, x_f, y_f)
    x, y = initialize_robot_positions(n, start_bbox, distance_from_edge=0)
    robots = [Robot(idx, x, y, max_speed=max_speed) for idx, (x, y) in enumerate(zip(x, y))]
    swarm = Swarm(robots)
    
    return swarm

# Swarm Initialization
swarm = initialize_swarm()

# Simulation variables initialization
simulation_lock = Lock()
simulation_state = {
    "running": True,
    "current_step": 0,
    "swarm": swarm
}

# Chat variables
initial_messages = [
    {"role": "ai", "content": "Hello! Welcome to **Dialogue Swarm** ;)"},
    {"role": "ai", "content": "Tell me how to group the drones and what behaviors the groups should have, and I'll do it for you."},
]

# Agent Initialization
agent = SwarmAgent(app, swarm, features_geojson)

# Agent variables initialization
agent_state = {
    "agent": agent,
    "messages": initial_messages.copy()
}

# Send message to the agent
def send_message(message):
    agent_state["messages"].append({"role": "user", "content": message})
    agent_state["messages"].append({"role": "ai", "content": "Waiting for AI response..."})
    response_content = agent_state["agent"].send_message(message)
    agent_state["messages"][-1] = {"role": "ai", "content": response_content}


# Simulation loop
def simulation_loop():
    while True:
        with simulation_lock:
            if simulation_state["running"]:
                simulation_state["swarm"].step()
                simulation_state["current_step"] += 1
        time.sleep(simulation_state["swarm"].step_ms / 1000.0)  # Convert ms to seconds

sim_thread = Thread(target=simulation_loop)
sim_thread.daemon = True
sim_thread.start()


### API Endpoints ###

@app.route('/state')
def get_state():
    """Return the current state of the simulation"""
    with simulation_lock:
        groups = []
        for group in simulation_state["swarm"].groups:
            robots = [{"idx": r.idx, "x": r.x, "y": r.y, "angle": r.angle, "target_angle": r.target_angle, "angle_diff": r.angle_diff,
                      "target_x": r.target_x, "target_y": r.target_y, "battery_level": r.battery_level}
                     for r in group.robots]
            groups.append({
                "idx": group.idx,
                "virtual_center": group.virtual_center,
                "state": group.bhvr["state"],
                "bhvr": group.bhvr,
                "robots": robots
            })
        return jsonify({
            "running": simulation_state["running"],
            "current_step": simulation_state["current_step"],
            "groups": groups,
        })

@app.route('/control', methods=['POST'])
def control():
    """Handle control commands from the client"""
    command = request.json.get('command')
    with simulation_lock:
        if command == 'reset':
            simulation_state["swarm"] = initialize_swarm()
            simulation_state["current_step"] = 0
            agent_state["agent"] = SwarmAgent(app, simulation_state["swarm"], features_geojson)
            agent_state["messages"] = initial_messages.copy()
        elif command == 'pause':
            simulation_state["running"] = not simulation_state["running"]
    return jsonify({"status": "ok"})

@app.route('/message', methods=['POST'])
def message():
    """Handle a new user message"""
    user_message = request.json.get('message')
    app.logger.info(f"User message: {user_message}")
    send_message(user_message)
    return jsonify({"status": "ok"})

@app.route('/chat')
def chat():
    """Return the current state of the chat"""
    return jsonify(agent_state["messages"])

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory('./public', 'index.html')

@app.route('/processed.geojson')
def processed_geojson():
    """Serve the processed GeoJSON file"""
    return send_from_directory('./data/features', os.path.join(SIM_CONFIG_NAME, 'processed.geojson'))

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the 'static' directory"""
    return send_from_directory('./public', path)

@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)