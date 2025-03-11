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

# CONFIGURATION FILES (within /data folder)
SIM_CONFIG_FILE = os.path.join(app.root_path, "data", "sim_config.json")
# FEATURES_FILE = os.path.join(app.root_path, "data", "features.json")

# Load config and features from JSON files
with open(SIM_CONFIG_FILE, "r") as file:
    SIM_CONFIG = json.load(file)
# with open(FEATURES_FILE, "r") as file:
#     FEATURES = json.load(file)

# Assign values to Python variables
ENV = SIM_CONFIG["env"]
SETTINGS = SIM_CONFIG["settings"]

# Simulation functions
def initialize_robot_positions(n, x_min=0, y_min=0, x_max=ENV["arena_width"], y_max=ENV["arena_height"], distance_from_edge=ENV["formation_radius"]):
    x = np.random.uniform(x_min + distance_from_edge, x_max - distance_from_edge, n)
    y = np.random.uniform(y_min + distance_from_edge, y_max - distance_from_edge, n)
    return x, y

# Initialize destination positions with some distance from the edge
def initialize_destinations(n, distance_from_edge=ENV["formation_radius"], arena_width=ENV["arena_width"], arena_height=ENV["arena_height"]):
    x = np.random.uniform(distance_from_edge, arena_width - distance_from_edge, n)
    y = np.random.uniform(distance_from_edge, arena_height - distance_from_edge, n)
    return x, y

# Initialize the swarm
def initialize_swarm(n=SETTINGS["number_of_robots"], x_i=SETTINGS["field_start_x"], y_i=SETTINGS["field_start_y"],
                     x_width=SETTINGS["field_width"], y_height=SETTINGS["field_height"], max_speed=SETTINGS["max_speed"]):
    x, y = initialize_robot_positions(n, x_i, y_i, x_i + x_width, y_i + y_height)
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
agent = SwarmAgent(app, swarm)

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
        time.sleep(0.01)

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
            robots = [{"idx": r.idx, "x": r.x, "y": r.y, "angle": r.angle, 
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
            "arena": {"width": ENV["arena_width"], "height": ENV["arena_height"]}
        })

@app.route('/control', methods=['POST'])
def control():
    """Handle control commands from the client"""
    command = request.json.get('command')
    with simulation_lock:
        if command == 'reset':
            simulation_state["swarm"] = initialize_swarm()
            simulation_state["current_step"] = 0
            agent_state["agent"] = SwarmAgent(app, simulation_state["swarm"])
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
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the 'static' directory"""
    return send_from_directory('./public', path)