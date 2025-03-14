"""
Swarm

This module contains the Swarm, Group, and Robot classes. 
It also has the SwarmAgent class, which is an agent that controls the swarm by interpreting user commands with an LLM.
"""

import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, DBSCAN

from openai import OpenAI
from typing import Dict, List, Tuple, Any
import geojson
import json

# CONSTANTS
MODEL_NAME = "gpt-4o-mini"
with open("mykeys/openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()

# POSSIBLE BEHAVIORS AND PARAMS
BEHAVIORS = {
    "form_and_move_around_shape": {
        "states": ["form", "rotate", "move"],
        "params": {
            "formation_shape": ["circle", "square", "triangle", "hexagon"],
            "formation_radius": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "name": ["building_1", "building_2", "river_geeslå"]
        }
    },
    "form_and_follow_trajectory": {
        "states": ["form", "rotate", "move"],
        "params": {
            "formation_shape": ["circle", "square", "triangle", "hexagon"],
            "formation_radius": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "trajectory": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        }
    },
    "random_walk": {
        "states": ["move"],
        "params": {}
    }
}

# PROMPTS
# Load system prompt
with open("prompts/swarm_agent.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# FEATURES
# Load features_mapped_to_names.json
with open("data/features_mapped_to_names.json", "r") as f:
    features_mapped_to_names = json.load(f)

# Load features.geojson
with open("data/features.geojson", "r") as f:
    features_geojson = geojson.load(f)

# Create list of unique_names that correspond to Polygon features
polygon_feature_names = [feature["properties"]["unique_name"] for feature in features_geojson["features"] if feature["geometry"]["type"] == "Polygon"]
# Sort them alphabetically
polygon_feature_names.sort()
# Create a list of unique_names that correspond to LineString features
linestring_feature_names = [feature["properties"]["unique_name"] for feature in features_geojson["features"] if feature["geometry"]["type"] == "LineString"]
# Sort them alphabetically
linestring_feature_names.sort()

# ARENA MAPPING FUNCTIONS
bbox = [55.39627171359563, 10.54412841796875, 55.41186640197955, 10.56884765625]
arena_width = 20
arena_height = 20
def latLonToArena(lat, lon):
    west = bbox[1] # min lon
    east = bbox[3] # max lon
    south = bbox[0] # min lat
    north = bbox[2] # max lat

    x = ((lon - west) / (east - west)) * arena_width
    y = ((north - lat) / (north - south)) * arena_height
    return x, y

# Given a bounding box and the arena size, map the coordinates of all features to the arena
# This is: long must be between [0, arena_width] and lat must be between [0, arena_height]
def map_features_to_arena(features_geojson):
    features = features_geojson["features"]
    for feature in features:
        if "geometry" in feature:
            if feature["geometry"]["type"] == "Polygon":
                for i, part in enumerate(feature["geometry"]["coordinates"]):
                    for j, coord in enumerate(part):
                        x, y = latLonToArena(coord[1], coord[0])
                        feature["geometry"]["coordinates"][i][j] = [x, y]
            elif feature["geometry"]["type"] == "LineString":
                for i, coord in enumerate(feature["geometry"]["coordinates"]):
                    x, y = latLonToArena(coord[1], coord[0])
                    feature["geometry"]["coordinates"][i] = [x, y]
    return features_geojson

# Remove all coordinates of LineString outside of the bounding box
def remove_outside_coordinates(features_geojson):
    features = features_geojson["features"]
    for feature in features:
        if "geometry" in feature:
            if feature["geometry"]["type"] == "LineString":
                feature["geometry"]["coordinates"] = [coord for coord in feature["geometry"]["coordinates"] if coord[0] >= 0 and coord[0] <= arena_width and coord[1] >= 0 and coord[1] <= arena_height]
    return features_geojson

# Map features to arena
features_geojson = map_features_to_arena(features_geojson)

# Remove outside coordinates
features_geojson = remove_outside_coordinates(features_geojson)

# CLIENT
client = OpenAI(api_key=OPENAI_API_KEY)

### CLASSES ###

class Robot:
    """
    A single robot agent in the swarm
    """
    
    def __init__(self, idx, x, y, max_speed=0.05):
        """
        Initialize a robot at specified coordinates
        
        Args:
            idx (int): Robot index
            x (float): Initial x-position
            y (float): Initial y-position
            max_speed (float): Maximum allowed speed
        """
        self.idx = idx
        self.x = x
        self.y = y
        self.max_speed = max_speed
        self.angle = 0.0
        self.update_target(x, y)
        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0
        self.battery_level = 1.0
    
    def update_target_angle(self, target_x, target_y):
        """Update target angle for robot"""
        target_angle = np.arctan2(target_y - self.y, target_x - self.x)
        self.target_angle = target_angle
    
    def update_target(self, target_x, target_y):
        """Update target position for robot"""
        self.target_x = target_x
        self.target_y = target_y
        self.update_target_angle(target_x, target_y)

    def move(self):
        """
        Move robot towards target position with rotation and velocity control
        """
        # Do not move if robot is out of battery
        if self.battery_level <= 0:
            return
        
        # Update battery level
        self._update_battery_level()
        
        # First align robot with assigned target angle
        if not self.is_robot_aligned():
            self.rotation_speed = np.sign(self.angle_diff) * max(0.1 * (abs(self.angle_diff) / np.pi), 0.005)
            # print('Robot is rotating', self.rotation_speed)
        # Move if it is not in position when aligned
        else:
            if not self.is_robot_in_position():
                self.vx = (self.dx / self.dist) * self.max_speed
                self.vy = (self.dy / self.dist) * self.max_speed
                # print('Robot is moving', self.vx, self.vy)

        # Update position based on velocity and rotation speed
        self._update_position()

    def _calculate_distance(self):
        """Calculate distance to target position"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = np.hypot(dx, dy)
        return dx, dy, dist
    
    def _calculate_angle_diff(self):
        """Calculate angle difference to target angle"""
        angle_diff = (self.target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
        return angle_diff

    def is_robot_in_position(self):
        """Check if robot is close to assigned target position"""
        self.dx, self.dy, self.dist = self._calculate_distance()
        return self.dist < 0.05
    
    def is_robot_aligned(self):
        """Check if robot is aligned with assigned target angle"""
        self.angle_diff = self._calculate_angle_diff()
        return abs(self.angle_diff) < 0.01
    
    def _update_position(self):
        """
        Update robot position based on current velocity and rotation speed
        """
        # Update position
        self.x = (self.x + self.vx)
        self.y = (self.y + self.vy)
        # Update angle and wrap to [0, 2*pi]
        self.angle = (self.angle + self.rotation_speed) % (2 * np.pi)
        # Reset velocity and rotation speed after movement
        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0

    def _update_battery_level(self):
        """Update robot battery level"""
        self.battery_level = self.battery_level - 1/10000;
        if self.battery_level <= 0:
            self.battery_level = 0

class Group:
    """
    A group of robots in the swarm that share a common behavior or task
    """
    
    def __init__(self, idx, robots):
        """
        Initialize a robot group
        
        Args:
            idx (int): Group index
            robots (list[Robot]): List of robots in the group
        """
        self.idx = idx
        self.robots = robots
        self.destination = (0, 0)
        self.bhvr = {
            "name": "", 
            "state": 0,
            "params": {},
            "data": {}
        }
        # Initialize virtual center
        self._update_virtual_center()

    def set_behavior(self, behavior_dict):
        """Set behavior for the group"""
        behavior_name = behavior_dict['name']
        behavior_params = behavior_dict['params'] if 'params' in behavior_dict else {}
        behavior_dict["data"] = {}

        if behavior_name == "form_and_follow_trajectory":
            # Assign formation positions
            n = len(self.robots)
            formation_pts = compute_formation_positions(n, behavior_params['formation_shape'], behavior_params['formation_radius'])
            
            # Create cost matrix
            cost_matrix = np.zeros((len(self.robots), len(formation_pts)))
            for i, robot in enumerate(self.robots):
                for j, pos in enumerate(formation_pts):
                    cost_matrix[i, j] = np.linalg.norm([robot.x - pos[0], robot.y - pos[1]])
                    
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            robot_idxs = [robot.idx for robot in self.robots]

            # Set formation positions in behavior data
            behavior_dict["data"]["formation_positions"] = {int(i): formation_pts[j].tolist() for i, j in zip(robot_idxs, col_ind)}
            behavior_dict["data"]["traj_index"] = 0

            # Initialize behavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict
        elif behavior_name == "random_walk":
            # Initialize behavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict

    def step(self):
        """Perform one simulation step for the group"""
        bhvr = self.bhvr

        # Calculate movement based on current behavior
        match bhvr["name"]:
            # State machine for formation and following trajectory
            case "form_and_follow_trajectory":
                match bhvr["state"]:
                    case 0:
                        for robot in self.robots:
                            robot.update_target(
                                self.virtual_center[0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                self.virtual_center[1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()
                        
                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            bhvr["state"] = 1

                    case 1:
                        for robot in self.robots:
                            robot.update_target_angle(
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()

                        # Check transition to next state
                        if all(r.is_robot_aligned() for r in self.robots):
                            bhvr["state"] = 2
                    
                    case 2:
                        for robot in self.robots:
                            robot.update_target(
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()

                        # Update virtual center 
                        self._update_virtual_center()

                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            # Check if traj_index is the last index
                            if bhvr["data"]["traj_index"] == len(bhvr["params"]["trajectory"]) - 1:
                                bhvr["state"] = -1
                                bhvr["data"]["traj_index"] = 0
                            else:
                                bhvr["state"] = 1
                                bhvr["data"]["traj_index"] += 1

            # State machine for random walk
            case "random_walk":
                for robot in self.robots:
                    # If they havent reached their target, move them
                    if not robot.is_robot_in_position():
                        robot.move()
                    # If they have reached their target, pick a new target
                    else: 
                        robot.update_target(robot.x + random.uniform(-0.5, 0.5), robot.y + random.uniform(-0.5, 0.5))

    def _update_robot_targets(self):
        """Update robot targets based on current behavior"""

    def _update_virtual_center(self):
        """Update virtual center to current robot positions average"""
        self.virtual_center = (
            np.mean([r.x for r in self.robots]),
            np.mean([r.y for r in self.robots])
        )

class Swarm:
    """
    A swarm of robots that can move and interact with each other forming groups
    """
    
    def __init__(self, robots):
        """
        Initialize a swarm of robots
        
        Args:
            robots (list[Robot]): List of robots in the swarm
        """  
        self.robots = robots
        self.groups = []
        self.group_idx_counter = 0
        for robot in self.robots:
            group = self.gen_group_by_ids([robot.idx])

        self.formation_shapes = ['circle', 'square', 'triangle', 'hexagon']

    def _get_group_by_idx(self, group_idx):
        """Get group by idx value. Find the group in the groups list with idx = group_idx"""
        groups = [group for group in self.groups if group.idx == group_idx]
        return groups[0]
    
    def assign_random_walk_behavior_to_group(self, group_idx):
        """Assign random_walk behavior to a group"""
        self._get_group_by_idx(group_idx).set_behavior({
            "name": "random_walk",
            "params": {}
        })

    def assign_form_and_follow_trajectory_behavior_to_group(self, group_idx, formation_shape="circle", formation_radius=1.0, trajectory=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]):
        """Assign form_and_follow_trajectory behavior to a group"""
        self._get_group_by_idx(group_idx).set_behavior({
            "name": "form_and_follow_trajectory",
            "params": {
                "formation_shape": formation_shape,
                "formation_radius": formation_radius,
                "trajectory": trajectory
            }
        })

    def gen_group_by_ids(self, robot_idxs):
        """
        Generate group based on robot IDs. 
        If the group is already assigned, it will be overwritten. 
        If the robots are assigned to a different group, they will be moved to the new group and removed from the old group.

        Args:
            robot_idxs (list[int]): List of robot IDs
        """
        # Iterate over the groups to see if they have any of the robots
        for i, group in enumerate(self.groups):
            if any(robot.idx in robot_idxs for robot in group.robots):
                # Remove robots from the old group
                self.groups[i].robots = [robot for robot in self.groups[i].robots if robot.idx not in robot_idxs]
        
        # Remove empty groups
        self.groups = [group for group in self.groups if group.robots]

        # Create the new group
        new_group = Group(self.group_idx_counter, [self.robots[i] for i in robot_idxs])
        self.group_idx_counter += 1

        # Add the new group
        self.groups.append(new_group)

        return new_group

    def gen_groups_by_lists_of_ids(self, lists_of_ids):
        """
        Generate groups based on lists of robot IDs

        Args:
            lists_of_ids (list[list[int]]): List of lists of robot IDs
        """
        self.groups = []
        for robot_ids in lists_of_ids:
            self.gen_group_by_ids(robot_ids)

    def gen_groups_by_clustering(self, num_groups):
        """
        Generate groups based on clustering and current robot positions
        
        Args:
            num_groups (int): Number of groups to create
        """
        x = [r.x for r in self.robots]
        y = [r.y for r in self.robots]
        coords = np.column_stack((x, y))

        kmeans = KMeans(n_clusters=num_groups, n_init=10).fit(coords)
        
        groups = []
        for group_idx in range(num_groups):
            robot_indices = np.where(kmeans.labels_ == group_idx)[0]
            group_robot_idxs = [self.robots[i].idx for i in robot_indices]
            group = self.gen_group_by_ids(group_robot_idxs)
            groups.append(group)
            
        self.groups = groups
    
    def step(self):
        """Perform one simulation step for entire swarm"""
        for group in self.groups:
            if group.bhvr["state"] == -1:
                continue
                
            group.step()

class SwarmAgent:
    def __init__(self, app, swarm: Swarm):
        self.app = app
        self.swarm = swarm
        self.tools = [
            self.gen_group_by_ids,
            self.gen_groups_by_clustering,
            self.random_walk,
            self.form_and_follow_trajectory,
            self.form_and_move_around_shape,
            self.form_and_move_to_shape
        ]
        self.memory = []
        self.system_prompt = SYSTEM_PROMPT
        
    def _format_memory(self):
        """Convert memory entries to proper message format"""
        formatted = []
        for entry in self.memory:
            if "tool_call_id" in entry:
                formatted.append({
                    "role": entry["role"],
                    "content": entry["content"],
                    "tool_call_id": entry["tool_call_id"]
                })
            elif "tool_calls" in entry:
                formatted.append({
                    "role": entry["role"],
                    "content": entry["content"],
                    "tool_calls": entry["tool_calls"]
                })
            else:
                formatted.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })
        return formatted
    
    def _get_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__,
                    "parameters": self._get_params_schema(tool)
                }
            } for tool in self.tools
        ]
    
    def _get_params_schema(self, func):
        # Implement parameter schema extraction based on func annotations
        if func.__name__ == "gen_group_by_ids":
            return {
                "type": "object",
                "properties": {
                    "robot_idxs": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["robot_idxs"]
            }
        elif func.__name__ == "gen_groups_by_clustering":
            return {
                "type": "object",
                "properties": {
                    "num_groups": {"type": "integer"}
                },
                "required": ["num_groups"]
            }
        elif func.__name__ == "random_walk":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"}
                },
                "required": ["group_idx"]
            }
        elif func.__name__ == "form_and_follow_trajectory":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"},
                    "formation_shape": {
                        "type": "string", 
                        "enum": ["circle", "square", "triangle", "hexagon"]
                    },
                    "formation_radius": {
                        "type": "number",
                        "minimum": 0.125,
                        "maximum": 1.0
                    },
                    "trajectory": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 10
                        },
                        "minItems": 1,
                        "maxItems": 1
                    }
                },
                "required": ["group_idx", "formation_shape", "formation_radius", "trajectory"]
            }
        elif func.__name__ == "form_and_move_around_shape":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"},
                    "formation_shape": {
                        "type": "string", 
                        "enum": ["circle", "square", "triangle", "hexagon"]
                    },
                    "formation_radius": {
                        "type": "number",
                        "minimum": 0.125,
                        "maximum": 1.0
                    },
                    "name": {
                        "type": "string",
                    }
                },
                "required": ["group_idx", "formation_shape", "formation_radius", "name"]
            }
        elif func.__name__ == "form_and_move_to_shape":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"},
                    "formation_shape": {
                        "type": "string", 
                        "enum": ["circle", "square", "triangle", "hexagon"]
                    },
                    "formation_radius": {
                        "type": "number",
                        "minimum": 0.125,
                        "maximum": 1.0
                    },
                    "name": {
                        "type": "string",
                    }
                },
                "required": ["group_idx", "formation_shape", "formation_radius", "name"]
            }
        
    def send_message(self, user_input: str):
        # Build current context strings
        robot_idxs = " ,".join(map(str, [robot.idx for robot in self.swarm.robots]))
        group_idxs = " ,".join(map(str, [group.idx for group in self.swarm.groups]))
        groups_str = ""
        for group in self.swarm.groups:
            group_robot_idxs = " ,".join(map(str, [robot.idx for robot in group.robots]))
            groups_str += f"[{group.idx}: [{group_robot_idxs}], "
        
        # Build message history with correct structure
        formatted_system_prompt = self.system_prompt.format(
            linestring_feature_names=linestring_feature_names,
            polygon_feature_names=polygon_feature_names,
            robot_idxs=robot_idxs, 
            groups_str=groups_str
        )

        # self.app.logger.info(f"Formatted system prompt: {formatted_system_prompt}")
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            *self._format_memory(),
            {"role": "user", "content": user_input}
        ]

        try :
            # First API call to get initial response
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=4192,
                messages=messages,
                tools=self._get_tool_schemas()
            )
            
            response_message = response.choices[0].message
            self.memory.append({"role": "user", "content": user_input})
            
            if response_message.tool_calls:
                
                # Process tool calls
                tool_responses = []
                for tool_call in response_message.tool_calls:
                    self.app.logger.info(f"Processing tool call: {tool_call.function.name}")
                    func = next(t for t in self.tools if t.__name__ == tool_call.function.name)
                    args = json.loads(tool_call.function.arguments)
                    self.app.logger.info(f"Calling {tool_call.function.name} with {args}")
                    result = func(**args)
                    # Store tool response
                    tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": result
                    })
                
                # Store assistant message with tool calls
                self.memory.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        } for call in response_message.tool_calls
                    ]
                })
                
                # Store tool responses
                self.memory.extend(tool_responses)
                
                # Get final response with full context
                # formatted_system_prompt = self.system_prompt.format(robot_idxs=robot_idxs, groups_str=groups_str)
                final_messages = [
                    {"role": "system", "content": formatted_system_prompt},
                    *self._format_memory()
                ]
                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0,
                    max_tokens=4192,
                    messages=final_messages
                )
                ai_message = final_response.choices[0].message.content
                self.memory.append({"role": "assistant", "content": ai_message})
                self.app.logger.info(f"Response after calls: {ai_message}")
                return ai_message
            else:
                self.memory.append({"role": "assistant", "content": response_message.content})
                self.app.logger.info(f"Response without calls: {response_message.content}")
                return response_message.content
        
        except Exception as e:
            self.app.logger.error(f"Error: {e}")
            return "There was an error processing your request. Please try clarifying your request."

    # Tool implementations
    def gen_group_by_ids(self, robot_idxs: List[int]):
        new_group = self.swarm.gen_group_by_ids(robot_idxs)
        return f"Drones {', '.join(map(str, robot_idxs))} grouped successfully in group {new_group.idx}"

    def gen_groups_by_clustering(self, num_groups: int):
        if num_groups < 0 or num_groups > len(self.swarm.robots):
            return "Invalid number of groups"
        self.swarm.gen_groups_by_clustering(num_groups)
        return f"Drones grouped successfully in {num_groups} groups"

    def random_walk(self, group_idx: int):
        self.swarm.assign_random_walk_behavior_to_group(group_idx)
        return f"random_walk behavior assigned to group {group_idx}"
    
    def form_and_follow_trajectory(self, group_idx: int,
                                    formation_shape: str,
                                    formation_radius: float,
                                    trajectory: List[Tuple[float, float]]): 
        # Validation logic
        if formation_radius < 0.125 or formation_radius > 1.0:
            return "Invalid radius value: must be between 0.125 and 1.0"
        if formation_shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if len(trajectory) < 1 or any(len(pos) != 2 for pos in trajectory):
            return "Invalid trajectory"
        
        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, formation_shape, formation_radius, trajectory
        )
        return f"form_and_follow_trajectory behavior assigned to group {group_idx} with formation shape {formation_shape}, radius {formation_radius}, and trajectory {trajectory}"

    def form_and_move_around_shape(self, group_idx: int,
                                    formation_shape: str,
                                    formation_radius: float,
                                    name: str):
        # Validation logic
        if formation_radius < 0.125 or formation_radius > 1.0:
            return "Invalid radius value: must be between 0.125 and 1.0"
        if formation_shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in polygon_feature_names + linestring_feature_names:
            return "Invalid feature name: must be one of the available features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = next(f for f in features_geojson["features"] if f["properties"]["unique_name"] == name)
        # Get the coordinates of the feature depending on its geometry type
        if feature["geometry"]["type"] == "Polygon":
            coords = feature["geometry"]["coordinates"][0]
        elif feature["geometry"]["type"] == "LineString":
            coords = feature["geometry"]["coordinates"]
        else:
            return "Invalid feature type"
        
        # self.app.logger.info(f"Feature coordinates: {coords}")

        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, formation_shape, formation_radius, coords
        )
        return f"form_and_move_around_shape behavior assigned to group {group_idx} with formation shape {formation_shape}, radius {formation_radius}, and feature name {name}"
    
    def form_and_move_to_shape(self, group_idx: int,
                                formation_shape: str,
                                formation_radius: float,
                                name: str):
        # Validation logic
        if formation_radius < 0.125 or formation_radius > 1.0:
            return "Invalid radius value: must be between 0.125 and 1.0"
        if formation_shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in polygon_feature_names + linestring_feature_names:
            return "Invalid feature name: must be one of the available features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = next(f for f in features_geojson["features"] if f["properties"]["unique_name"] == name)
        # Get the coordinates of the feature depending on its geometry type
        if feature["geometry"]["type"] == "Polygon":
            coords = feature["geometry"]["coordinates"][0]
        elif feature["geometry"]["type"] == "LineString":
            coords = feature["geometry"]["coordinates"]
        else:
            return "Invalid feature type"
        
        # If LineString, get the coordinate in the middle
        if feature["geometry"]["type"] == "LineString":
            center = coords[len(coords) // 2]
            coords = [center]
        # If Polygon, get the centroid
        else:
            x = np.mean([coord[0] for coord in coords])
            y = np.mean([coord[1] for coord in coords])
            coords = [[x, y]]
        
        # self.app.logger.info(f"Feature coordinates: {coords}")

        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, formation_shape, formation_radius, coords
        )
        return f"form_and_move_to_shape behavior assigned to group {group_idx} with formation shape {formation_shape}, radius {formation_radius}, and feature name {name}"

# AUXILIARY FUNCTIONS

def compute_formation_positions(n, formation_shape, formation_radius):
    """Calculate equally shaped positions along a shape's perimeter"""
    if formation_shape == 'circle':
        return compute_circle_positions(n, formation_radius)
    elif formation_shape == 'square':
        return compute_square_positions(n, formation_radius*2)
    elif formation_shape == 'triangle':
        return compute_triangle_positions(n, formation_radius*2)
    elif formation_shape == 'hexagon':
        return compute_hexagon_positions(n, formation_radius)

# Formation calculation functions   
def compute_circle_positions(n, radius):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def compute_square_positions(n, side_length):
    positions = []
    for i in range(n):
        t = i * (4.0 / n)
        side = int(t)
        pos_in_side = t - side
        if side == 0:
            x = -0.5 + pos_in_side
            y = 0.5
        elif side == 1:
            x = 0.5
            y = 0.5 - pos_in_side
        elif side == 2:
            x = 0.5 - pos_in_side
            y = -0.5
        else:
            x = -0.5
            y = -0.5 + pos_in_side
        positions.append([x * side_length, y * side_length])
    return np.array(positions)

def compute_triangle_vertices(side_length):
    h = np.sqrt(3) / 6 * side_length
    return np.array([[0, -2 * h], [0.5 * side_length, h], [-0.5 * side_length, h]])

def compute_triangle_positions(n, side_length):
    vertices = compute_triangle_vertices(side_length)
    positions = []
    for i in range(n):
        pos = i * (3 / n)
        if pos < 1:
            point = vertices[0] + pos * (vertices[1] - vertices[0])
        elif pos < 2:
            point = vertices[1] + (pos - 1) * (vertices[2] - vertices[1])
        else:
            point = vertices[2] + (pos - 2) * (vertices[0] - vertices[2])
        positions.append(point)
    return np.array(positions)

def compute_hexagon_vertices(side_length):
    """Compute the vertices of a regular hexagon centered at (0,0) with given side length."""
    h = np.sqrt(3) / 2 * side_length  # Height of an equilateral triangle (half hexagon height)
    
    return np.array([
        [side_length, 0],          # Right
        [0.5 * side_length, h],    # Top-right
        [-0.5 * side_length, h],   # Top-left
        [-side_length, 0],         # Left
        [-0.5 * side_length, -h],  # Bottom-left
        [0.5 * side_length, -h]    # Bottom-right
    ])

def compute_hexagon_positions(n, side_length):
    """Distribute n points evenly along the perimeter of a hexagon with given side length."""
    vertices = compute_hexagon_vertices(side_length)
    positions = []
    
    for i in range(n):
        pos = i * (6 / n)  # Normalize position around the hexagon (6 edges)
        edge_index = int(pos)  # Determine which edge the point belongs to
        t = pos - edge_index  # Fractional position along the edge
        
        point = vertices[edge_index] + t * (vertices[(edge_index + 1) % 6] - vertices[edge_index])
        positions.append(point)
    
    return np.array(positions)