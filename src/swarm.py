"""
Swarm

This module contains the Swarm, Group, and Robot classes. 
It also has the SwarmAgent class, which is an agent that controls the swarm by interpreting user commands with an LLM.
"""

import math
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, DBSCAN

from openai import OpenAI
from typing import Dict, List, Tuple, Any
import geojson
import json

from shapely import Polygon
from shapely.geometry import shape, Point, LineString, Polygon
from shapely.ops import transform
from pyproj import CRS, Transformer
from trajgenpy import Geometries

# CONSTANTS
MODEL_NAME = "gpt-4o-mini"
with open("mykeys/openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()

nato_phonetic_alphabet = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", 
    "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", 
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey", 
    "X-ray", "Yankee", "Zulu"
]

# POSSIBLE BEHAVIORS AND PARAMS
BEHAVIORS = {
    "form_and_follow_trajectory": {
        "states": ["form", "rotate", "move"],
        "params": {
            "formation_shape": ["circle", "square", "triangle", "hexagon"],
            "formation_radius": [5, 10, 15, 20], # meters
            "trajectory": [
                [0.0, 0.0], 
                [1.0, 1.0], 
                [2.0, 2.0], 
                [3.0, 3.0]
            ] # trajectory points
        }
    },
    "cover_shape": {
        "states": ["move"],
        "params": {
            "coords": [
                [0.0, 0.0], 
                [1.0, 1.0], 
                [2.0, 2.0], 
                [3.0, 3.0]
            ] # sweep trajectories points
        }
    },
    "random_walk": {
        "states": ["move"],
        "params": {}
    }
}

# PROMPTS
# Load system prompt
with open("data/prompts/swarm_agent.txt", "r") as f:
    SYSTEM_PROMPT = f.read()
# Load spatial awareness system prompt
with open("data/prompts/spatial_awareness.txt", "r") as f:
    SPATIAL_AWARENESS_SYSTEM_PROMPT = f.read()

# CLIENT
client = OpenAI(api_key=OPENAI_API_KEY)

### CLASSES ###

class Robot:
    """
    A single robot agent in the swarm.
    """

    def __init__(self, idx, x, y, max_speed=20.0):
        """
        Initialize a robot at specified coordinates.
        
        Args:
            idx (int): Robot index.
            x (float): Initial longitude.
            y (float): Initial latitude.
            max_speed (float): Maximum allowed speed in km/h.
        """
        self.idx = idx
        self.x = x
        self.y = y
        self.max_speed = max_speed
        self.max_rotation_speed = np.pi # rad/s
        self.min_rotation_speed = 0.5  # rad/s
        self.angle = 0.0  # Initial heading in radians

        self.target_x = x
        self.target_y = y
        self.target_angle = 0.0
        self.angle_diff = 0.0

        self.vx = 0.0  # Velocity in km/h
        self.vy = 0.0
        self.rotation_speed = 0.0
        self.battery_level = 1.0

        self.distance_th = 0.5  # 50 cm
        self.rotation_th = 5.0 # 2 degrees

        self.update_target(x, y)

    def update_target_angle(self, target_x, target_y):
        """Update target angle for robot using geodesic coordinates."""
        lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter((self.x, self.y))
        dx_in_meters = (target_x - self.x) / lon_per_meter
        dy_in_meters = (target_y - self.y) / lat_per_meter
        self.target_angle = math.atan2(dy_in_meters, dx_in_meters)

    def update_target(self, target_x, target_y):
        """Update target position for the robot."""
        self.target_x = target_x
        self.target_y = target_y
        self.update_target_angle(target_x, target_y)

    def move(self, step_ms):
        """
        Move robot towards target position with rotation and velocity control.
        """
        if self.battery_level <= 0:
            return

        self._update_battery_level()

        if not self.is_robot_aligned():
            self.rotation_speed = np.sign(self.angle_diff) * max(min(2*abs(self.angle_diff), self.max_rotation_speed), self.min_rotation_speed)
        else:
            if not self.is_robot_in_position():
                # self.vx = (self.dx_in_meters / self.dist_in_meters) * self.max_speed
                # self.vy = (self.dy_in_meters / self.dist_in_meters) * self.max_speed
                self.vx = np.cos(self.angle) * self.max_speed
                self.vy = np.sin(self.angle) * self.max_speed
                self.rotation_speed = np.sign(self.angle_diff) * max(min(2*abs(self.angle_diff), self.max_rotation_speed), self.min_rotation_speed)

        self._update_position(step_ms)

    def _calculate_distance(self):
        """Calculate distance to target position."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return dx, dy, np.hypot(dx, dy)

    def _calculate_distance_in_meters(self):
        """Calculate geospatial distance to target position in meters."""
        self.dx, self.dy, self.dist = self._calculate_distance()
        lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter((self.x, self.y))
        self.dx_in_meters = self.dx / lon_per_meter
        self.dy_in_meters = self.dy / lat_per_meter
        self.dist_in_meters = np.hypot(self.dx_in_meters, self.dy_in_meters)

    def _calculate_angle_diff(self):
        """Calculate and normalize angle difference to target."""
        self.angle_diff = (self.target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi

    def is_robot_in_position(self):
        """Check if robot is within a 0.25m tolerance of the target position."""
        self._calculate_distance_in_meters()
        return self.dist_in_meters < self.distance_th

    def is_robot_aligned(self):
        """Check if robot is aligned with target within 0.1 degrees."""
        self._calculate_angle_diff()
        return abs(self.angle_diff) < np.radians(self.rotation_th)

    def _update_position(self, step_ms):
        """
        Update robot position based on velocity and rotation speed.
        """
        vx_m_per_s = self.vx / 3.6
        vy_m_per_s = self.vy / 3.6

        displacement_x_m = vx_m_per_s * (step_ms / 1000)
        displacement_y_m = vy_m_per_s * (step_ms / 1000)

        lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter((self.x, self.y))
        self.x += displacement_x_m * lon_per_meter
        self.y += displacement_y_m * lat_per_meter

        self.angle = (self.angle + self.rotation_speed * (step_ms / 1000)) % (2 * np.pi)
        if self.angle > np.pi:
            self.angle -= 2 * np.pi

        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0

    def _update_battery_level(self):
        """Update robot battery level based on time."""
        self.battery_level = max(0, self.battery_level - (1 / 25000))

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
            behavior_dict["data"]["init_virtual_center"] = self.virtual_center
            formation_pts = compute_formation_coordinate_offsets(self.virtual_center, n, behavior_params["formation_shape"], behavior_params["formation_radius"])
            
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
        elif behavior_name == "cover_shape":
            n = len(self.robots)
            coords = behavior_params["coords"]

            # Divide the coords into n segments
            segments = np.array_split(coords, n)
            unassigned_segments = set(range(n))  # Keep track of available segment indices

            # Save a dictionary mapping robots to segments
            robot_segments = {}

            # Compute distances for each robot to each segment
            distances = []  # List of (robot_idx, segment_idx, distance, reversed) tuples

            for robot in self.robots:
                robot_pos = np.array([robot.x, robot.y])
                robot_idx = robot.idx

                for i, segment in enumerate(segments):
                    start_dist = np.linalg.norm(robot_pos - segment[0])
                    end_dist = np.linalg.norm(robot_pos - segment[-1])

                    if start_dist < end_dist:
                        distances.append((robot_idx, i, start_dist, False))  # Normal order
                    else:
                        distances.append((robot_idx, i, end_dist, True))  # Reversed order

            # Sort distances by closest match first
            distances.sort(key=lambda x: x[2])

            # Assign segments to robots in a fair way
            assigned_robots = set()
            for robot_idx, seg_idx, _, reversed_order in distances:
                if seg_idx in unassigned_segments and robot_idx not in assigned_robots:
                    assigned_robots.add(robot_idx)
                    unassigned_segments.remove(seg_idx)
                    robot_segments[robot_idx] = segments[seg_idx][::-1].tolist() if reversed_order else segments[seg_idx].tolist()

            # Save it to behavior data
            behavior_dict["data"]["segments"] = robot_segments
            behavior_dict["data"]["segment_indexes"] = {robot.idx: 0 for robot in self.robots}

            # Set beahavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict

        elif behavior_name == "random_walk":
            # Initialize behavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict

    def step(self, step_ms):
        """Perform one simulation step for the group"""
        bhvr = self.bhvr

        # Update virtual center
        self._update_virtual_center()

        # Calculate movement based on current behavior
        match bhvr["name"]:
            # State machine for formation and following trajectory
            case "form_and_follow_trajectory":
                match bhvr["state"]:
                    case 0:
                        for robot in self.robots:
                            robot.update_target(
                                bhvr["data"]["init_virtual_center"][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["data"]["init_virtual_center"][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move(step_ms)
                        
                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            bhvr["state"] = 1

                    case 1:
                        for robot in self.robots:
                            robot.update_target_angle(
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move(step_ms)

                        # Check transition to next state
                        if all(r.is_robot_aligned() for r in self.robots):
                            bhvr["state"] = 2
                    
                    case 2:
                        for robot in self.robots:
                            robot.update_target(
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["trajectory"][bhvr["data"]["traj_index"]][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move(step_ms)

                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            # Check if traj_index is the last index
                            if bhvr["data"]["traj_index"] == len(bhvr["params"]["trajectory"]) - 1:
                                bhvr["state"] = -1
                                bhvr["data"]["traj_index"] = 0
                            else:
                                bhvr["state"] = 1
                                bhvr["data"]["traj_index"] += 1

            # State machine for coverage
            case "cover_shape":
                for robot in self.robots:
                    # Get the robot's assigned segment
                    segment = bhvr["data"]["segments"][robot.idx]
                    segment_index = bhvr["data"]["segment_indexes"][robot.idx]

                    # Get the robot's target position
                    target_pos = segment[segment_index]

                    # Update the robot's target position
                    robot.update_target(target_pos[0], target_pos[1])

                    # Move the robot
                    robot.move(step_ms)

                    if robot.is_robot_in_position():
                        # Check if robot is at the end of its segment
                        if segment_index >= len(segment) - 1:
                            segment_index = -1  # Mark as finished
                        elif segment_index >= 0 and segment_index < len(segment) - 1:
                            # Increment the segment index
                            segment_index += 1

                        # Store the updated index back in behavior data
                        bhvr["data"]["segment_indexes"][robot.idx] = segment_index

                # Check if all robots have segments finished
                if all(index == -1 for index in bhvr["data"]["segment_indexes"].values()):
                    bhvr["state"] = -1


            # State machine for random walk
            case "random_walk":
                for robot in self.robots:
                    # If they havent reached their target, move them
                    if not robot.is_robot_in_position():
                        robot.move(step_ms)
                    # If they have reached their target, pick a new target
                    else: 
                        robot.update_target(robot.x + random.uniform(-0.5, 0.5), robot.y + random.uniform(-0.5, 0.5))

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
    
    def __init__(self, robots, step_ms=10):
        """
        Initialize a swarm of robots
        
        Args:
            robots (list[Robot]): List of robots in the swarm
        """  
        self.step_ms = step_ms
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

    def assign_cover_shape_behavior_to_group(self, group_idx, coords):
        """Assign cover_shape behavior to a group"""
        self._get_group_by_idx(group_idx).set_behavior({
            "name": "cover_shape",
            "params": {
                "coords": coords
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
                
            group.step(self.step_ms)

class SwarmAgent:
    def __init__(self, app, swarm: Swarm, features_geojson):
        self.app = app
        self.swarm = swarm
        self.features_geojson = features_geojson
        self._set_feature_name_lists()
        self._set_epsg_based_on_features()
        self.spatial_awareness_tools = [
            self.find_nearby_shapes_to_group,
            self.find_nearby_shapes_to_shape
        ]
        self.tools = [
            self.gen_group_by_ids,
            self.gen_groups_by_clustering,
            self.random_walk,
            self.form_and_follow_trajectory,
            self.form_and_move_around_shape,
            self.form_and_move_to_shape,
            self.cover_shape
        ]
        self.memory = []
        self.system_prompt = SYSTEM_PROMPT
        self.spatial_awareness_system_prompt = SPATIAL_AWARENESS_SYSTEM_PROMPT

        
    def _format_memory(self, memory=None):
        """Convert memory entries to proper message format"""
        formatted = []
        memory = memory if memory else self.memory
        for entry in memory:
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
    
    def _get_spatial_awareness_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__,
                    "parameters": self._get_params_schema(tool)
                }
            } for tool in self.spatial_awareness_tools
        ]
    
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
        if func.__name__ == "find_nearby_shapes_to_shape":
            return {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    }
                },
                "required": ["name"]
            }
        elif func.__name__ == "find_nearby_shapes_to_group":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"}
                },
                "required": ["group_idx"]
            }
        elif func.__name__ == "gen_group_by_ids":
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
                        "minimum": 5.0,
                        "maximum": 30.0
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
                        "minimum": 5.0,
                        "maximum": 30.0
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
                        "minimum": 5.0,
                        "maximum": 30.0
                    },
                    "name": {
                        "type": "string",
                    }
                },
                "required": ["group_idx", "formation_shape", "formation_radius", "name"]
            }
        elif func.__name__ == "cover_shape":
            return {
                "type": "object",
                "properties": {
                    "group_idx": {"type": "integer"},
                    "name": {
                        "type": "string",
                    },
                },
                "required": ["group_idx", "name"]
            }
        
    def _get_feature_by_unique_name(self, unique_name):
        """Get GeoJSON feature by unique name"""
        for feature in self.features_geojson["features"]:
            if "unique_name" not in feature["properties"]:
                continue
            if feature["properties"]["unique_name"] == unique_name:
                return feature
        return None
        
    def _set_feature_name_lists(self):
        # Create list of unique_names that correspond to Polygon features
        polygon_names = []
        linestring_names = []
        point_names = []

        for feature in self.features_geojson["features"]:
            try :
                if feature["geometry"]["type"] == "Polygon" or feature["geometry"]["type"] == "MultiPolygon":
                    polygon_names.append(feature["properties"]["unique_name"])
                elif feature["geometry"]["type"] == "LineString":
                    linestring_names.append(feature["properties"]["unique_name"])
                elif feature["geometry"]["type"] == "Point":
                    point_names.append(feature["properties"]["unique_name"])
            except KeyError:
                self.app.logger.warning(f"Feature without unique name with id: {feature["id"]} and properties: {feature["properties"]}")
                
        # Sort them alphabetically
        polygon_names.sort()
        linestring_names.sort()
        point_names.sort()

        # Store them in the class
        self.point_names = point_names
        self.linestring_names = linestring_names
        self.polygon_names = polygon_names
    
    def _set_epsg_based_on_features(self):
        """
        Set the EPSG code based on the features in the GeoJSON file.
        This is a placeholder function and should be implemented based on your requirements.
        """
        # Get feature with unique name "bbox_center"
        bbox_center = self._get_feature_by_unique_name("bbox_center")
        coords = bbox_center["geometry"]["coordinates"]
        self.epsg = guess_utm_crs(coords[0], coords[1])
        
    def send_message(self, user_input: str):
        # Build current context strings
        robot_idxs = " ,".join(map(str, [robot.idx for robot in self.swarm.robots]))
        group_idxs = " ,".join(map(str, [group.idx for group in self.swarm.groups]))
        groups_str = ""
        for group in self.swarm.groups:
            group_robot_idxs = " ,".join(map(str, [robot.idx for robot in group.robots]))
            groups_str += f"[{group.idx}: [{group_robot_idxs}], "

        # Build formatted spatial awareness system prompt
        formatted_spatial_awareness_system_prompt = self.spatial_awareness_system_prompt.format(
            point_names=self.point_names,
            linestring_names=self.linestring_names,
            polygon_names=self.polygon_names,
            robot_idxs=robot_idxs, 
            groups_str=groups_str
        )

        # Spatial awareness messages
        spatial_awareness_messages = [
            {"role": "system", "content": formatted_spatial_awareness_system_prompt},
            {"role": "user", "content": user_input}
        ]
        # Init spatial awareness memory
        spatial_awareness_memory = []

        # Get spatial awareness information
        spatial_awareness_info = ""
        try:
            # Call the API for tool calls
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=2048,
                messages=spatial_awareness_messages,
                tools=self._get_spatial_awareness_tool_schemas()
            )

            response_message = response.choices[0].message
            spatial_awareness_memory.append({"role": "user", "content": user_input})

            # Check if there are tool calls
            if response_message.tool_calls:
                # Process tool calls
                spatial_awareness_tool_responses = []
                for tool_call in response_message.tool_calls:
                    self.app.logger.info(f"Processing tool call: {tool_call.function.name}")
                    func = next(t for t in self.spatial_awareness_tools if t.__name__ == tool_call.function.name)
                    args = json.loads(tool_call.function.arguments)
                    self.app.logger.info(f"Calling {tool_call.function.name} with {args}")
                    result = func(**args)
                    spatial_awareness_tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": result
                    })

                # Store assistant message with tool calls
                spatial_awareness_memory.append({
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
                spatial_awareness_memory.extend(spatial_awareness_tool_responses)
                spatial_awareness_memory = self._format_memory(spatial_awareness_memory)
                final_spatial_awareness_messages = [
                    {"role": "system", "content": formatted_spatial_awareness_system_prompt},
                    *spatial_awareness_memory
                ]
                # Get final response with full context
                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0,
                    max_tokens=2048,
                    messages=final_spatial_awareness_messages
                )
                spatial_awareness_info = final_response.choices[0].message.content
                self.app.logger.info(f"Spatial awareness info with calls: {spatial_awareness_info}")

            else:
                spatial_awareness_info = response_message.content
                self.app.logger.info(f"Spatial awareness info without calls: {spatial_awareness_info}")
        except Exception as e:
            self.app.logger.error(f"Error: {e}")
            spatial_awareness_info = "Spatial awareness information could not be retrieved."

        # Build message history with correct structure
        formatted_system_prompt = self.system_prompt.format(
            robot_idxs=robot_idxs, 
            groups_str=groups_str,
            spatial_awareness_info=spatial_awareness_info
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
                max_tokens=4096,
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
    def find_nearby_shapes_to_shape(self, name: str):
        """
        Get nearby shapes based on the name of the feature
        """
        # Get the GeoJSON feature with unique_name = name
        source_feature = self._get_feature_by_unique_name(name)

        # Get the nearby features
        nearby_features = get_nearby_features_to_a_given_feature(source_feature, self.features_geojson["features"], epsg=self.epsg)
        
        # Get only the 10 closest features
        nearby_features = nearby_features[:10]

        # Stringify the nearby features in a string that gives name and distance
        nearby_features_str = ""
        for feature in nearby_features:
            feature_name = feature["name"]
            feature_distance = feature["distance"]
            nearby_features_str += f"{feature_name} ({feature_distance:.2f} m), "
        nearby_features_str = nearby_features_str[:-2]  # Remove last comma and space
        return f"Nearby features to {name}: {nearby_features_str}"
    
    def find_nearby_shapes_to_group(self, group_idx: int):
        """
        Get nearby shapes based on the group index
        """
        # Get the group by index
        group = self.swarm._get_group_by_idx(group_idx)
        
        # Get the virtual center of the group
        group_virtual_center = group.virtual_center

        # Get the nearby features
        nearby_features = get_nearby_features_to_a_given_coord(group_virtual_center, self.features_geojson["features"], epsg=self.epsg)
        
        # Get only the 10 closest features
        nearby_features = nearby_features[:10]

        # Stringify the nearby features in a string that gives name and distance
        nearby_features_str = ""
        for feature in nearby_features:
            feature_name = feature["name"]
            feature_distance = feature["distance"]
            nearby_features_str += f"{feature_name} ({feature_distance:.2f} m), "
        nearby_features_str = nearby_features_str[:-2] # Remove last comma and space
        return f"Nearby features to group {group_idx}: {nearby_features_str}"
    
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
        if formation_radius < 5 or formation_radius > 30:
            return "Invalid radius value: must be between 5 and 30"
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
        if formation_radius < 5 or formation_radius > 30:
            return "Invalid radius value: must be between 5 and 30"
        if formation_shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in self.polygon_names + self.linestring_names:
            return "Invalid feature name: must be one of the available multipolygon, polygon, or linestring features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = self._get_feature_by_unique_name(name)
        if feature is None:
            return "Feature not found"
        
        # Get the coordinates of the feature depending on its geometry type
        if feature["geometry"]["type"] == "Polygon" or feature["geometry"]["type"] == "MultiPolygon":
            if feature["geometry"]["type"] == "MultiPolygon":
                coords = feature["geometry"]["coordinates"][0][0]
            else:
                coords = feature["geometry"]["coordinates"][0]

            # Reorder coordinates so that initial position is the closest to the center of the group
            # If it is the coordinate at index 2, shift that coordinate to be the first one and put any previous ones at the end
            group_virtual_center = self.swarm._get_group_by_idx(group_idx).virtual_center
            closest_coord_idx = np.argmin([np.linalg.norm(np.array(coord) - np.array(group_virtual_center)) for coord in coords])
            coords = coords[closest_coord_idx:] + coords[1:closest_coord_idx] + [coords[closest_coord_idx]]

        elif feature["geometry"]["type"] == "LineString":
            coords = feature["geometry"]["coordinates"]
            # Check if the first or the last coordinate is the closest to the center of the group
            # If the last one is closer than the first one, reverse the order of the coordinates
            group_virtual_center = self.swarm._get_group_by_idx(group_idx).virtual_center
            closest_coord_idx = np.argmin([np.linalg.norm(np.array(coord) - np.array(group_virtual_center)) for coord in coords])
            if closest_coord_idx > len(coords) // 2:
                coords = coords[::-1]
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
        if formation_radius < 5 or formation_radius > 30:
            return "Invalid radius value: must be between 5 and 30"
        if formation_shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in self.polygon_names + self.linestring_names + self.point_names:
            return "Invalid feature name: must be one of the available features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = self._get_feature_by_unique_name(name)
        if feature is None:
            return "Feature not found"
        
        # Get the coordinates of the feature depending on its geometry type
        if feature["geometry"]["type"] == "Polygon" or feature["geometry"]["type"] == "MultiPolygon":
            if feature["geometry"]["type"] == "MultiPolygon":
                coords = feature["geometry"]["coordinates"][0][0]
            else:
                coords = feature["geometry"]["coordinates"][0]

            x = np.mean([coord[0] for coord in coords])
            y = np.mean([coord[1] for coord in coords])
            coords = [[x, y]]
        elif feature["geometry"]["type"] == "LineString":
            coords = feature["geometry"]["coordinates"]
            center = coords[len(coords) // 2]
            coords = [center]
        elif feature["geometry"]["type"] == "Point":
            coords = [feature["geometry"]["coordinates"]]
        else:
            return "Invalid feature type"
        
        # self.app.logger.info(f"Feature coordinates: {coords}")

        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, formation_shape, formation_radius, coords
        )
        return f"form_and_move_to_shape behavior assigned to group {group_idx} with formation shape {formation_shape}, radius {formation_radius}, and feature name {name}"
    
    def cover_shape(self, group_idx: int, name: str):
        # Validation logic
        if name not in self.polygon_names:
            return "Invalid feature name: must be one of the available multipolygon or polygon features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = self._get_feature_by_unique_name(name)
        if feature is None:
            return "Feature not found"
        
        # Get the coordinates of the feature depending on its geometry type
        if feature["geometry"]["type"] == "MultiPolygon":
            # Outer polygon
            poly_coords = feature["geometry"]["coordinates"][0][0]
            poly = Polygon(poly_coords)

            geo_poly = Geometries.GeoPolygon(poly)
            geo_poly.set_crs(self.epsg) # EPSG:2062 for coords in Spain
            geo_poly.buffer(-2) # Distance in meters from the border of the polygon

            # Inner polygons or holes
            holes = []
            for coords in feature["geometry"]["coordinates"][0][1:]:
                hole = Polygon(coords)
                holes.append(hole)
            
            # Create geo holes multipolygon
            geo_holes = Geometries.GeoMultiPolygon(holes)
            geo_holes.set_crs(self.epsg) # EPSG:2062 for coords in Spain
            geo_holes.buffer(2) # Distance in meters from the border of the holes

            # Build polygons list object
            polygons_list = Geometries.decompose_polygon(
                geo_poly.get_geometry(),
                obstacles=geo_holes.get_geometry(),
            )
        elif feature["geometry"]["type"] == "Polygon":
            # Polygon
            poly_coords = feature["geometry"]["coordinates"][0]
            poly = Polygon(poly_coords)
            geo_poly = Geometries.GeoPolygon(poly)
            geo_poly.set_crs(self.epsg) # EPSG:2062 for coords in Spain
            geo_poly.buffer(-2) # Distance in meters from the border of the polygon

            # Build polygons list object
            polygons_list = Geometries.decompose_polygon(
                geo_poly.get_geometry(),
                obstacles=None,
            )

        else:
            return "Invalid feature type"
        
        # Set offset (separation between sweeps depending on overlap ratio, altitude in meters and field of view in degrees)
        offset = Geometries.get_sweep_offset(0.1, 30, 60)
        coords = []
        for decomposed_poly in polygons_list:
            sweeps_connected = Geometries.generate_sweep_pattern(
                decomposed_poly, offset, clockwise=True, connect_sweeps=True
            )
            # self.app.logger.info(sweeps_connected)
            # Get the coordinates of the sweeps given that they are linestrings
            for sweep in sweeps_connected:
                # Turn sweep linestring into trajectory
                sweep_trajectory = Geometries.GeoTrajectory(sweep, crs=self.epsg)
                # Turn them back into geodesic coordinates
                sweep_trajectory.set_crs("WGS84")
                coords += list(sweep_trajectory.geometry.coords)

        # Reorder coordinates so that initial position is the closest to the center of the group
        group_virtual_center = self.swarm._get_group_by_idx(group_idx).virtual_center
        closest_coord_idx = np.argmin([np.linalg.norm(np.array(coord) - np.array(group_virtual_center)) for coord in coords])
        coords = coords[closest_coord_idx:] + coords[:closest_coord_idx]

        # self.app.logger.info(f"Coords: {coords}")
        self.swarm.assign_cover_shape_behavior_to_group(
            group_idx, coords
        )
        
        return f"cover_shape behavior assigned to group {group_idx} with feature name {name}"

# AUXILIARY FUNCTIONS
def checkIfCoordInPolygon(coords, polygon):
    """
    Check if a coordinate point is inside a polygon.
    
    Args:
        coords (list): List of coordinates [lon, lat].
        polygon (list): List of polygon coordinates [[lon1, lat1], [lon2, lat2], ...].
        
    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    lon, lat = coords
    poly = Polygon(polygon)
    return poly.contains(Point(lon, lat))

def transform_geometry(geom, source_epsg=4326, target_epsg=3857):
    """
    Transforms a Shapely geometry from source_epsg (WGS84) to target_epsg (meters).
    Default target CRS is EPSG:3857 (Web Mercator).
    """
    transformer = Transformer.from_crs(CRS.from_epsg(source_epsg), CRS.from_epsg(target_epsg), always_xy=True)
    return transform(transformer.transform, geom)  # Apply transformation

def compute_distance(feature1, feature2, target_epsg=3857):
    """
    Computes the shortest distance between two GeoJSON features in meters.
    
    Parameters:
    - feature1, feature2: GeoJSON-like dictionaries with 'geometry'.
    - target_epsg: EPSG code for distance calculation (default: EPSG:3857).
    
    Returns:
    - Distance in meters.
    """
    # Convert GeoJSON geometries to Shapely objects
    geom1 = shape(feature1["geometry"])
    geom2 = shape(feature2["geometry"])

    # Reproject to a coordinate system that measures in meters
    geom1_meters = transform_geometry(geom1, target_epsg=target_epsg)
    geom2_meters = transform_geometry(geom2, target_epsg=target_epsg)

    # Compute shortest distance in meters
    return geom1_meters.distance(geom2_meters)

def get_nearby_features_to_a_given_coord(coord, features, epsg):
    """
    Get the closest features to a given coordinate point.
    """
    # Create a GeoJSON feature for the coordinate point
    feature_point = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coord
        },
        "properties": {}
    }
    
    closest_features = []
    
    for feature in features:
        # Skip features without unique_name
        if "unique_name" not in feature["properties"]:
            continue
        feature_name = feature["properties"]["unique_name"]
        
        # Compute distance between source feature and current feature
        target_epsg = int(epsg.split(":")[1])
        distance = compute_distance(feature_point, feature, target_epsg=target_epsg)
        
        # Append to closest features list
        closest_features.append({"name": feature_name, "distance": distance})

    # Sort results by distance
    return sorted(closest_features, key=lambda x: x["distance"])

def get_nearby_features_to_a_given_feature(source_feature, features, epsg):
    """
    Get nearby features to a given source feature.
    """
    # Get the GeoJSON feature with unique_name = name
    if source_feature is None:
        return "Feature not found"
    
    nearby_features = []
    
    for feature in features:
        if feature == source_feature:
            continue  # Skip the source feature itself
        # Skip features without unique_name
        if "unique_name" not in feature["properties"]:
            continue
        feature_name = feature["properties"]["unique_name"]
        
        # Compute distance between source feature and current feature
        target_epsg = int(epsg.split(":")[1])
        distance = compute_distance(source_feature, feature, target_epsg=target_epsg)
        
        # Append to nearby features list
        nearby_features.append({"name": feature_name, "distance": distance})

    # Sort results by distance
    return sorted(nearby_features, key=lambda x: x["distance"])

def guess_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1  # Calculate UTM zone
    epsg = 32600 + zone if lat >= 0 else 32700 + zone  # 326 for Northern, 327 for Southern Hemisphere
    return f"EPSG:{epsg}"

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
    
def calculate_lon_lat_per_meter(coord):
    """
    Calculate the approximate conversion factors for longitude and latitude per meter.
    """
    x_center, y_center = coord

    # Convert meters to degrees (approximate)
    lat_per_meter = 1 / 111320  # degrees per meter
    lon_per_meter = 1 / (111320 * np.cos(np.radians(y_center)))  # degrees per meter

    return lon_per_meter, lat_per_meter
    
def compute_formation_coordinate_offsets(coord, n, formation_shape, formation_radius):
    if n <= 0:
        raise ValueError("Number of robots must be greater than 0")
    if n == 1:
        return np.array([[0, 0]])
    
    # Compute what a meter corresponds to in the current coordinate system
    lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter(coord)

    # Turn formation offsets in meters into degrees
    formation_positions = compute_formation_positions(n, formation_shape, formation_radius)
    formation_positions[:, 0] *= lon_per_meter
    formation_positions[:, 1] *= lat_per_meter
    return formation_positions

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