"""
Agents

This modules handles the agents interacting with LLM models.
It also defines how the agents collaborate with each other and with the swarm.
"""

import os
import math
import random
import numpy as np
from functools import wraps

from litellm import _turn_on_debug
_turn_on_debug()
from litellm import completion
from typing import Dict, List, Tuple, Any
import geojson
import json

from shapely.geometry import shape, Point, LineString, Polygon
from shapely.ops import transform, nearest_points
from pyproj import CRS, Transformer
from trajgenpy import Geometries

from src.swarm import Swarm

# Function parameters metadata decorator for tool schemas
def add_parameters_schema(params_schema):
    """
    Decorator to add parameters metadata to a function.
    """
    def decorator(func):
        func.__params_schema__ = params_schema
        return func
    return decorator

# CONSTANTS
MODEL_NAMES = [
    "openai/gpt-4o-mini",
    "gemini/gemini-2.0-flash",
    "ollama/llama3.1"
]
API_BASE_URLS = {
    "openai/gpt-4o-mini": "",
    "gemini/gemini-2.0-flash": "",
    "ollama/llama3.1": "http://localhost:11434"
}

MODEL_NAME = MODEL_NAMES[2]
with open("mykeys/openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()
with open("mykeys/gemini_api_key.txt", "r") as f:
    GEMINI_API_KEY = f.read()

# Set API keys as environment variables for LiteLLM
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# PROMPTS
# Load swarm agent system prompt template
with open("data/prompts/swarm_agent_system.txt", "r") as f:
    SWARM_AGENT_SYSTEM_PROMPT = f.read()
# Load swarm agent user prompt template
with open("data/prompts/swarm_agent_user.txt", "r") as f:
    SWARM_AGENT_USER_PROMPT = f.read()
# Load spatial agent system prompt template
with open("data/prompts/spatial_agent_system.txt", "r") as f:
    SPATIAL_AGENT_SYSTEM_PROMPT = f.read()
# Load spatial agent user prompt template
with open("data/prompts/spatial_agent_user.txt", "r") as f:
    SPATIAL_AGENT_USER_PROMPT = f.read()

# CLASSES
class MultiAgent:
    """
    Class to handle multi-agent functionalities.
    """
    def __init__(self, app, swarm: Swarm, features_geojson):
        self.app = app
        self.spatial_agent = SpatialAgent(app, swarm, features_geojson)
        self.swarm_agent = SwarmAgent(app, swarm, features_geojson)

    def send_message(self, user_input: str):
        """
        Handle a message from the user
        """
        # First get spatial information from the spatial agent
        spatial_info = self.spatial_agent.send_message(user_input)

        # Then call the swarm agent with the spatial information
        swarm_info = self.swarm_agent.send_message(user_input, spatial_info)

        return swarm_info

class Agent:
    """
    Class to inherit from for all agents.
    """
    def __init__(self, app):
        self.app = app
        self.swarm = None
        self.memory = []
        self.system_prompt = ""
        self.user_prompt = ""
        self.tools = []
        self.features_geojson = None
        self.point_names = []
        self.linestring_names = []
        self.polygon_names = []
        self.epsg = None
    
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
    
    def _get_tool_schemas(self):
        tool_schemas = []
        for tool in self.tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "strict": True,
                    "name": tool.__name__,
                    "description": tool.__doc__.strip(),
                    "parameters": tool.__params_schema__
                }
            }
            # Add tool schema to the list of schemas
            tool_schemas.append(tool_schema)

        # Return the list of tool schemas
        return tool_schemas
    
    def send_message(self):
        """
        To be implemented in subclasses deriving from this class.
        This method should handle the message sending process to the LLM.
        """
        raise NotImplementedError("send_message method not implemented in Agent class.")
    

class SpatialAgent(Agent):
    """
    Class to handle spatial awareness information regarding robots and geospatial features.
    """
    def __init__(self, app, swarm: Swarm, features_geojson):
        self.app = app
        self.swarm = swarm

        # Features handling
        self.features_geojson = features_geojson
        self.point_names, self.linestring_names, self.polygon_names = get_feature_name_lists(features_geojson)
        self.epsg = get_epsg_based_on_features(features_geojson)

        # Tools
        self.tools = [
            self.find_nearby_features_to_feature,
            self.find_nearby_features_to_group
        ]

        # Memory
        self.memory = []

        # System and user prompts
        self.system_prompt = SPATIAL_AGENT_SYSTEM_PROMPT
        self.user_prompt = SPATIAL_AGENT_USER_PROMPT

    def send_message(self, user_input):
        # Build current context strings
        robot_idxs = " ,".join(map(str, [robot.idx for robot in self.swarm.robots]))
        group_idxs = " ,".join(map(str, [group.idx for group in self.swarm.groups]))
        groups_str = ""
        for group in self.swarm.groups:
            group_robot_idxs = " ,".join(map(str, [robot.idx for robot in group.robots]))
            groups_str += f"[{group.idx}: [{group_robot_idxs}], "

        # Build formatted prompts
        formatted_system_prompt = self.system_prompt
        formatted_user_prompt = self.user_prompt.format(
            robot_idxs=robot_idxs,
            groups_str=groups_str,
            point_names=self.point_names,
            linestring_names=self.linestring_names,
            polygon_names=self.polygon_names,
            user_input=user_input
        )

        # Initialize messages
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        # Reset memory
        self.memory = []

        try:
            # Call the API for tool calls
            response = completion(
                model=MODEL_NAME,
                api_base=API_BASE_URLS[MODEL_NAME],
                temperature=0,
                max_tokens=2048,
                messages=messages,
                tools=self._get_tool_schemas()
            )

            response_message = response.choices[0].message
            self.memory.append({"role": "user", "content": formatted_user_prompt})

            # Tool calls
            if response_message.tool_calls:
                tool_responses = []
                for tool_call in response_message.tool_calls:
                    self.app.logger.info(f"Processing tool call: {tool_call.function.name}")
                    func = next(t for t in self.tools if t.__name__ == tool_call.function.name)
                    args = json.loads(tool_call.function.arguments)
                    self.app.logger.info(f"Calling {tool_call.function.name} with {args}")
                    result = func(**args)
                    self.app.logger.info(f"Result: {result}")
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
                final_messages = [
                    {"role": "system", "content": formatted_system_prompt},
                    *self._format_memory()
                ]
                # Get final response with full context
                final_response = completion(
                    model=MODEL_NAME,
                    api_base=API_BASE_URLS[MODEL_NAME],
                    temperature=0,
                    max_tokens=2048,
                    messages=final_messages
                )
                final_response_message = final_response.choices[0].message
                self.app.logger.info(f"Spatial awareness info with calls: {final_response_message.content}")
                return final_response_message.content
            # No tool calls
            else:
                self.app.logger.info(f"Spatial awareness info without calls: {response_message.content}")
                return response_message.content
            
        # Show detailed error message
        except Exception as e:
            self.app.logger.error(e)
            return "Spatial awareness information could not be retrieved due to an error."
    
    # Tool implementations
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "name": {
                "description": "Feature name to find nearby features to (must be within valid feature names lists).",
                "type": "string"
            }
        },
        "required": ["name"],
        "additionalProperties": False
    })
    def find_nearby_features_to_feature(self, name: str):
        """
        Get nearby map features to a given feature with their respective distances and angles. Where angle in degrees is 0 degrees North, 90 degrees East, 180 degrees South and 270 degrees West.
        """
        # Get the GeoJSON feature with unique_name = name
        source_feature = get_feature_by_unique_name(self.features_geojson, name)

        # Get the nearby features
        nearby_features = get_nearby_features_to_a_given_feature(source_feature, self.features_geojson["features"], epsg=self.epsg)
        
        # Get only the 10 closest features
        nearby_features = nearby_features[:10]

        # Stringify the nearby features in a string that gives name and distance
        result_str = ""
        intersecting_features_str = ""
        nearby_features_str = ""
        for feature in nearby_features:
            feature_name = feature["name"]
            feature_distance = feature["distance"]
            if feature_distance > 0:
                feature_angle = feature["angle"]
                nearby_features_str += f"{feature_name} ({feature_distance:.2f} m, {feature_angle:.1f}°), "
            else:
                intersecting_features_str += f"{feature_name}, "

        if intersecting_features_str:
            intersecting_features_str = intersecting_features_str[:-2]
            result_str += f"Intersecting features with {name}: {intersecting_features_str}\n"
        nearby_features_str = nearby_features_str[:-2]  # Remove last comma and space
        result_str += f"Nearby features to {name}: {nearby_features_str}"
        return result_str
    
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to find nearby features to (group must exist).",
                "type": "integer"
            }
        },
        "required": ["group_idx"],
        "additionalProperties": False
    })
    def find_nearby_features_to_group(self, group_idx: int):
        """
        Get nearby map features to a given group with their respective distances and angles. Where angle in degrees is 0 degrees North, 90 degrees East, 180 degrees South and 270 degrees West.
        """
        # Get the group by index
        group = self.swarm._get_group_by_idx(group_idx)
        
        # Get the virtual center of the group
        group_virtual_center = group.virtual_center

        # Get the nearby features
        nearby_features = get_nearby_features_to_a_given_coord(group_virtual_center, self.features_geojson["features"], epsg=self.epsg)
        
        # Get only the 20 closest features
        nearby_features = nearby_features[:20]

        # Stringify the nearby features in a string that gives name and distance
        result_str = ""
        intersecting_features_str = ""
        nearby_features_str = ""
        for feature in nearby_features:
            feature_name = feature["name"]
            feature_distance = feature["distance"]
            if feature_distance > 0:
                feature_angle = feature["angle"]
                nearby_features_str += f"{feature_name} ({feature_distance:.2f} m, {feature_angle:.1f}°), "
            else:
                intersecting_features_str += f"{feature_name}, "
        if intersecting_features_str:
            intersecting_features_str = intersecting_features_str[:-2]
            result_str += f"Group {group_idx} is within: {intersecting_features_str}\n"
            
        nearby_features_str = nearby_features_str[:-2] # Remove last comma and space
        result_str += f"Nearby features to group {group_idx}: {nearby_features_str}"
        return result_str
    

class SwarmAgent(Agent):
    """
    Class to handle swarm agent functionalities.
    """
    def __init__(self, app, swarm: Swarm, features_geojson):
        self.app = app
        self.swarm = swarm

        # Features handling
        self.features_geojson = features_geojson
        self.point_names, self.linestring_names, self.polygon_names = get_feature_name_lists(features_geojson)
        self.epsg = get_epsg_based_on_features(features_geojson)

        # Tools
        self.tools = [
            self.gen_group_by_ids,
            self.gen_groups_by_clustering,
            self.random_walk,
            self.form_and_follow_trajectory,
            self.form_and_move_around_shape,
            self.form_and_move_to_shape,
            self.cover_shape
        ]

        # Memory
        self.memory = []

        # System and user prompts
        self.system_prompt = SWARM_AGENT_SYSTEM_PROMPT
        self.user_prompt = SWARM_AGENT_USER_PROMPT

        # self.cover_shape(0, "natural_heath_4")
        # self.cover_shape(0, "natural_heath_8")
        
    def send_message(self, user_input: str, spatial_info: str = ""):
        # Build current context strings
        robot_idxs = " ,".join(map(str, [robot.idx for robot in self.swarm.robots]))
        group_idxs = " ,".join(map(str, [group.idx for group in self.swarm.groups]))
        groups_str = ""
        for group in self.swarm.groups:
            group_robot_idxs = " ,".join(map(str, [robot.idx for robot in group.robots]))
            groups_str += f"[{group.idx}: [{group_robot_idxs}], "

        # Build formatted prompts
        formatted_system_prompt = self.system_prompt
        formatted_user_prompt = self.user_prompt.format(
            robot_idxs=robot_idxs,
            groups_str=groups_str,
            spatial_info=spatial_info,
            user_input=user_input
        )


        # self.app.logger.info(f"Formatted system prompt: {formatted_system_prompt}")
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            *self._format_memory(),
            {"role": "user", "content": formatted_user_prompt}
        ]

        try :
            # First API call (potentially with tool calls)
            response = completion(
                model=MODEL_NAME,
                api_base=API_BASE_URLS[MODEL_NAME],
                temperature=0,
                max_tokens=4096,
                messages=messages,
                tools=self._get_tool_schemas()
            )
            
            response_message = response.choices[0].message
            self.memory.append({"role": "user", "content": formatted_user_prompt})
            
            # Tool calls
            if response_message.tool_calls:
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
                
                # Get final response using tool responses and full context
                final_messages = [
                    {"role": "system", "content": formatted_system_prompt},
                    *self._format_memory()
                ]
                final_response = completion(
                    model=MODEL_NAME,
                    api_base=API_BASE_URLS[MODEL_NAME],
                    temperature=0,
                    max_tokens=4192,
                    messages=final_messages
                )
                final_response_message = final_response.choices[0].message
                self.memory.append({"role": "assistant", "content": final_response_message.content})
                self.app.logger.info(f"Response after calls: {final_response_message.content}")
                return final_response_message.content
            # No tool calls
            else:
                self.memory.append({"role": "assistant", "content": response_message.content})
                self.app.logger.info(f"Response without calls: {response_message.content}")
                return response_message.content
        
        except Exception as e:
            self.app.logger.error(f"Error: {e}")
            return "There was an error processing your request. Please try clarifying your request."

    # Tool implementations
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "robot_idxs": {
                "description": "List of robot IDs to group.",
                "type": "array",
                "items": {"type": "integer"}
            }
        },
        "required": ["robot_idxs"], 
        "additionalProperties": False
    })
    def gen_group_by_ids(self, robot_idxs: List[int]):
        """
        Generate a group of robots by their IDs.
        """
        new_group = self.swarm.gen_group_by_ids(robot_idxs)
        return f"Drones {', '.join(map(str, robot_idxs))} grouped successfully in group {new_group.idx}"

    @add_parameters_schema({
        "type": "object",
        "properties": {
            "num_groups": {
                "description": "Number of groups to generate (must be between 0 and the number of robots in the swarm).",
                "type": "integer"
            }
        },
        "required": ["num_groups"],
        "additionalProperties": False
    })
    def gen_groups_by_clustering(self, num_groups: int):
        """
        Generate groups of robots by proximity clustering.
        """
        if num_groups < 0 or num_groups > len(self.swarm.robots):
            return "Invalid number of groups"
        self.swarm.gen_groups_by_clustering(num_groups)
        return f"Drones grouped successfully in {num_groups} groups"

    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to assign random walk behavior to.",
                "type": "integer"
            }
        },
        "required": ["group_idx"],
        "additionalProperties": False
    })
    def random_walk(self, group_idx: int):
        """
        Assign random walk behavior to a group of robots.
        """
        self.swarm.assign_random_walk_behavior_to_group(group_idx)
        return f"random_walk behavior assigned to group {group_idx}"
    
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to assign form and follow trajectory behavior to.",
                "type": "integer"
            },
            "shape": {
                "description": "Formation shape.",
                "type": "string", 
                "enum": ["circle", "square", "triangle", "hexagon"]
            },
            "radius": {
                "description": "Formation radius in meters (must be between 10 and 50).",
                "type": "number",
            },
            "trajectory": {
                "description": "Trajectory coordinates (must be a list of lon/lat pairs).",
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            }
        },
        "required": ["group_idx", "shape", "radius", "trajectory"],
        "additionalProperties": False
    })
    def form_and_follow_trajectory(self, group_idx: int, shape: str, radius: float, trajectory: List[Tuple[float, float]]):
        """
        Assign form and follow trajectory behavior to a group of robots.
        """
        # Validation logic
        if radius < 5 or radius > 50:
            return "Invalid radius value: must be between 5 and 50"
        if shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if len(trajectory) < 1 or any(len(pos) != 2 for pos in trajectory):
            return "Invalid trajectory"
        
        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, shape, radius, trajectory
        )
        return f"form_and_follow_trajectory behavior assigned to group {group_idx} with formation shape {shape}, radius {radius}, and trajectory {trajectory}"

    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to assign form and move around shape behavior to.",
                "type": "integer"
            },
            "shape": {
                "description": "Formation shape.",
                "type": "string", 
                "enum": ["circle", "square", "triangle", "hexagon"]
            },
            "radius": {
                "description": "Formation radius in meters (must be between 10 and 50).", 
                "type": "number",
            },
            "name": {
                "description": "Feature name to move around (must be within valid polygon and linestring names lists).",
                "type": "string"
            }
        },
        "required": ["group_idx", "shape", "radius", "name"],
        "additionalProperties": False
    })
    def form_and_move_around_shape(self, group_idx: int, shape: str, radius: float, name: str):
        """
        Assign form and move around shape behavior to a group of robots.
        """
        # Validation logic
        if radius < 5 or radius > 50:
            return "Invalid radius value: must be between 5 and 50"
        if shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in self.polygon_names + self.linestring_names:
            return "Invalid feature name: must be one of the available multipolygon, polygon, or linestring features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = get_feature_by_unique_name(self.features_geojson, name)
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
            group_idx, shape, radius, coords
        )
        return f"form_and_move_around_shape behavior assigned to group {group_idx} with formation shape {shape}, radius {radius}, and feature name {name}"
    
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to assign form and move to shape behavior to.",
                "type": "integer"
            },
            "shape": {
                "description": "Formation shape.",
                "type": "string", 
                "enum": ["circle", "square", "triangle", "hexagon"]
            },
            "radius": {
                "description": "Formation radius in meters (must be between 10 and 50).",
                "type": "number",
            },
            "name": {
                "description": "Feature name to move to (must be within valid feature names lists).",
                "type": "string"
            }
        },
        "required": ["group_idx", "shape", "radius", "name"],
        "additionalProperties": False
    })
    def form_and_move_to_shape(self, group_idx: int, shape: str, radius: float, name: str):
        """
        Assign form and move to shape behavior to a group of robots.
        """
        # Validation logic
        if radius < 5 or radius > 50:
            return "Invalid radius value: must be between 5 and 50"
        if shape not in ["circle", "square", "triangle", "hexagon"]:
            return "Invalid formation shape: must be 'circle', 'square', 'triangle', or 'hexagon'"
        if name not in self.polygon_names + self.linestring_names + self.point_names:
            return "Invalid feature name: must be one of the available features"
        
        # Get the GeoJSON feature with unique_name = name
        feature = get_feature_by_unique_name(self.features_geojson, name)
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
            # coords = feature["geometry"]["coordinates"]
            # center = coords[len(coords) // 2]
            # Get nearest coordinate to the center of the group
            coords = feature["geometry"]["coordinates"]
            group_virtual_center = self.swarm._get_group_by_idx(group_idx).virtual_center
            closest_coord_idx = np.argmin([np.linalg.norm(np.array(coord) - np.array(group_virtual_center)) for coord in coords])
            closest_coord = coords[closest_coord_idx]
            coords = [closest_coord]
        elif feature["geometry"]["type"] == "Point":
            coords = [feature["geometry"]["coordinates"]]
        else:
            return "Invalid feature type"
        
        # self.app.logger.info(f"Feature coordinates: {coords}")

        self.swarm.assign_form_and_follow_trajectory_behavior_to_group(
            group_idx, shape, radius, coords
        )
        return f"form_and_move_to_shape behavior assigned to group {group_idx} with formation shape {shape}, radius {radius}, and feature name {name}"
    
    @add_parameters_schema({
        "type": "object",
        "properties": {
            "group_idx": {
                "description": "Group index to assign cover shape behavior to.",
                "type": "integer"
            },
            "name": {
                "description": "Feature name to cover (must be within valid polygon names list).",
                "type": "string"
            }
        },
        "required": ["group_idx", "name"],
        "additionalProperties": False
    })
    def cover_shape(self, group_idx: int, name: str):
        """
        Assign cover shape behavior to a group of robots.
        """
        try:
            # Validation logic
            if name not in self.polygon_names:
                return "Invalid feature name: must be one of the available multipolygon or polygon features"
            
            # Get the GeoJSON feature with unique_name = name
            feature = get_feature_by_unique_name(self.features_geojson, name)
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
            
            # Set offset (separation between sweeps depending on overlap ratio, altitude and fov)
            overlap = 0.1 # 10% overlap
            altitude = 20 # Altitude in meters
            fov = 30 # Field of view in degrees
            offset = Geometries.get_sweep_offset(overlap, altitude, fov)
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
        
        except Exception as e:
            self.app.logger.error(f"Error: {e}")
            return "There was an error processing your request. Please try clarifying your request."

# AUXILIARY FUNCTIONS
def get_feature_by_unique_name(features_geojson, unique_name):
    """Get GeoJSON feature by unique name"""
    for feature in features_geojson["features"]:
        if "unique_name" not in feature["properties"]:
            continue
        if feature["properties"]["unique_name"] == unique_name:
            return feature
    return None
    
def get_feature_name_lists(features_geojson):
    """
    Get the list of unique names for each type of feature in the GeoJSON file.
    """
    # Create list of unique_names that correspond to Polygon features
    polygon_names = []
    linestring_names = []
    point_names = []

    for feature in features_geojson["features"]:
        try :
            if feature["geometry"]["type"] == "Polygon" or feature["geometry"]["type"] == "MultiPolygon":
                polygon_names.append(feature["properties"]["unique_name"])
            elif feature["geometry"]["type"] == "LineString":
                linestring_names.append(feature["properties"]["unique_name"])
            elif feature["geometry"]["type"] == "Point":
                point_names.append(feature["properties"]["unique_name"])
        except KeyError:
            print(f"Feature without unique name with id: {feature["id"]} and properties: {feature["properties"]}")
            
    # Sort them alphabetically
    polygon_names.sort()
    linestring_names.sort()
    point_names.sort()

    # Return the lists
    return point_names, linestring_names, polygon_names

def get_epsg_based_on_features(features_geojson):
    """
    Get the EPSG code based on the features in the GeoJSON file.
    """
    # Get feature with unique name "bbox_center"
    bbox_center = get_feature_by_unique_name(features_geojson, "bbox_center")
    coords = bbox_center["geometry"]["coordinates"]
    epsg = guess_utm_crs(coords[0], coords[1])
    return epsg

def transform_geometry(geom, source_epsg=4326, target_epsg=3857):
    """
    Transforms a Shapely geometry from source_epsg (WGS84) to target_epsg (meters).
    Default target CRS is EPSG:3857 (Web Mercator).
    """
    transformer = Transformer.from_crs(CRS.from_epsg(source_epsg), CRS.from_epsg(target_epsg), always_xy=True)
    return transform(transformer.transform, geom)  # Apply transformation

def compute_distance_and_angle(feature1, feature2, target_epsg=3857):
    """
    Computes the shortest distance between two GeoJSON features and the angle between them.
    The distance is calculated in meters using the specified EPSG code.
    The angle is calculated in degrees, where 0 degrees is North and 90 degrees is East.
    
    Parameters:
    - feature1, feature2: GeoJSON-like dictionaries with 'geometry'.
    - target_epsg: EPSG code for distance calculation (default: EPSG:3857).
    
    Returns:
    - Distance in meters.
    - Angle in degrees.
    """
    # Convert GeoJSON geometries to Shapely objects
    geom1 = shape(feature1["geometry"])
    geom2 = shape(feature2["geometry"])

    # Reproject to a coordinate system that measures in meters
    geom1_meters = transform_geometry(geom1, target_epsg=target_epsg)
    geom2_meters = transform_geometry(geom2, target_epsg=target_epsg)

    # Compute shortest distance in meters
    distance_in_meters = geom1_meters.distance(geom2_meters)

    # If the distance is 0, return 0 and angle 0
    if distance_in_meters == 0:
        return 0, 0
    
    # # Get the centroid of the two geometries
    # point1 = geom1_meters.centroid
    # point2 = geom2_meters.centroid
    # x1, y1 = centroid1.x, centroid1.y
    # x2, y2 = centroid2.x, centroid2.y

    # Get the closest points on each feature
    point1, point2 = nearest_points(geom1_meters, geom2_meters)

    # Compute angle between the two features in degrees, where 0 degrees is North and 90 degrees is East
    angle_radians = math.atan2(point2.x - point1.x, point2.y - point1.y)
    angle_degrees = math.degrees(angle_radians) % 360  # Normalize to [0, 360)

    return distance_in_meters, angle_degrees

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
        distance, angle = compute_distance_and_angle(feature_point, feature, target_epsg=target_epsg)
        
        # Append to closest features list
        closest_features.append({"name": feature_name, "distance": distance, "angle": angle})

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
        distance, angle = compute_distance_and_angle(source_feature, feature, target_epsg=target_epsg)
        
        # Append to nearby features list
        nearby_features.append({"name": feature_name, "distance": distance, "angle": angle})

    # Sort results by distance
    return sorted(nearby_features, key=lambda x: x["distance"])

def guess_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1  # Calculate UTM zone
    epsg = 32600 + zone if lat >= 0 else 32700 + zone  # 326 for Northern, 327 for Southern Hemisphere
    return f"EPSG:{epsg}"

