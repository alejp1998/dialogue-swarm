"""
Swarm

This module contains the Swarm, Group, and Robot classes.
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
from shapely.ops import transform, nearest_points
from pyproj import CRS, Transformer
from trajgenpy import Geometries

# CONSTANTS
NATO_PHONETIC_ALPHABET = [
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
        self.max_rotation_speed = 180 # degrees/s
        self.min_rotation_speed = 45 # degrees/s
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

    def update_target_angle(self, target_x, target_y):
        """
        Update target angle for robot using geodesic coordinates.
        0 degrees must correspond to the North direction.
        90 degrees must correspond to the East direction.
        """
        lon_per_meter, lat_per_meter = calculate_lon_lat_per_meter((self.x, self.y))
        dx_in_meters = (target_x - self.x) / lon_per_meter
        dy_in_meters = (target_y - self.y) / lat_per_meter
        target_angle_radians = math.atan2(dx_in_meters, dy_in_meters)
        # Turn target angle in rads into degrees where north is 0 degrees and 90 degrees is east
        self.target_angle = math.degrees(target_angle_radians) % 360

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
            self._calculate_rotation_speed()
        else:
            if not self.is_robot_in_position():
                self._calculate_speed()
                self._calculate_rotation_speed()

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

    def _calculate_speed(self):
        """Calculate speed based on current angle and distance to target."""
        angle_in_rads = -math.radians(self.angle) + np.pi/2
        self.vx = np.cos(angle_in_rads) * self.max_speed
        self.vy = np.sin(angle_in_rads) * self.max_speed

    def _calculate_angle_diff(self):
        """Calculate and normalize angle difference [-180, 180) degrees."""
        angle_diff = (self.target_angle - self.angle)
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        self.angle_diff = angle_diff

    def _calculate_rotation_speed(self):
        """Calculate rotation speed based on angle difference."""
        if self.angle_diff > 0:
            self.rotation_speed = max(min(self.angle_diff, self.max_rotation_speed), self.min_rotation_speed)
        else:
            self.rotation_speed = min(max(self.angle_diff, -self.max_rotation_speed), -self.min_rotation_speed)

    def is_robot_in_position(self):
        """Check if robot is close enough to target position."""
        self._calculate_distance_in_meters()
        return self.dist_in_meters < self.distance_th

    def is_robot_aligned(self):
        """Check if robot is aligned with target."""
        self._calculate_angle_diff()
        return abs(self.angle_diff) < self.rotation_th

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

        self.angle = (self.angle + self.rotation_speed * (step_ms / 1000)) % 360

        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0

    def _update_battery_level(self):
        """Update robot battery level based on time."""
        self.battery_level = max(0, self.battery_level - (1 / 50000))

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