# Import modules
import numpy as np
import pyswarms as ps

# Cost Function
# x is a numpy array with shape (n_particles, dimensions)

num_trajectories = 10
num_waypoints = 100
num_coordinates = 3
altitude_coordinate = 2
max_altitude = 10
min_altitude = 0

# trajectory is ndarray with shape (num_waypoints, num_coordinates)
def cost_length(trajectory):
    diffs = trajectory[:-1] - trajectory[1:]
    distances = np.apply_along_axis(np.linalg.norm, 1, diffs)
    return np.sum(distances)

def cost_altitude(trajectory):
    if altitude_coordinate == None:
        return 0
    altitudes = trajectory[:,altitude_coordinate]
    avg_altitude = np.sum(altitudes) / num_waypoints
    return (avg_altitude - min_altitude) / (max_altitude - min_altitude)

def cost_dangerzones(trajectory):
    return 0 # TODO

def cost_power(trajectory):
    return 0 # TODO

def cost_collision(trajectory):
    return 0 # TODO

def cost_fuel(trajectory):
    return 0 # TODO

def cost_smoothing(trajectory):
    return 0 # TODO

def cost_total_trajectory(trajectory):
    return cost_length(trajectory) + cost_altitude(trajectory) + \
        cost_dangerzones(trajectory) + cost_power(trajectory) + \
        cost_collision(trajectory) + cost_fuel(trajectory) + \
        cost_smoothing(trajectory)

# particle is ndarray with shape (1, num_waypoints * num_coordinates)
def cost_total_particle(particle):
    trajectory = particle.reshape((num_waypoints, num_coordinates))
    return cost_total_trajectory(trajectory)

# x is ndarray with shape (num_trajectories, num_waypoints * num_coordinates)
# returns ndarray with shape (num_trajectories, 1)
#   each cell represents the cost of a trajectory
def objective(x):
    return np.apply_along_axis(cost_total_particle, 1, x)
    

from pyswarms.utils.functions import single_obj as fx
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=num_trajectories, \
    dimensions=(num_waypoints * num_coordinates), options=options)
cost, pos = optimizer.optimize(objective, print_step=100, iters=1000, verbose=3)

