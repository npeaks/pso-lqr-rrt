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
penalty = 3

# trajectory is ndarray with shape (num_waypoints, num_coordinates)]

def get_length(trajectory):
    diffs = trajectory[:-1] - trajectory[1:]
    distances = np.apply_along_axis(np.linalg.norm, 1, diffs)
    return np.sum(distances)

def cost_length(trajectory):
    return 1 - (np.linalg.norm(trajectory[0]-trajectory[-1]), get_length(trajectory))

def cost_altitude(trajectory):
    if altitude_coordinate == None:
        return 0
    altitudes = trajectory[:,altitude_coordinate]
    avg_altitude = np.sum(altitudes) / num_waypoints
    return (avg_altitude - min_altitude) / (max_altitude - min_altitude)

# dangerzones is an ndarray with shape(num zones, 3)
# we use a cylindrical approximation, x,y,d
dangerzones = np.ndarray([[1,2,3],[4,5,6],[7,8,9],[10,11,1]])

def cost_dangerzones(trajectory):
    totalLength = 0.0
    for p in trajectory:

    def is_in_zone(p):
        for zone in dangerzones:
            if (np.linalg.norm(p[:2]-zone[:2]) < zone[2]):
                return True
        return False
    contained = np.apply_along_axis(is_in_zone, 1, trajectory)
    diameters = np.sum(dangerzones, 0)[2]
    for i in range(1, len(contained)):
        if contained[i] and contained[i-1]:
            totalLength += np.linalg.norm(trajectory[i]-trajectory[i-1])
    return min(1,max(0,totalLength/diameters))

def cost_power(trajectory):
    return 0 # TODO

def cost_collision(trajectory):
    length = 0
    for i in range(1,len(trajectory)):
        if trajectory[i][2] <= 0 and trajectory[i-1][2] <= 0:
            length += np.linalg.norm(trajectory[i]-trajectory[i-1])
    if length == 0:
        return 0
    return penalty + length/get_length(trajectory) # TODO

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
# coords should be 3, x,y,z; This is because we are looking at the quadcopter
# navigation stuff
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

