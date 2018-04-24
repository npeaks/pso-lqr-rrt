# Import modules
import numpy as np
import pyswarms as ps

# must be higher than worst possible cost without it (3)
penalty_constant = 5

num_trajectories = 10
num_waypoints = 100
num_coordinates = 3

time_between_waypoints = .1

altitude_coordinate = 2

start_coords = np.array([[0, 0, 0]])
end_coords = np.array([[1, 1, 1]])

# choose based on environment
max_altitude = 10
min_altitude = 0

# properties of the Earth
air_density = 1.292 # roughly sea level, room temperature
gravitational_acceleration = 9.81

# properties of robot (bebop2)
robot_mass = 0.5
drag_coefficient = 0.25 # estimate for quadcopter
robot_area = 1/3 * .328 * .382 # assuming 1/3 cover of top-down dimensions

# estimate motor force given bebop2 max vertical speed is 6 m/s
def air_resistance(velocity):
    return .5 * air_density * drag_coefficient * robot_area * \
        np.square(velocity)

max_force = robot_mass * gravitational_acceleration + air_resistance(6)

# returns column vector of 1 if waypoint is underground, 0 otherwise
# edit this function to check for collisions beyond just ground
def is_colliding(trajectory):
    mask_altitude = np.zeros((trajectory.shape[1], 1))
    mask_altitude[altitude_coordinate] = [1]
    altitudes = np.dot(trajectory, mask_altitude)
    return altitudes < 0


# trajectory is ndarray with shape (num_waypoints, num_coordinates)

# Length cost function
def length(trajectory):
    differences = trajectory[1:] - trajectory[:-1]
    distances = np.apply_along_axis(np.linalg.norm, 1, differences)
    return np.sum(distances)

def cost_length(trajectory):
    length_line = np.linalg.norm(end_coords - start_coords)
    length_actual = length(trajectory)
    return 1 - length_line / length_actual


# Altitude cost function
def cost_altitude(trajectory):
    if altitude_coordinate == None:
        return 0
    altitudes = trajectory[:,altitude_coordinate]
    avg_altitude = np.sum(altitudes) / num_waypoints
    return (avg_altitude - min_altitude) / (max_altitude - min_altitude)


# Danger zone cost function
def cost_dangerzones(trajectory):
    return 0 # TODO

# Power cost function
def gravitational_force(shape):
    F_g = np.zeros(shape)
    F_g[:, altitude_coordinate] = np.ones(shape[0])
    F_g *= robot_mass * gravitational_acceleration
    return F_g

def length_nonfeasible(trajectory):
    # calculate dynamics
    position_differences = trajectory[1:] - trajectory[:-1]
    velocities = position_differences / time_between_waypoints
    velocity_differences = velocities[1:] - velocities[:-1]
    accelerations = velocity_differences / time_between_waypoints
    # make shapes match
    accelerations = np.vstack((accelerations, \
        np.zeros((1, accelerations.shape[1]))))
    # calculate forces
    forces = robot_mass * accelerations + air_resistance(velocities) + \
        gravitational_force(accelerations.shape)
    # find where too much force is used
    overall_forces = np.apply_along_axis(np.linalg.norm, 1, forces)
    mask_nonfeasible = overall_forces > max_force
    # calculate length of infeasible part of trajectory
    distances = np.apply_along_axis(np.linalg.norm, 1, position_differences)
    return np.sum(distances * mask_nonfeasible)


def cost_power(trajectory):
    L_nonfeasible = length_nonfeasible(trajectory)
    if L_nonfeasible:
        return penalty_constant + L_nonfeasible / length(trajectory)
    return 0


# Collision cost function

def cost_collision(trajectory):
    mask_collision_position = is_colliding(trajectory)
    if np.sum(mask_collision_position):
        mask_collision_distance = np.logical_or(mask_collision_position[1:], \
            mask_collision_position[:-1])
        position_differences = trajectory[1:] - trajectory[:-1]
        distances = np.apply_along_axis(np.linalg.norm, 1, position_differences)
        return np.sum(mask_collision_distance * distances)
    return 0


# Fuel cost function
def cost_fuel(trajectory):
    return 0 # TODO


# Smoothing cost function
def cost_smoothing(trajectory):
    return 0 # TODO


# Overall cost function
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

