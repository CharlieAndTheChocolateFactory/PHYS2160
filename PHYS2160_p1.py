import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

############################# part (a) #############################

# Constants
g = 9.807  # gravitational acceleration (m/s^2)
b_over_m = 1.28e5  # drag coefficient per unit mass (1/s)\
c0_over_m = 1.46e-4  # drag coefficient per unit mass (1/s)

def projectile_model_case1(z, t):
    x, y, vx, vy = z
    v = np.sqrt(vx**2 + vy**2)
    dxdt = vx
    dydt = vy
    dvxdt = -b_over_m * vx
    dvydt = -g - b_over_m * vy
    return [dxdt, dydt, dvxdt, dvydt]

def projectile_model_case2(z, t):
    x, y, vx, vy = z
    v = np.sqrt(vx**2 + vy**2)  # Calculate the magnitude of velocity
    dxdt = vx
    dydt = vy
    dvxdt = -c0_over_m * vx * v
    dvydt = -c0_over_m * vy * v - g
    return [dxdt, dydt, dvxdt, dvydt]

def projectile_model_case3(z, t):
    x, y, vx, vy = z
    v = np.sqrt(vx**2 + vy**2)  # Calculate the magnitude of velocity
    
    # Constants
    c0_over_m = 1.46e-4  # drag coefficient per unit mass (1/s)
    T_0 = 288.2  # Reference temperature (K)
    L = 0.0065  # Temperature lapse rate (K/m)
    
    # Calculate altitude-dependent drag coefficient
    if y > 0:
        c_over_m = c0_over_m * (1 - (L * y) / T_0) ** 4.256
    else:
        c_over_m = c0_over_m
    
    dxdt = vx
    dydt = vy
    dvxdt = -c_over_m * vx * v
    dvydt = -c_over_m * vy * v - 9.807  # gravitational acceleration (m/s^2)
    
    return [dxdt, dydt, dvxdt, dvydt]

# this is the function that will be used to solve the ODEs
def solve_trajectory_odeint(v0, theta_deg, model):
    theta = np.radians(theta_deg)
    initial_conditions = [0, 0, v0 * np.cos(theta), v0 * np.sin(theta)]
    t = np.linspace(0, v0 * 5, 99999)
    sol = odeint(model, initial_conditions, t)
    x, y = sol[:, 0], sol[:, 1]
    valid_indices = y >= 0
    time_of_flight = t[valid_indices][-1]
    return x[valid_indices], y[valid_indices], t[valid_indices], time_of_flight

# this function will be used to get the results of the trajectory, also for part (b), (c) and (d)
def trajectory_results(v0s, angles, model):
    results = {}
    range_arr = []
    height_arr = []
    time_of_flight_arr = []

    for v0 in v0s:
        for theta in angles:
            key = (v0, theta)
            x, y, t, tf = solve_trajectory_odeint(v0, theta, model)
            range_val = max(x)
            height_val = max(y)
            results[key] = {'x': x, 'y': y, 't': t, 'tf': tf, 'range': range_val, 'height': height_val}
            range_arr.append(range_val)
            height_arr.append(height_val)
            time_of_flight_arr.append(tf)

    return results, range_arr, height_arr, time_of_flight_arr

# this function will be used to plot the trajectories
def plot_trajectories(v0s, angles, model):
    results, range_arr, height_arr, time_of_flight_arr = trajectory_results(v0s, angles, model)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array of axes for easier indexing

    idx = 0
    for i, v0 in enumerate(v0s):
        for theta in angles:
            key = (v0, theta)
            x = results[key]['x']
            y = results[key]['y']
            axs[i].plot(x, y, label=f'Theta = {theta}°')
            
        axs[i].set_title(f'v0 = {v0} m/s')
        axs[i].set_xlabel('x(t)')
        axs[i].set_ylabel('y(t)')
        axs[i].legend(fontsize='small', loc= 'best')
        axs[i].grid(True)
        axs[i].set_ylim(bottom=0)
        # Automatically extend x-axis limits with margin for better visualization
        current_x_limit = axs[i].get_xlim()
        axs[i].set_xlim(current_x_limit[0], current_x_limit[1] * 1.35)
    plt.tight_layout()
    plt.show()

    return range_arr, height_arr, time_of_flight_arr

# define the angles
angles = np.linspace(0, 90, 19)
num_of_speeds = 6

# plot the trajectories for case 1
v0s_case1 = np.linspace(10e-5, 6*10e-5, num_of_speeds)  # 6 different speeds
v0s_case1 = np.around(v0s_case1, decimals=7)
range_arr_case1, height_arr_case1, time_of_flight_arr_case1 = plot_trajectories(v0s_case1, angles, projectile_model_case1)

# plot the trajectories for case 2
v0s_case2 = np.linspace(500, 3*500, num_of_speeds)  # 6 different speeds
range_arr_case2, height_arr_case2, time_of_flight_arr_case2 = plot_trajectories(v0s_case2, angles, projectile_model_case2)

# plot the trajectories for case 3
v0s_case3 = np.linspace(500, 3*500, num_of_speeds)  # 6 different speeds
range_arr_case3 , height_arr_case3, time_of_flight_arr_case3 = plot_trajectories(v0s_case3, angles, projectile_model_case3)

############################# part (b)(c)(d) #############################

# this is a function that will be used to plot the data for part (b) (c) and (d)
def plot_data(arrays, v0s, ylabel, title, angles):
    plt.figure(figsize=(10, 6))  # Set the size of the plot
    for i, array in enumerate(arrays):
        plt.plot(angles, array, label=f'{ylabel} for v0 = {v0s[i]} m/s')
    plt.title(title)
    plt.xlabel('Theta (degrees)')
    plt.ylabel(f'{ylabel}')
    plt.legend()
    plt.grid(True)
    plt.show()

#split the range arrays obtained from part (a) into 6 parts for each speed
range_arr_case1 = np.array_split(range_arr_case1, num_of_speeds)
range_arr_case2 = np.array_split(range_arr_case2, num_of_speeds)
range_arr_case3 = np.array_split(range_arr_case3, num_of_speeds)
height_arr_case1 = np.array_split(height_arr_case1, num_of_speeds)
height_arr_case2 = np.array_split(height_arr_case2, num_of_speeds)
height_arr_case3 = np.array_split(height_arr_case3, num_of_speeds) 
time_of_flight_arr_case1 = np.array_split(time_of_flight_arr_case1, num_of_speeds)
time_of_flight_arr_case2 = np.array_split(time_of_flight_arr_case2, num_of_speeds)
time_of_flight_arr_case3 = np.array_split(time_of_flight_arr_case3, num_of_speeds)

#assign cases into the tuples and use for loop to plot them
cases = [
(range_arr_case1, v0s_case1, 'Range (m)', 'Case 1', angles),
(range_arr_case2, v0s_case2, 'Range (m)', 'Case 2', angles),
(range_arr_case3, v0s_case3, 'Range (m)', 'Case 3', angles),
(height_arr_case1, v0s_case1, 'Height (m)', 'Case 1', angles),
(height_arr_case2, v0s_case2, 'Height (m)', 'Case 2', angles),
(height_arr_case3, v0s_case3, 'Height (m)', 'Case 3', angles),
(time_of_flight_arr_case1, v0s_case1, 'Time of Flight (s)', 'Case 1', angles),
(time_of_flight_arr_case2, v0s_case2, 'Time of Flight (s)', 'Case 2', angles),
(time_of_flight_arr_case3, v0s_case3, 'Time of Flight (s)', 'Case 3', angles)]

for data, v0s, ylabel, title_prefix, angles in cases:
    plot_data(data, v0s, ylabel, title_prefix, angles)

