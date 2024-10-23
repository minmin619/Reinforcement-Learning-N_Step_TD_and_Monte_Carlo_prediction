import numpy as np

# Define the grid size
grid_size = 5

# Initialize the value function with zeros
value_function = np.zeros((grid_size, grid_size))

# Discount factor
gamma = 0.9

# Rewards
reward_A = 10
reward_B = 5

# Tolerance for convergence
tolerance = 1e-3

# Define the states A and B and their corresponding next states
state_A = (0, 1)
state_A_prime = (4, 1)
state_B = (1, 3)
state_B_prime = (4, 3)

# Define the directions (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_valid(state):
    return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size

# Value Iteration
delta = float('inf')
iteration = 0
while delta > tolerance:
    delta = 0
    new_value_function = np.copy(value_function)
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            if state == state_A:
                new_value_function[i, j] = reward_A + gamma * value_function[state_A_prime]
            elif state == state_B:
                new_value_function[i, j] = reward_B + gamma * value_function[state_B_prime]
            else:
                values = []
                for d in directions:
                    next_state = (i + d[0], j + d[1])
                    if is_valid(next_state):
                        values.append(value_function[next_state])
                if values:
                    new_value_function[i, j] = max(values) * gamma

    # Update delta for convergence check
    delta = max(delta, np.max(np.abs(new_value_function - value_function)))
    value_function = new_value_function
    iteration += 1

# Print the value function rounded to three decimal places
np.set_printoptions(precision=3)
value_function_rounded = np.round(value_function, 3)
value_function_rounded
