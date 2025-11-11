import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- ANSI Color Codes for Terminal Output ---
RESET = '\033[0m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'

# --- 1. Define the ODE and its Parameters ---

# The function f(t, x) for the ODE: x'(t) = f(t, x)
# ODE: x'(t) = (1/10) * sin(t) + (1/5) * arctan(x)
def f(t, x):
    """
    The right-hand side of the ODE: x'(t) = f(t, x).
    This function is non-linear and non-separable, making it analytically intractable.
    """
    return (1/10) * np.sin(t) + (1/5) * np.arctan(x)

# Initial condition
x0 = 0.0
t0 = 0.0

# Numerical Parameters
T_MAX = 2.0            # End time of the interval
N_ITERATIONS = 5       # Number of Picard iterations (N)
N_STEPS_T = 1000       # Number of time steps for discretization (Integration steps, M)
h = (T_MAX - t0) / N_STEPS_T # Step size

# Generate the fixed time grid
T = np.linspace(t0, T_MAX, N_STEPS_T + 1)

# --- 2. Numerical Integration (Composite Trapezoidal Rule) ---

def integrate_picard(T, g_values):
    """
    Numerically computes the definite integral I(t) = integral_t0^t g(s) ds
    for all points in the T grid, using the Composite Trapezoidal Rule.

    g_values is the array representing the integrand g(s) = f(s, x_n(s))
    at each point in the T grid.

    Returns: An array of integrated values corresponding to each point in T.
    """
    integral_values = np.zeros_like(T)
    
    # Calculate the integral for each sub-interval and accumulate
    # I(t_j) = I(t_{j-1}) + h * (g(t_{j-1}) + g(t_j)) / 2
    for j in range(1, len(T)):
        # Area of trapezoid in [T[j-1], T[j]]
        trap_area = h * (g_values[j-1] + g_values[j]) / 2.0
        # Accumulate the total integral up to T[j]
        integral_values[j] = integral_values[j-1] + trap_area

    return integral_values

# --- 3. Picard Iteration Solver ---

def picard_solver(f, T, x0, N_iterations):
    """
    Performs N_iterations of the Picard iteration process.

    Returns: A list of NumPy arrays, where each array is the function x_n(t)
             sampled over the time grid T.
    """
    
    # Initialize the list of iterates
    iterates = []

    # --- Iteration 0: Initial Guess ---
    x_n = np.full_like(T, x0)
    iterates.append(x_n)
    print(f"{BOLD}{CYAN}Iteration 0:{RESET} Initial guess {BOLD}x_0(t) = {x0}{RESET}")
    
    # --- Iteration Loop (n = 0 to N-1) ---
    for n in range(N_iterations):
        print(f"\n{BOLD}--- Starting Iteration {n+1} ---{RESET}")

        # 1. Evaluate the Integrand g_n(s) = f(s, x_n(s))
        g_n_values = f(T, x_n)
        
        # 2. Compute the Integral I_n(t)
        integral_I_n = integrate_picard(T, g_n_values)
        
        # 3. Compute the next iterate x_{n+1}(t)
        x_next = x0 + integral_I_n
        
        # Add the new iterate and set it for the next loop
        iterates.append(x_next)
        x_n = x_next
        
        # Print results with color
        print(f"{YELLOW}  - Status:{RESET} Iterate {BOLD}x_{n+1}{RESET} calculated.")
        print(f"{YELLOW}  - Check:{RESET} x_{n+1}(0) = {x_next[0]:.4f}")
        print(f"{YELLOW}  - Final Value:{RESET} x_{n+1}({T_MAX}) = {x_next[-1]:.4f}")
        
    return iterates

# --- 4. Execution and Visualization ---

if __name__ == "__main__":
    
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"{BOLD}{GREEN} Solving ODE via Banach Fixed Point Theorem (Picard Iteration) {RESET}")
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f" ODE: {BOLD}x'(t) = (1/10)sin(t) + (1/5)arctan(x){RESET} with x(0)={x0}")
    print(f" Interval: [{t0}, {T_MAX}], Steps: {N_STEPS_T}, Iterations: {N_ITERATIONS}")
    
    # Run the solver
    picard_solutions = picard_solver(f, T, x0, N_ITERATIONS)
    
    # Determine plot limits for stable animation
    y_min = min([np.min(x_n) for x_n in picard_solutions])
    y_max = max([np.max(x_n) for x_n in picard_solutions])
    y_buffer = (y_max - y_min) * 0.1
    
    # --- GIF Generation Logic ---
    
    FIG_SIZE = (10, 6)
    GIF_FILEPATH = 'picard_convergence.gif'
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(r'Picard Iteration Convergence (Fixed Point Method)', fontsize=14)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('x(t)', fontsize=12)
    ax.grid(True, linestyle='dotted', alpha=0.6)
    ax.set_xlim(t0, T_MAX)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    line, = ax.plot([], [], lw=2, color='darkred', label='Current Iterate')
    text = ax.text(0.5, 0.95, '', transform=ax.transAxes, fontsize=15, ha='center')

    # Store all previous iterates as dimmed lines
    previous_lines = []

    def init_anim():
        """Initialization function for the animation."""
        line.set_data([], [])
        text.set_text('')
        for pl in previous_lines:
            pl.remove()
        previous_lines.clear()
        return [line, text]

    def update_anim(i):
        """Update function for the animation."""
        x_n = picard_solutions[i]
        
        # Plot the current iterate (i) as the main, thick line
        line.set_data(T, x_n)
        
        # Update the iteration label
        text.set_text(r'Iteration $N = %d$' % i)
        
        # Add the previous iterate (i-1) as a light gray line for history
        if i > 0:
            x_prev = picard_solutions[i-1]
            # Use a dashed, lighter line for history
            pl, = ax.plot(T, x_prev, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=r'$x_{%d}(t)$' % (i-1))
            previous_lines.append(pl)
        
        # Ensure the final line (last iterate) is colored bold red
        if i == N_ITERATIONS:
            line.set_color('red')
            line.set_linewidth(3)

        return [line, text] + previous_lines


    print(f"\n{BOLD}{CYAN}--- Generating Convergence Animation ({GIF_FILEPATH}) ---{RESET}")
    # Create the animation object
    # The frames argument must include the initial guess (x0) + all N_ITERATIONS
    anim = animation.FuncAnimation(
        fig, 
        update_anim, 
        init_func=init_anim,
        frames=len(picard_solutions), # Total frames: 0 to N_ITERATIONS
        interval=1000, # 1000ms delay between frames
        repeat=True,
        blit=False # Blit=True can sometimes cause issues with complex updates
    )
    
    # Save the animation as a GIF
    anim.save(GIF_FILEPATH, writer='pillow', fps=1)
    print(f"{BOLD}{GREEN}SUCCESS!{RESET} Convergence GIF saved to: {YELLOW}{GIF_FILEPATH}{RESET} ")


    # --- Print Convergence Check ---
    x_last = picard_solutions[-1]
    x_prev = picard_solutions[-2]
    
    max_diff = np.max(np.abs(x_last - x_prev))
    
    print("\n" + BOLD + CYAN + "--- Convergence Analysis ---" + RESET)
    print(f"{YELLOW}Max difference between x_{N_ITERATIONS} and x_{N_ITERATIONS-1} (Proxy for Error):{RESET} {max_diff:.6e}")
    
    # Error Estimation using the BPFT formula
    L = 0.2
    k = L * T_MAX 
    
    if k < 1:
        error_bound_estimate = max_diff * k / (1 - k)
        print(f"{GREEN}Calculated k (Contraction Factor):{RESET} {k:.2f}")
        print(f"{GREEN}Estimated Truncation Error (E_trunc) bound:{RESET} {error_bound_estimate:.6e}")
    else:
        print(f"{RED}Warning:{RESET} k = {k:.2f}. Contraction condition not met globally.")