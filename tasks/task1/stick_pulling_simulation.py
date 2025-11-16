"""
Stick Pulling Simulation - Task 1.2
Step-by-step implementation of robot stick pulling simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: Understand the System Setup
# =============================================================================
"""
System Parameters:
- N robots
- M = 20 stick sides
- Robots wait ω_atStick = 7 time steps at one stick side before moving
- If 2+ robots meet at same stick side, they pull it out together
- Stick resets immediately after removal
"""

# Constants
M_STICK_SIDES = 20
WAIT_TIME_AT_STICK = 7
NUM_SIMULATION_STEPS = 1000
NUM_RUNS = 5000
C_QUADRATIC = 0.12  # Coefficient for quadratic commute time model


# =============================================================================
# STEP 2: Model Commute Times Between Stick Sides
# =============================================================================

def compute_commute_time_linear(N, zeta):
    """
    Linear commute time model: T_l(N) = N + ζ
    
    Parameters:
    -----------
    N : int
        Number of robots
    zeta : int
        Random delay (0, 1, or 2)
    
    Returns:
    --------
    int
        Commute time in time steps
    """
    return N + zeta


def compute_commute_time_quadratic(N, zeta):
    """
    Quadratic commute time model: T_q(N) = cN² + ζ
    
    Parameters:
    -----------
    N : int
        Number of robots
    zeta : int
        Random delay (0, 1, or 2)
    
    Returns:
    --------
    int
        Commute time in time steps
    """
    return int(C_QUADRATIC * N * N) + zeta


def get_random_zeta():
    """
    Get random delay ζ ∈ {0, 1, 2}
    
    Returns:
    --------
    int
        Random value from {0, 1, 2}
    """
    return random.randint(0, 2)


# =============================================================================
# STEP 3: Design Data Structures
# =============================================================================

class Robot:
    """
    Represents a single robot in the simulation.
    """
    def __init__(self, robot_id, initial_stick_side):
        self.id = robot_id
        self.current_stick_side = initial_stick_side  # Stick side index (1-20)
        self.waiting_time = 0  # Time steps waited at current stick
        self.in_transit = False  # Whether robot is moving between sticks
        self.transit_time_remaining = 0  # Steps remaining in transit
        self.target_stick_side = None  # Destination stick side
    
    def __repr__(self):
        status = "transit" if self.in_transit else f"stick_{self.current_stick_side}"
        return f"Robot_{self.id}({status}, wait={self.waiting_time}, transit_remaining={self.transit_time_remaining})"


class StickPullingSimulation:
    """
    Main simulation class for stick pulling task.
    """
    def __init__(self, num_robots, commute_time_model='linear'):
        """
        Initialize simulation.
        
        Parameters:
        -----------
        num_robots : int
            Number of robots (N)
        commute_time_model : str
            'linear' or 'quadratic' commute time model
        """
        self.N = num_robots
        self.commute_time_model = commute_time_model
        self.robots = []
        self.sticks_pulled = 0
        self.step_count = 0
        
        # Initialize robots at random stick sides
        self._initialize_robots()
    
    def _initialize_robots(self):
        """Initialize robots at random stick sides."""
        for i in range(self.N):
            initial_stick = random.randint(1, M_STICK_SIDES)
            robot = Robot(i, initial_stick)
            self.robots.append(robot)
    
    def _compute_commute_time(self):
        """Compute commute time based on selected model."""
        zeta = get_random_zeta()
        if self.commute_time_model == 'linear':
            return compute_commute_time_linear(self.N, zeta)
        else:  # quadratic
            return compute_commute_time_quadratic(self.N, zeta)
    
    def _get_robots_at_stick(self, stick_side):
        """Get list of robots currently waiting at a specific stick side."""
        return [r for r in self.robots 
                if r.current_stick_side == stick_side 
                and not r.in_transit]
    
    def _check_and_pull_sticks(self):
        """
        Check for sticks with 2+ robots and pull them out.
        Robots that pull a stick immediately start moving to new stick.
        """
        # Group robots by stick side
        stick_groups = defaultdict(list)
        for robot in self.robots:
            if not robot.in_transit:
                stick_groups[robot.current_stick_side].append(robot)
        
        # Check each stick side
        for stick_side, robots_at_stick in stick_groups.items():
            if len(robots_at_stick) >= 2:
                # Pull the stick!
                self.sticks_pulled += 1
                
                # Send all robots at this stick to new random stick sides
                for robot in robots_at_stick:
                    # Pick new random stick (different from current)
                    new_stick = robot.current_stick_side
                    while new_stick == robot.current_stick_side:
                        new_stick = random.randint(1, M_STICK_SIDES)
                    
                    # Set robot to transit
                    robot.in_transit = True
                    robot.target_stick_side = new_stick
                    robot.transit_time_remaining = self._compute_commute_time()
                    robot.waiting_time = 0  # Reset waiting counter
    
    def step(self):
        """
        Execute one simulation step.
        
        Returns:
        --------
        int
            Number of sticks pulled in this step
        """
        self.step_count += 1
        sticks_pulled_this_step = 0
        
        # STEP 5a: Update robots waiting at stick sides
        for robot in self.robots:
            if not robot.in_transit:
                robot.waiting_time += 1
                
                # If waiting time reaches threshold, start moving
                if robot.waiting_time >= WAIT_TIME_AT_STICK:
                    # Pick new random stick (different from current)
                    new_stick = robot.current_stick_side
                    while new_stick == robot.current_stick_side:
                        new_stick = random.randint(1, M_STICK_SIDES)
                    
                    # Set robot to transit
                    robot.in_transit = True
                    robot.target_stick_side = new_stick
                    robot.transit_time_remaining = self._compute_commute_time()
                    robot.waiting_time = 0
        
        # STEP 5b: Update robots in transit
        for robot in self.robots:
            if robot.in_transit:
                robot.transit_time_remaining -= 1
                
                # If transit complete, place robot at new stick
                if robot.transit_time_remaining <= 0:
                    robot.current_stick_side = robot.target_stick_side
                    robot.in_transit = False
                    robot.waiting_time = 0  # Reset waiting counter
                    robot.target_stick_side = None
        
        # STEP 5c: Check for sticks pulled out (2+ robots at same stick)
        sticks_before = self.sticks_pulled
        self._check_and_pull_sticks()
        sticks_pulled_this_step = self.sticks_pulled - sticks_before
        
        return sticks_pulled_this_step
    
    def run(self, num_steps=NUM_SIMULATION_STEPS):
        """
        Run simulation for specified number of steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of simulation steps
        
        Returns:
        --------
        int
            Total number of sticks pulled
        """
        self.sticks_pulled = 0
        self.step_count = 0
        
        for _ in range(num_steps):
            self.step()
        
        return self.sticks_pulled


# =============================================================================
# STEP 6: Run for Different N from 2 to 20
# =============================================================================

def run_multiple_simulations(num_robots, commute_model='linear', 
                            num_runs=NUM_RUNS, num_steps=NUM_SIMULATION_STEPS):
    """
    Run multiple independent simulations for given number of robots.
    
    Parameters:
    -----------
    num_robots : int
        Number of robots (N)
    commute_model : str
        'linear' or 'quadratic'
    num_runs : int
        Number of independent runs
    num_steps : int
        Number of steps per run
    
    Returns:
    --------
    float
        Average number of sticks pulled across all runs
    """
    total_sticks = 0
    
    for run in range(num_runs):
        sim = StickPullingSimulation(num_robots, commute_model)
        sticks_pulled = sim.run(num_steps)
        total_sticks += sticks_pulled
        
        if (run + 1) % 1000 == 0:
            print(f"  Completed {run + 1}/{num_runs} runs for N={num_robots}...")
    
    return total_sticks / num_runs


# =============================================================================
# STEP 7: Plot Results
# =============================================================================

def plot_results(robot_counts, avg_sticks_linear, avg_sticks_quadratic):
    """
    Plot relative capacity vs system size for both commute time models.
    
    Parameters:
    -----------
    robot_counts : list
        List of N values (number of robots)
    avg_sticks_linear : list
        Average sticks pulled for linear model
    avg_sticks_quadratic : list
        Average sticks pulled for quadratic model
    """
    # Calculate relative capacity (normalized to N=2)
    baseline_linear = avg_sticks_linear[0]  # N=2
    baseline_quadratic = avg_sticks_quadratic[0]  # N=2
    
    relative_capacity_linear = [s / baseline_linear for s in avg_sticks_linear]
    relative_capacity_quadratic = [s / baseline_quadratic for s in avg_sticks_quadratic]
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    plt.plot(robot_counts, relative_capacity_linear, 
            marker='o', linewidth=2, markersize=6,
            label='Linear Model: T_l(N) = N + ζ', color='blue')
    
    plt.plot(robot_counts, relative_capacity_quadratic, 
            marker='s', linewidth=2, markersize=6,
            label='Quadratic Model: T_q(N) = 0.12N² + ζ', color='red')
    
    plt.xlabel('System Size N (Number of Robots)', fontsize=12)
    plt.ylabel('Relative Capacity (normalized to N=2)', fontsize=12)
    plt.title('Stick Pulling Simulation: Relative Capacity vs System Size\n'
              f'({NUM_RUNS} runs per N, {NUM_SIMULATION_STEPS} steps per run)', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(robot_counts)
    plt.tight_layout()
    plt.savefig('outputs/stick_pulling_results.png', dpi=150)
    plt.show()
    
    # Also plot absolute values
    plt.figure(figsize=(14, 8))
    
    plt.plot(robot_counts, avg_sticks_linear, 
            marker='o', linewidth=2, markersize=6,
            label='Linear Model: T_l(N) = N + ζ', color='blue')
    
    plt.plot(robot_counts, avg_sticks_quadratic, 
            marker='s', linewidth=2, markersize=6,
            label='Quadratic Model: T_q(N) = 0.12N² + ζ', color='red')
    
    plt.xlabel('System Size N (Number of Robots)', fontsize=12)
    plt.ylabel('Average Sticks Pulled', fontsize=12)
    plt.title('Stick Pulling Simulation: Average Sticks Pulled vs System Size\n'
              f'({NUM_RUNS} runs per N, {NUM_SIMULATION_STEPS} steps per run)', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(robot_counts)
    plt.tight_layout()
    plt.savefig('outputs/stick_pulling_absolute.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run complete stick pulling simulation.
    """
    print("="*70)
    print("STICK PULLING SIMULATION - TASK 1.2")
    print("="*70)
    
    # Define robot counts to test
    robot_counts = list(range(2, 21))  # N from 2 to 20
    
    print(f"\nRunning simulations for N = {robot_counts[0]} to {robot_counts[-1]}")
    print(f"Number of runs per N: {NUM_RUNS}")
    print(f"Steps per run: {NUM_SIMULATION_STEPS}")
    print(f"Stick sides: {M_STICK_SIDES}")
    print(f"Wait time at stick: {WAIT_TIME_AT_STICK} steps")
    print("\nThis will take a while... Please be patient!\n")
    
    # Run simulations for linear model
    print("\n" + "="*70)
    print("LINEAR COMMUTE TIME MODEL: T_l(N) = N + ζ")
    print("="*70)
    avg_sticks_linear = []
    
    for N in robot_counts:
        print(f"\nProcessing N = {N} robots (Linear model)...")
        avg_sticks = run_multiple_simulations(N, 'linear', NUM_RUNS, NUM_SIMULATION_STEPS)
        avg_sticks_linear.append(avg_sticks)
        print(f"  Average sticks pulled: {avg_sticks:.2f}")
    
    # Run simulations for quadratic model
    print("\n" + "="*70)
    print("QUADRATIC COMMUTE TIME MODEL: T_q(N) = 0.12N² + ζ")
    print("="*70)
    avg_sticks_quadratic = []
    
    for N in robot_counts:
        print(f"\nProcessing N = {N} robots (Quadratic model)...")
        avg_sticks = run_multiple_simulations(N, 'quadratic', NUM_RUNS, NUM_SIMULATION_STEPS)
        avg_sticks_quadratic.append(avg_sticks)
        print(f"  Average sticks pulled: {avg_sticks:.2f}")
    
    # Plot results
    print("\n" + "="*70)
    print("GENERATING PLOTS...")
    print("="*70)
    plot_results(robot_counts, avg_sticks_linear, avg_sticks_quadratic)
    
    # Print summary
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print("\nSummary:")
    print(f"  Linear model - N=2: {avg_sticks_linear[0]:.2f} sticks, N=20: {avg_sticks_linear[-1]:.2f} sticks")
    print(f"  Quadratic model - N=2: {avg_sticks_quadratic[0]:.2f} sticks, N=20: {avg_sticks_quadratic[-1]:.2f} sticks")
    
    # Calculate relative capacities
    rel_cap_linear_20 = avg_sticks_linear[-1] / avg_sticks_linear[0]
    rel_cap_quadratic_20 = avg_sticks_quadratic[-1] / avg_sticks_quadratic[0]
    
    print(f"\nRelative Capacity at N=20 (normalized to N=2):")
    print(f"  Linear model: {rel_cap_linear_20:.3f}")
    print(f"  Quadratic model: {rel_cap_quadratic_20:.3f}")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # np.random.seed(42)
    # random.seed(42)
    
    main()

