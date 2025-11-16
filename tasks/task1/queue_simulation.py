"""
Complete Queue Simulation Implementation
Step-by-step implementation of Poisson distribution and queue processing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: Plot Poisson Distributions
# =============================================================================

def plot_poisson_distributions(lambda_values, max_x=15):
    """
    Task 1: Plot Poisson distributions for λ = 0.01, 0.1, 0.5, 1
    
    Parameters:
    -----------
    lambda_values : list
        List of λ values to plot
    max_x : int
        Maximum value of X (number of jobs) to plot
    """
    x_values = np.arange(0, max_x + 1)
    
    plt.figure(figsize=(12, 8))
    
    for lam in lambda_values:
        # Calculate P(X=i) for each i
        probabilities = [poisson.pmf(i, lam) for i in x_values]
        
        plt.plot(x_values, probabilities, marker='o', linewidth=2, 
                label=f'λ = {lam}', markersize=5)
    
    plt.xlabel('Number of Jobs (X)', fontsize=12)
    plt.ylabel('Probability P(X=i)', fontsize=12)
    plt.title('Task 1: Poisson Distribution for Different λ Values', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('outputs/poisson_distributions.png', dpi=150)
    plt.show()
    print("✓ Task 1 completed: Poisson distributions plotted")


# =============================================================================
# STEP 2: Sample Jobs from Poisson Distribution
# =============================================================================

def sample_poisson_jobs(lam):
    """
    Task 2: Sample number of incoming jobs using Poisson distribution
    
    Parameters:
    -----------
    lam : float
        Rate parameter λ
    
    Returns:
    --------
    int
        Number of jobs arriving (sampled from Poisson distribution)
    """
    # Sample from Poisson distribution
    return np.random.poisson(lam)


# =============================================================================
# STEP 3: Simulate Queue Processing
# =============================================================================

def simulate_queue(lam, num_steps, processing_time, verbose=False):
    """
    Task 3: Simulate queue with arrivals and processing
    
    At each time step:
    1. Sample new arrivals from Poisson distribution
    2. Add new arrivals to waiting queue
    3. Process one job (if available) after processing_time steps
    4. Track queue length
    
    Parameters:
    -----------
    lam : float
        Arrival rate λ
    num_steps : int
        Number of simulation steps
    processing_time : int
        Number of steps each job takes to process
    verbose : bool
        If True, print detailed information
    
    Returns:
    --------
    tuple
        (average_queue_length, queue_lengths, total_arrivals, total_processed)
    """
    # Queue: list of jobs, each job is represented by its remaining processing time
    queue = deque()  # Jobs waiting to be processed
    processing_job = None  # Current job being processed
    processing_remaining = 0  # Remaining processing time
    
    queue_lengths = []  # Track queue length at each step
    total_arrivals = 0
    total_processed = 0
    
    if verbose:
        print(f"\nStarting simulation: λ={lam}, steps={num_steps}, processing_time={processing_time}")
    
    for step in range(num_steps):
        # STEP 3a: Sample new arrivals from Poisson distribution
        new_arrivals = sample_poisson_jobs(lam)
        total_arrivals += new_arrivals
        
        # Add new arrivals to queue
        for _ in range(new_arrivals):
            queue.append(processing_time)  # Each job needs processing_time steps
        
        # STEP 3b: Process one job per time step
        if processing_job is None and len(queue) > 0:
            # Start processing a new job
            processing_job = queue.popleft()
            processing_remaining = processing_job
        
        if processing_job is not None:
            processing_remaining -= 1
            if processing_remaining <= 0:
                # Job completed
                total_processed += 1
                processing_job = None
                processing_remaining = 0
        
        # Record queue length (waiting jobs only, not the one being processed)
        queue_length = len(queue)
        queue_lengths.append(queue_length)
    
    average_queue_length = np.mean(queue_lengths)
    
    if verbose:
        print(f"  Total arrivals: {total_arrivals}")
        print(f"  Total processed: {total_processed}")
        print(f"  Jobs left in queue: {len(queue)}")
        print(f"  Average queue length: {average_queue_length:.3f}")
    
    return average_queue_length, queue_lengths, total_arrivals, total_processed


# =============================================================================
# STEP 4: Run Single Simulation (2000 steps, λ=0.1, processing_time=4)
# =============================================================================

def task_4_single_simulation():
    """
    Task 4: Run simulation for 2000 steps with λ=0.1, processing_time=4
    """
    print("\n" + "="*60)
    print("TASK 4: Single Simulation")
    print("="*60)
    
    lam = 0.1
    num_steps = 2000
    processing_time = 4
    
    avg_queue_length, queue_lengths, total_arrivals, total_processed = \
        simulate_queue(lam, num_steps, processing_time, verbose=True)
    
    # Plot queue length over time
    plt.figure(figsize=(14, 6))
    plt.plot(queue_lengths, linewidth=0.5, alpha=0.7)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Queue Length', fontsize=12)
    plt.title(f'Task 4: Queue Length Over Time (λ={lam}, processing_time={processing_time})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=float(avg_queue_length), color='r', linestyle='--', 
                label=f'Average: {avg_queue_length:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/task4_queue_over_time.png', dpi=150)
    plt.show()
    
    print("✓ Task 4 completed")
    return avg_queue_length


# =============================================================================
# STEP 5: Multiple Independent Runs (200 runs, rates 0.005-0.25)
# =============================================================================

def task_5_multiple_runs(processing_time=4):
    """
    Task 5: Perform 200 runs of 2000 steps for rates 0.005 to 0.25 (increment 0.005)
    Calculate and plot average waiting list length as function of rate
    
    Parameters:
    -----------
    processing_time : int
        Processing time per job
    """
    print("\n" + "="*60)
    print(f"TASK 5: Multiple Runs (processing_time={processing_time})")
    print("="*60)
    
    # Define rate range
    rates = np.arange(0.005, 0.255, 0.005)  # 0.005 to 0.25 inclusive
    num_runs = 200
    num_steps = 2000
    
    avg_queue_lengths = []
    
    print(f"Running {num_runs} simulations for each of {len(rates)} rate values...")
    print("This may take a few minutes...")
    
    for i, lam in enumerate(rates):
        run_avg_queues = []
        
        # Perform multiple independent runs
        for run in range(num_runs):
            avg_queue, _, _, _ = simulate_queue(lam, num_steps, processing_time)
            run_avg_queues.append(avg_queue)
        
        # Average across all runs for this rate
        overall_avg = np.mean(run_avg_queues)
        avg_queue_lengths.append(overall_avg)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(rates)} rates...")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(rates, avg_queue_lengths, linewidth=2, marker='o', markersize=3)
    plt.xlabel('Arrival Rate λ', fontsize=12)
    plt.ylabel('Average Queue Length', fontsize=12)
    plt.title(f'Task 5: Average Queue Length vs Arrival Rate\n({num_runs} runs, {num_steps} steps, processing_time={processing_time})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/task5_queue_vs_rate_pt{processing_time}.png', dpi=150)
    plt.show()
    
    print("✓ Task 5 completed")
    return rates, avg_queue_lengths


# =============================================================================
# STEP 6: Compare Different Processing Times
# =============================================================================

def task_6_compare_processing_times():
    """
    Task 6: Compare processing_time=2 and processing_time=4
    For processing_time=2: rates 0.005 to 0.5 (increment 0.005)
    """
    print("\n" + "="*60)
    print("TASK 6: Compare Different Processing Times")
    print("="*60)
    
    # First, run for processing_time=4 (rates 0.005-0.25)
    print("\nRunning simulations for processing_time=4...")
    rates_4, avg_queues_4 = task_5_multiple_runs(processing_time=4)
    
    # Then, run for processing_time=2 (rates 0.005-0.5)
    print("\nRunning simulations for processing_time=2...")
    rates_2 = np.arange(0.005, 0.505, 0.005)  # 0.005 to 0.5 inclusive
    num_runs = 200
    num_steps = 2000
    processing_time_2 = 2
    
    avg_queue_lengths_2 = []
    
    print(f"Running {num_runs} simulations for each of {len(rates_2)} rate values...")
    
    for i, lam in enumerate(rates_2):
        run_avg_queues = []
        
        for run in range(num_runs):
            avg_queue, _, _, _ = simulate_queue(lam, num_steps, processing_time_2)
            run_avg_queues.append(avg_queue)
        
        overall_avg = np.mean(run_avg_queues)
        avg_queue_lengths_2.append(overall_avg)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(rates_2)} rates...")
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Plot processing_time=4 (only for overlapping rates: 0.005-0.25)
    plt.plot(rates_4, avg_queues_4, linewidth=2, marker='o', markersize=4,
            label='Processing Time = 4 steps', color='blue')
    
    # Plot processing_time=2 (for 0.005-0.5)
    plt.plot(rates_2, avg_queue_lengths_2, linewidth=2, marker='s', markersize=3,
            label='Processing Time = 2 steps', color='red')
    
    plt.xlabel('Arrival Rate λ', fontsize=12)
    plt.ylabel('Average Queue Length', fontsize=12)
    plt.title('Task 6: Comparison of Average Queue Length\n(200 runs, 2000 steps each)', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/task6_comparison.png', dpi=150)
    plt.show()
    
    # Print comparison analysis
    print("\n" + "="*60)
    print("ANALYSIS AND COMPARISON:")
    print("="*60)
    print("\nKey Observations:")
    print("1. With processing_time=2, the system can handle higher arrival rates")
    print("   before queue length grows significantly")
    print("2. The critical rate (where queue starts growing rapidly) is higher")
    print("   when processing_time=2 (around λ=0.4-0.5) compared to processing_time=4 (around λ=0.2-0.25)")
    print("3. This makes sense: faster processing (2 steps) means the system")
    print("   can clear jobs faster, allowing higher arrival rates")
    print("4. For processing_time=4, the system becomes unstable at lower rates")
    print("   because jobs arrive faster than they can be processed")
    print("\nMathematical insight:")
    print("  Stability condition: λ < μ, where μ = 1/processing_time")
    print("  For processing_time=2: μ = 0.5, so stable when λ < 0.5")
    print("  For processing_time=4: μ = 0.25, so stable when λ < 0.25")
    
    print("\n✓ Task 6 completed")
    
    return rates_4, avg_queues_4, rates_2, avg_queue_lengths_2


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUEUE SIMULATION - STEP BY STEP IMPLEMENTATION")
    print("="*60)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Task 1: Plot Poisson Distributions
    print("\nExecuting Task 1...")
    lambda_values = [0.01, 0.1, 0.5, 1.0]
    plot_poisson_distributions(lambda_values, max_x=15)
    
    # Task 2a: Sampling is demonstrated in simulate_queue function
    
    # Task 2b: Queue simulation is implemented in simulate_queue function
    
    # Task 2: Single simulation
    task_4_single_simulation()
    
    # Task 3: Multiple runs (processing_time=4)
    task_5_multiple_runs(processing_time=4)
    
    # Task 4: Compare processing times
    task_6_compare_processing_times()
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED!")
    print("="*60)