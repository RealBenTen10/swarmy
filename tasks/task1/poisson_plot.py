"""
Poisson Distribution Plot Implementation

This script implements the Poisson distribution for modeling job arrivals
in a computer system based on queuing theory.

Formula: P(X=i) = e^(-λ) * λ^i / i!
Where λ = α * Δt, and since Δt = 1, we have λ = α
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def poisson_probability(i, alpha):
    """
    Calculate the Poisson probability P(X=i) for given i and α.
    
    Parameters:
    -----------
    i : int
        Number of jobs (the random variable X)
    alpha : float
        Rate parameter (λ = α since Δt = 1)
    
    Returns:
    --------
    float
        Probability P(X=i)
    """
    lam = alpha  # λ = α * Δt, and Δt = 1
    # Calculate using the Poisson PMF formula: P(X=i) = e^(-λ) * λ^i / i!
    # We can use scipy's poisson.pmf for numerical stability
    return poisson.pmf(i, lam)


def plot_poisson_distributions(alpha_values, max_i=20):
    """
    Plot Poisson distributions for different α values.
    
    Parameters:
    -----------
    alpha_values : list
        List of α values to plot
    max_i : int
        Maximum value of i (number of jobs) to plot
    """
    i_values = np.arange(0, max_i + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot for each α value
    for alpha in alpha_values:
        # Calculate probabilities for all i values
        probabilities = [poisson_probability(i, alpha) for i in i_values]
        
        # Plot with labels
        plt.plot(i_values, probabilities, marker='o', linewidth=2, 
                label=f'α = {alpha}', markersize=4)
    
    # Customize the plot
    plt.xlabel('Number of Jobs (i)', fontsize=12)
    plt.ylabel('Probability P(X=i)', fontsize=12)
    plt.title('Poisson Distribution: P(X=i) for Different Arrival Rates α', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(i_values[::2])  
    plt.ylim(bottom=0)  
    
    plt.text(0.02, 0.98, 
             f'Formula: P(X=i) = e^(-λ) × λ^i / i!\nλ = α (since Δt = 1)',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the α values as specified in the problem
    alpha_values = [0.01, 0.1, 0.5, 1.0]
    
    # For α = 0.01 and 0.1, we need a wider range to see the distribution
    # For α = 0.5 and 1.0, most probability mass is within 0-10
    # Let's use max_i = 15 to capture all relevant probabilities
    max_i = 15
    
    print("Plotting Poisson distributions...")
    print(f"α values: {alpha_values}")
    print(f"Range of i (jobs): 0 to {max_i}")
    print("\nExpected behavior:")
    print("- α = 0.01: Very low probability, mostly at i=0")
    print("- α = 0.1: Low probability, mostly at i=0")
    print("- α = 0.5: Moderate probability, spread around i=0-2")
    print("- α = 1.0: Higher probability, spread around i=0-3")
    
    # Create the plot
    plot_poisson_distributions(alpha_values, max_i)

