from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=1000)
def poisson_prob(lambda_param: float, n: int) -> float:
    """Cached computation of Poisson probability."""
    return (lambda_param**n * np.exp(-lambda_param)) / math.factorial(n)


class TransitionCache:
    def __init__(self, max_size=10000):
        self._cache = {}
        self.max_size = max_size

    def get(self, state, action):
        """Get cached transitions for state-action pair."""
        return self._cache.get((state, action))

    def set(self, state, action, transitions):
        """Cache transitions for state-action pair."""
        if len(self._cache) >= self.max_size:
            # Simple cache clearing strategy: remove everything when full
            self._cache.clear()
        self._cache[(state, action)] = transitions


import numpy as np
import matplotlib.pyplot as plt


def plot_policy(policy_array, max_cars=20):
    """
    Create a contour plot of the policy.

    Args:
        policy_array: 2D numpy array of actions for each state
        max_cars: Maximum number of cars (default 20)
    """
    # Create coordinate grids
    x = np.arange(max_cars + 1)
    y = np.arange(max_cars + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    # Policy array is a 2D array, with rows corresponding to location 1 and columns to location 2.
    # Use contourf for filled contours with reversed colormap
    contour = plt.contourf(
        X, Y, policy_array, levels=np.arange(-5.5, 6.5, 1), cmap="RdYlBu_r"
    )
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label(
        "Number of Cars Moved\n(Negative: Location 2 → 1, Positive: Location 1 → 2)"
    )

    # Add contour lines with labels
    line_contours = plt.contour(
        X, Y, policy_array, levels=np.arange(-5, 6), colors="black", linewidths=0.5
    )
    plt.clabel(line_contours, inline=True, fontsize=8, fmt="%d")

    # Labels and title
    plt.xlabel("Location 1")
    plt.ylabel("Location 2")
    plt.title("Optimal Policy: Number of Cars to Move Between Locations")

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    return plt


def save_policy_plot(policy, filename="policy_plot.png"):
    """
    Save a visualization of the policy to a file.

    Args:
        agent: PolicyIterationAgent instance
        filename: Output filename
    """
    plt = plot_policy(policy)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
