from functools import lru_cache
import math
import numpy as np

@lru_cache(maxsize=1000)
def poisson_prob(lambda_param: float, n: int) -> float:
    """Cached computation of Poisson probability."""
    return (lambda_param ** n * np.exp(-lambda_param)) / math.factorial(n)

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