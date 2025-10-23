import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy.linalg import toeplitz
from best_lambda import find_best_lambda

@dataclass
class Task:
    n_1:   int
    n_2: int
    mu_1: List[float]
    mu_2:  List[float]
    alpha_1: float
    alpha_2:  float 
    d: int

    def metadata(self, task_id, include_lambda: bool) -> dict:
        """Return the exact prompt block to feed the LLM."""
        meta = {
            "task_id" : task_id,
            "n1": self.n_1,
            "n2": self.n_2,
            "mu1": self.mu_1.astype(float).tolist(),
            "mu2": self.mu_2.astype(float).tolist(),
            "alpha_1": self.alpha_1.item(),
            "alpha_2": self.alpha_2.item(),
        }

        if include_lambda:
            meta['lambda_star'] =  float(self.lambda_star()[0])

        return meta
    
    
    def vectorize(self) -> np.ndarray:
        """Return a vector representation of the task."""
        # Convert alphas to scalars if they're arrays
        alpha_1_scalar = self.alpha_1.item() if hasattr(self.alpha_1, 'item') else self.alpha_1
        alpha_2_scalar = self.alpha_2.item() if hasattr(self.alpha_2, 'item') else self.alpha_2
        
        return np.array([
            self.n_1, self.n_2,
            *self.mu_1, *self.mu_2,
            alpha_1_scalar, alpha_2_scalar
        ], dtype=float)
    
    def lambda_star(self):
        return find_best_lambda(means=np.array([self.mu_1,self.mu_2]),
                                cov = [toeplitz(self.alpha_1**(np.arange(self.d))), toeplitz(self.alpha_2**(np.arange(self.d)))],
                                n_vec=[self.n_1, self.n_2])


def sample_task(
    d: int,
) -> Task:
    """
    Returns a random Task where
    """

    eps = np.random.uniform(low=0,high=2,size=d)
    eps = np.round(eps,2)
    mu1 = np.ones((d, ))
    mu2 = -eps*mu1
    alpha1 = np.round(np.random.uniform(low=0,high=0.9,size=1),2)
    alpha2 = np.round(np.random.uniform(low=0,high=0.9,size=1),2)
    N = np.random.randint(low=10, high=500, size=2)
    n1 = N[0].item()
    n2 = N[1].item()


    return Task(
        n_1=n1,
        n_2=n2,
        mu_1=mu1,
        mu_2=mu2,
        alpha_1=alpha1,
        alpha_2=alpha2,
        d=d,
        )