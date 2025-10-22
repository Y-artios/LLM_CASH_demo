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
    
    
    def vectorize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple (X, y) where:
        - X is a numpy array containing the attributes (n1, n2, mu1, mu2, alpha1, alpha2)
        - y is an array containing lambda_star
        """
        # Flatten mu_1 and mu_2 if they are arrays
        mu1_flat = np.array(self.mu_1).flatten()
        mu2_flat = np.array(self.mu_2).flatten()
        
        # Create the feature vector X by concatenating all attributes
        x = np.concatenate([
            [self.n_1, self.n_2],           # scalar values
            mu1_flat,                       # flattened mu_1
            mu2_flat,                       # flattened mu_2
            [self.alpha_1, self.alpha_2]    # scalar values
        ])
        
        # Get lambda_star as the target
        y = np.array([self.lambda_star()])
        
        return x, y
    
    def lambda_star(self) -> float:
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


def create_dataset(context_tasks: List[Task]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a list of tasks into a dataset (X, Y) for classifier training.
    
    Args:
        context_tasks: List of Task objects
    
    Returns:
        Tuple (X, Y) where:
        - X is a 2D numpy array of shape (n_tasks, n_features) containing task features
        - Y is a 1D numpy array of shape (n_tasks,) containing lambda_star values
    """
    if not context_tasks:
        return np.array([]).reshape(0, 0), np.array([])
    
    # Vectorize each task and collect features and targets
    X_list = []
    y_list = []
    
    for task in context_tasks:
        task_X, task_y = task.vectorize()
        X_list.append(task_X)
        y_list.append(task_y[0])  # Extract scalar from array
    
    # Stack all features and targets
    X = np.vstack(X_list)  # Shape: (n_tasks, n_features)
    Y = np.array(y_list)   # Shape: (n_tasks,)
    
    return X, Y
