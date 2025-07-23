from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseClusterModel(ABC):
    """Abstract base class for all clustering models."""
    
    def __init__(self, **kwargs):
        """Initialize the clustering model.
        
        Args:
            **kwargs: Model-specific parameters
        """
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterModel':
        """Fit the model to the data.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            self: Returns the instance itself
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for the input data.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            Array of cluster labels
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict cluster labels.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            Array of cluster labels
        """
        return self.fit(X).predict(X)
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the model's parameters.
        
        Returns:
            Dictionary of parameter names mapped to their values
        """
        pass
    
    def set_params(self, **params) -> 'BaseClusterModel':
        """Set the model's parameters.
        
        Args:
            **params: Parameter names and their values to set
            
        Returns:
            self: Returns the instance itself
        """
        self.model_params.update(params)
        return self
    
    def score(self, X: np.ndarray) -> float:
        """Compute a clustering score for the given data.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            Clustering score (higher is better)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        
        from sklearn.metrics import silhouette_score
        try:
            labels = self.predict(X)
            # Silhouette score requires at least 2 clusters and at least 2 samples per cluster
            if len(set(labels)) < 2 or min([sum(labels == i) for i in set(labels)]) < 2:
                return -1.0
            return silhouette_score(X, labels)
        except Exception as e:
            logger.warning(f"Error computing silhouette score: {str(e)}")
            return -1.0
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get the cluster centers if available.
        
        Returns:
            Array of cluster centers, or None if not applicable
        """
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances if available.
        
        Returns:
            Array of feature importances, or None if not applicable
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return self.model.coef_
        return None
