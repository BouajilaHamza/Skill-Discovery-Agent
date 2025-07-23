import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline
from .base_model import BaseClusterModel
import logging

logger = logging.getLogger(__name__)

class TextClusterModel(BaseClusterModel):
    """Clustering model for text data."""
    
    def __init__(self, 
                 model_type: str = 'kmeans',
                 n_clusters: int = 5,
                 use_dim_reduction: bool = True,
                 n_components: int = 100,
                 random_state: int = 42,
                 **kwargs):
        """Initialize the text clustering model.
        
        Args:
            model_type: Type of clustering algorithm ('kmeans', 'dbscan', 'agglomerative', 'gmm', 'optics')
            n_clusters: Number of clusters (not used for DBSCAN/OPTICS)
            use_dim_reduction: Whether to use dimensionality reduction
            n_components: Number of components for dimensionality reduction
            random_state: Random seed
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_type=model_type,
            n_clusters=n_clusters,
            use_dim_reduction=use_dim_reduction,
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        
        self.model_type = model_type.lower()
        self.n_clusters = n_clusters
        self.use_dim_reduction = use_dim_reduction
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = None
        self.dim_reducer = None
        self.scaler = StandardScaler()
        self._init_model()
    
    def _init_model(self):
        """Initialize the clustering model based on model_type."""
        if self.model_type == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 'n_components', 'random_state']}
            )
        elif self.model_type == 'dbscan':
            self.model = DBSCAN(
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 'n_components', 'random_state']}
            )
        elif self.model_type == 'agglomerative':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 'n_components', 'random_state']}
            )
        elif self.model_type == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 'n_components', 'random_state']}
            )
        elif self.model_type == 'optics':
            self.model = OPTICS(
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 'n_components', 'random_state']}
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize vectorizer and dimensionality reducer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        if self.use_dim_reduction:
            self.dim_reducer = Pipeline([
                ('svd', TruncatedSVD(n_components=min(100, self.n_components))),
                ('pca', PCA(n_components=self.n_components, random_state=self.random_state))
            ])
    
    def _preprocess_text(self, texts: list) -> np.ndarray:
        """Preprocess text data into numerical features."""
        if not texts or not any(isinstance(t, str) for t in texts):
            return np.array([])
        
        # Convert to list of strings (handle None values)
        texts = [str(t) if t is not None else '' for t in texts]
        
        # Vectorize text
        try:
            X = self.vectorizer.fit_transform(texts)
            
            # Apply dimensionality reduction if enabled
            if self.use_dim_reduction and self.dim_reducer is not None:
                X = self.dim_reducer.fit_transform(X)
            
            # Scale features
            if X.shape[0] > 0:
                X = self.scaler.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
            
            return X
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return np.array([])
    
    def fit(self, X: list) -> 'TextClusterModel':
        """Fit the model to the text data."""
        if not X or len(X) == 0:
            logger.warning("No data provided for fitting")
            return self
        
        try:
            # Preprocess text
            X_processed = self._preprocess_text(X)
            
            if X_processed.size == 0:
                logger.warning("No valid features after preprocessing")
                return self
            
            # Fit the model
            if hasattr(self.model, 'fit'):
                self.model.fit(X_processed)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            self.is_fitted = False
            return self
    
    def predict(self, X: list) -> np.ndarray:
        """Predict cluster labels for the input text data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not X or len(X) == 0:
            return np.array([])
        
        try:
            # Preprocess text
            X_processed = self._preprocess_text(X)
            
            if X_processed.size == 0:
                return np.array([-1] * len(X))  # Return -1 for all samples if preprocessing fails
            
            # Predict clusters
            if hasattr(self.model, 'predict'):
                return self.model.predict(X_processed)
            elif hasattr(self.model, 'fit_predict'):
                return self.model.fit_predict(X_processed)
            elif hasattr(self.model, 'labels_'):
                return self.model.labels_
            else:
                raise RuntimeError("Model does not support prediction")
                
        except Exception as e:
            logger.error(f"Error predicting clusters: {str(e)}")
            return np.array([-1] * len(X))  # Return -1 for all samples on error
    
    def get_feature_names(self) -> list:
        """Get the feature names after vectorization."""
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out().tolist()
        elif hasattr(self.vectorizer, 'get_feature_names'):
            return self.vectorizer.get_feature_names()
        else:
            return []
    
    def get_top_terms_per_cluster(self, n_terms: int = 10) -> Dict[int, list]:
        """Get the top terms for each cluster."""
        if not self.is_fitted or not hasattr(self.model, 'cluster_centers_'):
            return {}
        
        try:
            feature_names = self.get_feature_names()
            if not feature_names:
                return {}
            
            cluster_centers = self.model.cluster_centers_
            if hasattr(cluster_centers, 'toarray'):
                cluster_centers = cluster_centers.toarray()
            
            top_terms = {}
            for i, center in enumerate(cluster_centers):
                top_indices = np.argsort(center)[-n_terms:][::-1]
                top_terms[i] = [feature_names[idx] for idx in top_indices 
                              if idx < len(feature_names)]
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Error getting top terms: {str(e)}")
            return {}
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model's parameters."""
        params = {
            'model_type': self.model_type,
            'n_clusters': self.n_clusters,
            'use_dim_reduction': self.use_dim_reduction,
            'n_components': self.n_components,
            'random_state': self.random_state
        }
        params.update(self.model_params)
        return params
