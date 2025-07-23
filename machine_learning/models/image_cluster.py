import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from .base_model import BaseClusterModel
import logging

logger = logging.getLogger(__name__)

class ImageClusterModel(BaseClusterModel):
    """Clustering model for image data."""
    
    def __init__(self, 
                 model_type: str = 'kmeans',
                 n_clusters: int = 5,
                 use_dim_reduction: bool = True,
                 n_components: int = 128,
                 feature_type: str = 'deep',  # 'deep', 'color', 'texture', 'all'
                 random_state: int = 42,
                 **kwargs):
        """Initialize the image clustering model.
        
        Args:
            model_type: Type of clustering algorithm ('kmeans', 'dbscan', 'gmm', 'optics')
            n_clusters: Number of clusters (not used for DBSCAN/OPTICS)
            use_dim_reduction: Whether to use dimensionality reduction
            n_components: Number of components for dimensionality reduction
            feature_type: Type of features to use ('deep', 'color', 'texture', 'all')
            random_state: Random seed
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_type=model_type,
            n_clusters=n_clusters,
            use_dim_reduction=use_dim_reduction,
            n_components=n_components,
            feature_type=feature_type,
            random_state=random_state,
            **kwargs
        )
        
        self.model_type = model_type.lower()
        self.n_clusters = n_clusters
        self.use_dim_reduction = use_dim_reduction
        self.n_components = n_components
        self.feature_type = feature_type.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.dim_reducer = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the clustering model based on model_type."""
        if self.model_type == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 
                              'n_components', 'feature_type', 'random_state']}
            )
        elif self.model_type == 'dbscan':
            self.model = DBSCAN(
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 
                              'n_components', 'feature_type', 'random_state']}
            )
        elif self.model_type == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 
                              'n_components', 'feature_type', 'random_state']}
            )
        elif self.model_type == 'optics':
            self.model = OPTICS(
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['model_type', 'n_clusters', 'use_dim_reduction', 
                              'n_components', 'feature_type', 'random_state']}
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize dimensionality reducer
        if self.use_dim_reduction:
            self.dim_reducer = Pipeline([
                ('pca', PCA(n_components=min(100, self.n_components), 
                           random_state=self.random_state)),
                ('tsne', TSNE(n_components=min(2, self.n_components), 
                             random_state=self.random_state))
            ])
    
    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from images based on feature_type."""
        if not images or len(images) == 0:
            return np.array([])
        
        try:
            features = []
            
            for img in images:
                if img is None or not isinstance(img, np.ndarray):
                    continue
                    
                img_features = []
                
                # Extract color features (histogram)
                if self.feature_type in ['color', 'all']:
                    hist = self._extract_color_histogram(img)
                    img_features.extend(hist)
                
                # Extract texture features (LBP, Haralick, etc.)
                if self.feature_type in ['texture', 'all']:
                    texture = self._extract_texture_features(img)
                    img_features.extend(texture)
                
                # Use deep features if available or if feature_type is 'deep' or 'all'
                if self.feature_type in ['deep', 'all'] and hasattr(img, 'deep_features'):
                    img_features.extend(img.deep_features)
                
                features.append(img_features)
            
            if not features:
                return np.array([])
                
            features = np.array(features)
            
            # Apply dimensionality reduction if enabled
            if self.use_dim_reduction and self.dim_reducer is not None and len(features) > 0:
                features = self.dim_reducer.fit_transform(features)
            
            # Scale features
            if len(features) > 0:
                features = self.scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {str(e)}")
            return np.array([])
    
    def _extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract color histogram features from an image."""
        try:
            # Convert to HSV color space
            import cv2
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Compute color histogram
            hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 256])
            hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
            
            # Normalize the histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Concatenate histograms
            return np.concatenate([hist_h, hist_s, hist_v])
            
        except Exception as e:
            logger.warning(f"Error extracting color histogram: {str(e)}")
            return np.zeros(bins * 3)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP (Local Binary Pattern)."""
        try:
            import cv2
            from skimage.feature import local_binary_pattern
            
            # Convert to grayscale
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Compute LBP
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                 range=(0, n_points + 2))
            
            # Normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            logger.warning(f"Error extracting texture features: {str(e)}")
            return np.zeros(26)  # Default size for LBP features
    
    def fit(self, X: List[np.ndarray]) -> 'ImageClusterModel':
        """Fit the model to the image data."""
        if not X or len(X) == 0:
            logger.warning("No data provided for fitting")
            return self
        
        try:
            # Extract features
            X_features = self._extract_features(X)
            
            if X_features.size == 0:
                logger.warning("No valid features after extraction")
                return self
            
            # Fit the model
            if hasattr(self.model, 'fit'):
                self.model.fit(X_features)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            self.is_fitted = False
            return self
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict cluster labels for the input image data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not X or len(X) == 0:
            return np.array([])
        
        try:
            # Extract features
            X_features = self._extract_features(X)
            
            if X_features.size == 0:
                return np.array([-1] * len(X))  # Return -1 for all samples if feature extraction fails
            
            # Predict clusters
            if hasattr(self.model, 'predict'):
                return self.model.predict(X_features)
            elif hasattr(self.model, 'fit_predict'):
                return self.model.fit_predict(X_features)
            elif hasattr(self.model, 'labels_'):
                return self.model.labels_
            else:
                raise RuntimeError("Model does not support prediction")
                
        except Exception as e:
            logger.error(f"Error predicting clusters: {str(e)}")
            return np.array([-1] * len(X))  # Return -1 for all samples on error
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model's parameters."""
        params = {
            'model_type': self.model_type,
            'n_clusters': self.n_clusters,
            'use_dim_reduction': self.use_dim_reduction,
            'n_components': self.n_components,
            'feature_type': self.feature_type,
            'random_state': self.random_state
        }
        params.update(self.model_params)
        return params
