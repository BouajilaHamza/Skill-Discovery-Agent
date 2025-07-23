import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModel
import hashlib
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature extraction and embedding generation for files."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """Initialize the feature engineer.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run models on ('cuda', 'mps', 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        self.model_name = model_name
        self._load_models()
        
        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_models(self):
        """Load the required ML models."""
        logger.info(f"Loading models on device: {self.device}")
        
        # Text model
        try:
            self.text_model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded text model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load text model: {str(e)}")
            raise
        
        # Image model (using a pre-trained model from Hugging Face)
        try:
            model_name = "google/vit-base-patch16-224"
            self.image_processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.image_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.image_model.eval()
            logger.info(f"Loaded image model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load image model: {str(e)}")
            # We can still work with text even if image model fails
    
    def generate_embeddings(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for a file based on its content and metadata.
        
        Args:
            file_data: Dictionary containing file metadata and content
            
        Returns:
            Dictionary with original data and generated embeddings
        """
        if not file_data or 'content' not in file_data:
            return {}
        
        try:
            features = {}
            
            # Generate metadata features
            if 'metadata' in file_data:
                features['metadata'] = self._extract_metadata_features(file_data['metadata'])
            
            # Generate content-based features
            if file_data.get('content_type') == 'document' and 'content' in file_data:
                features['content'] = self._generate_text_embeddings(file_data['content'])
            elif file_data.get('content_type') == 'image' and 'path' in file_data:
                features['content'] = self._generate_image_embeddings(file_data['path'])
            
            # Combine features
            combined_features = self._combine_features(features, file_data)
            
            # Add to results
            result = file_data.copy()
            result['features'] = {
                'embeddings': combined_features,
                'feature_types': list(features.keys())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {file_data.get('path', 'unknown')}: {str(e)}")
            file_data['error'] = str(e)
            return file_data
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features from file metadata."""
        # Simple one-hot encoding of metadata
        features = {}
        
        # File type
        file_type = metadata.get('file_type', 'other')
        features[f"type_{file_type}"] = 1.0
        
        # Extension (simple hash as a feature)
        ext = metadata.get('extension', '').lower().lstrip('.')
        if ext:
            ext_hash = int(hashlib.md5(ext.encode()).hexdigest(), 16) % 1000 / 1000.0
            features["ext_hash"] = ext_hash
        
        # Size (log scale)
        size_mb = metadata.get('size_bytes', 0) / (1024 * 1024)
        features["size_log"] = np.log1p(size_mb)
        
        # Convert to numpy array
        return np.array(list(features.values()))
    
    def _generate_text_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text content."""
        if not text or not self.text_model:
            return np.array([])
        
        try:
            # Clean and chunk text if too long
            text = text.strip()
            if not text:
                return np.array([])
                
            # Generate embedding
            embedding = self.text_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            return np.array([])
    
    def _generate_image_embeddings(self, image_path: str) -> np.ndarray:
        """Generate embeddings for image content."""
        if not self.image_model or not self.image_processor:
            return np.array([])
            
        try:
            # Open and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate features
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                # Use the [CLS] token representation as the image embedding
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating image embeddings for {image_path}: {str(e)}")
            return np.array([])
    
    def _combine_features(self, features: Dict[str, np.ndarray], file_data: Dict[str, Any]) -> np.ndarray:
        """Combine different feature types into a single feature vector."""
        combined = []
        
        # Add metadata features if available
        if 'metadata' in features and len(features['metadata']) > 0:
            combined.append(features['metadata'])
        
        # Add content features if available
        if 'content' in features and len(features['content']) > 0:
            combined.append(features['content'])
        
        if not combined:
            return np.array([])
            
        # Concatenate all features
        return np.concatenate(combined)

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Example file data (would come from file processor and content extractor)
    example_file = {
        'path': 'example.txt',
        'content_type': 'document',
        'content': 'This is a sample document for testing the feature extraction.',
        'metadata': {
            'file_type': 'document',
            'extension': '.txt',
            'size_bytes': 1024
        }
    }
    
    # Generate embeddings
    result = feature_engineer.generate_embeddings(example_file)
    
    if 'features' in result:
        print(f"Generated {len(result['features']['embeddings'])}-dimensional embedding")
        print(f"Feature types: {', '.join(result['features']['feature_types'])}")
    else:
        print("Failed to generate embeddings")
