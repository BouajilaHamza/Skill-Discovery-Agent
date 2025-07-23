from pathlib import Path
from typing import List, Set, Dict, Optional
import os

class FileOrganizerConfig:
    """Configuration for the file organizer."""
    
    def __init__(self):
        # File types to process
        self.document_extensions = {
            '.txt', '.pdf', '.doc', '.docx', '.odt', '.rtf',
            '.csv', '.xls', '.xlsx', '.ods', '.ppt', '.pptx',
            '.md', '.html', '.htm'
        }
        
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
            '.webp', '.svg', '.heic', '.heif'
        }
        
        self.all_supported_extensions = self.document_extensions.union(self.image_extensions)
        
        # Default directories
        self.default_source_dir = str(Path.home() / 'Downloads')
        self.default_output_dir = str(Path.home() / 'OrganizedFiles')
        
        # Clustering parameters
        self.default_num_clusters = 5
        self.feature_weights = {
            'file_type': 0.3,
            'content': 0.7
        }
        
        # Processing settings
        self.max_file_size_mb = 50  # Skip files larger than this
        self.extract_metadata = True
        self.recursive = True
        
        # Logging
        self.log_level = 'INFO'
        self.log_file = 'file_organizer.log'

    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.all_supported_extensions

# Global configuration instance
config = FileOrganizerConfig()
