import os
import magic
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from datetime import datetime
import mimetypes
from ..config.organizer_config import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles file system operations and metadata extraction."""
    
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        self._setup_mime_types()
    
    def _setup_mime_types(self):
        """Ensure all supported extensions have proper MIME type mappings."""
        mimetypes.add_type('application/x-ole-storage', '.doc')
        mimetypes.add_type('application/x-ole-storage', '.xls')
        mimetypes.add_type('application/x-ole-storage', '.ppt')
    
    def scan_directory(self, directory: str) -> List[Dict]:
        """Scan a directory and return file metadata.
        
        Args:
            directory: Path to the directory to scan
            
        Returns:
            List of dictionaries containing file metadata
        """
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    file_meta = self.get_file_metadata(file_path)
                    if file_meta:
                        files.append(file_meta)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
                    continue
        
        return files
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata or None if file should be skipped
        """
        try:
            file_path = os.path.abspath(file_path)
            file_stat = os.stat(file_path)
            
            # Skip files that are too large
            file_size_mb = file_stat.st_size / (1024 * 1024)
            if file_size_mb > config.max_file_size_mb:
                logger.debug(f"Skipping large file: {file_path} ({file_size_mb:.2f} MB)")
                return None
            
            # Get file extension and check if supported
            file_ext = os.path.splitext(file_path)[1].lower()
            if not config.is_supported_file(file_path):
                return None
            
            # Basic file info
            file_info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'extension': file_ext,
                'size_bytes': file_stat.st_size,
                'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(file_stat.st_atime).isoformat(),
                'parent_dir': os.path.dirname(file_path),
                'file_type': self._get_file_type(file_path, file_ext),
                'mime_type': self._get_mime_type(file_path),
                'hash': self._calculate_file_hash(file_path)
            }
            
            # Add additional metadata based on file type
            if file_ext in config.document_extensions:
                file_info['content_type'] = 'document'
            elif file_ext in config.image_extensions:
                file_info['content_type'] = 'image'
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            return None
    
    def _get_file_type(self, file_path: str, ext: str) -> str:
        """Determine the type of file."""
        if ext in config.document_extensions:
            return 'document'
        elif ext in config.image_extensions:
            return 'image'
        return 'other'
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get the MIME type of a file."""
        try:
            return self.mime.from_file(file_path)
        except:
            return 'application/octet-stream'
    
    def _calculate_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for block in iter(lambda: f.read(block_size), b''):
                    sha256.update(block)
            return sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {str(e)}")
            return ''

# Example usage
if __name__ == "__main__":
    processor = FileProcessor()
    files = processor.scan_directory("~/Downloads")
    print(f"Found {len(files)} files")
    if files:
        print("Sample file:", files[0])
