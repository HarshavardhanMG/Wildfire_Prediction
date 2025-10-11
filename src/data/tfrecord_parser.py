"""
TFRecord parsing utility for the Next Day Wildfire Spread dataset.

This module provides functions to parse and explore TFRecord files containing
geospatial and meteorological data for wildfire prediction.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os


class TFRecordParser:
    """Parser for Next Day Wildfire Spread TFRecord files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the TFRecord parser.
        
        Args:
            data_dir: Directory containing TFRecord files
        """
        self.data_dir = data_dir
        self.feature_description = None
        
    def get_tfrecord_files(self, pattern: str = "*.tfrecord") -> List[str]:
        """
        Get list of TFRecord files in the data directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of TFRecord file paths
        """
        import glob
        file_pattern = os.path.join(self.data_dir, pattern)
        return glob.glob(file_pattern)
    
    def parse_single_example(self, example_proto):
        """
        Parse a single example from TFRecord.
        
        Args:
            example_proto: Serialized example
            
        Returns:
            Parsed features
        """
        if self.feature_description is None:
            # Create a basic feature description - this will be updated based on actual data
            self.feature_description = {
                # Common features in wildfire datasets
                'elevation': tf.io.FixedLenFeature([], tf.string),
                'th': tf.io.FixedLenFeature([], tf.string),
                'vs': tf.io.FixedLenFeature([], tf.string),
                'tmmn': tf.io.FixedLenFeature([], tf.string),
                'tmmx': tf.io.FixedLenFeature([], tf.string),
                'sph': tf.io.FixedLenFeature([], tf.string),
                'pr': tf.io.FixedLenFeature([], tf.string),
                'pdsi': tf.io.FixedLenFeature([], tf.string),
                'NDVI': tf.io.FixedLenFeature([], tf.string),
                'population': tf.io.FixedLenFeature([], tf.string),
                'erc': tf.io.FixedLenFeature([], tf.string),
                'PrevFireMask': tf.io.FixedLenFeature([], tf.string),
                'FireMask': tf.io.FixedLenFeature([], tf.string),
            }
        
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    def decode_feature(self, feature_tensor, dtype=tf.float32):
        """
        Decode a feature tensor from string format.
        
        Args:
            feature_tensor: Tensor in string format
            dtype: Target data type
            
        Returns:
            Decoded tensor
        """
        return tf.io.parse_tensor(feature_tensor, dtype)
    
    def explore_tfrecord_structure(self, file_path: str, num_examples: int = 1) -> Dict[str, Any]:
        """
        Explore the structure of a TFRecord file.
        
        Args:
            file_path: Path to TFRecord file
            num_examples: Number of examples to examine
            
        Returns:
            Dictionary containing feature information
        """
        dataset = tf.data.TFRecordDataset(file_path)
        
        feature_info = {}
        
        for i, example in enumerate(dataset.take(num_examples)):
            if i == 0:
                # Parse the first example to understand the structure
                parsed = tf.train.Example.FromString(example.numpy())
                
                for feature_name, feature in parsed.features.feature.items():
                    feature_type = None
                    feature_shape = None
                    
                    if feature.HasField('bytes_list'):
                        feature_type = 'bytes'
                        feature_shape = len(feature.bytes_list.value)
                        
                        # Try to decode as tensor to get actual shape
                        try:
                            decoded = tf.io.parse_tensor(feature.bytes_list.value[0], tf.float32)
                            feature_shape = decoded.shape.as_list()
                            feature_type = f'tensor_{decoded.dtype.name}'
                        except:
                            pass
                            
                    elif feature.HasField('float_list'):
                        feature_type = 'float'
                        feature_shape = len(feature.float_list.value)
                        
                    elif feature.HasField('int64_list'):
                        feature_type = 'int64'
                        feature_shape = len(feature.int64_list.value)
                    
                    feature_info[feature_name] = {
                        'type': feature_type,
                        'shape': feature_shape
                    }
        
        return feature_info
    
    def load_sample_data(self, file_path: str, num_examples: int = 5) -> Dict[str, List]:
        """
        Load sample data from TFRecord file.
        
        Args:
            file_path: Path to TFRecord file
            num_examples: Number of examples to load
            
        Returns:
            Dictionary containing sample data
        """
        dataset = tf.data.TFRecordDataset(file_path)
        
        samples = {}
        
        for i, example in enumerate(dataset.take(num_examples)):
            parsed = tf.train.Example.FromString(example.numpy())
            
            for feature_name, feature in parsed.features.feature.items():
                if feature_name not in samples:
                    samples[feature_name] = []
                
                if feature.HasField('bytes_list'):
                    try:
                        # Try to decode as tensor
                        decoded = tf.io.parse_tensor(feature.bytes_list.value[0], tf.float32)
                        samples[feature_name].append(decoded.numpy())
                    except:
                        samples[feature_name].append(feature.bytes_list.value[0])
                        
                elif feature.HasField('float_list'):
                    samples[feature_name].append(list(feature.float_list.value))
                    
                elif feature.HasField('int64_list'):
                    samples[feature_name].append(list(feature.int64_list.value))
        
        return samples
    
    def create_feature_summary(self, data_dir: str = None) -> pd.DataFrame:
        """
        Create a summary of all features across multiple TFRecord files.
        
        Args:
            data_dir: Directory containing TFRecord files (uses self.data_dir if None)
            
        Returns:
            DataFrame with feature summary
        """
        if data_dir is None:
            data_dir = self.data_dir
            
        tfrecord_files = self.get_tfrecord_files()
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {data_dir}")
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Explore the first file to get feature structure
        feature_info = self.explore_tfrecord_structure(tfrecord_files[0])
        
        # Create summary DataFrame
        summary_data = []
        for feature_name, info in feature_info.items():
            summary_data.append({
                'feature_name': feature_name,
                'data_type': info['type'],
                'shape': str(info['shape']),
                'description': self._get_feature_description(feature_name)
            })
        
        return pd.DataFrame(summary_data)
    
    def _get_feature_description(self, feature_name: str) -> str:
        """
        Get description for known features.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature description
        """
        descriptions = {
            'elevation': 'Elevation above sea level (meters)',
            'th': 'Wind direction (degrees)',
            'vs': 'Wind speed (m/s)',
            'tmmn': 'Minimum temperature (K)',
            'tmmx': 'Maximum temperature (K)',
            'sph': 'Specific humidity (kg/kg)',
            'pr': 'Precipitation (mm)',
            'pdsi': 'Palmer Drought Severity Index',
            'NDVI': 'Normalized Difference Vegetation Index',
            'population': 'Population density',
            'erc': 'Energy Release Component',
            'PrevFireMask': 'Previous fire occurrence mask',
            'FireMask': 'Current fire occurrence mask (target)',
        }
        
        return descriptions.get(feature_name, 'Unknown feature')


def main():
    """Main function to demonstrate TFRecord parsing."""
    # Initialize parser with the dataset directory
    data_dir = "ndws_western_dataset"
    parser = TFRecordParser(data_dir)
    
    try:
        # Get list of TFRecord files
        tfrecord_files = parser.get_tfrecord_files()
        print(f"Found {len(tfrecord_files)} TFRecord files:")
        for file in tfrecord_files[:5]:  # Show first 5 files
            print(f"  - {os.path.basename(file)}")
        
        if tfrecord_files:
            # Explore structure of the first file
            print(f"\nExploring structure of: {os.path.basename(tfrecord_files[0])}")
            feature_info = parser.explore_tfrecord_structure(tfrecord_files[0])
            
            print("\nFeature Information:")
            for name, info in feature_info.items():
                print(f"  {name}: {info['type']} - Shape: {info['shape']}")
            
            # Create feature summary
            print("\nCreating feature summary...")
            summary_df = parser.create_feature_summary()
            print(summary_df.to_string(index=False))
            
            # Load sample data
            print(f"\nLoading sample data from: {os.path.basename(tfrecord_files[0])}")
            sample_data = parser.load_sample_data(tfrecord_files[0], num_examples=2)
            
            print(f"\nSample data loaded for {len(sample_data)} features")
            for feature_name, data in sample_data.items():
                if isinstance(data[0], np.ndarray):
                    print(f"  {feature_name}: Array shape {data[0].shape}, dtype {data[0].dtype}")
                else:
                    print(f"  {feature_name}: {type(data[0])}")
                    
        else:
            print("No TFRecord files found in the specified directory.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the TFRecord files are in the correct directory.")


if __name__ == "__main__":
    main()