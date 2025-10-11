"""
EDA and Baseline Utilities for Wildfire Prediction Project

This module contains utility functions for exploratory data analysis
and baseline model evaluation used in the wildfire prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy import stats


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score for binary classification).
    
    Args:
        y_true: True binary mask
        y_pred: Predicted binary mask  
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    
    # Flatten arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    # Calculate Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    
    return dice


def jaccard_index(y_true, y_pred, smooth=1e-6):
    """Calculate Jaccard Index (IoU)."""
    
    y_true_f = y_true.flatten() > 0.5
    y_pred_f = y_pred.flatten() > 0.5
    
    intersection = np.sum(y_true_f & y_pred_f)
    union = np.sum(y_true_f | y_pred_f)
    
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall, and F1 score."""
    
    y_true_f = y_true.flatten() > 0.5
    y_pred_f = y_pred.flatten() > 0.5
    
    tp = np.sum(y_true_f & y_pred_f)  # True Positive
    fp = np.sum(~y_true_f & y_pred_f)  # False Positive  
    fn = np.sum(y_true_f & ~y_pred_f)  # False Negative
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


class PersistenceModel:
    """
    Simple persistence baseline model for wildfire prediction.
    
    This model assumes that the fire mask at time t+1 will be 
    identical to the fire mask at time t.
    """
    
    def __init__(self):
        self.name = "Persistence Model"
        self.description = "Predicts fire_mask(t+1) = fire_mask(t)"
    
    def predict(self, prev_fire_mask):
        """
        Make prediction using persistence assumption.
        
        Args:
            prev_fire_mask: Fire mask at time t
            
        Returns:
            Predicted fire mask at time t+1 (same as input)
        """
        return prev_fire_mask.copy()
    
    def evaluate(self, samples):
        """
        Evaluate the persistence model on a set of samples.
        
        Args:
            samples: List of sample dictionaries containing PrevFireMask and FireMask
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        metrics = {
            'dice_scores': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'jaccard_indices': [],
            'sample_stats': []
        }
        
        for i, sample in enumerate(samples):
            if 'PrevFireMask' not in sample or 'FireMask' not in sample:
                continue
            
            # Get fire masks and handle different shapes
            prev_fire = self._prepare_mask(sample['PrevFireMask'])
            true_fire = self._prepare_mask(sample['FireMask'])
            
            # Make prediction
            pred_fire = self.predict(prev_fire)
            
            # Calculate metrics
            dice = dice_coefficient(true_fire, pred_fire)
            precision, recall, f1 = precision_recall_f1(true_fire, pred_fire)
            jaccard = jaccard_index(true_fire, pred_fire)
            
            metrics['dice_scores'].append(dice)
            metrics['precisions'].append(precision)
            metrics['recalls'].append(recall)
            metrics['f1_scores'].append(f1)
            metrics['jaccard_indices'].append(jaccard)
            
            # Sample statistics
            sample_stat = {
                'sample_id': i + 1,
                'prev_fire_pixels': np.sum(prev_fire > 0.5),
                'true_fire_pixels': np.sum(true_fire > 0.5),
                'total_pixels': prev_fire.size,
                'dice': dice,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'jaccard': jaccard
            }
            
            metrics['sample_stats'].append(sample_stat)
        
        # Calculate summary statistics
        if metrics['dice_scores']:
            metrics['summary'] = {
                'mean_dice': np.mean(metrics['dice_scores']),
                'std_dice': np.std(metrics['dice_scores']),
                'min_dice': np.min(metrics['dice_scores']),
                'max_dice': np.max(metrics['dice_scores']),
                'mean_precision': np.mean(metrics['precisions']),
                'mean_recall': np.mean(metrics['recalls']),
                'mean_f1': np.mean(metrics['f1_scores']),
                'mean_jaccard': np.mean(metrics['jaccard_indices']),
                'n_samples': len(metrics['dice_scores'])
            }
        
        return metrics
    
    def _prepare_mask(self, mask):
        """Prepare mask by handling different shapes."""
        if len(mask.shape) == 3:
            return mask[:, :, 0] if mask.shape[2] == 1 else mask.mean(axis=2)
        return mask


def get_feature_descriptions():
    """Get standardized feature descriptions for the wildfire dataset."""
    
    return {
        'elevation': {
            'description': 'Elevation above sea level',
            'unit': 'meters',
            'expected_range': (0, 4000),
            'type': 'continuous'
        },
        'th': {
            'description': 'Wind direction from north',
            'unit': 'degrees',
            'expected_range': (0, 360),
            'type': 'continuous'
        },
        'vs': {
            'description': 'Wind speed',
            'unit': 'm/s',
            'expected_range': (0, 30),
            'type': 'continuous'
        },
        'tmmn': {
            'description': 'Minimum temperature',
            'unit': 'Kelvin',
            'expected_range': (240, 320),
            'type': 'continuous'
        },
        'tmmx': {
            'description': 'Maximum temperature',
            'unit': 'Kelvin',
            'expected_range': (250, 330),
            'type': 'continuous'
        },
        'sph': {
            'description': 'Specific humidity',
            'unit': 'kg/kg',
            'expected_range': (0, 0.02),
            'type': 'continuous'
        },
        'pr': {
            'description': 'Precipitation',
            'unit': 'mm',
            'expected_range': (0, 100),
            'type': 'continuous'
        },
        'pdsi': {
            'description': 'Palmer Drought Severity Index',
            'unit': 'index',
            'expected_range': (-10, 10),
            'type': 'continuous'
        },
        'NDVI': {
            'description': 'Normalized Difference Vegetation Index',
            'unit': 'index',
            'expected_range': (-1, 1),
            'type': 'continuous'
        },
        'population': {
            'description': 'Population density',
            'unit': 'people/km²',
            'expected_range': (0, 1000),
            'type': 'continuous'
        },
        'erc': {
            'description': 'Energy Release Component',
            'unit': 'index',
            'expected_range': (0, 150),
            'type': 'continuous'
        },
        'PrevFireMask': {
            'description': 'Previous fire occurrence (fire_mask_t)',
            'unit': 'binary',
            'expected_range': (0, 1),
            'type': 'binary'
        },
        'FireMask': {
            'description': 'Current fire occurrence (fire_mask_t+1) - TARGET',
            'unit': 'binary',
            'expected_range': (0, 1),
            'type': 'binary'
        }
    }


def analyze_fire_patterns(samples):
    """
    Analyze fire patterns across samples.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Dictionary containing fire pattern statistics
    """
    
    patterns = {
        'fire_persistence': [],    # Fire continues from t to t+1
        'fire_extinction': [],     # Fire at t but not at t+1
        'fire_ignition': [],       # No fire at t but fire at t+1
        'no_fire': [],             # No fire at either t or t+1
        'total_fire_t': [],        # Total fire pixels at time t
        'total_fire_t1': [],       # Total fire pixels at time t+1
    }
    
    for sample in samples:
        if 'PrevFireMask' not in sample or 'FireMask' not in sample:
            continue
            
        prev_fire = sample['PrevFireMask']
        curr_fire = sample['FireMask']
        
        # Handle shapes
        if len(prev_fire.shape) == 3:
            prev_fire = prev_fire[:, :, 0] if prev_fire.shape[2] == 1 else prev_fire.mean(axis=2)
        if len(curr_fire.shape) == 3:
            curr_fire = curr_fire[:, :, 0] if curr_fire.shape[2] == 1 else curr_fire.mean(axis=2)
        
        # Binary masks
        prev_binary = prev_fire > 0.5
        curr_binary = curr_fire > 0.5
        
        # Calculate patterns
        persistence = np.sum(prev_binary & curr_binary)
        extinction = np.sum(prev_binary & ~curr_binary)
        ignition = np.sum(~prev_binary & curr_binary)
        no_fire = np.sum(~prev_binary & ~curr_binary)
        
        patterns['fire_persistence'].append(persistence)
        patterns['fire_extinction'].append(extinction)
        patterns['fire_ignition'].append(ignition)
        patterns['no_fire'].append(no_fire)
        patterns['total_fire_t'].append(np.sum(prev_binary))
        patterns['total_fire_t1'].append(np.sum(curr_binary))
    
    # Calculate summary statistics
    summary = {}
    for key, values in patterns.items():
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return patterns, summary