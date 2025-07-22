"""
Configuration management
"""

import yaml
import os
import math
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration
    """
    # GPU device override
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        if os.environ['CUDA_VISIBLE_DEVICES'] == '':
            config['feature_extraction']['device'] = 'cpu'
        else:
            config['feature_extraction']['device'] = 'cuda'
    
    # Redis configuration
    if 'REDIS_URL' in os.environ:
        config['cache']['l2_host'] = os.environ['REDIS_URL']
    
    # Batch size optimization based on available memory
    if 'BATCH_SIZE' in os.environ:
        try:
            batch_size = int(os.environ['BATCH_SIZE'])
            config['feature_extraction']['batch_size'] = batch_size
            config['batch_processing']['batch_size'] = batch_size
        except ValueError:
            pass
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values
    """
    # Check required sections
    required_sections = ['frame_extraction', 'feature_extraction', 'index', 'cache']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate numeric ranges
    if config['frame_extraction']['sample_rate'] <= 0:
        raise ValueError("sample_rate must be positive")
    
    if config['feature_extraction']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    # Fix level_generation_factor if missing
    if 'level_generation_factor' not in config['index']['hnsw']:
        config['index']['hnsw']['level_generation_factor'] = 1.0 / math.log(2.0)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    """
    return {
        'system': {
            'name': 'video-search',
            'version': '1.0.0'
        },
        'frame_extraction': {
            'sample_rate': 1.0,
            'max_frames_per_video': 3600,
            'frame_size': [224, 224],
            'scene_detection_threshold': 30.0
        },
        'feature_extraction': {
            'model_name': 'openai/clip-vit-base-patch32',
            'batch_size': 32,
            'device': 'auto',
            'dimension': 512
        },
        'index': {
            'type': 'hnsw',
            'dimension': 512,
            'hnsw': {
                'M': 16,
                'ef_construction': 200,
                'ef_search': 50,
                'max_M': 16,
                'level_generation_factor': 1.0 / math.log(2.0)
            }
        },
        'cache': {
            'l1_capacity': 2000,
            'l2_host': 'redis://localhost:6379',
            'ttl_seconds': 300,
            'enable_cache': True
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'max_upload_size': 5368709120,
            'request_timeout': 30
        },
        'batch_processing': {
            'batch_size': 32,
            'timeout_ms': 10,
            'max_concurrent_batches': 4
        },
        'performance': {
            'use_memory_mapping': True,
            'enable_gpu_acceleration': True,
            'num_threads': 4
        }
    }
