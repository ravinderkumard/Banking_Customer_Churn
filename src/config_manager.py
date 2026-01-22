# src/config_manager.py

import os
import yaml
import logging
from typing import Dict


class ConfigManager:
    def __init__(self, config_dir="config"):
        import yaml
        import ast
        
        # Load configs
        with open(os.path.join(config_dir, 'dataset_config.yaml'), 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        with open(os.path.join(config_dir, 'model_config.yaml'), 'r') as f:
            self.models_config = yaml.safe_load(f)
        
        # Convert string parameters to appropriate types
        self._convert_parameters()
    
    def _convert_parameters(self):
        """Convert string parameters to appropriate Python types"""
        for model_name, models_config in self.models_config.get("models", {}).items():
            params = models_config.get("parameters", {})
            converted_params = {}
            
            for key, value in params.items():
                # Try to convert scientific notation strings to float
                if isinstance(value, str) and 'e' in value.lower() and '-' in value:
                    try:
                        # Check if it's scientific notation
                        if 'e-' in value.lower():
                            converted_value = float(value)
                            converted_params[key] = converted_value
                            continue
                    except:
                        pass
                
                # Keep original value
                converted_params[key] = value
            
            # Update parameters
            models_config['parameters'] = converted_params
    
    def get_dataset_config(self):
        return self.dataset_config.get("dataset", {})
    
    def get_models_config(self, model_name=None):
        if model_name:
            return self.models_config.get("models", {}).get(model_name, {})
        return self.models_config.get("models", {})
    def get_evaluation_config(self):
        """Get evaluation configuration"""
        return self.models_config.get('evaluation', {})
        
    def get_all_models(self):
        return list(self.get_models_config().keys())
    
    def should_use_scaled(self, model_name):
        models_config = self.get_models_config(model_name)
        return models_config.get("use_scaled", False)
    
    def get_model_parameters(self, model_name):
        models_config = self.get_model_config(model_name)
        return models_config.get("parameters", {})
    
    def should_optimize_hyperparams(self, model_name):
        models_config = self.get_models_config(model_name)
        return models_config.get("optimize_hyperparams", False)
    
    def get_param_grid(self, model_name):
        models_config = self.get_models_config(model_name)
        return models_config.get("param_grid", {})

    def get_output_config(self):
        """Get output configuration"""
        return self.models_config.get('output', {})