"""
Model Factory - Creates models based on configuration
"""
import importlib
from typing import Dict, Any
import logging
from sklearn.model_selection import GridSearchCV

# Model Factory Class
class ModelFactory:
    def __init__(self, config_manager):
        self.config = config_manager.get_models_config()
    
    def create_model(self, model_name):
        """Create model instance from configuration"""
        model_config = self.config.get(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        # Get class path
        class_path = model_config.get('class')
        if not class_path:
            raise ValueError(f"No class specified for model '{model_name}'")
        
        # Dynamically import and create model
        try:
            # module_name, class_name = class_path.rsplit('.', 1)
            # module = importlib.import_module(module_name)
            # model_class = getattr(module, class_name)
            model_class = self._load_class(class_path)
            
            # Get parameters
            params = model_config.get('parameters', {})
            
            # Create model instance
            model = model_class(**params)
            return model
            
        except (ImportError, AttributeError) as e:
            print(f"Error creating model {model_name}: {e}")
            # Fallback for common models
            return self._create_fallback_model(model_name, params)
    
    def _create_fallback_model(self, model_name, params):
        """Fallback method for common models"""
        fallback_map = {
            'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
            'DecisionTree': 'sklearn.tree.DecisionTreeClassifier',
            'KNN': 'sklearn.neighbors.KNeighborsClassifier',
            'GaussianNB': 'sklearn.naive_bayes.GaussianNB',
            'RandomForest': 'sklearn.ensemble.RandomForestClassifier',
            'XGBoost': 'xgboost.XGBClassifier'
        }
        
        if model_name in fallback_map:
            class_path = fallback_map[model_name]
            module_name, class_name = class_path.rsplit('.', 1)
            
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                return model_class(**params)
            except:
                pass
        
        raise ValueError(f"Cannot create model {model_name}")
    
    def build_scaler(self):
        scaling_cfg = self.config.dataset_config["dataset"]["preprocessing"].get(
            "scaling", {}
        )

        if not scaling_cfg.get("enabled", False):
            return None

        cls = self._load_class(scaling_cfg["class"])

        self.logger.info(f"Building scaler: {scaling_cfg['class']}")
        return cls()

    def _load_class(self, class_path: str):
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)