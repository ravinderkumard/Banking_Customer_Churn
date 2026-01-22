
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Environment setup complete!")

# Create config directory if it doesn't exist
os.makedirs('config', exist_ok=True)

from src.config_manager import ConfigManager

# Recreate instance with updated manager
config_manager = ConfigManager()

from src.data_loader import DataLoader

# Load data
data_loader = DataLoader(config_manager)
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_and_preprocess()

# Model Factory Class
from src.model_factory import ModelFactory

# Create model factory
model_factory = ModelFactory(config_manager)

# Evaluation Engine Class

from src.evaluation_engine import EvaluationEngine

# Create evaluation engine
eval_engine = EvaluationEngine(config_manager)

# Main Execution: Implement All Models
logger.info("\n IMPLEMENTING ALL MODELS FROM CONFIG")
print("="*70)

all_metrics = []

# Get models from config
models_config = config_manager.get_models_config()

for model_name, model_config in models_config.items():
    logger.info(f"\n Processing: {model_name}")
    
    try:
        # Create model
        model = model_factory.create_model(model_name)
        
        # Check if model needs scaled data
        use_scaled = model_config.get('use_scaled', False)
        
        # Select appropriate data
        if use_scaled:
            X_train_data = X_train_scaled
            X_test_data = X_test_scaled
            data_type = "scaled"
        else:
            X_train_data = X_train
            X_test_data = X_test
            data_type = "unscaled"
        
        logger.info(f"   Using {data_type} data")
        
        # Evaluate model
        metrics = eval_engine.evaluate_model(
            model, model_name, 
            X_train_data, X_test_data, 
            y_train, y_test, 
            use_scaled=use_scaled
        )
        
        if metrics:
            all_metrics.append(metrics)
            logger.info(f"   {model_name} completed successfully")
        else:
            logger.error(f"   {model_name} evaluation failed")
            
    except Exception as e:
        logger.error(f"   Error with {model_name}: {e}")
        continue

logger.info(f"\n COMPLETED: {len(all_metrics)}/{len(models_config)} models")