import os
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Evaluation Engine Class
class EvaluationEngine:
    def __init__(self, config_manager):
        self.eval_config = config_manager.get_evaluation_config()
        self.output_config = config_manager.get_output_config()
        
        # Create output directory
        output_dir = self.output_config.get('output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        self.output_dir = output_dir
    
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test, use_scaled=False):
        """Evaluate a single model with all metrics"""
        
        print(f"\n{'='*70}")
        logger.info(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Train model
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate all metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
            
            # Display results
            self._display_results(metrics, y_test, y_pred)
            
            # Create plots
            if self.output_config.get('save_plots', True):
                self._create_plots(model, model_name, y_test, y_pred, y_pred_proba, X_train.columns,self.output_config.get('show_plots', True))
            
            # Save model
            if self.output_config.get('save_models', True):
                self._save_model(model, model_name)
            
            if self.output_config.get('save_metrics', True):
                self.save_metrics(metrics,model_name)

            return metrics
            
        except Exception as e:
            print(f" Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_metrics(self, y_test, y_pred, y_pred_proba, model_name):
        """Calculate all required metrics"""

        from sklearn.metrics import (
            accuracy_score, roc_auc_score, precision_score, recall_score,
            f1_score, matthews_corrcoef
        )
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        # AUC Score
        if y_pred_proba is not None:
            metrics['AUC'] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics['AUC'] = None
        
        return metrics
    
    def save_metrics(self, metrics, model_name):
    
        import os
        os.makedirs('outputs/metrics', exist_ok=True)
        
        # Add timestamp
        metrics_with_meta = metrics.copy()
        metrics_with_meta['evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        filename = f'outputs/metrics/{model_name.lower().replace(" ", "_")}_metrics.json'
        with open(filename, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        print(f"✓ Metrics saved to: {filename}")

    def _display_results(self, metrics, y_test, y_pred):
        """Display evaluation results"""
        print(f"\n PERFORMANCE METRICS:")
        print("-" * 50)
        
        for key, value in metrics.items():
            if key != 'Model' and value is not None:
                print(f"{key:12}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n CONFUSION MATRIX:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Classification Report
        print(f"\n CLASSIFICATION REPORT:")
        # print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    
    def _create_plots(self, model, model_name, y_test, y_pred, y_pred_proba, feature_names, show_plots):
        """Create visualization plots"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.3f}')
            axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('ROC Curve')
        
        # 3. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            
            axes[2].barh(range(len(indices)), importances[indices])
            axes[2].set_yticks(range(len(indices)))
            axes[2].set_yticklabels([feature_names[i] for i in indices])
            axes[2].set_xlabel('Importance')
            axes[2].set_title('Top 10 Features')
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[-10:]  # Top 10
            
            colors = ['red' if c < 0 else 'blue' for c in coef[indices]]
            axes[2].barh(range(len(indices)), coef[indices], color=colors)
            axes[2].set_yticks(range(len(indices)))
            axes[2].set_yticklabels([feature_names[i] for i in indices])
            axes[2].set_xlabel('Coefficient')
            axes[2].set_title('Top 10 Coefficients')
        else:
            axes[2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                        ha='center', va='center', fontsize=12)
            axes[2].set_title('Feature Analysis')
        
        plt.suptitle(f'{model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'plots', f'{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        if show_plots:
            plt.show()
        print(f" Plot saved: {plot_path}")
    
    def _save_model(self, model, model_name):
        """Save trained model"""
        model_path = os.path.join(self.output_dir, 'models', f'{model_name.lower().replace(" ", "_")}.pkl')
        joblib.dump(model, model_path)
        print(f" Model saved: {model_path}")
