import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import joblib
import json
from datetime import datetime

def evaluate_model(model,model_name,X_train,X_test,y_train,y_test,use_scaled=False,save_model=True):
    
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    print(f"Training {model_name} ...")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else None

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test,y_pred),
        'precision': precision_score(y_test,y_pred,zero_division=0),
        'recall':recall_score(y_test,y_pred,zero_division=0),
        'f1_score': f1_score(y_test,y_pred,zero_division=0),
        'mcc': matthews_corrcoef(y_test,y_pred)
    }

    if y_pred_proba is not None:
        metrics['auc_score'] = roc_auc_score(y_test,y_pred_proba)
    else:
        metrics['auc_score'] = None
        print("Warning: Model doesnot support predict_proba(), AUC score set to None")

    print(f"{model_name} Performance Metrics:")
    print("-"*40)
    for metric, value in metrics.items():
        if metric!='model_name':
            if value is not None:
                print(f"{metric.replace('_',' ').title():15}:{value:.4f}")
            else:
                print(f"{metric.replace('_',' ').title():15}: N/A")
    
    print(f"\nClassification Report:")
    print("-"*40)
    print(classification_report(y_test,y_pred,target_names=['Retained','Churned']))

    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm,model_name)

    if y_pred_proba is not None:
        plot_roc_curve(y_test,y_pred_proba,model_name)
    
    if save_model:
        save_model_file(model,model_name,use_scaled)

    save_metrics(metrics,model_name)

    return metrics

def plot_confusion_matrix(cm,model_name):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=['Predicted Retained','Predicted Churned'],
                yticklabels=['Actual Retained','Actual Churned'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'data/plots/{model_name.lower().replace(" ","_")}_confusion_matrix.png',dpi=100,bbox_inches='tight')

    tn,fp,fn,tp = cm.ravel()

    print(f"\nConfusion Matrix Details:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")

def plot_roc_curve(y_test,y_pred_proba,model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=1)
    plt.fill_between(fpr, tpr, alpha=0.1, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../data/plots/{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=100, bbox_inches='tight')
    plt.show()

def save_model_file(model, model_name, use_scaled):
    
    import os
    os.makedirs('models', exist_ok=True)
    
    # Create filename
    filename = model_name.lower().replace(' ', '_')
    if use_scaled:
        filename += '_scaled'
    filename += '.pkl'
    
    filepath = f'models/{filename}'
    joblib.dump(model, filepath)
    print(f"\n✓ Model saved to: {filepath}")

def save_metrics(metrics, model_name):
    
    import os
    os.makedirs('data/metrics', exist_ok=True)
    
    # Add timestamp
    metrics_with_meta = metrics.copy()
    metrics_with_meta['evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    filename = f'data/metrics/{model_name.lower().replace(" ", "_")}_metrics.json'
    with open(filename, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    
    print(f"✓ Metrics saved to: {filename}")

def compare_models(all_metrics):
    
    
    print("MODEL COMPARISON TABLE")
    print(f"{'='*70}")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns
    columns_order = ['model_name', 'accuracy', 'auc_score', 'precision', 
                     'recall', 'f1_score', 'mcc']
    df = df[columns_order]
    
    # Format for display
    display_df = df.copy()
    for col in ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    print(display_df.to_string(index=False))
    
    # Find best model for each metric
    print(f"\n{'='*70}")
    print("BEST PERFORMING MODELS")
    print(f"{'='*70}")
    
    numeric_metrics = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc']
    for metric in numeric_metrics:
        if metric in df.columns:
            # Filter out None values
            valid_metrics = df[df[metric].notnull()]
            if len(valid_metrics) > 0:
                best_idx = valid_metrics[metric].idxmax()
                best_model = df.loc[best_idx, 'model_name']
                best_value = df.loc[best_idx, metric]
                print(f"{metric.replace('_', ' ').title():15}: {best_model} ({best_value:.4f})")
    
    return df