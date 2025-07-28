import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_display_metrics(model_type="synthetic"):
    """Load and display evaluation metrics"""
    
    if model_type == "synthetic":
        metrics_file = "models/evaluation_metrics.json"
        model_name = "Synthetic Data Model"
    else:
        metrics_file = "models/comprehensive_metrics.json"
        model_name = "Real Data Model"
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {model_name}")
        print(f"{'='*60}")
        
        if model_type == "synthetic":
            # Display synthetic model metrics
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            print(f"AUC Score: {metrics['auc_score']:.4f}")
            
            print(f"\nConfusion Matrix:")
            cm = np.array(metrics['confusion_matrix'])
            print(f"                 Predicted")
            print(f"                Normal  Anomalous")
            print(f"Actual Normal    {cm[0,0]:6d}    {cm[0,1]:6d}")
            print(f"Actual Anomalous {cm[1,0]:6d}    {cm[1,1]:6d}")
            
        else:
            # Display real data model metrics
            perf = metrics['model_performance']
            print(f"Accuracy:    {perf['accuracy']:.4f}")
            print(f"Precision:   {perf['precision']:.4f}")
            print(f"Recall:      {perf['recall']:.4f}")
            print(f"Sensitivity: {perf['sensitivity']:.4f}")
            print(f"Specificity: {perf['specificity']:.4f}")
            print(f"F1 Score:    {perf['f1_score']:.4f}")
            print(f"AUC Score:   {perf['auc_score']:.4f}")
            print(f"Avg Precision: {perf['average_precision']:.4f}")
            
            print(f"\nDataset Information:")
            dataset = metrics['dataset_info']
            print(f"Total Samples:     {dataset['total_samples']:,}")
            print(f"Training Samples:  {dataset['training_samples']:,}")
            print(f"Test Samples:      {dataset['test_samples']:,}")
            print(f"Normal Cases:      {dataset['normal_cases']:,}")
            print(f"Abnormal Cases:    {dataset['abnormal_cases']:,}")
            print(f"Abnormality Rate:  {dataset['abnormality_rate']:.2%}")
            
            print(f"\nTraining Information:")
            training = metrics['training_history']
            print(f"Epochs Trained:       {training['epochs']}")
            print(f"Final Train Accuracy: {training['final_train_accuracy']:.4f}")
            print(f"Final Val Accuracy:   {training['final_val_accuracy']:.4f}")
            
            print(f"\nConfusion Matrix:")
            cm_info = metrics['confusion_matrix']
            print(f"                 Predicted")
            print(f"                Normal  Abnormal")
            print(f"Actual Normal    {cm_info['true_negatives']:6d}    {cm_info['false_positives']:6d}")
            print(f"Actual Abnormal  {cm_info['false_negatives']:6d}    {cm_info['true_positives']:6d}")
        
        # Calculate additional insights
        if model_type == "synthetic":
            cm = np.array(metrics['confusion_matrix'])
        else:
            cm_info = metrics['confusion_matrix']
            cm = np.array([[cm_info['true_negatives'], cm_info['false_positives']], 
                          [cm_info['false_negatives'], cm_info['true_positives']]])
        
        # Clinical interpretation
        print(f"\n{'='*60}")
        print("CLINICAL INTERPRETATION")
        print(f"{'='*60}")
        
        tn, fp, fn, tp = cm.ravel()
        
        print(f"True Positives (Correctly identified anomalies):  {tp}")
        print(f"True Negatives (Correctly identified normal):     {tn}")
        print(f"False Positives (False alarms):                   {fp}")
        print(f"False Negatives (Missed anomalies):               {fn}")
        
        print(f"\nClinical Significance:")
        if fn > 0:
            print(f"⚠️  {fn} anomalies were missed (False Negatives)")
            print("   → Could lead to delayed treatment")
        
        if fp > 0:
            print(f"⚠️  {fp} false alarms (False Positives)")
            print("   → Could lead to unnecessary interventions")
        
        # Model recommendations
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nModel Recommendations:")
        if precision < 0.8:
            print("📊 Consider improving precision to reduce false alarms")
        if recall < 0.8:
            print("📊 Consider improving recall to catch more anomalies")
        if precision > 0.9 and recall > 0.9:
            print("✅ Excellent performance for clinical screening")
        
        print(f"\n{'='*60}")
        
    except FileNotFoundError:
        print(f"❌ Metrics file not found: {metrics_file}")
        print("Please run the training script first.")
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")

def compare_models():
    """Compare synthetic and real data models if both exist"""
    synthetic_file = Path("models/evaluation_metrics.json")
    real_file = Path("models/comprehensive_metrics.json")
    
    if not synthetic_file.exists() or not real_file.exists():
        print("❌ Both model metrics files needed for comparison")
        return
    
    with open(synthetic_file, 'r') as f:
        synthetic_metrics = json.load(f)
    
    with open(real_file, 'r') as f:
        real_metrics = json.load(f)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Metric':<15} {'Synthetic Data':<15} {'Real Data':<15} {'Difference':<15}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    # Compare key metrics
    synthetic_acc = synthetic_metrics['accuracy']
    real_acc = real_metrics['model_performance']['accuracy']
    print(f"{'Accuracy':<15} {synthetic_acc:<15.4f} {real_acc:<15.4f} {real_acc-synthetic_acc:<+15.4f}")
    
    synthetic_prec = synthetic_metrics['precision']
    real_prec = real_metrics['model_performance']['precision']
    print(f"{'Precision':<15} {synthetic_prec:<15.4f} {real_prec:<15.4f} {real_prec-synthetic_prec:<+15.4f}")
    
    synthetic_rec = synthetic_metrics['recall']
    real_rec = real_metrics['model_performance']['recall']
    print(f"{'Recall':<15} {synthetic_rec:<15.4f} {real_rec:<15.4f} {real_rec-synthetic_rec:<+15.4f}")
    
    synthetic_f1 = synthetic_metrics['f1_score']
    real_f1 = real_metrics['model_performance']['f1_score']
    print(f"{'F1 Score':<15} {synthetic_f1:<15.4f} {real_f1:<15.4f} {real_f1-synthetic_f1:<+15.4f}")
    
    synthetic_auc = synthetic_metrics['auc_score']
    real_auc = real_metrics['model_performance']['auc_score']
    print(f"{'AUC Score':<15} {synthetic_auc:<15.4f} {real_auc:<15.4f} {real_auc-synthetic_auc:<+15.4f}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type in ["synthetic", "real"]:
            load_and_display_metrics(model_type)
        elif model_type == "compare":
            compare_models()
        else:
            print("Usage: python view_evaluation_results.py [synthetic|real|compare]")
    else:
        # Display both if available
        print("Checking for available evaluation results...")
        
        synthetic_exists = Path("models/evaluation_metrics.json").exists()
        real_exists = Path("models/comprehensive_metrics.json").exists()
        
        if synthetic_exists:
            load_and_display_metrics("synthetic")
        
        if real_exists:
            load_and_display_metrics("real")
        
        if synthetic_exists and real_exists:
            compare_models()
        
        if not synthetic_exists and not real_exists:
            print("❌ No evaluation results found. Please run training scripts first.")
