import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

def load_and_display_metrics(model_type="auto"):
    """Load and display evaluation metrics with auto-detection"""
    
    print(f"\n🔍 SEARCHING FOR AVAILABLE MODEL METRICS...")
    print(f"{'='*70}")
    
    # Auto-detect available models
    available_models = []
    
    # Check for different model types
    model_files = {
        'sklearn': 'models/sklearn_comprehensive_metrics.json',
        'tensorflow_176k': 'models/medical_metrics_176k.json', 
        'tensorflow_medical': 'models/medical_metrics_comprehensive.json',
        'tensorflow_synthetic': 'models/evaluation_metrics.json'
    }
    
    for model_name, file_path in model_files.items():
        if Path(file_path).exists():
            available_models.append((model_name, file_path))
            print(f"✅ Found {model_name} model metrics: {file_path}")
    
    if not available_models:
        print("❌ No model metrics found!")
        print("Please train a model first.")
        return
    
    # If specific model type requested, use it; otherwise use the first available
    if model_type == "auto":
        model_name, metrics_file = available_models[0]
        print(f"\n🎯 Auto-selected: {model_name} model")
    else:
        # Find requested model type
        found = False
        for model_name, file_path in available_models:
            if model_type.lower() in model_name.lower():
                metrics_file = file_path
                found = True
                break
        
        if not found:
            print(f"❌ Model type '{model_type}' not found!")
            print(f"Available models: {[m[0] for m in available_models]}")
            return
    
    # Load and display metrics
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"📊 MODEL EVALUATION RESULTS - {model_name.upper()}")
        print(f"{'='*70}")
        
        # Display based on model type
        if 'sklearn' in model_name:
            display_sklearn_metrics(metrics)
        elif 'tensorflow' in model_name or 'medical' in metrics_file:
            display_tensorflow_metrics(metrics)
        else:
            display_generic_metrics(metrics)
            
        # Additional analysis
        print(f"\n{'='*70}")
        print(f"📈 PERFORMANCE ANALYSIS")
        print(f"{'='*70}")
        analyze_model_performance(metrics, model_name)
        
        # Clinical interpretation
        print(f"\n{'='*70}")
        print(f"🩺 CLINICAL INTERPRETATION")
        print(f"{'='*70}")
        provide_clinical_interpretation(metrics, model_name)
        
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")

def display_sklearn_metrics(metrics):
    """Display scikit-learn model metrics"""
    print(f"🤖 Model Type: {metrics.get('model_type', 'Scikit-learn Ensemble')}")
    print(f"📊 Data Source: {metrics.get('data_source', 'Real Medical Data')}")
    print(f"📅 Training Date: {metrics.get('training_date', 'Unknown')}")
    
    if 'individual_models' in metrics:
        print(f"\n🔍 INDIVIDUAL MODEL PERFORMANCE:")
        print(f"{'-'*50}")
        
        for model_name, scores in metrics['individual_models'].items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {scores.get('accuracy', 0):.4f}")
            print(f"  Precision: {scores.get('precision', 0):.4f}")
            print(f"  Recall:    {scores.get('recall', 0):.4f}")
            print(f"  F1 Score:  {scores.get('f1', 0):.4f}")
            print(f"  AUC Score: {scores.get('auc', 0):.4f}")
    
    # Feature importance
    if 'feature_names' in metrics:
        print(f"\n🔍 FEATURE ANALYSIS:")
        print(f"Total Features: {metrics.get('total_features', len(metrics['feature_names']))}")

def display_tensorflow_metrics(metrics):
    """Display TensorFlow/medical model metrics"""
    
    # Model info
    if 'model_info' in metrics:
        model_info = metrics['model_info']
        print(f"🤖 Model: {model_info.get('name', 'Medical LSTM')}")
        print(f"📊 Version: {model_info.get('version', '1.0')}")
        print(f"🏥 Data Source: {model_info.get('trained_on', 'Real Medical Data')}")
        print(f"📅 Training Date: {model_info.get('training_date', 'Unknown')}")
        if 'total_parameters' in model_info:
            print(f"⚙️  Parameters: {model_info['total_parameters']:,}")
    
    # Performance metrics
    if 'performance_metrics' in metrics:
        perf = metrics['performance_metrics']
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"{'-'*40}")
        print(f"Accuracy:    {perf.get('accuracy', 0):.4f} ({perf.get('accuracy', 0)*100:.2f}%)")
        print(f"Sensitivity: {perf.get('sensitivity', perf.get('recall', 0)):.4f} ({perf.get('sensitivity', perf.get('recall', 0))*100:.2f}%)")
        print(f"Specificity: {perf.get('specificity', 0):.4f} ({perf.get('specificity', 0)*100:.2f}%)")
        print(f"Precision:   {perf.get('precision', 0):.4f} ({perf.get('precision', 0)*100:.2f}%)")
        print(f"F1 Score:    {perf.get('f1_score', 0):.4f} ({perf.get('f1_score', 0)*100:.2f}%)")
        print(f"AUC Score:   {perf.get('auc_score', 0):.4f}")
        if 'average_precision' in perf:
            print(f"Avg Precision: {perf['average_precision']:.4f}")
    
    # Dataset info
    if 'dataset_info' in metrics:
        dataset = metrics['dataset_info']
        print(f"\n📊 DATASET INFORMATION:")
        print(f"{'-'*40}")
        print(f"Total Sequences: {dataset.get('total_sequences', 0):,}")
        if 'databases' in dataset:
            print(f"Databases: {', '.join(dataset['databases'])}")
        print(f"Normal Cases: {dataset.get('normal_cases', 0):,}")
        print(f"Abnormal Cases: {dataset.get('abnormal_cases', 0):,}")
        if dataset.get('total_sequences', 0) > 0:
            abnormal_rate = dataset.get('abnormal_cases', 0) / dataset.get('total_sequences', 1)
            print(f"Abnormality Rate: {abnormal_rate:.2%}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm_info = metrics['confusion_matrix']
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"{'-'*40}")
        print(f"                 Predicted")
        print(f"                Normal  Abnormal")
        print(f"Actual Normal    {cm_info.get('true_negatives', 0):6d}    {cm_info.get('false_positives', 0):6d}")
        print(f"Actual Abnormal  {cm_info.get('false_negatives', 0):6d}    {cm_info.get('true_positives', 0):6d}")

def display_generic_metrics(metrics):
    """Display generic metrics for unknown model types"""
    print(f"📊 Generic Model Metrics:")
    
    # Try to find common metric names
    common_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
    
    for metric in common_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")

def analyze_model_performance(metrics, model_name):
    """Analyze and provide insights on model performance"""
    
    # Extract key metrics
    if 'performance_metrics' in metrics:
        perf = metrics['performance_metrics']
        accuracy = perf.get('accuracy', 0)
        sensitivity = perf.get('sensitivity', perf.get('recall', 0))
        specificity = perf.get('specificity', 0)
        precision = perf.get('precision', 0)
        f1 = perf.get('f1_score', 0)
        auc = perf.get('auc_score', 0)
    elif 'individual_models' in metrics:
        # For sklearn ensemble, use best performing model
        best_model = max(metrics['individual_models'].items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        perf = best_model[1]
        accuracy = perf.get('accuracy', 0)
        sensitivity = perf.get('recall', 0)
        specificity = 0  # Not available in sklearn metrics
        precision = perf.get('precision', 0)
        f1 = perf.get('f1', 0)
        auc = perf.get('auc', 0)
    else:
        print("⚠️  Unable to analyze performance - metrics format not recognized")
        return
    
    print(f"🎯 PERFORMANCE GRADE:")
    
    # Overall grade
    if accuracy >= 0.99:
        grade = "A+ (EXCEPTIONAL)"
        color = "🟢"
    elif accuracy >= 0.95:
        grade = "A (EXCELLENT)"
        color = "🟢"
    elif accuracy >= 0.90:
        grade = "B (GOOD)"
        color = "🟡"
    elif accuracy >= 0.85:
        grade = "C (ACCEPTABLE)"
        color = "🟠"
    else:
        grade = "D (NEEDS IMPROVEMENT)"
        color = "🔴"
    
    print(f"{color} Overall Grade: {grade}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    # Specific strengths and weaknesses
    print(f"\n💪 STRENGTHS:")
    if accuracy >= 0.99:
        print(f"   🏆 Exceptional accuracy - world-class performance")
    if sensitivity >= 0.95:
        print(f"   🎯 Excellent sensitivity - catches most anomalies")
    if specificity >= 0.90:
        print(f"   ✅ High specificity - low false alarm rate")
    if f1 >= 0.95:
        print(f"   ⚖️  Excellent balance between precision and recall")
    if auc >= 0.95:
        print(f"   📊 Outstanding discrimination ability")
    
    print(f"\n⚠️  AREAS FOR ATTENTION:")
    if sensitivity < 0.95:
        print(f"   🚨 Sensitivity could be improved ({sensitivity:.1%}) - may miss some anomalies")
    if specificity < 0.85 and specificity > 0:
        print(f"   📢 Specificity could be improved ({specificity:.1%}) - may cause false alarms")
    if precision < 0.90:
        print(f"   🎯 Precision could be improved ({precision:.1%}) - some predictions may be incorrect")

def provide_clinical_interpretation(metrics, model_name):
    """Provide clinical interpretation of the results"""
    
    # Extract confusion matrix if available
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        tp = cm.get('true_positives', 0)
        tn = cm.get('true_negatives', 0)
        fp = cm.get('false_positives', 0)
        fn = cm.get('false_negatives', 0)
        
        print(f"🏥 CLINICAL IMPACT ANALYSIS:")
        print(f"{'-'*50}")
        
        if fn > 0:
            print(f"🚨 CRITICAL: {fn} anomalies were missed")
            print(f"   → Could lead to delayed diagnosis and treatment")
            print(f"   → Review these cases for pattern analysis")
        else:
            print(f"✅ EXCELLENT: No missed anomalies in test set")
        
        if fp > 0:
            print(f"⚠️  {fp} false alarms generated")
            print(f"   → May cause unnecessary anxiety and procedures")
            print(f"   → Consider adjusting decision threshold")
        else:
            print(f"✅ PERFECT: No false alarms in test set")
        
        print(f"\n🎯 CLINICAL RECOMMENDATIONS:")
        
        # Performance-based recommendations
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            sensitivity = perf.get('sensitivity', perf.get('recall', 0))
            specificity = perf.get('specificity', 0)
            
            if sensitivity >= 0.99 and specificity >= 0.95:
                print(f"✅ READY FOR CLINICAL DEPLOYMENT")
                print(f"   → Exceeds medical device standards")
                print(f"   → Suitable for primary screening")
                print(f"   → Consider regulatory approval process")
            elif sensitivity >= 0.95 and specificity >= 0.90:
                print(f"✅ SUITABLE FOR CLINICAL SCREENING")
                print(f"   → Good for secondary screening")
                print(f"   → Requires medical supervision")
                print(f"   → Consider additional validation")
            else:
                print(f"⚠️  REQUIRES IMPROVEMENT FOR CLINICAL USE")
                print(f"   → Suitable for research and development")
                print(f"   → Need higher sensitivity/specificity")
                print(f"   → Consider more training data")
    
    print(f"\n📋 REGULATORY CONSIDERATIONS:")
    print(f"   🔬 Research Use: Currently approved")
    print(f"   🏥 Clinical Use: Requires medical validation")
    print(f"   📜 FDA Approval: Would require clinical trials")
    print(f"   🌍 International: Check local regulations")
    
    print(f"\n⚖️  MEDICAL DISCLAIMER:")
    print(f"   This AI model is for research and educational purposes.")
    print(f"   Always consult qualified healthcare professionals.")
    print(f"   Not intended for emergency diagnosis or life-critical decisions.")

def compare_all_available_models():
    """Compare all available models"""
    print(f"\n🔄 COMPARING ALL AVAILABLE MODELS")
    print(f"{'='*70}")
    
    model_files = {
        'Scikit-learn Ensemble': 'models/sklearn_comprehensive_metrics.json',
        'TensorFlow 176K': 'models/medical_metrics_176k.json', 
        'TensorFlow Medical': 'models/medical_metrics_comprehensive.json',
        'TensorFlow Synthetic': 'models/evaluation_metrics.json'
    }
    
    available_models = []
    
    for model_name, file_path in model_files.items():
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                available_models.append((model_name, metrics))
            except:
                continue
    
    if len(available_models) < 2:
        print("⚠️  Need at least 2 models for comparison")
        return
    
    print(f"📊 COMPARISON TABLE:")
    print(f"{'-'*70}")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*70}")
    
    for model_name, metrics in available_models:
        # Extract metrics based on model type
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            acc = perf.get('accuracy', 0)
            prec = perf.get('precision', 0)
            rec = perf.get('sensitivity', perf.get('recall', 0))
            f1 = perf.get('f1_score', 0)
        elif 'individual_models' in metrics:
            # Use best model from ensemble
            best_model = max(metrics['individual_models'].items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            perf = best_model[1]
            acc = perf.get('accuracy', 0)
            prec = perf.get('precision', 0)
            rec = perf.get('recall', 0)
            f1 = perf.get('f1', 0)
        else:
            acc = prec = rec = f1 = 0
        
        print(f"{model_name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    
    # Find best model
    best_model = max(available_models, 
                    key=lambda x: x[1].get('performance_metrics', {}).get('accuracy', 
                                  max(x[1].get('individual_models', {}).values(), 
                                      key=lambda y: y.get('accuracy', 0), default={}).get('accuracy', 0)))
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")

def main():
    """Main function with enhanced options"""
    import sys
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type == "compare":
            compare_all_available_models()
        else:
            load_and_display_metrics(model_type)
    else:
        # Auto-detect and display all available models
        load_and_display_metrics("auto")
        
        # Also show comparison if multiple models exist
        print(f"\n" + "="*70)
        compare_all_available_models()

if __name__ == "__main__":
    main()
