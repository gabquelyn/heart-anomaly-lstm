# Heart Anomaly Detection LSTM API

A machine learning API that uses LSTM neural networks to detect heart anomalies from ECG, SpO2, Heart Rate, and Blood Pressure data.

## Features

- **LSTM Encoder Model**: Deep learning model trained to detect anomalous patterns in cardiac data
- **Multi-modal Input**: Processes ECG, Heart Rate, SpO2, Systolic BP, and Diastolic BP
- **REST API**: Easy-to-use HTTP endpoints for predictions
- **Batch Processing**: Support for single and batch predictions
- **Real-time Inference**: Fast prediction responses

## Quick Start

### 1. Install Dependencies

\`\`\`bash
# Install dependencies for real medical data
pip install -r requirements_real_data.txt

# Node.js dependencies for API
npm install
\`\`\`

### 2. Download Real Medical Data

\`\`\`bash
# Download MIT-BIH Arrhythmia Database
python scripts/download_real_medical_data.py
\`\`\`

### 3. Train the Medical Model

\`\`\`bash
# Train model on real medical data
python scripts/train_model_real_medical_data.py
\`\`\`

### 4. Start the API Server

\`\`\`bash
# Start medical-grade API server
node server-medical-data.js
\`\`\`

The API will be available at `http://localhost:3000`

## API Endpoints

### Health Check
\`\`\`
GET /health
\`\`\`

### Model Information
\`\`\`
GET /model/info
\`\`\`

### Single Prediction
\`\`\`
POST /predict
Content-Type: application/json

{
  "data": [
    [ecg_value, hr_value, spo2_value, bp_sys_value, bp_dia_value],
    // ... more timesteps (typically 100 timesteps)
  ]
}
\`\`\`

### Batch Prediction
\`\`\`
POST /predict/batch
Content-Type: application/json

{
  "sequences": [
    [[ecg, hr, spo2, bp_sys, bp_dia], ...], // Sequence 1
    [[ecg, hr, spo2, bp_sys, bp_dia], ...], // Sequence 2
    // ... more sequences
  ]
}
\`\`\`

## Data Format

Each input sequence should contain timesteps with 5 features:
1. **ECG**: Electrocardiogram signal value
2. **Heart Rate**: Beats per minute
3. **SpO2**: Oxygen saturation percentage (0-100)
4. **BP Systolic**: Systolic blood pressure (mmHg)
5. **BP Diastolic**: Diastolic blood pressure (mmHg)

## Example Usage

\`\`\`javascript
const response = await fetch('http://localhost:3000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: [
      [0.5, 75, 98, 120, 80],  // Timestep 1
      [0.3, 76, 97, 118, 79],  // Timestep 2
      // ... 98 more timesteps
    ]
  })
});

const result = await response.json();
console.log(result.predictions);
\`\`\`

## Model Architecture

- **Input Layer**: Accepts sequences of shape (timesteps, 5_features)
- **LSTM Layers**: 3 stacked LSTM layers with dropout and batch normalization
- **Dense Layers**: Fully connected layers for classification
- **Output**: Binary classification (normal vs anomalous)

## Model Evaluation

### Comprehensive Metrics

The training script provides detailed evaluation including:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified
- **Specificity**: Proportion of actual negatives correctly identified  
- **F1 Score**: Harmonic mean of precision and recall
- **AUC Score**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

### View Evaluation Results

\`\`\`bash
# View medical model results  
python scripts/view_evaluation_results.py

# View detailed medical metrics
python scripts/train_model_real_medical_data.py
\`\`\`

### Generated Files

After training, you'll find these evaluation files:

**Medical Data Model:**
- `models/medical_metrics_comprehensive.json` - Comprehensive metrics
- `models/medical_evaluation_comprehensive.png` - Detailed visualization plots

### Clinical Interpretation

The evaluation includes clinical significance analysis:
- **False Negatives**: Missed anomalies (could delay treatment)
- **False Positives**: False alarms (unnecessary interventions)
- **Sensitivity**: Critical for not missing serious conditions
- **Specificity**: Important to avoid alarm fatigue

### Performance Benchmarks

**Target Performance for Clinical Use:**
- Sensitivity (Recall) > 95% (minimize missed anomalies)
- Specificity > 90% (minimize false alarms)
- F1 Score > 90% (balanced performance)
- AUC > 0.95 (excellent discrimination)

## Testing

Run the test script to verify the API:

\`\`\`bash
node test-api.js
\`\`\`

## Response Format

\`\`\`json
{
  "predictions": {
    "sequence_index": 0,
    "anomaly_probability": 0.23,
    "is_anomalous": false,
    "confidence": 0.77,
    "risk_level": "Low"
  },
  "model_info": {
    "features_used": ["ECG", "Heart_Rate", "SpO2", "BP_Systolic", "BP_Diastolic"],
    "threshold": 0.5
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
\`\`\`

## Notes

- The model is trained on synthetic data for demonstration purposes
- For production use, train with real medical data and proper validation
- Always consult medical professionals for actual diagnosis
- This is for educational/research purposes only
