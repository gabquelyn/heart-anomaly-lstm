const express = require("express")
const cors = require("cors")
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs")
const path = require("path")

const app = express()
const PORT = process.env.PORT || 3000

// Middleware
app.use(cors())
app.use(express.json())

// Global variables
let model = null
let scalerParams = null
let modelMetrics = null
const featureNames = ["ECG", "Heart_Rate", "SpO2", "BP_Systolic", "BP_Diastolic"]

// Load TensorFlow model in Docker environment
async function loadTensorFlowModel() {
  try {
    console.log("🐳 Loading TensorFlow model in Docker environment...")

    // Check for different model file names
    const modelPaths = [
      "models/heart_anomaly_medical_lstm_176k.h5",
      "models/heart_anomaly_medical_lstm.h5",
      "models/best_medical_model.h5",
    ]

    let modelPath = null
    for (const path of modelPaths) {
      if (fs.existsSync(path)) {
        modelPath = path
        break
      }
    }

    if (modelPath) {
      model = await tf.loadLayersModel(`file://${modelPath}`)
      console.log(`✅ TensorFlow model loaded from: ${modelPath}`)
    } else {
      console.log("⚠️  No trained model found. Please train a model first.")
      console.log("   Run: docker run -v $(pwd)/models:/app/models heart-anomaly-training:latest")
      return false
    }

    // Load scaler parameters
    const scalerPaths = ["models/scaler_params_medical_176k.json", "models/scaler_params_medical.json"]

    let scalerPath = null
    for (const path of scalerPaths) {
      if (fs.existsSync(path)) {
        scalerPath = path
        break
      }
    }

    if (scalerPath) {
      const scalerData = fs.readFileSync(scalerPath, "utf8")
      scalerParams = JSON.parse(scalerData)
      console.log(`✅ Scaler parameters loaded from: ${scalerPath}`)
    } else {
      console.log("⚠️  Scaler parameters not found")
      return false
    }

    // Load model metrics
    const metricsPaths = ["models/medical_metrics_176k.json", "models/medical_metrics_comprehensive.json"]

    let metricsPath = null
    for (const path of metricsPaths) {
      if (fs.existsSync(path)) {
        metricsPath = path
        break
      }
    }

    if (metricsPath) {
      const metricsData = fs.readFileSync(metricsPath, "utf8")
      modelMetrics = JSON.parse(metricsData)

      console.log("\n🩺 TensorFlow Model Performance:")
      console.log(`   Accuracy:    ${(modelMetrics.performance_metrics.accuracy * 100).toFixed(1)}%`)
      console.log(`   Sensitivity: ${(modelMetrics.performance_metrics.sensitivity * 100).toFixed(1)}%`)
      console.log(`   Specificity: ${(modelMetrics.performance_metrics.specificity * 100).toFixed(1)}%`)
      console.log(`   Precision:   ${(modelMetrics.performance_metrics.precision * 100).toFixed(1)}%`)
      console.log(`   F1 Score:    ${(modelMetrics.performance_metrics.f1_score * 100).toFixed(1)}%`)
      console.log(`   AUC:         ${modelMetrics.performance_metrics.auc_score.toFixed(3)}`)
    }

    console.log("\n📊 Model Configuration:")
    console.log(`   Input shape: [${model.inputs[0].shape.slice(1).join(", ")}]`)
    console.log(`   Features: ${featureNames.join(", ")}`)
    console.log(`   Environment: Docker Container`)
    console.log(`   TensorFlow: ${tf.version.tfjs}`)

    return true
  } catch (error) {
    console.error("❌ Error loading TensorFlow model:", error)
    return false
  }
}

// Standardize input data using scaler
function standardizeData(data) {
  if (!scalerParams) {
    throw new Error("Scaler parameters not loaded")
  }

  return data.map((sequence) => {
    return sequence.map((timestep) => {
      return timestep.map((feature, featIdx) => {
        return (feature - scalerParams.mean[featIdx]) / scalerParams.std[featIdx]
      })
    })
  })
}

// Enhanced medical prediction with TensorFlow
app.post("/predict", async (req, res) => {
  try {
    if (!model) {
      return res.status(503).json({
        error: "TensorFlow model not loaded",
        message: "Please train a model first using the training Docker container",
      })
    }

    const { data, patient_info, device_id } = req.body

    if (!data) {
      return res.status(400).json({
        error: "Missing data",
        message: "Please provide ECG and vital signs data",
      })
    }

    // Validate and prepare data
    const inputData = Array.isArray(data[0]) ? [data] : [data]

    // Standardize data
    const standardizedData = standardizeData(inputData)

    // Make TensorFlow prediction
    const inputTensor = tf.tensor3d(standardizedData)
    const prediction = model.predict(inputTensor)
    const predictionData = await prediction.data()

    // Clean up tensors
    inputTensor.dispose()
    prediction.dispose()

    // Enhanced medical interpretation
    const probability = predictionData[0]
    const isAnomalous = probability > 0.5
    const confidence = isAnomalous ? probability : 1 - probability

    // Medical risk stratification
    let riskLevel, clinicalInterpretation, urgency, recommendations

    if (probability > 0.8) {
      riskLevel = "High"
      urgency = "Immediate"
      clinicalInterpretation = "Strong indication of cardiac anomaly. Immediate medical evaluation required."
      recommendations = [
        "🚨 URGENT: Immediate cardiology consultation",
        "📊 Obtain 12-lead ECG immediately",
        "🩸 Check cardiac enzymes (troponin, CK-MB)",
        "💊 Review current medications",
        "🏥 Consider emergency department evaluation",
      ]
    } else if (probability > 0.6) {
      riskLevel = "Medium-High"
      urgency = "Within 24 hours"
      clinicalInterpretation = "Probable cardiac anomaly detected. Prompt medical evaluation recommended."
      recommendations = [
        "📞 Contact primary care physician within 24 hours",
        "📊 Schedule ECG and echocardiogram",
        "📝 Document symptoms and timing",
        "💊 Review medication compliance",
        "🔍 Monitor for symptom progression",
      ]
    } else if (probability > 0.3) {
      riskLevel = "Medium"
      urgency = "Within 1 week"
      clinicalInterpretation = "Borderline findings detected. Routine follow-up recommended."
      recommendations = [
        "📅 Schedule routine cardiology follow-up",
        "📊 Consider Holter monitor if symptomatic",
        "📝 Keep symptom diary",
        "🏃 Continue regular exercise as tolerated",
        "💊 Maintain current medications",
      ]
    } else {
      riskLevel = "Low"
      urgency = "Routine"
      clinicalInterpretation = "No significant cardiac anomaly detected. Continue routine care."
      recommendations = [
        "✅ Continue standard cardiac care",
        "📅 Routine follow-up as scheduled",
        "🏃 Maintain healthy lifestyle",
        "📊 Annual cardiac screening",
        "📞 Contact physician if symptoms develop",
      ]
    }

    const response = {
      device_id: device_id || "unknown",
      timestamp: new Date().toISOString(),
      prediction: {
        anomaly_probability: probability,
        is_anomalous: isAnomalous,
        confidence: confidence,
        risk_level: riskLevel,
        urgency: urgency,
      },
      clinical_assessment: {
        interpretation: clinicalInterpretation,
        recommendations: recommendations,
        clinical_significance:
          probability > 0.7
            ? "High clinical significance"
            : probability > 0.5
              ? "Moderate clinical significance"
              : "Low clinical significance",
      },
      patient_context: patient_info || null,
      model_info: {
        model_type: "TensorFlow CNN-LSTM",
        trained_on: "Real Medical Data (176K+ sequences)",
        features_analyzed: featureNames,
        environment: "Docker Container",
        tensorflow_version: tf.version.tfjs,
        accuracy: modelMetrics ? `${(modelMetrics.performance_metrics.accuracy * 100).toFixed(1)}%` : "Unknown",
        sensitivity: modelMetrics ? `${(modelMetrics.performance_metrics.sensitivity * 100).toFixed(1)}%` : "Unknown",
        threshold: 0.5,
      },
      processing_time: Date.now(),
      medical_disclaimer:
        "🩺 TensorFlow-based cardiac analysis for screening purposes only. Always consult qualified healthcare professionals.",
    }

    console.log(`✅ TensorFlow prediction - Risk: ${riskLevel} (${(probability * 100).toFixed(1)}%)`)
    res.json(response)
  } catch (error) {
    console.error("❌ TensorFlow prediction error:", error)
    res.status(500).json({
      error: "Prediction failed",
      message: error.message,
    })
  }
})

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    model_loaded: model !== null,
    model_type: "TensorFlow CNN-LSTM",
    environment: "Docker Container",
    tensorflow_version: tf.version.tfjs,
    timestamp: new Date().toISOString(),
  })
})

// Model info endpoint
app.get("/model/info", (req, res) => {
  if (!model) {
    return res.status(503).json({
      error: "TensorFlow model not loaded",
      message: "Please train a model first",
    })
  }

  res.json({
    model_name: "TensorFlow Heart Anomaly Detection - Docker",
    version: "3.0 TensorFlow Docker",
    model_type: "CNN-LSTM Neural Network",
    environment: "Docker Container",
    tensorflow_version: tf.version.tfjs,
    features: featureNames,
    model_architecture: {
      input_shape: model.inputs[0].shape,
      output_shape: model.outputs[0].shape,
      total_params: model.countParams ? model.countParams() : "Unknown",
    },
    performance: modelMetrics
      ? {
          accuracy: `${(modelMetrics.performance_metrics.accuracy * 100).toFixed(1)}%`,
          sensitivity: `${(modelMetrics.performance_metrics.sensitivity * 100).toFixed(1)}%`,
          specificity: `${(modelMetrics.performance_metrics.specificity * 100).toFixed(1)}%`,
          precision: `${(modelMetrics.performance_metrics.precision * 100).toFixed(1)}%`,
          f1_score: `${(modelMetrics.performance_metrics.f1_score * 100).toFixed(1)}%`,
          auc: modelMetrics.performance_metrics.auc_score.toFixed(3),
        }
      : "Metrics not available",
    disclaimer: "🩺 TensorFlow-based cardiac analysis in Docker. For medical screening support only.",
  })
})

// Start server
async function startDockerServer() {
  console.log("🐳 TENSORFLOW HEART ANOMALY DETECTION IN DOCKER")
  console.log("=" * 60)

  const modelLoaded = await loadTensorFlowModel()

  app.listen(PORT, () => {
    console.log(`\n🏥 TensorFlow Heart Anomaly Detection Server (Docker)`)
    console.log(`🔗 Server running on: http://localhost:${PORT}`)
    console.log(`🔗 Health check: http://localhost:${PORT}/health`)
    console.log(`📊 Model info: http://localhost:${PORT}/model/info`)
    console.log(`🩺 Prediction: POST http://localhost:${PORT}/predict`)

    if (modelLoaded) {
      console.log("\n✅ Ready for TensorFlow-based cardiac analysis!")
      console.log("🧠 Deep learning model loaded successfully in Docker")
    } else {
      console.log("\n⚠️  Model not loaded. To train a model:")
      console.log("   docker run -v $(pwd)/models:/app/models heart-anomaly-training:latest")
    }

    console.log(`\n🐳 Docker Environment Info:`)
    console.log(`   TensorFlow.js: ${tf.version.tfjs}`)
    console.log(`   Node.js: ${process.version}`)
    console.log(`   Platform: ${process.platform}`)
  })
}

startDockerServer()
