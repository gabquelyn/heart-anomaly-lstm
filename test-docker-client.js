const fetch = require("node-fetch")

// Test client for Docker-based TensorFlow API
const API_URL = "http://localhost:3000"

async function testDockerAPI() {
  console.log("🐳 TESTING TENSORFLOW DOCKER API")
  console.log("=" * 50)

  try {
    // Test health check
    console.log("\n🔍 Testing health check...")
    const healthResponse = await fetch(`${API_URL}/health`)
    const healthData = await healthResponse.json()

    console.log("✅ Health Check:")
    console.log(`   Status: ${healthData.status}`)
    console.log(`   Model Loaded: ${healthData.model_loaded}`)
    console.log(`   TensorFlow: ${healthData.tensorflow_version}`)
    console.log(`   Environment: ${healthData.environment}`)

    if (!healthData.model_loaded) {
      console.log("\n⚠️  Model not loaded. Please train a model first:")
      console.log("   ./train-in-docker.sh")
      return
    }

    // Test model info
    console.log("\n📊 Testing model info...")
    const infoResponse = await fetch(`${API_URL}/model/info`)
    const infoData = await infoResponse.json()

    console.log("✅ Model Info:")
    console.log(`   Model: ${infoData.model_name}`)
    console.log(`   Type: ${infoData.model_type}`)
    console.log(`   Accuracy: ${infoData.performance?.accuracy || "Unknown"}`)
    console.log(`   Sensitivity: ${infoData.performance?.sensitivity || "Unknown"}`)

    // Test prediction with normal data
    console.log("\n📡 Testing normal cardiac prediction...")
    const normalData = Array(100)
      .fill(0)
      .map((_, i) => [
        Math.random() * 0.5 - 0.25 + Math.sin(i * 0.1), // Normal ECG
        75 + Math.random() * 10, // Normal HR
        97 + Math.random() * 3, // Normal SpO2
        120 + Math.random() * 10, // Normal BP
        80 + Math.random() * 5, // Normal BP
      ])

    const normalPayload = {
      data: normalData,
      patient_info: {
        patient_id: "DOCKER_TEST_001",
        age: 65,
        gender: "M",
      },
      device_id: "DOCKER_ECG_001",
    }

    const normalResponse = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(normalPayload),
    })

    const normalResult = await normalResponse.json()

    console.log("✅ Normal Data Prediction:")
    console.log(`   Risk Level: ${normalResult.prediction.risk_level}`)
    console.log(`   Probability: ${(normalResult.prediction.anomaly_probability * 100).toFixed(1)}%`)
    console.log(`   Model Type: ${normalResult.model_info.model_type}`)
    console.log(`   TensorFlow: ${normalResult.model_info.tensorflow_version}`)

    // Test prediction with abnormal data
    console.log("\n📡 Testing abnormal cardiac prediction...")
    const abnormalData = Array(100)
      .fill(0)
      .map((_, i) => [
        Math.random() * 1.5 - 0.75 + Math.sin(i * 0.2), // Irregular ECG
        150 + Math.random() * 30, // Tachycardia
        88 + Math.random() * 8, // Hypoxemia
        180 + Math.random() * 20, // Hypertension
        110 + Math.random() * 15, // High diastolic
      ])

    const abnormalPayload = {
      data: abnormalData,
      patient_info: {
        patient_id: "DOCKER_TEST_002",
        age: 70,
        gender: "F",
      },
      device_id: "DOCKER_ECG_002",
    }

    const abnormalResponse = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(abnormalPayload),
    })

    const abnormalResult = await abnormalResponse.json()

    console.log("✅ Abnormal Data Prediction:")
    console.log(`   Risk Level: ${abnormalResult.prediction.risk_level}`)
    console.log(`   Probability: ${(abnormalResult.prediction.anomaly_probability * 100).toFixed(1)}%`)
    console.log(`   Urgency: ${abnormalResult.prediction.urgency}`)
    console.log(`   Clinical: ${abnormalResult.clinical_assessment.interpretation}`)

    console.log("\n🎉 Docker TensorFlow API tests completed successfully!")
  } catch (error) {
    console.error("❌ Test failed:", error.message)
    console.log("\n💡 Make sure the Docker container is running:")
    console.log("   docker-compose up -d")
  }
}

testDockerAPI()
