import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import Toolbar from "./components/Toolbar";
import ImageClassifier from "./components/ImageClassifier";
import TabularClassifier from "./components/TabularClassifier";
import ConfusionMatrix from "./components/ConfusionMatrix";
import Logs from "./components/Logs";

const API_BASE = window.__MCP_API_BASE__ || "http://localhost:8000";
// If backend labels are generic (class_0, class_1, class_2), show these explicit names instead.
// Order: benign (0), malignant (1), normal (2)
const EXPLICIT_CLASS_NAMES = ["benign", "malignant", "normal"]; // index 0,1,2
const EXPLICIT_CLASS_NAMES_TEXTUAL = ["Benign", "Malignant"]; // index 0,1

function App() {
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  // Image classification controls
  const [modelPath, setModelPath] = useState(
    "saved_models/multitask_model_latest.keras"
  );
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  // Tabular classification controls
  // features can be provided as JSON array or comma-separated values
  const [featuresText, setFeaturesText] = useState("[]");
  const [tabularPredicting, setTabularPredicting] = useState(false);
  const [tabularResult, setTabularResult] = useState(null);
  // Confusion matrix controls
  const [modality, setModality] = useState("tabular"); // 'tabular' | 'image'
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(8); // used for tabular
  const [numImgClasses, setNumImgClasses] = useState(2); // used when modality === 'tabular'
  const [numTabFeatures, setNumTabFeatures] = useState(30); // used when modality === 'image'
  const [cmImageUrl, setCmImageUrl] = useState(null);
  // Log scroll is handled inside Logs component

  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const txt = await res.text();
      console.log("Health:", txt);
      setHealth(txt);
    } catch (e) {
      console.error("Health check failed:", e);
      setHealth("DOWN");
    }
  }

  async function callTool(path, payload = {}) {
    setRunning(true);
    pushLog(`Calling ${path}...`, "info");
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }

      if (!res.ok) {
        pushLog(
          `Error ${res.status}: ${
            typeof data === "string" ? data : JSON.stringify(data)
          }`,
          "error"
        );
      } else {
        pushLog(
          typeof data === "string" ? data : JSON.stringify(data, null, 2),
          "success"
        );
      }
    } catch (e) {
      pushLog(String(e), "error");
    }
    setRunning(false);
  }

  async function fetchConfusionMatrixImage(payload = {}) {
    setRunning(true);
    pushLog(`Calling /confusion_matrix_image...`, "info");
    try {
      const res = await fetch(`${API_BASE}/confusion_matrix_image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        pushLog(`Error ${res.status}: ${text}`, "error");
      } else {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setCmImageUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
        pushLog(`Confusion matrix image updated.`, "success");
      }
    } catch (e) {
      pushLog(String(e), "error");
    }
    setRunning(false);
  }

  async function predictImage() {
    if (!imageFile) {
      pushLog("Please choose an image first.", "error");
      return;
    }
    if (!modelPath) {
      pushLog(
        "Please provide a model path or set MODEL_PATH on the server.",
        "error"
      );
      return;
    }
    setPredicting(true);
    setPrediction(null);
    pushLog("Calling /predict_image...", "info");
    try {
      const fd = new FormData();
      fd.append("image", imageFile);
      fd.append("model_path", modelPath);
      // Optionally: fd.append('image_size', '256,256');

      const res = await fetch(`${API_BASE}/predict_image`, {
        method: "POST",
        body: fd,
      });

      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }

      if (!res.ok) {
        pushLog(
          `Error ${res.status}: ${
            typeof data === "string" ? data : JSON.stringify(data)
          }`,
          "error"
        );
      } else {
        setPrediction(data);
        pushLog(JSON.stringify(data, null, 2), "success");
      }
    } catch (e) {
      pushLog(String(e), "error");
    }
    setPredicting(false);
  }

  // Load a sample feature vector from backend (/load_tabular_data) and populate featuresText
  async function loadSampleFeatures() {
    pushLog("Loading sample features from /load_tabular_data...", "info");
    try {
      const res = await fetch(`${API_BASE}/load_tabular_data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (!res.ok) {
        pushLog(`Error ${res.status}: ${JSON.stringify(data)}`, "error");
        return;
      }
      // prefer a validation sample if present
      const sample =
        (data.X_val && data.X_val[0]) || (data.X_train && data.X_train[0]);
      if (sample) {
        setFeaturesText(JSON.stringify(sample));
        pushLog("Sample features loaded.", "success");
      } else {
        pushLog("No sample features found in response.", "error");
      }
    } catch (e) {
      pushLog(String(e), "error");
    }
  }

  // Call backend /predict_tabular with a JSON array of 30 features
  async function predictTabular() {
    let features;
    try {
      // accept JSON array or comma-separated values
      if (featuresText.trim().startsWith("[")) {
        features = JSON.parse(featuresText);
      } else {
        features = featuresText
          .split(/[,\n\s]+/)
          .map((s) => (s === "" ? null : Number(s)))
          .filter((v) => v !== null && !Number.isNaN(v));
      }
    } catch (e) {
      pushLog("Failed to parse features: provide JSON array or CSV", "error");
      return;
    }

    if (!Array.isArray(features) || features.length !== 30) {
      pushLog(
        "features must be an array of 30 numbers (sklearn breast cancer)",
        "error"
      );
      return;
    }

    setTabularPredicting(true);
    setTabularResult(null);
    pushLog("Calling /predict_tabular...", "info");
    try {
      const payload = { features };
      const res = await fetch(`${API_BASE}/predict_tabular`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        pushLog(`Error ${res.status}: ${JSON.stringify(data)}`, "error");
      } else {
        setTabularResult(data);
        pushLog(JSON.stringify(data, null, 2), "success");
      }
    } catch (e) {
      pushLog(String(e), "error");
    }
    setTabularPredicting(false);
  }

  function onSelectImage(e) {
    const f = e.target.files?.[0];
    setImageFile(f || null);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
      setImagePreview(null);
    }
    if (f) {
      const url = URL.createObjectURL(f);
      setImagePreview(url);
    }
  }

  function pushLog(message, kind = "info") {
    const entry = {
      id: Date.now() + Math.random(),
      time: new Date(),
      text: typeof message === "string" ? message : JSON.stringify(message),
      kind, // 'info' | 'success' | 'error'
    };
    setLogs((l) => [...l, entry]);
  }

  function clearLogs() {
    setLogs([]);
  }

  // Log auto-scroll is encapsulated in Logs component

  return (
    <div className="container">
      <Header health={health} />
      <Toolbar
        running={running}
        onHealth={checkHealth}
        onLoadImageData={() => callTool("/load_image_data")}
        onLoadTabularData={() => callTool("/load_tabular_data")}
        onBuildModel={() => callTool("/build_multitask_model")}
        onTrainTextBranch={() => callTool("/train_text_branch")}
        onTrainImageBranch={() => callTool("/train_image_branch")}
        onTrainFull={() => callTool("/train_and_evaluate_full_pipeline")}
      />

      <ImageClassifier
        modelPath={modelPath}
        setModelPath={setModelPath}
        imageFile={imageFile}
        onSelectImage={onSelectImage}
        imagePreview={imagePreview}
        predicting={predicting}
        prediction={prediction}
        onPredict={predictImage}
        explicitClassNames={EXPLICIT_CLASS_NAMES}
      />
      <TabularClassifier
        featuresText={featuresText}
        setFeaturesText={setFeaturesText}
        tabularPredicting={tabularPredicting || running}
        tabularResult={tabularResult}
        onLoadSample={loadSampleFeatures}
        onPredict={predictTabular}
        explicitTextualNames={EXPLICIT_CLASS_NAMES_TEXTUAL}
      />
      <ConfusionMatrix
        modality={modality}
        setModality={setModality}
        epochs={epochs}
        setEpochs={setEpochs}
        batchSize={batchSize}
        setBatchSize={setBatchSize}
        numImgClasses={numImgClasses}
        setNumImgClasses={setNumImgClasses}
        numTabFeatures={numTabFeatures}
        setNumTabFeatures={setNumTabFeatures}
        running={running}
        cmImageUrl={cmImageUrl}
        onPlot={() => {
          const payload = {
            modality,
            epochs,
            ...(modality === "tabular"
              ? { batch_size: batchSize, num_img_classes: numImgClasses }
              : {}),
            ...(modality === "image"
              ? { num_tab_features: numTabFeatures }
              : {}),
          };
          fetchConfusionMatrixImage(payload);
        }}
      />

  <Logs logs={logs} onClear={clearLogs} />

      
    </div>
  );
}

export default App;
