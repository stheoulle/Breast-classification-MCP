import React from "react";

export default function ImageClassifier({
  modelPath,
  setModelPath,
  imageFile,
  onSelectImage,
  imagePreview,
  predicting,
  prediction,
  onPredict,
  explicitClassNames,
}) {
  return (
    <section
      className="panel"
      style={{
        background: "#f5f7fb",
        borderRadius: 8,
        padding: "18px 20px",
        marginBottom: 24,
        boxShadow: "0 2px 8px rgba(19,23,30,0.04)",
      }}
    >
      <h2 style={{ color: "#314570", marginBottom: 10 }}>Classify Image</h2>
      <div
        className="grid"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
          marginBottom: 12,
        }}
      >
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontWeight: 500 }}>Model path</span>
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="saved_models/multitask_model_latest.keras"
            disabled={predicting}
            style={{
              padding: "8px",
              borderRadius: 6,
              border: "1px solid #dbeafe",
            }}
          />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontWeight: 500 }}>Choose image</span>
          <input
            type="file"
            accept="image/*"
            onChange={onSelectImage}
            disabled={predicting}
            style={{
              padding: "8px",
              borderRadius: 6,
              border: "1px solid #dbeafe",
            }}
          />
        </label>
      </div>
      <div className="actions" style={{ marginTop: 10 }}>
        <button
          disabled={predicting || !imageFile || !modelPath}
          onClick={onPredict}
          style={{
            fontWeight: 600,
            background: "#314570",
            color: "#fff",
            border: "none",
          }}
        >
          {predicting ? "Predicting…" : "Predict Image"}
        </button>
      </div>
      {imagePreview && (
        <div
          className="preview"
          style={{
            marginTop: 12,
            background: "#fff",
            borderRadius: 8,
            padding: 12,
            boxShadow: "0 1px 4px rgba(19,23,30,0.04)",
          }}
        >
          <h3 style={{ margin: "0 0 8px", color: "#314570" }}>Preview</h3>
          <img
            src={imagePreview}
            alt="preview"
            style={{
              maxWidth: "100%",
              height: "auto",
              border: "1px solid #dbeafe",
              borderRadius: 6,
            }}
          />
        </div>
      )}
      {prediction && (
        <div
          className="results"
          style={{
            marginTop: 12,
            background: "#fff",
            borderRadius: 8,
            padding: 12,
            boxShadow: "0 1px 4px rgba(19,23,30,0.04)",
          }}
        >
          <h3 style={{ margin: "0 0 8px", color: "#314570" }}>Prediction</h3>
          {(() => {
            const idx = Number.isInteger(prediction?.predicted_index)
              ? prediction.predicted_index
              : null;
            const rawLabels = Array.isArray(prediction?.labels)
              ? prediction.labels
              : [];
            const generic =
              rawLabels.length > 0 && rawLabels.every((l) => /^class_\d+$/.test(l));
            const sameLen = rawLabels.length === explicitClassNames.length;
            const friendlyLabel =
              idx !== null && generic && sameLen
                ? explicitClassNames[idx] ?? prediction.predicted_label
                : prediction.predicted_label ?? "-";
            return (
              <>
                <div>
                  <strong>Label:</strong> {friendlyLabel}
                </div>
                <div>
                  <strong>Index:</strong> {idx ?? "-"}
                </div>
              </>
            );
          })()}
          {Array.isArray(prediction.probabilities) && (
            <div style={{ marginTop: 8 }}>
              <strong>Probabilities:</strong>
              <ul>
                {(() => {
                  const rawLabels = Array.isArray(prediction.labels)
                    ? prediction.labels
                    : prediction.probabilities.map((_, i) => `class_${i}`);
                  const generic = rawLabels.every((l) => /^class_\d+$/.test(l));
                  const sameLen = rawLabels.length === explicitClassNames.length;
                  const labelsToShow = generic && sameLen ? explicitClassNames : rawLabels;
                  return labelsToShow.map((lab, i) => (
                    <li key={`${lab}-${i}`}>
                      {lab}:{" "}
                      {prediction.probabilities[i]?.toFixed?.(4) ?? String(prediction.probabilities[i])}
                    </li>
                  ));
                })()}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
