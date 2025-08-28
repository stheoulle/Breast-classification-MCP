import React from "react";

export default function TabularClassifier({
  featuresText,
  setFeaturesText,
  tabularPredicting,
  tabularResult,
  onLoadSample,
  onPredict,
  explicitTextualNames,
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
      <h2 style={{ color: "#314570", marginBottom: 10 }}>Classify Tabular</h2>
      <div style={{ marginBottom: 8 }}>
        <div style={{ marginBottom: 6 }}>
          Enter features (30 numbers) as a JSON array or CSV. You can also
          load a sample from the backend.
        </div>
        <textarea
          value={featuresText}
          onChange={(e) => setFeaturesText(e.target.value)}
          rows={4}
          style={{
            width: "100%",
            padding: 8,
            borderRadius: 6,
            border: "1px solid #dbeafe",
          }}
        />
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={onLoadSample} disabled={tabularPredicting}>
          Load Sample Features
        </button>
        <button
          onClick={onPredict}
          disabled={tabularPredicting}
          style={{
            background: "#314570",
            color: "#fff",
            border: "none",
            fontWeight: 600,
          }}
        >
          {tabularPredicting ? "Predicting…" : "Predict Tabular"}
        </button>
      </div>
      {tabularResult && (
        <div
          style={{
            marginTop: 12,
            background: "#fff",
            borderRadius: 8,
            padding: 12,
          }}
        >
          <h3 style={{ margin: "0 0 8px", color: "#314570" }}>Result</h3>
          {(() => {
            const idx = Number(tabularResult?.predicted_class);
            const friendly = Number.isInteger(idx) && idx >= 0 && idx < explicitTextualNames.length
              ? explicitTextualNames[idx]
              : String(tabularResult?.predicted_class);
            return (
              <>
                <div>
                  <strong>Label:</strong> {friendly}
                </div>
                <div>
                  <strong>Index:</strong> {Number.isInteger(idx) ? idx : String(tabularResult?.predicted_class)}
                </div>
              </>
            );
          })()}
          <div>
            <strong>Probability:</strong>{" "}
            {typeof tabularResult.predicted_probability === "number"
              ? tabularResult.predicted_probability.toFixed(4)
              : String(tabularResult.predicted_probability)}
          </div>
        </div>
      )}
    </section>
  );
}
