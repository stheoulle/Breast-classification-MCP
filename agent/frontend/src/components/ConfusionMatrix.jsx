import React from "react";

export default function ConfusionMatrix({
  modality,
  setModality,
  epochs,
  setEpochs,
  batchSize,
  setBatchSize,
  numImgClasses,
  setNumImgClasses,
  numTabFeatures,
  setNumTabFeatures,
  running,
  onPlot,
  cmImageUrl,
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
      <h2 style={{ color: "#314570", marginBottom: 10 }}>Confusion Matrix</h2>
      <div
        className="grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3,1fr)",
          gap: 18,
          marginBottom: 12,
        }}
      >
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontWeight: 500 }}>Modality</span>
          <select
            value={modality}
            onChange={(e) => setModality(e.target.value)}
            disabled={running}
            style={{
              padding: "8px",
              borderRadius: 6,
              border: "1px solid #dbeafe",
            }}
          >
            <option value="tabular">Tabular</option>
            <option value="image">Image</option>
          </select>
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontWeight: 500 }}>Epochs</span>
          <input
            type="number"
            min="1"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value || "1", 10))}
            disabled={running}
            style={{
              padding: "8px",
              borderRadius: 6,
              border: "1px solid #dbeafe",
            }}
          />
        </label>
        {modality === "tabular" && (
          <>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontWeight: 500 }}>Batch size</span>
              <input
                type="number"
                min="1"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value || "1", 10))}
                disabled={running}
                style={{
                  padding: "8px",
                  borderRadius: 6,
                  border: "1px solid #dbeafe",
                }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontWeight: 500 }}>Num image classes</span>
              <input
                type="number"
                min="2"
                value={numImgClasses}
                onChange={(e) => setNumImgClasses(parseInt(e.target.value || "2", 10))}
                disabled={running}
                style={{
                  padding: "8px",
                  borderRadius: 6,
                  border: "1px solid #dbeafe",
                }}
              />
            </label>
          </>
        )}
        {modality === "image" && (
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontWeight: 500 }}>Num tabular features</span>
            <input
              type="number"
              min="1"
              value={numTabFeatures}
              onChange={(e) => setNumTabFeatures(parseInt(e.target.value || "1", 10))}
              disabled={running}
              style={{
                padding: "8px",
                borderRadius: 6,
                border: "1px solid #dbeafe",
              }}
            />
          </label>
        )}
      </div>
      <div className="actions" style={{ marginTop: 10 }}>
        <button
          disabled={running}
          onClick={onPlot}
          style={{
            fontWeight: 600,
            background: "#314570",
            color: "#fff",
            border: "none",
          }}
        >
          Plot Confusion Matrix (PNG)
        </button>
      </div>
      {cmImageUrl && (
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
          <h3 style={{ margin: "0 0 8px", color: "#314570" }}>
            Confusion Matrix Preview
          </h3>
          <img
            src={cmImageUrl}
            alt="Confusion matrix"
            style={{
              maxWidth: "100%",
              height: "auto",
              border: "1px solid #dbeafe",
              borderRadius: 6,
            }}
          />
        </div>
      )}
    </section>
  );
}
