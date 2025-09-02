import React, { useState } from "react";
import Plot from "react-plotly.js";

export default function ShapExplain({
  featuresText,
  setFeaturesText,
  explaining,
  shapResult,
  onExplain,
}) {
  const [topK, setTopK] = useState(10);
  const [backgroundSize, setBackgroundSize] = useState(100);
  const [nsamples, setNsamples] = useState(512);

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
      <h2 style={{ color: "#314570", marginBottom: 10 }}>Explain (SHAP)</h2>
      <div style={{ marginBottom: 8 }}>
        <div style={{ marginBottom: 6 }}>
          Enter the same 30 features you’d use for tabular prediction. Then
          request a SHAP explanation for this single instance.
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

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
        <label>
          <span style={{ marginRight: 6 }}>Top K</span>
          <input
            type="number"
            min={1}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value || "10", 10))}
            style={{ width: 80 }}
          />
        </label>
        <label>
          <span style={{ marginRight: 6 }}>Background size</span>
          <input
            type="number"
            min={10}
            value={backgroundSize}
            onChange={(e) => setBackgroundSize(parseInt(e.target.value || "100", 10))}
            style={{ width: 120 }}
          />
        </label>
        <label>
          <span style={{ marginRight: 6 }}>nsamples</span>
          <input
            type="number"
            min={50}
            step={50}
            value={nsamples}
            onChange={(e) => setNsamples(parseInt(e.target.value || "512", 10))}
            style={{ width: 120 }}
          />
        </label>
        <button
          onClick={() => onExplain({ topK, backgroundSize, nsamples })}
          disabled={explaining}
          style={{ background: "#314570", color: "#fff", border: "none", fontWeight: 600 }}
        >
          {explaining ? "Explaining…" : "Explain with SHAP"}
        </button>
      </div>

      {shapResult && !shapResult.error && (
        <div style={{ marginTop: 12, background: "#fff", borderRadius: 8, padding: 12 }}>
          <h3 style={{ margin: "0 0 8px", color: "#314570" }}>Explanation</h3>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div>
              <strong>Predicted probability:</strong>{" "}
              {typeof shapResult.predicted_probability === "number"
                ? shapResult.predicted_probability.toFixed(4)
                : String(shapResult.predicted_probability)}
            </div>
            <div>
              <strong>Base value:</strong>{" "}
              {typeof shapResult.base_value === "number"
                ? shapResult.base_value.toFixed(4)
                : String(shapResult.base_value)}
            </div>
            <div>
              <strong>Background:</strong> {shapResult.background_size}
            </div>
            <div>
              <strong>nsamples:</strong> {shapResult.nsamples}
            </div>
          </div>

          {/* SHAP Bar Chart (summary plot for single sample) */}
          {Array.isArray(shapResult.shap_values) && Array.isArray(shapResult.feature_names) && (
            <div style={{ marginTop: 18 }}>
              <h4 style={{ margin: "0 0 6px", color: "#314570" }}>SHAP Value Bar Chart</h4>
              <Plot
                data={[{
                  x: shapResult.feature_names,
                  y: shapResult.shap_values,
                  type: "bar",
                  marker: { color: shapResult.shap_values.map(v => v > 0 ? "#2b7cff" : "#ff4b4b") },
                }]}
                layout={{
                  width: 700,
                  height: 320,
                  margin: { l: 40, r: 10, t: 30, b: 80 },
                  xaxis: { title: "Feature", tickangle: -45 },
                  yaxis: { title: "SHAP value" },
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
              />
            </div>
          )}

          {/* SHAP Waterfall (force-like) plot for single sample */}
          {Array.isArray(shapResult.shap_values) && Array.isArray(shapResult.feature_names) && typeof shapResult.base_value === "number" && (
            <div style={{ marginTop: 18 }}>
              <h4 style={{ margin: "0 0 6px", color: "#314570" }}>SHAP Waterfall Plot</h4>
              <Plot
                data={[{
                  type: "waterfall",
                  orientation: "v",
                  x: shapResult.feature_names,
                  y: shapResult.shap_values,
                  base: shapResult.base_value,
                  measure: Array(30).fill("relative"),
                  text: shapResult.shap_values.map(v => v.toExponential(2)),
                  decreasing: { marker: { color: "#ff4b4b" } },
                  increasing: { marker: { color: "#2b7cff" } },
                  totals: { marker: { color: "#314570" } },
                }]}
                layout={{
                  width: 700,
                  height: 320,
                  margin: { l: 40, r: 10, t: 30, b: 80 },
                  xaxis: { title: "Feature", tickangle: -45 },
                  yaxis: { title: "SHAP value" },
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
              />
            </div>
          )}

          {/* SHAP Dependence Plot (scatter for each feature) */}
          {Array.isArray(shapResult.shap_values) && Array.isArray(shapResult.features) && Array.isArray(shapResult.feature_names) && (
            <div style={{ marginTop: 18 }}>
              <h4 style={{ margin: "0 0 6px", color: "#314570" }}>SHAP Dependence Plot</h4>
              <Plot
                data={[{
                  x: shapResult.features,
                  y: shapResult.shap_values,
                  text: shapResult.feature_names,
                  mode: "markers",
                  type: "scatter",
                  marker: { color: shapResult.shap_values, colorscale: "RdBu", size: 10 },
                }]}
                layout={{
                  width: 700,
                  height: 320,
                  margin: { l: 40, r: 10, t: 30, b: 60 },
                  xaxis: { title: "Feature value" },
                  yaxis: { title: "SHAP value" },
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
              />
            </div>
          )}

          {/* Top features table (unchanged) */}
          {Array.isArray(shapResult.top_features) && shapResult.top_features.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <h4 style={{ margin: "0 0 6px", color: "#314570" }}>Top features</h4>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px 4px" }}>#</th>
                      <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px 4px" }}>Feature</th>
                      <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>Value</th>
                      <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>SHAP</th>
                      <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>|SHAP|</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shapResult.top_features.map((f, i) => (
                      <tr key={`${f.index}-${i}`}>
                        <td style={{ padding: "6px 4px" }}>{i + 1}</td>
                        <td style={{ padding: "6px 4px" }}>{f.name}</td>
                        <td style={{ padding: "6px 4px", textAlign: "right" }}>
                          {typeof f.value === "number" ? f.value.toFixed(6) : String(f.value)}
                        </td>
                        <td style={{ padding: "6px 4px", textAlign: "right" }}>
                          {typeof f.shap === "number" ? f.shap.toExponential(3) : String(f.shap)}
                        </td>
                        <td style={{ padding: "6px 4px", textAlign: "right" }}>
                          {typeof f.abs_shap === "number" ? f.abs_shap.toExponential(3) : String(f.abs_shap)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {Array.isArray(shapResult.shap_values) && shapResult.shap_values.length === 30 && (
            <details style={{ marginTop: 10 }}>
              <summary>Raw SHAP values (30)</summary>
              <pre style={{ whiteSpace: "pre-wrap" }}>
                {JSON.stringify(shapResult.shap_values, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {shapResult && shapResult.error && (
        <div style={{ marginTop: 12, background: "#fff3f3", borderRadius: 8, padding: 12, color: "#8a1f1f" }}>
          <strong>Error:</strong> {String(shapResult.error)}
        </div>
      )}
    </section>
  );
}
