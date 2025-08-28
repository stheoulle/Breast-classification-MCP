import React from "react";

export default function Toolbar({ running, onHealth, onLoadImageData, onLoadTabularData, onBuildModel, onTrainTextBranch, onTrainImageBranch, onTrainFull }) {
  return (
    <section className="controls" style={{ marginBottom: 24 }}>
      <button onClick={onHealth} disabled={running}>Health</button>
      <button onClick={onLoadImageData} disabled={running}>Load Image Data</button>
      <button onClick={onLoadTabularData} disabled={running}>Load Tabular Data</button>
      <button onClick={onBuildModel} disabled={running}>Build Model</button>
      <button onClick={onTrainTextBranch} disabled={running}>Train Text Branch</button>
      <button onClick={onTrainImageBranch} disabled={running}>Train Image Branch</button>
      <button onClick={onTrainFull} disabled={running}>Train Full Pipeline</button>
    </section>
  );
}
