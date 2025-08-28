import React, { useEffect, useRef } from "react";

export default function Logs({ logs, onClear }) {
  const logEndRef = useRef(null);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [logs]);

  return (
    <section className="log" style={{ marginTop: 32 }}>
      <div className="log-header">
        <div className="log-title">
          <h2 style={{ color: "#314570", margin: 0 }}>Logs</h2>
          <span className="log-count">{logs.length}</span>
        </div>
        <div className="log-actions">
          <button onClick={onClear} disabled={!logs.length}>
            Clear
          </button>
        </div>
      </div>
      <div className="log-window">
        {logs.length === 0 ? (
          <div className="log-empty">
            No logs yet. Actions and results will appear here.
          </div>
        ) : (
          logs.map((l) => (
            <div key={l.id} className={`log-line kind-${l.kind}`}>
              <span className="log-time">{new Date(l.time).toLocaleTimeString()}</span>
              <span className="log-msg">{l.text}</span>
            </div>
          ))
        )}
        <div ref={logEndRef} />
      </div>
    </section>
  );
}
