import React, {useState, useEffect, useRef} from "react";

const API_BASE = window.__MCP_API_BASE__ || "http://localhost:8000";

function App(){
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const logEndRef = useRef(null);

  useEffect(()=>{
    checkHealth();
  },[]);

  async function checkHealth(){
    try{
      const res = await fetch(`${API_BASE}/health`);
      const txt = await res.text();
      console.log("Health:", txt);
      setHealth(txt);
    }catch(e){
        console.error("Health check failed:", e);
      setHealth("DOWN");
    }
  }

  async function callTool(path, payload = {}){
    setRunning(true);
    pushLog(`Calling ${path}...`, 'info');
    try{
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data;
      try{ data = JSON.parse(text); } catch { data = text; }

      if(!res.ok){
        pushLog(`Error ${res.status}: ${typeof data === 'string' ? data : JSON.stringify(data)}`, 'error');
      } else {
        pushLog(typeof data === 'string' ? data : JSON.stringify(data, null, 2), 'success');
      }
    }catch(e){
      pushLog(String(e), 'error');
    }
    setRunning(false);
  }

  function pushLog(message, kind = 'info'){
    const entry = {
      id: Date.now() + Math.random(),
      time: new Date(),
      text: typeof message === 'string' ? message : JSON.stringify(message),
      kind // 'info' | 'success' | 'error'
    };
    setLogs(l => [...l, entry]);
  }

  function clearLogs(){
    setLogs([]);
  }

  useEffect(()=>{
    // Auto-scroll to bottom when logs update
    logEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [logs]);

  return (
    <div className="container">
      <h1>MCP Async Frontend</h1>
      <div className="status">Health: <strong>{health ?? '...'}</strong></div>
      <div className="controls">
        <button onClick={()=>checkHealth()} disabled={running}>Health</button>
        <button onClick={()=>callTool('/load_image_data')} disabled={running}>Load Image Data</button>
        <button onClick={()=>callTool('/load_tabular_data')} disabled={running}>Load Tabular Data</button>
        <button onClick={()=>callTool('/build_multitask_model')} disabled={running}>Build Model</button>
        <button onClick={()=>callTool('/train_text_branch')} disabled={running}>Train Text Branch</button>
        <button onClick={()=>callTool('/train_image_branch')} disabled={running}>Train Image Branch</button>
        <button onClick={()=>callTool('/train_and_evaluate_full_pipeline')} disabled={running}>Train Full Pipeline</button>
      </div>

      <div className="log">
        <div className="log-header">
          <div className="log-title">
            <h2>Logs</h2>
            <span className="log-count">{logs.length}</span>
          </div>
          <div className="log-actions">
            <button onClick={clearLogs} disabled={!logs.length}>Clear</button>
          </div>
        </div>
        <div className="log-window">
          {logs.length === 0 ? (
            <div className="log-empty">No logs yet. Actions and results will appear here.</div>
          ) : (
            logs.map((l)=> (
              <div key={l.id} className={`log-line kind-${l.kind}`}>
                <span className="log-time">{l.time.toLocaleTimeString()}</span>
                <span className="log-msg">{l.text}</span>
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  );
}

export default App;
