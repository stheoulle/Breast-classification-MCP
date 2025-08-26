import React, {useState, useEffect} from "react";

const API_BASE = window.__MCP_API_BASE__ || "http://localhost:8000";

function App(){
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);

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
    pushLog(`Calling ${path}...`);
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
        pushLog(`Error ${res.status}: ${typeof data === 'string' ? data : JSON.stringify(data)}`);
      } else {
        pushLog(JSON.stringify(data, null, 2));
      }
    }catch(e){
      pushLog(String(e));
    }
    setRunning(false);
  }

  function pushLog(msg){
    setLogs(l=>[...l, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }

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
        <h2>Logs</h2>
        <div className="log-window">
          {logs.map((l,i)=> <div key={i} className="log-line">{l}</div>)}
        </div>
      </div>
    </div>
  );
}

export default App;
