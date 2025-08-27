import React, {useState, useEffect, useRef} from "react";

const API_BASE = window.__MCP_API_BASE__ || "http://localhost:8000";

function App(){
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  // Confusion matrix controls
  const [modality, setModality] = useState('tabular'); // 'tabular' | 'image'
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(8); // used for tabular
  const [numImgClasses, setNumImgClasses] = useState(2); // used when modality === 'tabular'
  const [numTabFeatures, setNumTabFeatures] = useState(30); // used when modality === 'image'
  const [cmImageUrl, setCmImageUrl] = useState(null);
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

  async function fetchConfusionMatrixImage(payload = {}){
    setRunning(true);
    pushLog(`Calling /confusion_matrix_image...`, 'info');
    try{
      const res = await fetch(`${API_BASE}/confusion_matrix_image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if(!res.ok){
        const text = await res.text();
        pushLog(`Error ${res.status}: ${text}`, 'error');
      } else {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setCmImageUrl(prev => {
          if(prev) URL.revokeObjectURL(prev);
          return url;
        });
        pushLog(`Confusion matrix image updated.`, 'success');
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

      <div className="panel">
        <h2>Confusion Matrix</h2>
        <div className="grid">
          <label>
            Modality
            <select value={modality} onChange={(e)=>setModality(e.target.value)} disabled={running}>
              <option value="tabular">Tabular</option>
              <option value="image">Image</option>
            </select>
          </label>
          <label>
            Epochs
            <input type="number" min="1" value={epochs} onChange={(e)=>setEpochs(parseInt(e.target.value||'1',10))} disabled={running} />
          </label>
          {modality === 'tabular' && (
            <>
              <label>
                Batch size
                <input type="number" min="1" value={batchSize} onChange={(e)=>setBatchSize(parseInt(e.target.value||'1',10))} disabled={running} />
              </label>
              <label>
                Num image classes
                <input type="number" min="2" value={numImgClasses} onChange={(e)=>setNumImgClasses(parseInt(e.target.value||'2',10))} disabled={running} />
              </label>
            </>
          )}
          {modality === 'image' && (
            <label>
              Num tabular features
              <input type="number" min="1" value={numTabFeatures} onChange={(e)=>setNumTabFeatures(parseInt(e.target.value||'1',10))} disabled={running} />
            </label>
          )}
        </div>
        <div className="actions">
          <button
            disabled={running}
            onClick={()=>{
              const payload = {
                modality,
                epochs,
                ...(modality === 'tabular' ? { batch_size: batchSize, num_img_classes: numImgClasses } : {}),
                ...(modality === 'image' ? { num_tab_features: numTabFeatures } : {}),
              };
              callTool('/confusion_matrix', payload);
            }}
          >
            Get Confusion Matrix
          </button>
          <button
            disabled={running}
            onClick={()=>{
              const payload = {
                modality,
                epochs,
                ...(modality === 'tabular' ? { batch_size: batchSize, num_img_classes: numImgClasses } : {}),
                ...(modality === 'image' ? { num_tab_features: numTabFeatures } : {}),
              };
              fetchConfusionMatrixImage(payload);
            }}
          >
            Plot Confusion Matrix (PNG)
          </button>
          <button
            disabled={running}
            onClick={()=>callTool('/confusion_matrix', { modality: 'tabular', epochs: 3, batch_size: 8, num_img_classes: 2 })}
          >
            Quick: Tabular
          </button>
          <button
            disabled={running}
            onClick={()=>callTool('/confusion_matrix', { modality: 'image', epochs: 2, num_tab_features: 30 })}
          >
            Quick: Image
          </button>
        </div>
        {cmImageUrl && (
          <div className="preview">
            <h3>Confusion Matrix Preview</h3>
            <img src={cmImageUrl} alt="Confusion matrix" style={{maxWidth:'100%', height:'auto', border:'1px solid #ddd'}} />
          </div>
        )}
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
