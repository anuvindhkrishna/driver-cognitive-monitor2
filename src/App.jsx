import React, { useRef, useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'

function computeEAR(landmarks, eyeIndices) {
  if (!Array.isArray(landmarks) || !Array.isArray(eyeIndices)) return 0
  const safePoint = i => {
    const idx = eyeIndices[i]
    if (!landmarks[idx] || typeof landmarks[idx][0] !== 'number') return { x: 0, y: 0 }
    return { x: landmarks[idx][0], y: landmarks[idx][1] }
  }
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)
  const A = dist(safePoint(1), safePoint(5))
  const B = dist(safePoint(2), safePoint(4))
  const C = dist(safePoint(0), safePoint(3))
  if (C === 0) return 0
  return (A + B) / (2.0 * C)
}

const LEFT_EYE = [33, 160, 158, 133, 153, 144]
const RIGHT_EYE = [263, 387, 385, 362, 380, 373]

export default function App(){
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const modelRef = useRef(null)
  const rafRef = useRef(null)
  const wsRef = useRef(null)

  const [status, setStatus] = useState('idle')
  const [drowsy, setDrowsy] = useState(false)
  const [blinkRate, setBlinkRate] = useState(0)
  const [ear, setEar] = useState(0)
  const [calibrating, setCalibrating] = useState(false)
  const [baselineEAR, setBaselineEAR] = useState(null)
  const [bp, setBp] = useState(null)
  const [spo2, setSpo2] = useState(null)
  const [mockMode, setMockMode] = useState(true)
  const [logRows, setLogRows] = useState([])

  const earHistoryRef = useRef([])
  const lastBlinkTRef = useRef(0)
  const modelLR = useRef(null)

  useEffect(()=>{
    let mounted = true
    ;(async ()=>{
      setStatus('loading-model')
      await tf.setBackend('webgl')
      modelRef.current = await faceLandmarksDetection.load(faceLandmarksDetection.SupportedPackages.mediapipeFacemesh)
      const res = await fetch('/model.json')
      modelLR.current = await res.json()
      setStatus('model-ready')
      try{
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 }, audio:false })
        if(!mounted) return
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setStatus('camera-ready')
        startLoop()
      }catch(err){
        console.error(err)
        setStatus('camera-error')
      }
    })()

    return ()=>{
      mounted = false
      if(rafRef.current) cancelAnimationFrame(rafRef.current)
      const s = videoRef.current && videoRef.current.srcObject
      if(s && s.getTracks) s.getTracks().forEach(t => t.stop())
      if(wsRef.current) wsRef.current.close()
    }
  },[])

  async function startLoop(){
    setStatus('running')
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !modelRef.current) return
    const ctx = canvas.getContext('2d')

    function resizeCanvas(){
      const w = video.videoWidth || 640
      const h = video.videoHeight || 480
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
        canvas.style.width = `${w}px`
        canvas.style.height = `${h}px`
      }
    }

    async function frame(){
      if (!video || !canvas || !modelRef.current) return
      if (video.readyState < 2){
        rafRef.current = requestAnimationFrame(frame)
        return
      }
      resizeCanvas()
      ctx.clearRect(0,0,canvas.width,canvas.height)
      ctx.drawImage(video,0,0,canvas.width,canvas.height)

      let predictions = []
      try{
        predictions = await modelRef.current.estimateFaces({input:video, predictIrises:false}) || []
      }catch(e){
        predictions = []
      }

      if(predictions.length>0){
        const mesh = predictions[0].scaledMesh || predictions[0].mesh || []
        ctx.fillStyle = 'rgba(0,255,160,0.4)'
        for(let i=0;i<mesh.length;i+=4){
          const p = mesh[i]
          if(!p) continue
          const x = p[0], y = p[1]
          if(typeof x === 'number' && typeof y === 'number') ctx.fillRect(x-1,y-1,2,2)
        }

        const leftEAR = computeEAR(mesh, LEFT_EYE)
        const rightEAR = computeEAR(mesh, RIGHT_EYE)
        const avgEAR = (leftEAR + rightEAR)/2
        if(Number.isFinite(avgEAR)) setEar(avgEAR)

        const now = Date.now()
        earHistoryRef.current.push({t:now, ear:avgEAR})
        earHistoryRef.current = earHistoryRef.current.filter(item => now - item.t < 10000)

        const EAR_BLINK_THRESH = baselineEAR ? Math.max(0.15, baselineEAR * 0.6) : 0.20

        const prev = earHistoryRef.current[earHistoryRef.current.length-2]
        if(prev && prev.ear <= EAR_BLINK_THRESH && avgEAR > EAR_BLINK_THRESH){
          const dt = now - lastBlinkTRef.current
          if(dt > 200){
            lastBlinkTRef.current = now
            setBlinkRate(p => Math.min(100, p + 1))
          }
        }

        // Prepare features for classifier: avgEAR, blinkRate over last 30s, lowEyeRatio = fraction of samples below threshold
        const avgEarWindow = earHistoryRef.current.reduce((a,b)=>a+b.ear,0) / Math.max(1, earHistoryRef.current.length)
        const blinks = blinkRate
        const lowEyeRatio = earHistoryRef.current.filter(it=>it.ear < (baselineEAR ? baselineEAR*0.6 : 0.18)).length / Math.max(1, earHistoryRef.current.length)

        const features = [avgEarWindow, blinks, lowEyeRatio]
        if(modelLR.current && modelLR.current.type === 'logistic-regression'){
          const w = modelLR.current.weights
          const b = modelLR.current.bias || 0
          let score = b
          for(let i=0;i<w.length && i<features.length;i++) score += w[i]*features[i]
          const prob = 1/(1+Math.exp(-score))
          const isDrowsy = prob > (modelLR.current.threshold || 0.5)
          setDrowsy(isDrowsy)
          if(isDrowsy){
            triggerAlert()
          }
        }else{
          // fallback heuristic
          const lowCount = earHistoryRef.current.reduce((acc, it) => acc + (it.ear < 0.14 ? 1 : 0), 0)
          setDrowsy(lowCount > 20)
          if(lowCount > 20) triggerAlert()
        }

        // Logging
        setLogRows(prev=>[...prev, {t:now, ear:avgEAR, blinkRate, drowsy}])
      }

      rafRef.current = requestAnimationFrame(frame)
    }
    rafRef.current = requestAnimationFrame(frame)
  }

  function triggerAlert(){
    try{
      const audio = new Audio('/beep.mp3')
      audio.play().catch(()=>{})
    }catch(e){}
    try{ navigator.vibrate && navigator.vibrate([200,100,200]) }catch(e){}
  }

  async function calibrateBaseline(){
    setCalibrating(true)
    setStatus('calibrating')
    const start = Date.now()
    const arr = []
    while(Date.now() - start < 2000){
      await new Promise(r => setTimeout(r, 120))
      if(ear > 0) arr.push(ear)
    }
    const mean = arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0
    setBaselineEAR(mean)
    setCalibrating(false)
    setStatus('running')
  }

  useEffect(()=>{
    if(!mockMode) return
    const id = setInterval(()=>{
      setBp({systolic: 110 + Math.round(Math.random()*20-10), diastolic: 70 + Math.round(Math.random()*12-6)})
      setSpo2(95 + Math.round(Math.random()*3-1))
    }, 3000)
    return ()=>clearInterval(id)
  },[mockMode])

  function connectWebSocket(){
    if(wsRef.current) wsRef.current.close()
    const ws = new WebSocket('ws://localhost:4000')
    ws.onopen = ()=>{ console.log('ws open') }
    ws.onmessage = (ev)=> {
      try{
        const data = JSON.parse(ev.data)
        if(data.bp) setBp(data.bp)
        if(data.spo2) setSpo2(data.spo2)
      }catch(e){}
    }
    wsRef.current = ws
  }

  function exportCsv(){
    const rows = [['timestamp','ear','blinkRate','drowsy']]
    logRows.forEach(r=> rows.push([new Date(r.t).toISOString(), r.ear, r.blinkRate, r.drowsy]))
    const csv = rows.map(r=> r.join(',')).join('\n')
    const blob = new Blob([csv], {type:'text/csv'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'driver-log.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="container" role="main">
      <div className="card card-animated">
        <div className="header">
          <div className="logo" aria-hidden>DC</div>
          <div>
            <h1 style={{margin:0}}>Driver Cognitive Monitor — Prototype</h1>
            <small>On-device face monitoring • WebSocket companion • Mock watch mode</small>
          </div>
        </div>

        <div className="row" style={{marginTop:12}}>
          <div className="column video-wrap card">
            <div style={{position:'relative'}}>
              <video ref={videoRef} style={{width:'100%', borderRadius:8}} muted playsInline aria-label="Driver camera feed" />
              <canvas ref={canvasRef} style={{width:'100%', height:'100%'}} aria-hidden />
            </div>
            <div style={{marginTop:10, display:'flex', gap:10, alignItems:'center'}} className="controls">
              <button onClick={calibrateBaseline} disabled={calibrating}>Calibrate baseline</button>
              <button onClick={()=>{ setMockMode(m=>!m) }}>{mockMode? 'Switch to Real Watch' : 'Switch to Mock Mode'}</button>
              <button onClick={connectWebSocket}>Connect Companion</button>
              <button onClick={exportCsv}>Export CSV</button>
              <div style={{marginLeft:'auto'}}>
                <small>Status: </small>
                <strong className={status === 'running' ? 'status-green' : 'status-red'}>{status}</strong>
              </div>
            </div>

            <div className="metrics">
              <div className="metric">EAR: {Number.isFinite(ear) ? ear.toFixed(3) : '—'}</div>
              <div className="metric">Baseline: {baselineEAR ? baselineEAR.toFixed(3) : '—'}</div>
              <div className="metric">Drowsy: {drowsy? 'YES' : 'NO'}</div>
              <div className="metric">BlinkRate (est): {blinkRate}</div>
            </div>

          </div>

          <div className="column card">
            <h3>Vitals from Companion</h3>
            <p><small>BP and SpO₂ will appear here when the companion mobile client sends them.</small></p>
            <div style={{marginTop:10}}>
              <div style={{display:'flex', gap:12}}>
                <div className="metric">Systolic: {bp ? bp.systolic : '—'}</div>
                <div className="metric">Diastolic: {bp ? bp.diastolic : '—'}</div>
                <div className="metric">SpO₂: {spo2 ? spo2 + '%' : '—'}</div>
              </div>

              <h4 style={{marginTop:14}}>Risk Assessment</h4>
              <div>
                {spo2 && spo2 < 92 ? <div className="alert red">Low SpO₂ — alert</div> : <div className="alert green">SpO₂ OK</div>}
                {bp && (bp.systolic>140 || bp.diastolic>90) ? <div className="alert red">High BP — alert</div> : <div className="alert green">BP OK</div>}
              </div>

              <div style={{marginTop:18}}>
                <h4>Recommendations</h4>
                <ul>
                  <li>When driver drowsy: play auditory alert, suggest a rest break.</li>
                  <li>When vitals abnormal: suggest immediate stop and medical check.</li>
                </ul>
              </div>

            </div>
          </div>
        </div>

        <div style={{marginTop:18}} className="card">
          <h3>Privacy & Deployment Notes</h3>
          <ul>
            <li>All camera processing is done on-device in this prototype (no server upload), improving privacy.</li>
            <li>WebSocket companion accepts forwarded smartwatch sensor data from a mobile client. Use secure transport in production.</li>
            <li>This is a research/demo prototype. For real deployment, use tested medical-grade sensors and go through regulatory certification.</li>
          </ul>
        </div>

      </div>
    </div>
  )
}
