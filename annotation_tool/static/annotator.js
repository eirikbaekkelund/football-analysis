"use strict";

// ---------------------------------------------------------------------------
// History (undo / redo)
// ---------------------------------------------------------------------------

const _hist = { past: [], future: [], MAX: 50 };

function _snap() { return state.anno ? JSON.parse(JSON.stringify(state.anno)) : null; }

function pushHistory() {
  const s = _snap(); if (!s) return;
  _hist.past.push(s);
  if (_hist.past.length > _hist.MAX) _hist.past.shift();
  _hist.future = [];
  _syncUndoRedo();
}

function undo() {
  if (!_hist.past.length) return;
  _hist.future.push(_snap());
  state.anno = _hist.past.pop();
  state.selected = null; updateClassDropdown();
  _syncUndoRedo(); renderFrame(); renderPitch();
}

function redo() {
  if (!_hist.future.length) return;
  _hist.past.push(_snap());
  state.anno = _hist.future.pop();
  state.selected = null; updateClassDropdown();
  _syncUndoRedo(); renderFrame(); renderPitch();
}

function _syncUndoRedo() {
  btnUndo.disabled = _hist.past.length === 0;
  btnRedo.disabled = _hist.future.length === 0;
}

function _resetHistory() { _hist.past = []; _hist.future = []; _syncUndoRedo(); }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _loadAbort = null;

const state = {
  videoId: null,
  videoMeta: null,
  frameIndices: [],
  currentPos: 0,
  frameImg: null,
  pitchImg: null,
  pitchKps: [],
  anno: null,
  selected: null,       // {type:'bbox'|'kp', idx:int}
  multiSelected: new Set(), // Set of "kp:idx"|"bbox:idx" for batch ops
  mode: 'select',
  addKpStep: null,
  pendingKpPixel: null, // frame coords
  // bbox draw
  bboxDrawStart: null,  // CSS canvas coords
  bboxDrawCurrent: null,
  // annotation drag
  isDragging: false,
  dragHistoryPushed: false,
  dragAnchor: null,     // CSS canvas coords at mousedown
  dragPreState: null,
  // zoom / pan
  zoom: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panAnchor: null,      // CSS canvas coords at pan start
  panStartX: 0,
  panStartY: 0,
  // pitch flip — V on by default so far-in-video = far-on-pitch
  flipH: false,
  flipV: true,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const frameCanvas  = document.getElementById('frame-canvas');
const pitchCanvas  = document.getElementById('pitch-canvas');
const fCtx         = frameCanvas.getContext('2d');
const pCtx         = pitchCanvas.getContext('2d');

const fileInput    = document.getElementById('file-input');
const intervalSel  = document.getElementById('interval-select');
const btnSelect    = document.getElementById('btn-select');
const btnDrawBbox  = document.getElementById('btn-draw-bbox');
const btnAddKp     = document.getElementById('btn-add-kp');
const btnUndo      = document.getElementById('btn-undo');
const btnRedo      = document.getElementById('btn-redo');
const btnFlipH     = document.getElementById('btn-flip-h');
const btnFlipV     = document.getElementById('btn-flip-v');
const btnExport    = document.getElementById('btn-export');
const btnPrev      = document.getElementById('btn-prev');
const btnNext      = document.getElementById('btn-next');
const btnDelete     = document.getElementById('btn-delete');
const btnClearFrame = document.getElementById('btn-clear-frame');
const btnSave       = document.getElementById('btn-save');
const frameCounter = document.getElementById('frame-counter');
const classLabel   = document.getElementById('class-label');
const classSelect  = document.getElementById('class-select');
const selectedInfo = document.getElementById('selected-info');
const statusMsg      = document.getElementById('status-msg');
const progressWrap   = document.getElementById('progress-wrap');
const progressFill   = document.getElementById('progress-bar-fill');
const progressLabel  = document.getElementById('progress-label');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BBOX_CLASSES = ['player', 'goalkeeper', 'referee', 'ball'];
const BBOX_COLORS  = ['#2196f3', '#4caf50', '#ffeb3b', '#ff9800'];
const MIN_ZOOM = 1, MAX_ZOOM = 8;

// ---------------------------------------------------------------------------
// Zoom / pan helpers
// ---------------------------------------------------------------------------

// CSS canvas coords → base canvas coords (pre-zoom drawing space)
function cssToBase(cx, cy) {
  return { x: (cx - state.panX) / state.zoom, y: (cy - state.panY) / state.zoom };
}

// Base canvas coords → frame pixel coords
function baseToFrame(bx, by) {
  return { x: bx * state.anno.width / frameCanvas.width, y: by * state.anno.height / frameCanvas.height };
}

// CSS canvas coords → frame pixel coords (combined)
function cssToFrame(cx, cy) {
  const b = cssToBase(cx, cy); return baseToFrame(b.x, b.y);
}

// Frame pixel coords → base canvas coords
function frameToBase(fx, fy) {
  return { x: fx * frameCanvas.width / state.anno.width, y: fy * frameCanvas.height / state.anno.height };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function clampPan() {
  if (state.zoom <= 1) { state.panX = 0; state.panY = 0; return; }
  state.panX = clamp(state.panX, frameCanvas.width  * (1 - state.zoom), 0);
  state.panY = clamp(state.panY, frameCanvas.height * (1 - state.zoom), 0);
}

function resetZoom() { state.zoom = 1; state.panX = 0; state.panY = 0; }

// Hit radius in base canvas coords — stays visually ~10px on screen
function hitRadius() { return 10 / state.zoom; }

// ---------------------------------------------------------------------------
// Multi-select helpers
// ---------------------------------------------------------------------------

function msKey(type, idx) { return `${type}:${idx}`; }
function msHas(type, idx) { return state.multiSelected.has(msKey(type, idx)); }
function msToggle(type, idx) {
  const k = msKey(type, idx);
  if (state.multiSelected.has(k)) state.multiSelected.delete(k); else state.multiSelected.add(k);
}
function msClear() { state.multiSelected.clear(); }
function _syncDeleteBtn() {
  const n = state.multiSelected.size;
  btnDelete.disabled = n === 0 && !state.selected;
  btnDelete.textContent = n > 1 ? `Delete (${n})` : 'Delete';
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function setStatus(msg, color = '#aaa') { statusMsg.textContent = msg; statusMsg.style.color = color; }
function dist(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

// ---------------------------------------------------------------------------
// Pitch flip helpers
// ---------------------------------------------------------------------------

function pitchKpCanvas(pk) {
  const sx = pitchCanvas.width  / state.pitchImg.naturalWidth;
  const sy = pitchCanvas.height / state.pitchImg.naturalHeight;
  let x = pk.canvas_px * sx, y = pk.canvas_py * sy;
  if (state.flipH) x = pitchCanvas.width  - x;
  if (state.flipV) y = pitchCanvas.height - y;
  return { x, y };
}

function unflipPitchClick(cx, cy) {
  let x = cx, y = cy;
  if (state.flipH) x = pitchCanvas.width  - x;
  if (state.flipV) y = pitchCanvas.height - y;
  return { x, y };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderFrame() {
  if (!state.frameImg || !state.anno) return;

  const wrap  = document.getElementById('frame-wrap');
  const scale = Math.min(wrap.clientWidth / state.anno.width, wrap.clientHeight / state.anno.height);
  frameCanvas.width  = Math.round(state.anno.width  * scale);
  frameCanvas.height = Math.round(state.anno.height * scale);

  const z = state.zoom, px = state.panX, py = state.panY;
  fCtx.setTransform(z, 0, 0, z, px, py);

  fCtx.drawImage(state.frameImg, 0, 0, frameCanvas.width, frameCanvas.height);

  const sx = frameCanvas.width  / state.anno.width;
  const sy = frameCanvas.height / state.anno.height;
  const lw = 1.5 / z;   // line width looks the same on screen regardless of zoom

  // Bboxes
  state.anno.bboxes.forEach((b, i) => {
    const x1 = b.x1 * sx, y1 = b.y1 * sy, x2 = b.x2 * sx, y2 = b.y2 * sy;
    const sel   = state.selected?.type === 'bbox' && state.selected.idx === i;
    const multi = msHas('bbox', i);
    fCtx.strokeStyle = sel ? '#fff' : multi ? '#ff9800' : (BBOX_COLORS[b.class_id] ?? '#fff');
    fCtx.lineWidth = (sel || multi ? 2.5 : lw);
    fCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = BBOX_CLASSES[b.class_id] ?? `cls${b.class_id}`;
    const conf  = b.confidence > 0 ? ` ${(b.confidence * 100).toFixed(0)}%` : '';
    fCtx.font = `${11 / z}px sans-serif`;
    fCtx.fillStyle = BBOX_COLORS[b.class_id] ?? '#fff';
    fCtx.fillText(label + conf, x1 + 2 / z, y1 - 3 / z);
  });

  // Keypoints
  const kpR = 5 / z;
  state.anno.keypoints.forEach((kp, i) => {
    const cx = kp.x * sx, cy = kp.y * sy;
    const sel   = state.selected?.type === 'kp' && state.selected.idx === i;
    const multi = msHas('kp', i);
    const color = kp.confidence <= 0 ? '#f44336' : kp.confidence < 0.5 ? '#ffeb3b' : '#4caf50';
    fCtx.beginPath(); fCtx.arc(cx, cy, kpR, 0, 2 * Math.PI);
    fCtx.fillStyle = color; fCtx.fill();
    if (sel || multi) {
      fCtx.beginPath(); fCtx.arc(cx, cy, kpR * 1.8, 0, 2 * Math.PI);
      fCtx.strokeStyle = sel ? '#fff' : '#ff9800'; fCtx.lineWidth = lw; fCtx.stroke();
    }
    fCtx.font = `${9 / z}px sans-serif`; fCtx.fillStyle = '#fff';
    fCtx.fillText(String(kp.index), cx + 6 / z, cy + 3 / z);
  });

  // Live bbox draw preview
  if (state.mode === 'draw_bbox' && state.bboxDrawStart && state.bboxDrawCurrent) {
    const s = cssToBase(state.bboxDrawStart.x,   state.bboxDrawStart.y);
    const e = cssToBase(state.bboxDrawCurrent.x, state.bboxDrawCurrent.y);
    fCtx.strokeStyle = '#fff'; fCtx.lineWidth = lw;
    fCtx.setLineDash([4 / z, 4 / z]);
    fCtx.strokeRect(s.x, s.y, e.x - s.x, e.y - s.y);
    fCtx.setLineDash([]);
  }

  // Add-KP step-1 indicator
  if (state.mode === 'add_kp' && state.addKpStep === 'pitch' && state.pendingKpPixel) {
    const b = frameToBase(state.pendingKpPixel.x, state.pendingKpPixel.y);
    fCtx.beginPath(); fCtx.arc(b.x, b.y, 7 / z, 0, 2 * Math.PI);
    fCtx.strokeStyle = '#ff69b4'; fCtx.lineWidth = lw; fCtx.stroke();
  }

  fCtx.resetTransform();
}

function renderPitch() {
  if (!state.pitchImg || !state.pitchKps.length) return;
  const wrap = document.getElementById('pitch-wrap');
  const imgW = state.pitchImg.naturalWidth, imgH = state.pitchImg.naturalHeight;
  const s = Math.min(wrap.clientWidth / imgW, wrap.clientHeight / imgH);
  pitchCanvas.width  = Math.round(imgW * s);
  pitchCanvas.height = Math.round(imgH * s);

  pCtx.save();
  if (state.flipH) { pCtx.translate(pitchCanvas.width,  0); pCtx.scale(-1,  1); }
  if (state.flipV) { pCtx.translate(0, pitchCanvas.height); pCtx.scale( 1, -1); }
  pCtx.drawImage(state.pitchImg, 0, 0, pitchCanvas.width, pitchCanvas.height);
  pCtx.restore();

  const annotatedIdx = new Set((state.anno?.keypoints ?? []).map(k => k.index));
  const selKpIdx = state.selected?.type === 'kp' && state.anno
    ? (state.anno.keypoints[state.selected.idx]?.index ?? null) : null;
  const multiKpIndices = new Set();
  state.multiSelected.forEach(k => {
    const [type, i] = k.split(':');
    if (type === 'kp' && state.anno?.keypoints[+i]) multiKpIndices.add(state.anno.keypoints[+i].index);
  });

  state.pitchKps.forEach(pk => {
    const { x: cx, y: cy } = pitchKpCanvas(pk);
    const isSel   = pk.index === selKpIdx;
    const isMulti = multiKpIndices.has(pk.index);
    const isAnn   = annotatedIdx.has(pk.index);
    pCtx.beginPath(); pCtx.arc(cx, cy, (isSel || isMulti) ? 7 : 4, 0, 2 * Math.PI);
    pCtx.fillStyle = isSel ? '#fff' : isMulti ? '#ff9800' : isAnn ? '#4caf50' : '#888'; pCtx.fill();
    if (isSel || isMulti) {
      pCtx.beginPath(); pCtx.arc(cx, cy, 11, 0, 2 * Math.PI);
      pCtx.strokeStyle = isSel ? '#fff' : '#ff9800'; pCtx.lineWidth = 2; pCtx.stroke();
    }
  });
}

// ---------------------------------------------------------------------------
// Frame loading
// ---------------------------------------------------------------------------

async function loadFrame(pos) {
  if (!state.videoId || !state.frameIndices.length) return;
  if (_loadAbort) _loadAbort.abort();
  _loadAbort = new AbortController();
  const signal = _loadAbort.signal;

  state.currentPos = pos;
  const idx = state.frameIndices[pos];
  frameCounter.textContent = `${pos + 1} / ${state.frameIndices.length}  (frame ${idx})`;
  btnPrev.disabled = pos === 0;
  btnNext.disabled = pos === state.frameIndices.length - 1;
  state.selected = null;
  msClear();
  resetZoom();
  _resetHistory();
  updateClassDropdown();
  setStatus('Loading…');

  let frameOk = false;
  await new Promise(resolve => {
    const img = new Image();
    img.onload  = () => { if (!signal.aborted) { state.frameImg = img; frameOk = true; } resolve(); };
    img.onerror = () => resolve();
    img.src = `/frame/${state.videoId}/${idx}?t=${Date.now()}`;
  });
  if (signal.aborted) return;
  if (!frameOk) { setStatus(`Frame ${idx} not readable (past end of video)`, '#f44'); return; }

  let anno = null;
  try { const r = await fetch(`/annotations/${state.videoId}/${idx}`, { signal }); if (r.ok) anno = await r.json(); } catch (_) {}
  if (signal.aborted) return;
  if (!anno) {
    setStatus('Auto-annotating…', '#ff9800');
    try { const r = await fetch(`/annotate/${state.videoId}/${idx}`, { method: 'POST', signal }); if (r.ok) anno = await r.json(); } catch (_) {}
  }
  if (signal.aborted) return;

  state.anno = anno ?? { frame_idx: idx, width: state.videoMeta.width, height: state.videoMeta.height, bboxes: [], keypoints: [] };
  const nb = state.anno.bboxes.length, nk = state.anno.keypoints.length;
  setStatus(`${nb} box${nb !== 1 ? 'es' : ''}, ${nk} keypoint${nk !== 1 ? 's' : ''}`);
  renderFrame(); renderPitch();
}

// ---------------------------------------------------------------------------
// Frame canvas — mouse events
// ---------------------------------------------------------------------------

frameCanvas.addEventListener('contextmenu', e => e.preventDefault());

frameCanvas.addEventListener('mousedown', evt => {
  if (!state.anno) return;
  evt.preventDefault();

  // ---- DRAW BBOX ----
  if (state.mode === 'draw_bbox') {
    state.bboxDrawStart   = { x: evt.offsetX, y: evt.offsetY };
    state.bboxDrawCurrent = { x: evt.offsetX, y: evt.offsetY };
    return;
  }

  // ---- ADD KP step 1 ----
  if (state.mode === 'add_kp' && state.addKpStep === null) {
    state.pendingKpPixel = cssToFrame(evt.offsetX, evt.offsetY);
    state.addKpStep = 'pitch';
    setStatus('Now click the matching point on the pitch →', '#ff69b4');
    renderFrame(); return;
  }

  // ---- SELECT / DRAG / PAN ----
  if (state.mode === 'select') {
    const base = cssToBase(evt.offsetX, evt.offsetY);
    const sx = frameCanvas.width  / state.anno.width;
    const sy = frameCanvas.height / state.anno.height;
    const hr = hitRadius();

    for (let i = 0; i < state.anno.keypoints.length; i++) {
      const kp = state.anno.keypoints[i];
      if (dist(base.x, base.y, kp.x * sx, kp.y * sy) <= hr) {
        if (evt.shiftKey) {
          msToggle('kp', i); _syncDeleteBtn(); renderFrame(); renderPitch(); return;
        }
        msClear();
        state.selected = { type: 'kp', idx: i };
        state.isDragging = true; state.dragHistoryPushed = false;
        state.dragAnchor = { x: evt.offsetX, y: evt.offsetY };
        state.dragPreState = _snap();
        updateClassDropdown(); _syncDeleteBtn();
        renderFrame(); renderPitch(); return;
      }
    }
    for (let i = 0; i < state.anno.bboxes.length; i++) {
      const b  = state.anno.bboxes[i];
      const x1 = b.x1 * sx, y1 = b.y1 * sy, x2 = b.x2 * sx, y2 = b.y2 * sy;
      if (base.x >= x1 && base.x <= x2 && base.y >= y1 && base.y <= y2) {
        if (evt.shiftKey) {
          msToggle('bbox', i); _syncDeleteBtn(); renderFrame(); renderPitch(); return;
        }
        msClear();
        state.selected = { type: 'bbox', idx: i };
        state.isDragging = true; state.dragHistoryPushed = false;
        state.dragAnchor = { x: evt.offsetX, y: evt.offsetY };
        state.dragPreState = _snap();
        updateClassDropdown(); _syncDeleteBtn();
        renderFrame(); renderPitch(); return;
      }
    }

    // Nothing hit — clear multi-select and start pan
    msClear(); state.selected = null; _syncDeleteBtn();
    updateClassDropdown();
    state.isPanning   = true;
    state.panAnchor   = { x: evt.offsetX, y: evt.offsetY };
    state.panStartX   = state.panX;
    state.panStartY   = state.panY;
    frameCanvas.style.cursor = 'grabbing';
    renderFrame(); renderPitch();
  }
});

frameCanvas.addEventListener('mousemove', evt => {
  // Live bbox draw
  if (state.mode === 'draw_bbox' && state.bboxDrawStart) {
    state.bboxDrawCurrent = { x: evt.offsetX, y: evt.offsetY };
    renderFrame(); return;
  }

  // Drag annotation
  if (state.mode === 'select' && state.isDragging && state.selected && state.dragAnchor) {
    const dx = evt.offsetX - state.dragAnchor.x;
    const dy = evt.offsetY - state.dragAnchor.y;
    if (!state.dragHistoryPushed && Math.hypot(dx, dy) > 5) {
      _hist.past.push(state.dragPreState);
      if (_hist.past.length > _hist.MAX) _hist.past.shift();
      _hist.future = []; _syncUndoRedo();
      state.dragHistoryPushed = true;
    }
    if (state.dragHistoryPushed) {
      // Convert CSS delta to frame delta (account for zoom)
      const fDx = dx / state.zoom * state.anno.width  / frameCanvas.width;
      const fDy = dy / state.zoom * state.anno.height / frameCanvas.height;
      if (state.selected.type === 'kp') {
        const orig = state.dragPreState.keypoints[state.selected.idx];
        const kp   = state.anno.keypoints[state.selected.idx];
        kp.x = clamp(orig.x + fDx, 0, state.anno.width);
        kp.y = clamp(orig.y + fDy, 0, state.anno.height);
      } else {
        const orig = state.dragPreState.bboxes[state.selected.idx];
        const b    = state.anno.bboxes[state.selected.idx];
        const w = orig.x2 - orig.x1, h = orig.y2 - orig.y1;
        b.x1 = clamp(orig.x1 + fDx, 0, state.anno.width  - w);
        b.y1 = clamp(orig.y1 + fDy, 0, state.anno.height - h);
        b.x2 = b.x1 + w; b.y2 = b.y1 + h;
      }
      renderFrame();
    }
    return;
  }

  // Pan
  if (state.isPanning && state.panAnchor) {
    state.panX = state.panStartX + (evt.offsetX - state.panAnchor.x);
    state.panY = state.panStartY + (evt.offsetY - state.panAnchor.y);
    clampPan(); renderFrame();
  }
});

frameCanvas.addEventListener('mouseup', evt => {
  // Finish bbox draw
  if (state.mode === 'draw_bbox' && state.bboxDrawStart) {
    const start = state.bboxDrawStart, end = { x: evt.offsetX, y: evt.offsetY };
    state.bboxDrawStart = null; state.bboxDrawCurrent = null;
    const sf = cssToFrame(start.x, start.y), ef = cssToFrame(end.x, end.y);
    if (Math.abs(ef.x - sf.x) < 5 || Math.abs(ef.y - sf.y) < 5) { renderFrame(); return; }
    const x1 = Math.min(sf.x, ef.x), y1 = Math.min(sf.y, ef.y);
    const x2 = Math.max(sf.x, ef.x), y2 = Math.max(sf.y, ef.y);
    const classId = parseInt(document.getElementById('bbox-class-fixed')?.value ?? '0');
    pushHistory();
    state.anno.bboxes.push({ x1, y1, x2, y2, class_id: classId, confidence: 0 });
    state.selected = { type: 'bbox', idx: state.anno.bboxes.length - 1 };
    updateClassDropdown(); _syncDeleteBtn();
    renderFrame(); renderPitch(); return;
  }
  // End annotation drag
  if (state.isDragging) { state.isDragging = false; state.dragAnchor = null; state.dragPreState = null; }
  // End pan
  if (state.isPanning) {
    state.isPanning = false; state.panAnchor = null;
    frameCanvas.style.cursor = state.zoom > 1 ? 'grab' : 'default';
  }
});

frameCanvas.addEventListener('mouseleave', () => {
  if (state.isDragging) { state.isDragging = false; state.dragAnchor = null; }
  if (state.isPanning)  { state.isPanning  = false; state.panAnchor  = null; frameCanvas.style.cursor = 'default'; }
  if (state.mode === 'draw_bbox') { state.bboxDrawStart = null; state.bboxDrawCurrent = null; renderFrame(); }
});

// Zoom with mouse wheel
frameCanvas.addEventListener('wheel', evt => {
  if (!state.anno) return;
  evt.preventDefault();
  const factor   = evt.deltaY < 0 ? 1.15 : 1 / 1.15;
  const oldZoom  = state.zoom;
  const newZoom  = clamp(oldZoom * factor, MIN_ZOOM, MAX_ZOOM);
  // Zoom centred on cursor
  state.panX = evt.offsetX - (evt.offsetX - state.panX) * (newZoom / oldZoom);
  state.panY = evt.offsetY - (evt.offsetY - state.panY) * (newZoom / oldZoom);
  state.zoom = newZoom;
  clampPan();
  frameCanvas.style.cursor = state.zoom > 1 ? 'grab' : 'default';
  renderFrame();
}, { passive: false });

// ---------------------------------------------------------------------------
// Pitch canvas — click
// ---------------------------------------------------------------------------

pitchCanvas.addEventListener('click', evt => {
  if (!state.anno || !state.pitchKps.length) return;
  const { x: ux, y: uy } = unflipPitchClick(evt.offsetX, evt.offsetY);
  const imgW = state.pitchImg.naturalWidth, imgH = state.pitchImg.naturalHeight;
  const sx = pitchCanvas.width / imgW, sy = pitchCanvas.height / imgH;
  let bestIdx = 0, bestDist = Infinity;
  state.pitchKps.forEach(pk => {
    const d = dist(ux, uy, pk.canvas_px * sx, pk.canvas_py * sy);
    if (d < bestDist) { bestDist = d; bestIdx = pk.index; }
  });

  if (state.mode === 'add_kp' && state.addKpStep === 'pitch' && state.pendingKpPixel) {
    pushHistory();
    state.anno.keypoints.push({ index: bestIdx, x: state.pendingKpPixel.x, y: state.pendingKpPixel.y, visible: true, confidence: 0 });
    state.selected = { type: 'kp', idx: state.anno.keypoints.length - 1 };
    state.addKpStep = null; state.pendingKpPixel = null;
    updateClassDropdown(); _syncDeleteBtn();
    setStatus('Keypoint added', '#4caf50'); setTimeout(() => setStatus(''), 1500);
    renderFrame(); renderPitch(); return;
  }

  if (state.mode === 'select') {
    // Try to select a nearby annotated keypoint by its pitch canvas position
    const HR = 15;
    let nearestAnnoIdx = -1, nearestAnnoDist = HR;
    state.anno.keypoints.forEach((kp, i) => {
      const pk = state.pitchKps.find(p => p.index === kp.index);
      if (!pk) return;
      const { x: cx, y: cy } = pitchKpCanvas(pk);
      const d = dist(evt.offsetX, evt.offsetY, cx, cy);
      if (d < nearestAnnoDist) { nearestAnnoDist = d; nearestAnnoIdx = i; }
    });

    if (nearestAnnoIdx >= 0) {
      if (evt.shiftKey) {
        msToggle('kp', nearestAnnoIdx); _syncDeleteBtn(); renderFrame(); renderPitch(); return;
      }
      msClear();
      state.selected = { type: 'kp', idx: nearestAnnoIdx };
      updateClassDropdown(); _syncDeleteBtn();
      renderFrame(); renderPitch(); return;
    }

    // Missed all annotated points — reassign currently selected kp's index
    if (!evt.shiftKey && state.selected?.type === 'kp') {
      pushHistory();
      state.anno.keypoints[state.selected.idx].index = bestIdx;
      updateClassDropdown(); renderFrame(); renderPitch();
    }
  }
});

// ---------------------------------------------------------------------------
// Class dropdown
// ---------------------------------------------------------------------------

function updateClassDropdown() {
  const sel = state.selected;
  if (!sel || !state.anno) {
    classLabel.style.display = 'none'; classSelect.style.display = 'none';
    selectedInfo.textContent = ''; return;
  }
  classLabel.style.display = 'inline'; classSelect.style.display = 'inline';
  const old = document.getElementById('bbox-class-fixed'); if (old) old.remove();
  if (sel.type === 'bbox') {
    const b = state.anno.bboxes[sel.idx];
    classSelect.innerHTML = BBOX_CLASSES.map((n, i) => `<option value="${i}" ${i === b.class_id ? 'selected' : ''}>${n}</option>`).join('');
    classLabel.textContent = 'Class:';
    selectedInfo.textContent = `bbox conf=${(b.confidence * 100).toFixed(0)}%`;
  } else {
    const kp = state.anno.keypoints[sel.idx];
    classSelect.innerHTML = state.pitchKps.map(pk => `<option value="${pk.index}" ${pk.index === kp.index ? 'selected' : ''}>${pk.index}: ${pk.name}</option>`).join('');
    classLabel.textContent = 'Keypoint:';
    selectedInfo.textContent = `kp #${kp.index} conf=${(kp.confidence * 100).toFixed(0)}%`;
  }
}

classSelect.addEventListener('change', () => {
  const sel = state.selected; if (!sel || !state.anno) return;
  pushHistory();
  const val = parseInt(classSelect.value);
  if (sel.type === 'bbox') state.anno.bboxes[sel.idx].class_id = val;
  if (sel.type === 'kp')   state.anno.keypoints[sel.idx].index = val;
  renderFrame(); renderPitch();
});

function ensureDrawBboxClassPicker() {
  if (document.getElementById('bbox-class-fixed')) return;
  const el = document.createElement('select');
  el.id = 'bbox-class-fixed';
  el.style.cssText = 'background:#2a2a2a;color:#e0e0e0;border:1px solid #444;border-radius:4px;padding:4px 8px;font-size:13px;';
  el.innerHTML = BBOX_CLASSES.map((n, i) => `<option value="${i}">${n}</option>`).join('');
  document.getElementById('class-row').appendChild(el);
  classLabel.style.display = 'inline'; classLabel.textContent = 'New box class:';
}

// ---------------------------------------------------------------------------
// Mode buttons
// ---------------------------------------------------------------------------

function setMode(m) {
  state.mode = m;
  state.addKpStep = null; state.pendingKpPixel = null;
  state.bboxDrawStart = null; state.bboxDrawCurrent = null; state.isDragging = false;
  btnSelect.classList.toggle('active',   m === 'select');
  btnDrawBbox.classList.toggle('active', m === 'draw_bbox');
  btnAddKp.classList.toggle('active',    m === 'add_kp');
  if (m === 'draw_bbox') { ensureDrawBboxClassPicker(); frameCanvas.style.cursor = 'crosshair'; }
  else { const old = document.getElementById('bbox-class-fixed'); if (old) old.remove(); updateClassDropdown(); frameCanvas.style.cursor = 'default'; }
  setStatus(m === 'add_kp' ? 'Click on the frame to place a keypoint' : '', '#ff69b4');
  renderFrame();
}

btnSelect.addEventListener('click',   () => setMode('select'));
btnDrawBbox.addEventListener('click', () => setMode('draw_bbox'));
btnAddKp.addEventListener('click',    () => setMode('add_kp'));

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------

btnUndo.addEventListener('click', undo);
btnRedo.addEventListener('click', redo);

// ---------------------------------------------------------------------------
// Pitch flip
// ---------------------------------------------------------------------------

btnFlipH.addEventListener('click', () => { state.flipH = !state.flipH; btnFlipH.classList.toggle('active', state.flipH); renderPitch(); });
btnFlipV.addEventListener('click', () => { state.flipV = !state.flipV; btnFlipV.classList.toggle('active', state.flipV); renderPitch(); });

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

function deleteSelected() {
  if (!state.anno) return;
  pushHistory();
  if (state.multiSelected.size > 0) {
    const bboxIdxs = [], kpIdxs = [];
    state.multiSelected.forEach(k => {
      const [type, i] = k.split(':');
      if (type === 'bbox') bboxIdxs.push(+i); else kpIdxs.push(+i);
    });
    // Also include single selected if not already in multi set
    if (state.selected?.type === 'bbox' && !msHas('bbox', state.selected.idx)) bboxIdxs.push(state.selected.idx);
    if (state.selected?.type === 'kp'   && !msHas('kp',   state.selected.idx)) kpIdxs.push(state.selected.idx);
    bboxIdxs.sort((a, b) => b - a).forEach(i => state.anno.bboxes.splice(i, 1));
    kpIdxs.sort((a, b) => b - a).forEach(i => state.anno.keypoints.splice(i, 1));
    msClear(); state.selected = null;
  } else {
    const sel = state.selected; if (!sel) return;
    if (sel.type === 'bbox') state.anno.bboxes.splice(sel.idx, 1);
    if (sel.type === 'kp')   state.anno.keypoints.splice(sel.idx, 1);
    state.selected = null;
  }
  _syncDeleteBtn(); updateClassDropdown(); renderFrame(); renderPitch();
}

btnDelete.addEventListener('click', deleteSelected);

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', evt => {
  const meta = evt.metaKey || evt.ctrlKey;
  if (meta && evt.shiftKey && (evt.key === 'z' || evt.key === 'Z')) { evt.preventDefault(); redo(); return; }
  if (meta && (evt.key === 'z' || evt.key === 'Z'))                 { evt.preventDefault(); undo(); return; }
  if (meta && (evt.key === '0'))                                     { evt.preventDefault(); resetZoom(); renderFrame(); return; }
  if ((evt.key === 'Delete' || evt.key === 'Backspace') && document.activeElement !== classSelect) deleteSelected();
  if (evt.key === 'Escape') {
    state.selected = null; state.addKpStep = null; state.pendingKpPixel = null;
    state.isDragging = false; state.isPanning = false;
    msClear(); _syncDeleteBtn(); updateClassDropdown(); setStatus(''); renderFrame(); renderPitch();
  }
  if (!meta && evt.key === 's') setMode('select');
  if (!meta && evt.key === 'b') setMode('draw_bbox');
  if (!meta && evt.key === 'k') setMode('add_kp');
  if (evt.key === 'ArrowLeft')  { evt.preventDefault(); navigateFrame(-1); }
  if (evt.key === 'ArrowRight') { evt.preventDefault(); navigateFrame(1); }
});

// ---------------------------------------------------------------------------
// Save / Export
// ---------------------------------------------------------------------------

btnSave.addEventListener('click', async () => {
  if (!state.videoId || !state.anno) return;
  const idx = state.frameIndices[state.currentPos];
  const r = await fetch(`/annotations/${state.videoId}/${idx}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state.anno) });
  setStatus(r.ok ? 'Saved ✓' : 'Save failed', r.ok ? '#4caf50' : '#f44');
  if (r.ok) setTimeout(() => setStatus(''), 1500);
});

btnClearFrame.addEventListener('click', async () => {
  if (!state.videoId || !state.anno) return;
  const idx = state.frameIndices[state.currentPos];
  const r = await fetch(`/annotations/${state.videoId}/${idx}`, { method: 'DELETE' });
  if (r.ok) {
    pushHistory();
    state.anno.bboxes = []; state.anno.keypoints = [];
    msClear(); state.selected = null; _syncDeleteBtn(); updateClassDropdown();
    renderFrame(); renderPitch();
    setStatus('Frame cleared', '#ff9800'); setTimeout(() => setStatus(''), 1500);
  }
});

btnExport.addEventListener('click', async () => {
  if (!state.videoId) return;
  if (state.anno) {
    const idx = state.frameIndices[state.currentPos];
    await fetch(`/annotations/${state.videoId}/${idx}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state.anno) });
  }
  setStatus('Exporting…', '#ff9800');
  const r = await fetch(`/export/${state.videoId}`);
  if (r.ok) {
    const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `${state.videoId}_annotations.zip`; a.click(); URL.revokeObjectURL(url);
    setStatus('Export complete ✓', '#4caf50'); setTimeout(() => setStatus(''), 2000);
  } else { setStatus(`Export failed: ${r.status}`, '#f44'); }
});

// ---------------------------------------------------------------------------
// Frame navigation
// ---------------------------------------------------------------------------

function navigateFrame(delta) {
  const next = state.currentPos + delta;
  if (next >= 0 && next < state.frameIndices.length) { loadFrame(next); saveSession(); }
}

btnPrev.addEventListener('click', () => navigateFrame(-1));
btnNext.addEventListener('click', () => navigateFrame(1));

// ---------------------------------------------------------------------------
// Background annotation progress polling
// ---------------------------------------------------------------------------

let _progressPoll = null;

function startProgressPolling(videoId) {
  if (_progressPoll) clearInterval(_progressPoll);
  progressWrap.style.display = 'flex';
  _progressPoll = setInterval(async () => {
    if (document.hidden) return;  // don't poll from background tabs
    const r = await fetch(`/progress/${videoId}`);
    if (!r.ok) return;
    const p = await r.json();
    const pct = p.total > 0 ? (p.done / p.total * 100) : 0;
    progressFill.style.width = `${pct}%`;
    progressLabel.textContent = `${p.done} / ${p.total}`;
    if (!p.running) {
      clearInterval(_progressPoll); _progressPoll = null;
      progressFill.style.width = '100%';
      setTimeout(() => { progressWrap.style.display = 'none'; }, 2000);
    }
  }, 3000);
}

// ---------------------------------------------------------------------------
// Session persistence (survives browser refresh as long as server is running)
// ---------------------------------------------------------------------------

const _SESSION_KEY = 'torchkick_session';

function saveSession() {
  if (!state.videoId) return;
  localStorage.setItem(_SESSION_KEY, JSON.stringify({
    videoId: state.videoId,
    videoMeta: state.videoMeta,
    frameIndices: state.frameIndices,
    currentPos: state.currentPos,
    intervalS: parseInt(intervalSel.value),
  }));
}

// ---------------------------------------------------------------------------
// Upload + sampling
// ---------------------------------------------------------------------------

async function applyInterval() {
  if (!state.videoId) return;
  const interval_s = parseInt(intervalSel.value);
  const r = await fetch(`/sample/${state.videoId}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ interval_s }) });
  if (r.ok) {
    const data = await r.json();
    state.frameIndices = data.frame_indices;
    setStatus(`${state.frameIndices.length} frames at ${interval_s}s interval`, '#4caf50');
    setTimeout(() => setStatus(''), 2000);
    btnPrev.disabled = false; btnNext.disabled = false; btnSave.disabled = false; btnExport.disabled = false; btnClearFrame.disabled = false;
    startProgressPolling(state.videoId);
    await loadFrame(0);
    saveSession();
  }
}

intervalSel.addEventListener('change', applyInterval);

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0]; if (!file) return;
  setStatus('Uploading…', '#ff9800');
  const form = new FormData(); form.append('file', file);
  const r = await fetch('/upload', { method: 'POST', body: form });
  if (!r.ok) { setStatus('Upload failed', '#f44'); return; }
  const meta = await r.json();
  state.videoId = meta.video_id; state.videoMeta = meta;
  setStatus(`Uploaded (${meta.total_frames} frames @ ${meta.fps.toFixed(1)}fps)`, '#4caf50');
  await applyInterval();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function initPitch() {
  const [imgR, kpR, statusR] = await Promise.all([fetch('/pitch/image'), fetch('/pitch/keypoints'), fetch('/status')]);
  if (!imgR.ok || !kpR.ok) return;
  const blob = await imgR.blob();
  state.pitchKps = await kpR.json();
  await new Promise(resolve => {
    const img = new Image();
    img.onload = () => { state.pitchImg = img; resolve(); };
    img.src = URL.createObjectURL(blob);
  });
  if (statusR.ok) {
    const s = await statusR.json();
    const badge = (ok, label) => `<span style="padding:2px 7px;border-radius:3px;font-size:11px;background:${ok ? '#1b5e20' : '#4a0000'};color:${ok ? '#a5d6a7' : '#ef9a9a'}">${label}: ${ok ? 'ready' : 'no weights'}</span>`;
    const div = document.createElement('div');
    div.style.cssText = 'display:flex;gap:5px;align-items:center;';
    div.innerHTML = badge(!!s.player_model, 'players') + badge(!!s.pitch_model, 'pitch') + `<span style="font-size:11px;color:#888">${s.device}</span>`;
    document.getElementById('toolbar').appendChild(div);
  }
  btnFlipV.classList.add('active');
  renderPitch();

  // Restore session after a browser refresh (server must still be running)
  const saved = JSON.parse(localStorage.getItem(_SESSION_KEY) ?? 'null');
  if (saved?.videoId && saved?.frameIndices?.length) {
    // Verify the server still knows about this video
    const check = await fetch(`/frame/${saved.videoId}/${saved.frameIndices[0]}`);
    if (check.ok) {
      state.videoId = saved.videoId;
      state.videoMeta = saved.videoMeta;
      state.frameIndices = saved.frameIndices;
      if (saved.intervalS) intervalSel.value = String(saved.intervalS);
      btnPrev.disabled = false; btnNext.disabled = false; btnSave.disabled = false; btnExport.disabled = false; btnClearFrame.disabled = false;
      setStatus('Resuming session…', '#ff9800');
      await loadFrame(saved.currentPos ?? 0);
      setStatus('Session restored ✓', '#4caf50');
      setTimeout(() => setStatus(''), 2000);
    } else {
      localStorage.removeItem(_SESSION_KEY);
    }
  }
}

initPitch();
