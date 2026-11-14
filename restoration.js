(() => {
'use strict';

const API_BASE = (() => {
  if (window.OLD_PHOTO_BACKEND) return String(window.OLD_PHOTO_BACKEND).replace(/\/$/, '');
  if (location.protocol === 'file:') return 'http://127.0.0.1:8765';
  if (location.port === '8765') return '';
  return 'http://127.0.0.1:8765';
})();

const bridge = window.OldPhotoRestorer;
if (!bridge) {
  console.error('OldPhotoRestorer bridge is unavailable.');
  return;
}

const STAGES = [
  [0, 'Uploading photograph...'],
  [12, 'Analyzing photograph for cracks and scratches...'],
  [28, 'Building a damage mask for damaged pixels only...'],
  [42, 'Protecting faces and undamaged detail...'],
  [58, 'Reconstructing damaged areas with LaMa AI...'],
  [80, 'Restoring photographic texture and tone...'],
  [92, 'Finalizing at the original resolution...']
];

const panel = document.getElementById('panel');
const table = document.querySelector('.table');
if (!panel || !table) return;

const group = document.createElement('div');
group.className = 'group ai-group';
group.innerHTML = `
  <h2>AI RESTORATION <span class="ai-badge">LaMa</span></h2>
  <button class="primary ai-run" id="aiRestore" type="button">
    <span class="ai-spark" aria-hidden="true">✦</span> AUTOMATIC RESTORATION
  </button>
  <label class="ai-sensitivity" for="aiSensitivity">
    <span class="ai-sens-head"><span>DAMAGE SENSITIVITY</span><output id="aiSensitivityLabel">Balanced</output></span>
    <input type="range" id="aiSensitivity" min="0" max="100" value="50" step="1">
    <small>Lower keeps more of the original photo. Higher finds fainter cracks and scratches.</small>
  </label>
  <div class="ai-status" id="aiStatus" role="status" aria-live="polite">Upload a photo, then run automatic restoration.</div>
  <div class="ai-progress" id="aiProgress" hidden>
    <div class="ai-bar"><i id="aiBar"></i></div>
    <div class="ai-progress-meta"><span id="aiStage">Ready</span><b id="aiPercent">0%</b></div>
  </div>
  <button class="ai-download" id="aiDownload" type="button" disabled>DOWNLOAD RESTORED IMAGE</button>
  <small class="ai-note" id="aiEngineNote"><b>Face preservation is always on.</b> Only damaged pixels are rebuilt; original resolution and colour are kept.</small>
`;
const actions = panel.querySelector('.actions');
panel.insertBefore(group, actions || null);

const overlay = document.createElement('div');
overlay.className = 'ai-overlay';
overlay.id = 'aiOverlay';
overlay.hidden = true;
overlay.innerHTML = `
  <div class="ai-overlay-card" role="status" aria-live="polite">
    <div class="ai-spinner" aria-hidden="true"></div>
    <b>Restoring photograph</b>
    <span id="aiOverlayStatus">Analyzing damage...</span>
    <div class="ai-overlay-bar"><i id="aiOverlayBar"></i></div>
    <div class="ai-pct" id="aiOverlayPercent">0%</div>
    <small>Your original photo stays untouched. Faces, colour and resolution are protected.</small>
  </div>
`;
table.appendChild(overlay);

const runButton = document.getElementById('aiRestore');
const downloadButton = document.getElementById('aiDownload');
const statusBox = document.getElementById('aiStatus');
const progressBox = document.getElementById('aiProgress');
const bar = document.getElementById('aiBar');
const stageLabel = document.getElementById('aiStage');
const percentLabel = document.getElementById('aiPercent');
const overlayStatus = document.getElementById('aiOverlayStatus');
const overlayBar = document.getElementById('aiOverlayBar');
const overlayPercent = document.getElementById('aiOverlayPercent');
const engineNote = document.getElementById('aiEngineNote');
const sensitivity = document.getElementById('aiSensitivity');
const sensitivityLabel = document.getElementById('aiSensitivityLabel');

let running = false;
let currentXhr = null;
let restoredBlob = null;
let restoredMeta = null;
let stageTimer = null;
let startedAt = 0;
let currentProgress = 0;

try {
  const saved = localStorage.getItem('oldphoto-ai-sensitivity');
  if (saved !== null) sensitivity.value = saved;
} catch (_) {}

function toast(message) {
  if (bridge && bridge.toast) bridge.toast(message);
}
function setSensitivityLabel() {
  const value = Number(sensitivity.value);
  sensitivityLabel.textContent = value < 34 ? 'Conservative' : value > 66 ? 'Thorough' : 'Balanced';
}
setSensitivityLabel();
sensitivity.addEventListener('input', () => {
  setSensitivityLabel();
  try { localStorage.setItem('oldphoto-ai-sensitivity', sensitivity.value); } catch (_) {}
});

function setBusy(busy) {
  running = busy;
  runButton.classList.toggle('busy', busy);
  runButton.disabled = busy || !bridge.isReady();
  runButton.setAttribute('aria-busy', busy ? 'true' : 'false');
  downloadButton.disabled = busy || !restoredBlob;
}
function setStatus(message, kind) {
  statusBox.textContent = message;
  statusBox.classList.toggle('ok', kind === 'ok');
  statusBox.classList.toggle('err', kind === 'err');
}
function stageFor(progress) {
  let text = STAGES[0][1];
  for (const [start, label] of STAGES) if (progress >= start) text = label;
  return text;
}
function setProgress(progress, label) {
  currentProgress = Math.max(currentProgress, Math.min(100, progress));
  const value = Math.round(currentProgress);
  const text = label || stageFor(currentProgress);
  bar.style.width = value + '%';
  overlayBar.style.width = value + '%';
  percentLabel.textContent = value + '%';
  overlayPercent.textContent = value + '%';
  if (text) {
    stageLabel.textContent = text;
    overlayStatus.textContent = text;
  }
}
function showOverlay(show) {
  overlay.hidden = !show;
}
function stopTimers() {
  if (stageTimer) clearInterval(stageTimer);
  stageTimer = null;
}
function startProgress() {
  currentProgress = 0;
  startedAt = performance.now();
  progressBox.hidden = false;
  setProgress(2, STAGES[0][1]);
  stageTimer = setInterval(() => {
    if (!running) return;
    const elapsed = (performance.now() - startedAt) / 1000;
    const simulated = Math.min(90, 5 + Math.log1p(elapsed) * 27);
    if (simulated > currentProgress) setProgress(simulated);
  }, 450);
}
function finishProgress() {
  stopTimers();
  setProgress(100, 'Restoration complete');
}
function showError(message) {
  stopTimers();
  running = false;
  showOverlay(false);
  progressBox.hidden = true;
  setStatus('Restoration failed: ' + message, 'err');
  setBusy(false);
  toast(message);
}
function readMeta(xhr) {
  const get = name => xhr.getResponseHeader('X-Restore-' + name);
  return {
    engine: get('Engine') || 'unknown',
    damagePct: parseFloat(get('Damage-Percent') || '0') || 0,
    damagePixels: parseInt(get('Damage-Pixels') || '0', 10) || 0,
    faces: parseInt(get('Faces') || '0', 10) || 0,
    width: parseInt(get('Width') || '0', 10) || 0,
    height: parseInt(get('Height') || '0', 10) || 0,
    ms: parseInt(get('Ms') || '0', 10) || 0,
    grayscale: get('Grayscale') === 'true',
    mode: get('Mode') || '',
    tiles: parseInt(get('Tiles') || '0', 10) || 0,
    protectedFacePixels: parseInt(get('Protected-Face-Pixels') || '0', 10) || 0
  };
}
function describe(meta) {
  if (!meta.damagePixels || meta.damagePct <= 0.0001) {
    return 'No cracks or scratches were detected. Your original photo was left unchanged.';
  }
  const engine = meta.engine.startsWith('lama') ? 'LaMa AI' : meta.engine === 'opencv-telea' ? 'OpenCV fallback' : meta.engine;
  const parts = [
    `Rebuilt ${meta.damagePct.toFixed(2)}% of the photograph (${meta.damagePixels.toLocaleString()} damaged pixels).`,
    `${meta.width} x ${meta.height} preserved.`
  ];
  if (meta.faces > 0) parts.push(`${meta.faces} face${meta.faces === 1 ? '' : 's'} detected; ${meta.protectedFacePixels.toLocaleString()} facial pixels protected.`);
  parts.push(`Engine: ${engine}.`);
  if (meta.ms) parts.push(`${(meta.ms / 1000).toFixed(1)} s.`);
  return parts.join(' ');
}
function blobText(blob) {
  if (blob && blob.text) return blob.text();
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => resolve('');
    reader.readAsText(blob);
  });
}
async function errorFromResponse(xhr) {
  try {
    const text = await blobText(xhr.response);
    if (text) {
      const data = JSON.parse(text);
      if (data && data.error) return data.error;
    }
  } catch (_) {}
  if (xhr.status === 0) return 'The restoration server could not be reached. Start the backend with start.bat, then try again.';
  return 'The restoration server returned ' + xhr.status + '.';
}
function baseName() {
  return String(bridge.getSourceName() || 'photo').replace(/\.[^.]+$/, '') || 'photo';
}

function runRestoration() {
  if (running) return;
  if (!bridge.isReady()) {
    toast('Upload a photo or try the sample first.');
    return;
  }
  const token = bridge.getLoadToken();
  setBusy(true);
  restoredBlob = null;
  restoredMeta = null;
  downloadButton.disabled = true;
  setStatus('Preparing the photograph for restoration...', '');
  showOverlay(true);
  startProgress();

  bridge.getSourceBlob().then(blob => {
    if (!blob) throw new Error('The original photograph could not be prepared.');
    if (blob.size > 60 * 1024 * 1024) {
      throw new Error('This photograph is larger than 60 MB. Please use a smaller file.');
    }
    const form = new FormData();
    form.append('file', blob, baseName());
    form.append('sensitivity', (Number(sensitivity.value) / 100).toFixed(3));

    const xhr = new XMLHttpRequest();
    currentXhr = xhr;
    xhr.open('POST', API_BASE + '/restore', true);
    xhr.responseType = 'blob';
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      const ratio = event.loaded / Math.max(1, event.total);
      setProgress(Math.min(12, 2 + ratio * 10), 'Uploading photograph...');
    };
    xhr.onload = async () => {
      currentXhr = null;
      if (xhr.status !== 200) {
        showError(await errorFromResponse(xhr));
        return;
      }
      try {
        const meta = readMeta(xhr);
        await bridge.applyAiRestored(xhr.response, token);
        restoredBlob = xhr.response;
        restoredMeta = meta;
        finishProgress();
        downloadButton.disabled = false;
        const description = describe(meta);
        setStatus(description, meta.engine === 'opencv-telea' ? 'err' : 'ok');
        if (meta.engine === 'opencv-telea') {
          setStatus(description + ' The LaMa model is unavailable, so the fallback reconstructed the marked areas only.', 'err');
        }
        if (meta.engine === 'none' || !meta.damagePixels) {
          toast('No damage detected. The original photograph is unchanged.');
        } else {
          toast('Restoration complete. Drag the divider to compare.');
        }
        setTimeout(() => {
          showOverlay(false);
          setBusy(false);
        }, 650);
      } catch (error) {
        showError(error && error.message ? error.message : 'The restored image could not be displayed.');
      }
    };
    xhr.onerror = () => {
      currentXhr = null;
      showError('The restoration server could not be reached. Start the backend with start.bat, then try again.');
    };
    xhr.ontimeout = () => {
      currentXhr = null;
      showError('Restoration took too long. Try a smaller photograph or lower the damage sensitivity.');
    };
    xhr.onabort = () => {
      currentXhr = null;
    };
    setProgress(4, 'Uploading photograph...');
    xhr.send(form);
  }).catch(error => {
    showError(error && error.message ? error.message : 'The photograph could not be prepared.');
  });
}

function resetForNewPhoto() {
  if (currentXhr) {
    try { currentXhr.abort(); } catch (_) {}
    currentXhr = null;
  }
  stopTimers();
  running = false;
  showOverlay(false);
  progressBox.hidden = true;
  currentProgress = 0;
  restoredBlob = null;
  restoredMeta = null;
  downloadButton.disabled = true;
  setBusy(false);
  if (bridge.isReady()) {
    setStatus('Photo ready. Run automatic restoration to rebuild cracks, scratches and damaged areas.', '');
  } else {
    setStatus('Upload a photo, then run automatic restoration.', '');
  }
}

runButton.addEventListener('click', runRestoration);
downloadButton.addEventListener('click', () => {
  if (!restoredBlob) return;
  const url = URL.createObjectURL(restoredBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = baseName() + '-ai-restored.png';
  link.hidden = true;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast('Downloading the restored photograph at its original resolution.');
});
window.addEventListener('oldphoto:loaded', resetForNewPhoto);

setBusy(false);
setStatus('Upload a photo, then run automatic restoration.', '');

fetch(API_BASE + '/health', {cache: 'no-store'})
  .then(response => response.ok ? response.json() : null)
  .then(data => {
    if (!data) return;
    if (data.lama_model_present) {
      engineNote.innerHTML = '<b>LaMa AI model ready.</b> Only damaged pixels are rebuilt; original resolution and colour are kept.';
    } else if (data.ai_inpainting) {
      engineNote.innerHTML = '<b>AI model will be downloaded on first use.</b> Only damaged pixels are rebuilt; original resolution and colour are kept.';
    }
  })
  .catch(() => {
    engineNote.innerHTML = '<b>Start the backend with start.bat to enable AI restoration.</b> Face preservation is always on.';
  });
})();
