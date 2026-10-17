import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Browser geometry is a visual reference only. All measurements come from /api/sample.
const $ = (selector, root = document) => root.querySelector(selector);
const container = $('#scene-container');
const workspace = $('#workspace');
const windows = $('#windows');
const state = {
  bootstrap: null, map: null, revision: 0, mapRequest: 0, request: 0,
  vehicle: null, target: null, selected: 'vehicle', placement: null,
  placementHeight: 1.5, pending: null, sample: null, sampleRevision: -1,
  busy: false, loadingMap: false, sensorView: null, overview: null, transition: null,
  transitionRequest: 0, viewRequest: 0, aligned: null, windowZ: 10, nextWindow: 0,
};
const resultWindows = new Set();
let inspectorWindow = null;
let resultsWindow = null;
let toastTimer;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#142129');
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10000);
camera.up.set(0, 0, 1);
camera.position.set(18, -22, 18);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);
container.style.position = 'relative';
container.append($('#sensor-frame'), $('#aligned-overlay'));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 1);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.maxPolarAngle = Math.PI * 0.495;
controls.update();
scene.add(new THREE.HemisphereLight(0xe4f5ff, 0x50636b, 2));
const sun = new THREE.DirectionalLight(0xffffff, 2.6);
sun.position.set(12, -18, 30);
scene.add(sun);
const mapGroup = new THREE.Group();
const entityGroup = new THREE.Group();
const helperGroup = new THREE.Group();
scene.add(mapGroup, entityGroup, helperGroup);
const entityVisuals = {};
let grid = new THREE.GridHelper(60, 30, 0x486c76, 0x2c454e);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.015;
scene.add(grid);
const axes = new THREE.AxesHelper(1);
scene.add(axes);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const fluToCamera = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0),
));

function status(message) { $('#status').textContent = message; }
function toast(message, error = false) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.toggle('error', error);
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.hidden = true; }, error ? 8500 : 4500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`服务返回了无法解析的数据（HTTP ${response.status}）`); }
  if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
  return payload;
}
function disposeTree(group) {
  group.traverse(object => {
    object.geometry?.dispose();
    if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
    else object.material?.dispose();
  });
  group.clear();
}
function poseClone(pose) {
  return pose ? { position_m: [...pose.position_m], quaternion_xyzw: [...pose.quaternion_xyzw] } : null;
}
function entityName(kind) { return kind === 'vehicle' ? '飞机' : '目标'; }
function degrees(value) { return value * 180 / Math.PI; }
function radians(value) { return value * Math.PI / 180; }
function eulerFor(pose) {
  return new THREE.Euler().setFromQuaternion(new THREE.Quaternion().fromArray(pose.quaternion_xyzw), 'ZYX');
}
function currentPose(kind = state.selected) { return state.pending?.kind === kind ? state.pending.pose : state[kind]; }
function defaultPose() {
  return { position_m: [0, 0, state.placementHeight], quaternion_xyzw: [0, 0, 0, 1] };
}
function markEdited(message) {
  state.revision += 1;
  if (state.sample) updateStaleNotices();
  if (message) status(message);
  updateInspector();
  updateControls();
}
function isStale(sample = state.sample) { return sample && sample.clientRevision !== state.revision; }
function updateControls() {
  $('#sample').disabled = !state.map || !state.vehicle || state.busy || state.loadingMap || !!state.placement;
  $('#load-map').disabled = state.busy || state.loadingMap;
  $('#upload-map').disabled = state.busy || state.loadingMap;
  $('#place-vehicle').disabled = state.loadingMap;
  $('#place-target').disabled = state.loadingMap;
  $('#open-results').disabled = !state.sample;
  $('#return-overview').disabled = !state.sensorView && !state.transition;
  $('#place-vehicle').classList.toggle('active', state.placement === 'vehicle');
  $('#place-target').classList.toggle('active', state.placement === 'target');
}

function makeSphereGeometry(radius) {
  // Browser display approximation of the analytic sphere used by the GPU sensor.
  // Tessellation is for display only; ray measurements use the exact sphere.
  const latitude = 64; const longitude = 128;
  const points = [0, 0, radius];
  for (let row = 1; row < latitude; row += 1) {
    const theta = Math.PI * row / latitude;
    for (let col = 0; col < longitude; col += 1) {
      const phi = 2 * Math.PI * col / longitude;
      points.push(Math.sin(theta) * Math.cos(phi) * radius,
        Math.sin(theta) * Math.sin(phi) * radius, Math.cos(theta) * radius);
    }
  }
  const south = points.length / 3;
  points.push(0, 0, -radius);
  const indices = [];
  for (let col = 0; col < longitude; col += 1) {
    const next = (col + 1) % longitude;
    indices.push(0, col + 1, next + 1);
    for (let row = 0; row < latitude - 2; row += 1) {
      const a = 1 + row * longitude + col;
      const b = 1 + row * longitude + next;
      indices.push(a, a + longitude, b, b, a + longitude, b + longitude);
    }
    const last = 1 + (latitude - 2) * longitude;
    indices.push(last + col, south, last + next);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}
function makeEntityVisual(kind) {
  const group = new THREE.Group();
  group.userData.entity = kind;
  const radius = state.bootstrap.config[kind === 'vehicle' ? 'vehicle_radius_m' : 'target_radius_m'];
  const color = kind === 'vehicle' ? 0x6de0c2 : 0xffb66c;
  const sphere = new THREE.Mesh(makeSphereGeometry(radius),
    new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0.06 }));
  sphere.userData.entity = kind;
  group.add(sphere);
  group.userData.physicalSphere = sphere;
  if (kind === 'vehicle') {
    const armLength = Math.max(radius * 1.5, 0.45);
    for (const angle of [Math.PI / 4, -Math.PI / 4]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(armLength * 2, 0.045, 0.035),
        new THREE.MeshStandardMaterial({ color: 0x9fc6c0 }));
      arm.rotation.z = angle;
      group.add(arm);
    }
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, radius + 0.1),
      Math.max(1, radius * 3), color, 0.25, 0.15);
    group.add(arrow);
  }
  entityGroup.add(group);
  entityVisuals[kind] = group;
  return group;
}
function refreshEntities() {
  grid.visible = !state.sensorView;
  axes.visible = !state.sensorView;
  for (const kind of ['vehicle', 'target']) {
    const pose = currentPose(kind);
    let visual = entityVisuals[kind];
    if (pose && !visual) visual = makeEntityVisual(kind);
    if (!visual) continue;
    visual.visible = !!pose && !(kind === 'vehicle' && state.sensorView);
    if (pose) {
      visual.position.fromArray(pose.position_m);
      visual.quaternion.fromArray(pose.quaternion_xyzw);
      // Analytic spheres are rotation invariant. Keep the display mesh fixed
      // while the body arrow and arms follow the parent attitude.
      visual.userData.physicalSphere.quaternion.copy(visual.quaternion).invert();
    }
  }
  disposeTree(helperGroup);
  const selected = currentPose();
  if (!selected || state.sensorView) return;
  const position = new THREE.Vector3().fromArray(selected.position_m);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.012, 6, 64),
    new THREE.MeshBasicMaterial({ color: 0xb5f4df, transparent: true, opacity: 0.7 }));
  ring.position.copy(position);
  helperGroup.add(ring);
  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(position.x, position.y, 0), position,
  ]);
  const line = new THREE.Line(lineGeometry, new THREE.LineDashedMaterial({ color: 0x79ac9f, dashSize: 0.15, gapSize: 0.1 }));
  line.computeLineDistances();
  helperGroup.add(line);
  const floorMark = new THREE.Mesh(new THREE.RingGeometry(0.15, 0.19, 32),
    new THREE.MeshBasicMaterial({ color: 0x8fceba, side: THREE.DoubleSide }));
  floorMark.position.set(position.x, position.y, 0.02);
  helperGroup.add(floorMark);
}

function openWindow(title, { width = 400, height = 430, left, top, className = '', onClose } = {}) {
  const offset = state.nextWindow++;
  const element = document.createElement('section');
  element.className = `floating-window ${className}`;
  element.style.width = `${Math.min(width, workspace.clientWidth - 20)}px`;
  element.style.height = `${Math.min(height, workspace.clientHeight - 20)}px`;
  element.style.left = `${left ?? Math.max(20, workspace.clientWidth - width - 25 - (offset % 3) * 23)}px`;
  element.style.top = `${top ?? 25 + (offset % 5) * 28}px`;
  const titlebar = document.createElement('header');
  titlebar.className = 'window-titlebar';
  const titleElement = document.createElement('span');
  titleElement.className = 'window-title';
  titleElement.textContent = title;
  const opacityLabel = document.createElement('label');
  opacityLabel.className = 'window-opacity';
  opacityLabel.textContent = '透明度';
  const opacity = document.createElement('input');
  opacity.type = 'range'; opacity.min = '15'; opacity.max = '100'; opacity.value = '100';
  opacity.setAttribute('aria-label', `${title}透明度`);
  opacityLabel.appendChild(opacity);
  opacity.addEventListener('input', () => {
    element.style.opacity = Number(opacity.value) / 100;
    if (state.aligned?.window === instance) updateAlignedOverlay();
  });
  const close = document.createElement('button'); close.textContent = '×'; close.title = '关闭窗口';
  close.setAttribute('aria-label', '关闭窗口');
  titlebar.append(titleElement, opacityLabel, close);
  const body = document.createElement('div'); body.className = 'window-body';
  element.append(titlebar, body); windows.appendChild(element);
  const instance = { element, titleElement, body, opacity, close: () => {
    if (state.aligned?.window === instance) clearAlignedOverlay();
    element.remove(); onClose?.();
  } };
  close.onclick = instance.close;
  function raise() { element.style.zIndex = ++state.windowZ; }
  raise(); element.addEventListener('pointerdown', raise);
  titlebar.addEventListener('pointerdown', event => {
    if (event.target.closest('button,input,label')) return;
    event.preventDefault();
    const startX = event.clientX; const startY = event.clientY;
    const x = element.offsetLeft; const y = element.offsetTop;
    titlebar.setPointerCapture(event.pointerId);
    function move(moveEvent) {
      element.style.left = `${Math.max(0, Math.min(workspace.clientWidth - 100, x + moveEvent.clientX - startX))}px`;
      element.style.top = `${Math.max(0, Math.min(workspace.clientHeight - 35, y + moveEvent.clientY - startY))}px`;
    }
    function end() { titlebar.removeEventListener('pointermove', move); titlebar.removeEventListener('pointerup', end); }
    titlebar.addEventListener('pointermove', move); titlebar.addEventListener('pointerup', end, { once: true });
  });
  return instance;
}

function openInspector() {
  if (inspectorWindow) { inspectorWindow.element.style.zIndex = ++state.windowZ; return; }
  inspectorWindow = openWindow('场景 · 布置与位姿', { width: 292, height: 445, left: 18, top: 18,
    onClose: () => { inspectorWindow = null; } });
  inspectorWindow.body.innerHTML = `
    <div class="entity-select"><button id="select-vehicle">飞机</button><button id="select-target">目标</button></div>
    <div class="small" id="entity-state"></div>
    <div class="section-label">世界位置 / m</div>
    <div class="coordinate-grid"><label>X<input id="entity-x" type="number" step="0.1"></label><label>Y<input id="entity-y" type="number" step="0.1"></label><label>Z<input id="entity-z" type="number" step="0.1"></label></div>
    <div class="height-row"><button id="height-down" title="降低高度">−</button><input id="height-step" type="number" value="0.25" min="0.01" step="0.05" aria-label="高度步长"><span class="small">m / 步</span><button id="height-up" title="升高高度">＋</button></div>
    <div class="section-label">姿态 / deg · ZYX</div>
    <div class="coordinate-grid"><label>Roll<input id="entity-roll" type="number" step="5"></label><label>Pitch<input id="entity-pitch" type="number" step="5"></label><label>Yaw<input id="entity-yaw" type="number" step="5"></label></div>
    <div class="field-row"><button id="reposition">重新放置 / 定向</button><button id="remove-entity" class="quiet danger">移除</button></div>
    <div class="small">按下地图确定 XY，拖动指定朝向，松开确认。水平放置始终锁定当前 Z；Esc 取消。<br>自由观察：左键旋转、右键平移、滚轮缩放。</div>
    <div class="divider"></div><div class="small" id="backend-info"></div>`;
  for (const kind of ['vehicle', 'target']) $(`#select-${kind}`).onclick = () => {
    state.selected = kind;
    if (state.placement) setPlacement(kind);
    updateInspector(); refreshEntities();
  };
  for (const axis of ['x', 'y', 'z']) $(`#entity-${axis}`).addEventListener('change', updatePoseFromFields);
  for (const axis of ['roll', 'pitch', 'yaw']) $(`#entity-${axis}`).addEventListener('change', updatePoseFromFields);
  $('#height-down').onclick = () => changeHeight(-1);
  $('#height-up').onclick = () => changeHeight(1);
  $('#reposition').onclick = () => setPlacement(state.selected);
  $('#remove-entity').onclick = () => {
    if (!state[state.selected]) return;
    const kind = state.selected; state[kind] = null;
    cancelPlacement(); markEdited(`已移除${entityName(kind)}，现有采样保留为旧快照`); refreshEntities();
  };
  updateInspector();
}
function updateInspector() {
  if (!inspectorWindow) return;
  const pose = currentPose();
  for (const kind of ['vehicle', 'target']) $(`#select-${kind}`).classList.toggle('active', state.selected === kind);
  $('#entity-state').textContent = pose ? `${entityName(state.selected)}已放置 · 调整后需要重新采样` : `${entityName(state.selected)}尚未放置 · 当前放置高度 ${state.placementHeight.toFixed(2)} m`;
  const displayPose = pose || defaultPose();
  const euler = eulerFor(displayPose);
  const values = { x: displayPose.position_m[0], y: displayPose.position_m[1], z: displayPose.position_m[2],
    roll: degrees(euler.x), pitch: degrees(euler.y), yaw: degrees(euler.z) };
  for (const [name, value] of Object.entries(values)) {
    const input = $(`#entity-${name}`);
    if (document.activeElement !== input) input.value = value.toFixed(3);
    input.disabled = !pose && name !== 'z';
  }
  $('#remove-entity').disabled = !state[state.selected];
  $('#backend-info').textContent = state.bootstrap ? `正式 GPU sensor · Torch ${state.bootstrap.torch_version} · 浏览器仅展示` : '等待后端';
}
function updatePoseFromFields() {
  const values = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'].map(name => Number($(`#entity-${name}`).value));
  if (!values.every(Number.isFinite)) { toast('位姿必须是有限数字', true); updateInspector(); return; }
  state.placementHeight = values[2];
  const existing = currentPose();
  if (!existing) { updateInspector(); return; }
  const pose = { position_m: values.slice(0, 3), quaternion_xyzw: new THREE.Quaternion()
    .setFromEuler(new THREE.Euler(...values.slice(3).map(radians), 'ZYX')).toArray() };
  state[state.selected] = pose;
  state.pending = null;
  markEdited(`${entityName(state.selected)}位姿已更新`); refreshEntities();
}
function changeHeight(sign) {
  const step = Number($('#height-step')?.value || 0.25);
  if (!Number.isFinite(step) || step <= 0) return toast('高度步长必须大于零', true);
  const pose = currentPose();
  state.placementHeight = (pose?.position_m[2] ?? state.placementHeight) + sign * step;
  if (pose) {
    const next = poseClone(pose); next.position_m[2] = state.placementHeight;
    if (state.pending) state.pending.pose = next; else state[state.selected] = next;
    markEdited(`${entityName(state.selected)}高度 ${state.placementHeight.toFixed(2)} m`); refreshEntities();
  }
  updateInspector();
  if (state.placement) updatePlacementBanner();
}
function updatePlacementBanner() {
  const banner = $('#mode-banner');
  banner.hidden = !state.placement;
  if (state.placement) banner.textContent = `放置${entityName(state.placement)} · 按下定位置 → 拖动定朝向 → 松开确认 · Z=${state.placementHeight.toFixed(2)} m · Esc 取消`;
}
function setPlacement(kind) {
  if (state.loadingMap) return;
  if (!state.map) { toast('先加载一张地图'); return; }
  if (state.sensorView || state.transition) returnOverview(false);
  state.selected = kind; state.placement = kind; state.pending = null;
  state.placementHeight = state[kind]?.position_m[2] ?? state.placementHeight;
  controls.enabled = false;
  renderer.domElement.style.cursor = 'crosshair';
  openInspector(); updateInspector(); updatePlacementBanner(); updateControls();
}
function cancelPlacement() {
  state.pending = null; state.placement = null;
  controls.enabled = !state.sensorView;
  renderer.domElement.style.cursor = '';
  updatePlacementBanner(); updateControls(); refreshEntities(); updateInspector();
}
function horizontalHit(event, height) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  return raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -height), new THREE.Vector3());
}
let selectionDown = null;
renderer.domElement.addEventListener('pointerdown', event => {
  if (!state.placement) {
    selectionDown = event.button === 0 ? [event.clientX, event.clientY] : null;
    return;
  }
  if (event.button !== 0) return;
  const hit = horizontalHit(event, state.placementHeight);
  if (!hit) return;
  event.preventDefault();
  const pose = poseClone(state[state.placement]) || defaultPose();
  pose.position_m = hit.toArray();
  state.pending = { kind: state.placement, pose, start: hit, startPixel: [event.clientX, event.clientY], pointerId: event.pointerId };
  renderer.domElement.setPointerCapture(event.pointerId);
  refreshEntities(); updateInspector();
});
renderer.domElement.addEventListener('pointermove', event => {
  if (!state.pending || event.pointerId !== state.pending.pointerId) return;
  const hit = horizontalHit(event, state.placementHeight);
  if (!hit) return;
  const delta = hit.sub(state.pending.start);
  const pixelDistance = Math.hypot(event.clientX - state.pending.startPixel[0], event.clientY - state.pending.startPixel[1]);
  if (pixelDistance > 5 && delta.lengthSq() > 1e-8) {
    const oldEuler = eulerFor(state.pending.pose);
    oldEuler.z = Math.atan2(delta.y, delta.x);
    state.pending.pose.quaternion_xyzw = new THREE.Quaternion().setFromEuler(oldEuler).toArray();
  }
  refreshEntities(); updateInspector();
});
renderer.domElement.addEventListener('pointerup', event => {
  if (!state.pending || event.pointerId !== state.pending.pointerId) {
    if (selectionDown && !state.sensorView && Math.hypot(event.clientX - selectionDown[0], event.clientY - selectionDown[1]) < 5) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(entityGroup.children, true).find(result => result.object.userData.entity);
      if (hit) { state.selected = hit.object.userData.entity; openInspector(); updateInspector(); refreshEntities(); }
    }
    selectionDown = null;
    return;
  }
  const pending = state.pending;
  state[pending.kind] = poseClone(pending.pose);
  cancelPlacement(); markEdited(`${entityName(pending.kind)}放置完成 · 按“采样”检查正式 sensor 输出`);
});
renderer.domElement.addEventListener('pointercancel', () => { if (state.pending) cancelPlacement(); });
window.addEventListener('keydown', event => { if (event.key === 'Escape') cancelPlacement(); });

function displayMap(map) {
  disposeTree(mapGroup);
  // Merge by material class, without changing or simplifying any triangle.
  const groups = new Map();
  for (const object of map.objects) {
    const kind = object.kind || 'obstacle';
    if (!groups.has(kind)) groups.set(kind, []);
    groups.get(kind).push(object);
  }
  let triangleCount = 0;
  for (const [kind, objects] of groups) {
    const faceCount = objects.reduce((sum, object) => sum + object.triangles.length, 0);
    triangleCount += faceCount;
    const vertices = new Float32Array(faceCount * 9);
    let offset = 0;
    for (const object of objects) for (const triangle of object.triangles) for (const index of triangle) {
      vertices.set(object.vertices[index], offset); offset += 3;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: kind === 'ground' ? 0x344e59 : 0x8a9a9b, roughness: 0.88, metalness: 0,
      side: THREE.DoubleSide, flatShading: true,
    });
    mapGroup.add(new THREE.Mesh(geometry, material));
  }
  state.map = map;
  $('#empty-state').hidden = true;
  $('#scene-stats').textContent = `${map.name} · ${map.objects.length.toLocaleString()} 个物体 · ${triangleCount.toLocaleString()} 个三角形 · W / m / Z↑`;
  const box = new THREE.Box3().setFromObject(mapGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 5);
  grid.geometry.dispose();
  scene.remove(grid);
  grid = new THREE.GridHelper(Math.ceil(span / 10) * 10, Math.min(100, Math.ceil(span / 2)), 0x486c76, 0x2c454e);
  grid.rotation.x = Math.PI / 2; grid.position.set(center.x, center.y, -0.015); scene.add(grid);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(span * 0.55, -span * 0.65, span * 0.55));
  camera.up.set(0, 0, 1); camera.lookAt(center); controls.update();
  controls.maxDistance = span * 10;
  state.overview = null;
  refreshEntities();
}
async function loadMap(body) {
  if (state.busy || state.loadingMap) { toast('当前请求完成后再切换地图'); return; }
  const request = ++state.mapRequest;
  state.loadingMap = true; updateControls();
  status('加载完整地图几何…');
  try {
    const response = await api('/api/map', body);
    if (request !== state.mapRequest) return;
    cancelPlacement(); returnOverview(false);
    state.vehicle = null; state.target = null;
    displayMap(response.map);
    markEdited('地图已加载。选择“＋ 飞机”，按下定位置、拖动定朝向。');
    if (![...$('#map-select').options].some(option => option.value === response.map.id)) {
      const option = document.createElement('option'); option.value = response.map.id; option.textContent = response.map.name;
      $('#map-select').appendChild(option);
    }
    $('#map-select').value = response.map.id;
    openInspector();
  } catch (error) { toast(`地图加载失败：${error.message}`, true); status('地图加载失败'); }
  finally { if (request === state.mapRequest) { state.loadingMap = false; updateControls(); } }
}

function decodeArray(field) {
  if (!field || !Array.isArray(field.shape) || ![2, 3].includes(field.shape.length)
      || !field.shape.every(size => Number.isInteger(size) && size > 0)) throw new Error('诊断数组缺少有效二维或三维 shape');
  const bytes = Uint8Array.from(atob(field.data), char => char.charCodeAt(0));
  const ArrayType = { float32: Float32Array, int32: Int32Array, uint8: Uint8Array }[field.dtype];
  if (!ArrayType) throw new Error(`未知图像类型 ${field.dtype}`);
  const result = new ArrayType(bytes.buffer);
  if (result.length !== field.shape.reduce((size, axis) => size * axis, 1)) throw new Error('诊断数组字节数与 shape 不一致');
  return result;
}
function decodeSample(sample) {
  for (const view of sample.views) {
    view.arrays = {};
    for (const field of ['raw_distance', 'raw_object_index', 'range_m', 'target_mask', 'target_coverage', 'target_only_distance',
      'target_only_object_index', 'hit_point_sensor_m', 'triangle_index', 'selected_subray', 'subray_distance',
      'subray_object_index', 'target_only_subray_distance', 'target_only_subray_object_index']) {
      if (view[field]) view.arrays[field] = decodeArray(view[field]);
    }
    if (view.hit_point_sensor_m && (view.hit_point_sensor_m.shape.length !== 3 || view.hit_point_sensor_m.shape[2] !== 3)) {
      throw new Error('命中点诊断数组必须是 H×W×3');
    }
  }
  return sample;
}
async function sampleScene() {
  if (!state.map || !state.vehicle || state.busy || state.placement) return;
  const revision = state.revision;
  const request = ++state.request;
  const payload = { request_id: `browser-${Date.now()}-${request}`, map_id: state.map.id,
    vehicle: poseClone(state.vehicle), target: state.target ? { position_m: [...state.target.position_m] } : null };
  state.busy = true; updateControls(); $('#sample').textContent = '采样中…'; $('#sample').classList.add('busy');
  status('正式 sensor 正在 GPU 上采样与后处理…');
  try {
    const response = decodeSample(await api('/api/sample', payload));
    if (request !== state.request) return;
    if (response.request_id !== payload.request_id || response.map_id !== payload.map_id) throw new Error('服务返回的采样标识与请求不一致');
    response.clientRevision = revision;
    response.requestSnapshot = payload;
    state.sample = response; state.sampleRevision = revision;
    openResults(true); updateStaleNotices();
    const queries = response.diagnostics?.geometry_info?.trace_calls;
    status(`采样 #${response.sample_id} 完成 · ${response.views.length} 路 · 场景版本 ${response.scene_version}${queries !== undefined ? ` · 累计 ${queries} 批射线查询` : ''}${isStale(response) ? ' · 场景已变化，此结果为旧快照' : ''}`);
    toast(`采样 #${response.sample_id} 完成。打开原始距离图，或“进入视角”检查。`);
  } catch (error) { toast(`采样失败：${error.message}`, true); status('采样失败；未替换已有结果'); }
  finally { state.busy = false; $('#sample').textContent = '采样'; $('#sample').classList.remove('busy'); updateControls(); }
}

const modeNames = {
  subray_distance: '子射线 · 原始距离', subray_object_index: '子射线 · 物体 ID',
  raw_distance: '归约后 · 原始距离', raw_object_index: '归约后 · 物体 ID', range_m: '最终距离',
  target_mask: '目标 mask / hint', target_coverage: '目标覆盖率',
  target_only_distance: '仅目标 · 代表距离', target_only_object_index: '仅目标 · 代表 ID',
  target_only_subray_distance: '仅目标 · 子射线距离', target_only_subray_object_index: '仅目标 · 子射线 ID',
};
function samplingLabel(view) {
  const count = view.rays_per_cell ?? 1;
  return view.sampling === 'weighted_quantile_subrays'
    ? `${count} 条/格 · 加权 ${Math.round((view.reduction?.quantile ?? .2) * 100)}% 分位`
    : `${count} 条/像素 · 最近命中`;
}
function viewModeName(view, mode) { return modeNames[mode]; }
function samplingDescription(view) {
  return view.sampling === 'weighted_quantile_subrays'
    ? '先查看子射线命中，再看加权分位选出的距离；少量子射线命中目标时，最终一格仍可能选择背景。'
    : '深度选最近子射线；目标覆盖率由全部可见目标子射线统计，可能与代表深度的物体不同。';
}
function isSubrayMode(mode) { return mode.includes('subray_'); }
function imageShape(view, mode) { return { height: view[mode].shape[0], width: view[mode].shape[1] }; }
function makeStaleNotice(sample) {
  const notice = document.createElement('div'); notice.className = 'stale-notice';
  notice.textContent = '场景已变化：当前窗口是采样时的旧快照，请重新采样。';
  notice.hidden = !isStale(sample); return notice;
}
function updateStaleNotices() {
  if (resultsWindow?.sample) resultsWindow.stale.hidden = !isStale(resultsWindow.sample);
  for (const result of resultWindows) result.stale.hidden = !isStale(result.sample);
  updateAlignedOverlay();
}
function openResults(replace = false) {
  if (!state.sample) return;
  if (resultsWindow && !replace) { resultsWindow.element.style.zIndex = ++state.windowZ; return; }
  if (resultsWindow) resultsWindow.close();
  const sample = state.sample;
  resultsWindow = openWindow(`采样 #${sample.sample_id} · 五路观测`, { width: 355, height: 590,
    onClose: () => { resultsWindow = null; } });
  resultsWindow.sample = sample;
  const stale = makeStaleNotice(sample); resultsWindow.stale = stale;
  resultsWindow.body.appendChild(stale);
  const description = document.createElement('div'); description.className = 'small';
  description.textContent = '窗口可拖动、缩放、调透明度。子射线与归约结果来自正式 sensor 的同一次采样。点击“进入视角”平滑移动到实际安装位姿。';
  resultsWindow.body.appendChild(description);
  for (const view of sample.views) {
    const card = document.createElement('div'); card.className = 'view-card';
    const heading = document.createElement('h3'); heading.textContent = view.label;
    const size = document.createElement('small'); size.textContent = `${view.config.width} × ${view.config.height} · ${view.config.hfov_deg}° / ${view.config.vfov_deg}°`;
    heading.appendChild(size); card.appendChild(heading);
    const sampling = document.createElement('div'); sampling.className = 'sampling-label';
    sampling.textContent = samplingLabel(view);
    sampling.title = samplingDescription(view);
    card.appendChild(sampling);
    const row = document.createElement('div'); row.className = 'field-row';
    const enter = document.createElement('button'); enter.textContent = '进入视角'; enter.onclick = () => flyToView(view, sample);
    row.appendChild(enter);
    for (const [mode, label] of [['subray_distance', '子射线'], ['subray_object_index', '射线 ID'],
      ['raw_distance', '归约距离'], ['range_m', '最终距离'], ['target_mask', 'mask'], ['target_coverage', '覆盖率']]) {
      if (!view.arrays[mode]) continue;
      const button = document.createElement('button'); button.textContent = label; button.onclick = () => openImageWindow(view, sample, mode); row.appendChild(button);
    }
    card.appendChild(row); resultsWindow.body.appendChild(card);
  }
  const metadata = document.createElement('div'); metadata.className = 'sample-meta';
  const info = sample.diagnostics?.geometry_info || {};
  metadata.textContent = `scene_version: ${sample.scene_version}  t_ref: ${sample.t_ref ?? 'null'}\n地图: ${sample.map_id}\n后端: ${info.backend ?? '—'}\n几何上传: ${info.geometry_uploads ?? '—'}  BVH 构建: ${info.mesh_bvh_builds ?? '—'}  变换更新: ${info.object_transform_updates ?? '—'}\n结果仅在展示边界从 GPU 下载。`;
  resultsWindow.body.appendChild(metadata);
}

const stops = [[22, 51, 83], [27, 141, 154], [138, 214, 113], [243, 204, 81], [223, 94, 57]];
function distanceColor(value, maximum) {
  if (Number.isNaN(value)) return [220, 71, 183];
  if (!Number.isFinite(value)) return [110, 179, 225];
  if (value <= 0) return [5, 12, 18];
  const location = Math.max(0, Math.min(0.999999, value / maximum)) * (stops.length - 1);
  const index = Math.floor(location); const fraction = location - index;
  return stops[index].map((channel, i) => Math.round(channel + (stops[index + 1][i] - channel) * fraction));
}
function identityColor(id) {
  if (id < 0) return [5, 12, 18];
  const color = new THREE.Color().setHSL((id * 0.61803398875 + 0.1) % 1, 0.6, 0.55);
  return [color.r * 255, color.g * 255, color.b * 255];
}
function drawImage(canvas, view, mode) {
  const { width, height } = imageShape(view, mode);
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d');
  const image = context.createImageData(width, height);
  const data = view.arrays[mode];
  for (let index = 0; index < data.length; index += 1) {
    const rgb = mode.endsWith('object_index') ? identityColor(data[index])
      : mode === 'target_mask' ? (data[index] ? [109, 224, 194] : [8, 18, 25])
      : distanceColor(data[index], mode === 'target_coverage' ? 1 : view.config.max_range_m);
    image.data.set([...rgb, 255], index * 4);
  }
  context.putImageData(image, 0, 0);
  if (mode === 'target_mask' && view.target_hint?.[0] > 0) {
    const [, left, top, right, bottom] = view.target_hint;
    context.strokeStyle = '#ffb66c'; context.lineWidth = Math.max(1, width / 160);
    context.strokeRect(left * width, top * height, (right - left) * width, (bottom - top) * height);
  }
}
function formatRange(value) {
  if (Number.isNaN(value)) return 'NaN（无效）';
  if (value === Infinity) return '+inf（已覆盖，无量程内命中）';
  if (value === -Infinity) return '-inf';
  return `${value.toFixed(5)} m`;
}
function openImageWindow(view, sample, mode) {
  const win = openWindow(`${view.label} · ${viewModeName(view, mode)} · #${sample.sample_id}`, { width: view.config.width === 8 ? 390 : 480, height: 490,
    onClose: () => { win.imageObserver?.disconnect(); resultWindows.delete(win); } });
  win.body.classList.add('image-body'); win.view = view; win.sample = sample; win.mode = mode;
  resultWindows.add(win);
  win.stale = makeStaleNotice(sample); win.body.appendChild(win.stale);
  const row = document.createElement('div'); row.className = 'result-controls';
  const selector = document.createElement('select'); selector.setAttribute('aria-label', '图像内容');
  for (const key of Object.keys(modeNames)) if (view.arrays[key]) {
    const option = document.createElement('option'); option.value = key; option.textContent = viewModeName(view, key); selector.appendChild(option);
  }
  selector.value = mode;
  const fly = document.createElement('button'); fly.textContent = '进入视角'; fly.onclick = () => flyToView(view, sample);
  const align = document.createElement('button'); align.textContent = '贴合视口'; align.title = '平滑进入传感器视角，并以本窗口透明度叠加图像';
  align.onclick = async () => {
    if (state.aligned?.window === win) { clearAlignedOverlay(); align.textContent = '贴合视口'; return; }
    const success = await flyToView(view, sample);
    if (success) {
      state.aligned = { window: win, view, sample }; win.opacity.value = '55'; win.element.style.opacity = '.55';
      align.textContent = '取消叠图'; updateAlignedOverlay();
    }
  };
  const pass = document.createElement('button'); pass.textContent = '穿透'; pass.title = '切换鼠标穿透；标题栏仍可操作，窗口内容不拦截场景鼠标';
  pass.setAttribute('aria-label', '鼠标穿透');
  pass.onclick = () => {
    win.element.classList.toggle('passthrough');
    const enabled = win.element.classList.contains('passthrough');
    pass.classList.toggle('active', enabled); pass.textContent = enabled ? '穿透中' : '穿透';
    pass.setAttribute('aria-pressed', String(enabled));
  };
  const titlebar = $('.window-titlebar', win.element);
  titlebar.insertBefore(pass, titlebar.lastElementChild);
  row.append(selector, fly, align); win.body.appendChild(row);
  const description = document.createElement('div'); description.className = 'zone-description';
  description.textContent = `${samplingLabel(view)} · ${samplingDescription(view)}`;
  win.body.appendChild(description);
  const stage = document.createElement('div'); stage.className = 'image-stage';
  const canvas = document.createElement('canvas'); canvas.setAttribute('aria-label', `${view.label}${viewModeName(view, mode)}`);
  stage.appendChild(canvas); win.body.appendChild(stage); win.canvas = canvas;
  const resizeImage = () => {
    const { width, height } = imageShape(view, win.mode);
    const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
    canvas.style.width = `${Math.floor(width * scale)}px`;
    canvas.style.height = `${Math.floor(height * scale)}px`;
  };
  win.imageObserver = new ResizeObserver(resizeImage);
  win.imageObserver.observe(stage);
  const readout = document.createElement('div'); readout.className = 'pixel-readout';
  readout.textContent = '悬停查看实际采样数据；可切换子射线与归约结果。';
  win.body.appendChild(readout);
  const legend = document.createElement('div'); legend.className = 'legend';
  legend.innerHTML = `<span>0 m</span><span class="gradient"></span><span>${view.config.max_range_m} m</span><span><i class="swatch" style="background:#dc47b7"></i>NaN</span><span><i class="swatch" style="background:#6eb3e1"></i>inf</span>`;
  win.body.appendChild(legend);
  const hint = document.createElement('div'); hint.className = 'hint-description';
  if (view.target_hint) hint.textContent = `hint [detected,left,top,right,bottom] = [${view.target_hint.map(value => Number(value).toFixed(3)).join(', ')}]\n可见比例 ${Number(view.visible_fraction).toFixed(4)} · 无遮挡子射线 ${view.unoccluded_samples ?? '—'} · 无遮挡像素 ${view.unoccluded_pixels}`;
  win.body.appendChild(hint);
  selector.onchange = () => {
    win.mode = selector.value; drawImage(canvas, view, win.mode);
    win.titleElement.textContent = `${view.label} · ${viewModeName(view, win.mode)} · #${sample.sample_id}`;
    canvas.setAttribute('aria-label', `${view.label}${viewModeName(view, win.mode)}`);
    legend.hidden = win.mode === 'target_mask' || win.mode.endsWith('object_index');
    legend.innerHTML = win.mode === 'target_coverage'
      ? '<span>0%</span><span class="gradient"></span><span>100%</span>'
      : `<span>0 m</span><span class="gradient"></span><span>${view.config.max_range_m} m</span><span><i class="swatch" style="background:#dc47b7"></i>NaN</span><span><i class="swatch" style="background:#6eb3e1"></i>inf</span>`;
    resizeImage();
    updateAlignedOverlay();
  };
  stage.addEventListener('pointermove', event => {
    const bounds = canvas.getBoundingClientRect();
    const { width, height } = imageShape(view, win.mode);
    const col = Math.floor((event.clientX - bounds.left) / bounds.width * width);
    const rowIndex = Math.floor((event.clientY - bounds.top) / bounds.height * height);
    if (col < 0 || rowIndex < 0 || col >= width || rowIndex >= height) return;
    const index = rowIndex * width + col;
    const subray = isSubrayMode(win.mode);
    const targetOnly = win.mode.startsWith('target_only');
    const idField = subray ? (targetOnly ? 'target_only_subray_object_index' : 'subray_object_index')
      : targetOnly ? 'target_only_object_index' : 'raw_object_index';
    const id = view.arrays[idField][index];
    const object = id >= 0 ? view.object_ids[id] ?? `未知 ID ${id}` : '无命中';
    const selected = win.mode === 'target_only_distance' ? `\n仅目标距离 ${formatRange(view.arrays[win.mode][index])}`
      : win.mode === 'target_only_object_index' ? `\n仅目标 ID ${view.arrays[win.mode][index]}` : '';
    const triangle = subray || targetOnly ? undefined : view.arrays.triangle_index?.[index];
    const hit = subray || targetOnly ? undefined : view.arrays.hit_point_sensor_m?.slice(index * 3, index * 3 + 3);
    let surface = '';
    if (!subray && !targetOnly) {
      if (triangle !== undefined) surface += `\n三角形 ID ${triangle >= 0 ? triangle : id >= 0 ? '解析球面' : '无'}`;
      if (hit) surface += `\n命中点 S/FLU [${id >= 0 && hit.every(Number.isFinite) ? [...hit].map(value => value.toFixed(5)).join(', ') : '无命中'}] m`;
      if (view.arrays.selected_subray) surface += `\n选中子射线 ${view.arrays.selected_subray[index]}`;
    }
    const distanceField = subray ? (targetOnly ? 'target_only_subray_distance' : 'subray_distance')
      : targetOnly ? 'target_only_distance' : 'raw_distance';
    const measurement = subray || targetOnly ? '' : ` → 最终 ${formatRange(view.arrays.range_m[index])}`;
    const coverage = !subray && view.arrays.target_coverage ? ` · 目标覆盖 ${(view.arrays.target_coverage[index] * 100).toFixed(0)}%` : '';
    readout.textContent = `[${subray ? '子射线 ' : ''}行 ${rowIndex}, 列 ${col}] · ${subray ? '' : '代表射线：'}${object} (${id})\n原始 ${formatRange(view.arrays[distanceField][index])}${measurement}${coverage}${surface}${selected}`;
  });
  selector.onchange();
  return win;
}

function localViewport() {
  const width = container.clientWidth; const height = container.clientHeight;
  if (!state.sensorView) return { left: 0, top: 0, width, height };
  const aspect = state.sensorView.config.width / state.sensorView.config.height;
  const imageWidth = Math.min(width, height * aspect);
  const imageHeight = imageWidth / aspect;
  return { left: (width - imageWidth) / 2, top: (height - imageHeight) / 2, width: imageWidth, height: imageHeight };
}
function positionOverlay(element) {
  const rect = localViewport();
  Object.assign(element.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
}
function updateProjection() {
  if (!container.clientWidth || !container.clientHeight) return;
  if (state.sensorView) {
    const config = state.sensorView.config;
    const near = 0.001;
    const horizontal = near * Math.tan(radians(config.hfov_deg / 2));
    const vertical = near * Math.tan(radians(config.vfov_deg / 2));
    // Set both independent FOVs, exactly as the backend's normalized pixel rays.
    camera.projectionMatrix.makePerspective(-horizontal, horizontal, vertical, -vertical, near, 10000);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  } else {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
  }
  const frame = $('#sensor-frame');
  frame.hidden = !state.sensorView;
  if (state.sensorView) {
    positionOverlay(frame);
    $('.sensor-tag', frame).textContent = `${state.sensorView.label} · ${state.sensorView.config.width}×${state.sensorView.config.height} · ${samplingLabel(state.sensorView)} · 浏览器参考视野`;
  }
  if (state.aligned) positionOverlay($('#aligned-overlay'));
}
function resizeScene() {
  renderer.setSize(container.clientWidth, container.clientHeight, false);
  updateProjection();
}
new ResizeObserver(resizeScene).observe(container);
window.addEventListener('resize', resizeScene);

function clearAlignedOverlay() {
  state.aligned = null;
  $('#aligned-overlay').hidden = true;
}
function updateAlignedOverlay() {
  const element = $('#aligned-overlay');
  if (!state.aligned || state.sensorView !== state.aligned.view) { element.hidden = true; return; }
  const { window: win, sample, view } = state.aligned;
  element.hidden = false;
  element.classList.toggle('zone-overlay', view.config.width === 8);
  element.style.setProperty('--zone-width', `${100 / view.config.width}%`);
  element.style.setProperty('--zone-height', `${100 / view.config.height}%`);
  drawImage($('canvas', element), view, win.mode);
  $('canvas', element).style.opacity = Number(win.opacity.value) / 100;
  $('.overlay-caption', element).textContent = `#${sample.sample_id} · ${viewModeName(view, win.mode)} · ${samplingLabel(view)} · ${isStale(sample) ? '旧场景快照 · ' : ''}叠加 ${win.opacity.value}% · 鼠标穿透`;
  positionOverlay(element);
}
function saveOverview() {
  if (state.overview) return;
  state.overview = { position: camera.position.clone(), quaternion: camera.quaternion.clone(), target: controls.target.clone() };
}
function trajectoryAt(trajectory, elapsed) {
  const times = trajectory.times;
  let left = 0; let right = times.length - 1;
  while (right - left > 1) {
    const middle = Math.floor((left + right) / 2);
    if (times[middle] <= elapsed) left = middle; else right = middle;
  }
  const ratio = Math.max(0, Math.min(1, (elapsed - times[left]) / Math.max(1e-9, times[right] - times[left])));
  const position = new THREE.Vector3().fromArray(trajectory.positions_m[left])
    .lerp(new THREE.Vector3().fromArray(trajectory.positions_m[right]), ratio);
  const progress = trajectory.progress ? trajectory.progress[left] + (trajectory.progress[right] - trajectory.progress[left]) * ratio
    : elapsed / trajectory.duration_s;
  return { position, progress: Math.max(0, Math.min(1, progress)) };
}
async function transitionTo(position, quaternion, onComplete) {
  const request = ++state.transitionRequest;
  state.transition?.resolve(false);
  state.transition = null;
  controls.enabled = false;
  const startQuaternion = camera.quaternion.clone();
  const startPosition = camera.position.clone();
  updateControls();
  try {
    const trajectory = await api('/api/transition', { start_position_m: startPosition.toArray(), end_position_m: position.toArray(), duration_s: 1.15 });
    if (request !== state.transitionRequest) return false;
    if (!Array.isArray(trajectory.times) || trajectory.times.length < 2 || !Array.isArray(trajectory.positions_m)
        || trajectory.times.length !== trajectory.positions_m.length) throw new Error('轨迹接口返回了无效采样');
    return await new Promise(resolve => {
      state.transition = { request, started: performance.now(), trajectory, startQuaternion,
        endQuaternion: quaternion, endPosition: position, onComplete, resolve };
      updateControls();
    });
  } catch (error) {
    toast(`视角切换失败：${error.message}`, true);
    if (request === state.transitionRequest) { state.transition = null; controls.enabled = !state.sensorView; updateControls(); }
    return false;
  }
}
async function flyToView(view, sample) {
  const viewRequest = ++state.viewRequest;
  if (state.placement) cancelPlacement();
  saveOverview();
  if (state.sensorView !== view) clearAlignedOverlay();
  state.sensorView = view;
  state.sensorSample = sample;
  refreshEntities(); updateProjection(); updateControls();
  const position = new THREE.Vector3().fromArray(view.pose.position_m);
  // A Three camera looks down -Z with +Y up. Sensor FLU is +X forward / +Y left / +Z up.
  const quaternion = new THREE.Quaternion().fromArray(view.pose.quaternion_xyzw).multiply(fluToCamera);
  status(`正在沿五次轨迹进入 ${view.label} · 采样 #${sample.sample_id} 的实际传感器位姿`);
  const success = await transitionTo(position, quaternion, () => {
    status(`${view.label} · 已匹配双 FOV 与安装外参 · 采样 #${sample.sample_id}${isStale(sample) ? '（旧场景快照）' : ''}`);
  });
  if (!success && state.viewRequest === viewRequest) returnOverview(false);
  return success;
}
async function returnOverview(animated = true) {
  state.viewRequest += 1;
  clearAlignedOverlay();
  if (!state.overview) {
    state.sensorView = null; state.transitionRequest += 1;
    state.transition?.resolve(false); state.transition = null;
    controls.enabled = !state.placement; updateProjection(); updateControls(); refreshEntities(); return;
  }
  const overview = state.overview;
  const finish = () => {
    state.sensorView = null; state.overview = null;
    camera.position.copy(overview.position); camera.quaternion.copy(overview.quaternion);
    controls.target.copy(overview.target); controls.enabled = !state.placement;
    camera.up.set(0, 0, 1); controls.update(); updateProjection(); updateControls(); refreshEntities();
    status('已返回总览 · 左键旋转，右键平移，滚轮缩放');
  };
  if (!animated) {
    state.transitionRequest += 1; state.transition?.resolve(false); state.transition = null; finish();
  } else {
    const success = await transitionTo(overview.position, overview.quaternion, finish);
    if (!success && state.overview === overview) finish();
  }
}

let sceneWindow = null;
function toggleSceneWindow() {
  if (sceneWindow) { sceneWindow.close(); return; }
  sceneWindow = openWindow('3D 场景 · 拖动 / 缩放 · 关闭恢复最大化', {
    width: Math.min(920, workspace.clientWidth - 60), height: Math.min(650, workspace.clientHeight - 60), left: 30, top: 25,
    onClose: () => {
      workspace.insertBefore(container, workspace.firstChild);
      sceneWindow.resizeObserver?.disconnect();
      container.style.flex = ''; sceneWindow = null;
      $('#scene-window').textContent = '场景窗口化'; resizeScene();
    },
  });
  sceneWindow.body.style.padding = '0'; sceneWindow.body.style.overflow = 'hidden'; sceneWindow.body.style.gap = '0';
  sceneWindow.body.appendChild(container); container.style.flex = '1';
  $('#scene-window').textContent = '场景最大化';
  sceneWindow.resizeObserver = new ResizeObserver(() => updateProjection());
  sceneWindow.resizeObserver.observe(sceneWindow.element);
  // The scene frame / overlay follow window movement as well as resize.
  sceneWindow.element.addEventListener('pointermove', () => updateProjection());
  resizeScene();
}

function animate(now) {
  requestAnimationFrame(animate);
  if (state.transition) {
    const transition = state.transition;
    const elapsed = Math.min((now - transition.started) / 1000, transition.trajectory.duration_s);
    const frame = trajectoryAt(transition.trajectory, elapsed);
    camera.position.copy(frame.position);
    camera.quaternion.copy(transition.startQuaternion).slerp(transition.endQuaternion, frame.progress);
    if (elapsed >= transition.trajectory.duration_s) {
      camera.position.copy(transition.endPosition); camera.quaternion.copy(transition.endQuaternion);
      state.transition = null; transition.onComplete?.(); transition.resolve(true); updateControls();
    }
  } else if (controls.enabled) controls.update();
  const view = localViewport();
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, container.clientWidth, container.clientHeight);
  renderer.setClearColor(state.sensorView ? 0x050b10 : 0x142129);
  renderer.clear();
  renderer.setViewport(view.left, container.clientHeight - view.top - view.height, view.width, view.height);
  renderer.setScissor(view.left, container.clientHeight - view.top - view.height, view.width, view.height);
  renderer.setScissorTest(true);
  renderer.render(scene, camera);
  renderer.setScissorTest(false);
}

$('#load-map').onclick = () => loadMap({ map_id: $('#map-select').value });
$('#upload-map').onclick = () => $('#map-file').click();
$('#map-file').onchange = async event => {
  const file = event.target.files[0];
  if (file) await loadMap({ filename: file.name, obj_text: await file.text() });
  event.target.value = '';
};
$('#place-vehicle').onclick = () => setPlacement('vehicle');
$('#place-target').onclick = () => setPlacement('target');
$('#sample').onclick = sampleScene;
$('#open-inspector').onclick = openInspector;
$('#open-results').onclick = () => openResults();
$('#return-overview').onclick = () => returnOverview();
$('#scene-window').onclick = toggleSceneWindow;

async function bootstrap() {
  try {
    const response = await api('/api/bootstrap');
    state.bootstrap = response;
    const select = $('#map-select'); select.replaceChildren();
    for (const map of response.maps) {
      const option = document.createElement('option'); option.value = map.id; option.textContent = map.label; select.appendChild(option);
    }
    $('#connection').textContent = 'GPU 服务'; $('#connection').classList.add('ready');
    status('后端已连接 · 请选择地图');
    const initialMap = response.maps.find(map => /(?:^|\D)20(?:\D|$)/.test(`${map.id} ${map.label}`)) || response.maps[0];
    if (initialMap) await loadMap({ map_id: initialMap.id });
    else toast('没有可用地图，可以导入 OBJ');
  } catch (error) {
    $('#connection').textContent = '未连接'; status('服务连接失败');
    toast(`无法连接在线后端：${error.message}。请检查 Docker 服务。`, true);
  }
}
resizeScene(); requestAnimationFrame(animate); bootstrap();
