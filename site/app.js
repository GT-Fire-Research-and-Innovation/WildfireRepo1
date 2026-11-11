/* GT Fire Prediction: live NIFC perimeters / incidents / VIIRS hotspots (fetched in the browser)
   and the model's forecast drawn as its own 150 m cells (data/index.json + data/forecast/*.json,
   written by live/publish.py). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ sources
  const ARC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services";
  const URLS = {
    perimeters: `${ARC}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query`,
    incidents: `${ARC}/WFIGS_Incident_Locations_Current/FeatureServer/0/query`,
    viirs: "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/Satellite_VIIRS_Thermal_Hotspots_and_Fire_Activity/FeatureServer/0/query",
  };
  const LIVE_MS = 5 * 60e3, INDEX_MS = 10 * 60e3, HOT_MIN_ZOOM = 6;
  // chance of burning, 10 classes: 1-9 %, 10-19 %, ... 90-100 %
  const RAMP = ["#fff4c2", "#ffe17a", "#ffc94a", "#ffad33", "#ff8c1f", "#f56a10", "#e2480d", "#c8300f", "#a51d14", "#6e0f12"];
  const START_FILL = "#2a211c", START_EDGE = "#b42318", ACTUAL_EDGE = "#0b7285";
  const CROWN = { 1: "#9bd49a", 2: "#2f7d32" }, EMBER = { 1: "#d6c8ff", 2: "#9f84f2", 3: "#6a3fd4" };

  // ------------------------------------------------------------------ helpers
  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids) if (k !== null && k !== undefined && k !== false) n.append(k);
    return n;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem("gtfp:" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("gtfp:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const ac = (a) => (a === null || a === undefined || isNaN(a)) ? "–" : `${Math.round(a).toLocaleString()} ac`;
  const when = (t) => {
    const d = t ? new Date(t) : null;
    return d && !isNaN(d) ? d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }) : "–";
  };
  const clock = (t) => { const d = new Date(t); return isNaN(d) ? "–" : d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" }); };
  const ago = (t) => {
    if (!t) return "";
    const m = Math.round((Date.now() - +new Date(t)) / 60000);
    if (m < 1) return "just now";
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const st = (s) => (s || "").replace("US-", "");
  async function getJSON(url, params, ms = 30000) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), ms);
    try {
      const r = await fetch(params ? `${url}?${new URLSearchParams(params)}` : url, { signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally { clearTimeout(t); }
  }
  function toast(msg, ms = 3500) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), ms); }
  const b64 = (s) => { const bin = atob(s), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; };

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  if (store.get("theme", null)) root.dataset.theme = store.get("theme");
  const dark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("#btn-theme").addEventListener("click", () => { root.dataset.theme = dark() ? "light" : "dark"; store.set("theme", root.dataset.theme); });

  // ------------------------------------------------------------------ map
  L.Popup.mergeOptions({ autoPanPadding: [16, 16], maxWidth: 320 });
  const v0 = store.get("view", [39.5, -112, 5]);
  const map = L.map("map", { zoomControl: false, worldCopyJump: true, minZoom: 3, maxZoom: 18 }).setView([v0[0], v0[1]], v0[2]);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.control.scale({ position: "bottomright" }).addTo(map);
  map.on("moveend", () => { const c = map.getCenter(); store.set("view", [+c.lat.toFixed(4), +c.lng.toFixed(4), map.getZoom()]); });
  [["overlay", 250], ["hill", 260], ["roads", 270], ["labels", 280], ["cells", 400], ["perim", 420], ["hot", 430], ["inc", 440], ["hi", 450]]
    .forEach(([n, z]) => { map.createPane(n); map.getPane(n).style.zIndex = z; });
  map.getPane("roads").style.pointerEvents = map.getPane("labels").style.pointerEvents = map.getPane("hill").style.pointerEvents = map.getPane("overlay").style.pointerEvents = "none";

  const esri = (s) => `https://server.arcgisonline.com/ArcGIS/rest/services/${s}/MapServer/tile/{z}/{y}/{x}`;
  const usgs = (s) => `https://basemap.nationalmap.gov/arcgis/rest/services/${s}/MapServer/tile/{z}/{y}/{x}`;
  const USGS_ATT = 'USGS <a href="https://www.usgs.gov/programs/national-geospatial-program/national-map" target="_blank" rel="noopener">The National Map</a>';
  const BASES = {
    usgsTopo: { name: "USGS Topo", note: "Contours, trails, roads (US)", url: usgs("USGSTopo"), o: { maxNativeZoom: 16, attribution: USGS_ATT } },
    usgsImagery: { name: "USGS Imagery + Topo", note: "Aerial photos with contours (US)", url: usgs("USGSImageryTopo"), o: { maxNativeZoom: 16, attribution: USGS_ATT } },
    esriTopo: { name: "Esri Topographic", note: "Worldwide topo", url: esri("World_Topo_Map"), o: { maxNativeZoom: 19, attribution: "Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS" } },
    imagery: { name: "Satellite imagery", note: "High-resolution aerial", url: esri("World_Imagery"), o: { maxNativeZoom: 19, attribution: "Esri, Maxar, Earthstar Geographics" } },
    openTopo: { name: "OpenTopoMap", note: "Terrain style", url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", o: { maxNativeZoom: 17, attribution: "&copy; OpenStreetMap contributors, SRTM | OpenTopoMap (CC-BY-SA)" } },
    darkGray: { name: "Dark gray", note: "Low-light canvas", url: esri("Canvas/World_Dark_Gray_Base"), o: { maxNativeZoom: 16, attribution: "Esri, HERE, Garmin, OpenStreetMap contributors" } },
  };
  const yday = new Date(Date.now() - 86400e3).toISOString().slice(0, 10);
  const OVERLAYS = {
    roads: { name: "Roads", note: "Highways and local roads", tile: [esri("Reference/World_Transportation"), { pane: "roads", maxNativeZoom: 19 }], on: false },
    labels: { name: "Place names", note: "Towns and boundaries", tile: [esri("Reference/World_Boundaries_and_Places"), { pane: "labels", maxNativeZoom: 19 }], on: false },
    hillshade: { name: "Hillshade", note: "Shaded terrain relief", tile: [esri("Elevation/World_Hillshade"), { pane: "hill", maxNativeZoom: 16, opacity: 0.45, className: "hillshade" }], on: false },
    viirsImage: { name: `NASA VIIRS image (${yday})`, note: "Yesterday's true-color satellite view: smoke plumes, burn scars",
      tile: [`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_NOAA20_CorrectedReflectance_TrueColor/default/${yday}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
        { pane: "overlay", maxNativeZoom: 9, opacity: 0.85, attribution: "NASA GIBS / VIIRS NOAA-20" }], on: false },
    hotspots: { name: "Satellite hotspots, 24 h", note: "NASA VIIRS fire detections (zoom in to load)", on: true },
    perimeters: { name: "NIFC perimeters", note: "Current interagency fire perimeters", on: true },
    incidents: { name: "Incident locations", note: "Wildfires of 10+ acres", on: true },
  };
  const MODEL = {
    pred: { name: "Predicted growth", note: "Chance each 150 m cell burns in the forecast window", on: true },
    start: { name: "Fire now", note: "Burning or burned at forecast start (perimeter + 48 h detections)", on: true },
    grid: { name: "Cell outlines", note: "Draw each forecast cell when zoomed in", on: true },
    crown: { name: "Crown fire potential", note: "Where fire would move into the canopy if it arrives", on: false },
    ember: { name: "Ember landing", note: "Where embers would land downwind of intense fire", on: false },
    actual: { name: "What actually burned", note: "Replay example only", on: true },
  };
  const saved = store.get("layers", {});
  for (const reg of [OVERLAYS, MODEL]) for (const [k, l] of Object.entries(reg)) if (saved[k] !== undefined) l.on = saved[k];
  let predOpacity = store.get("predOpacity", 0.8);
  const saveLayers = () => store.set("layers", Object.fromEntries([...Object.entries(OVERLAYS), ...Object.entries(MODEL)].map(([k, l]) => [k, l.on])));

  let baseLayer = null, baseKey = null;
  function setBase(k) {
    if (baseLayer) map.removeLayer(baseLayer);
    baseLayer = L.tileLayer(BASES[k].url, { maxZoom: 18, ...BASES[k].o }).addTo(map);
    baseKey = k; store.set("base", k);
    document.querySelectorAll('input[name="base"]').forEach((i) => (i.checked = i.value === k));
    if (cellLayer) cellLayer.redraw();
  }
  const tiles = {};
  for (const [k, l] of Object.entries(OVERLAYS)) if (l.tile) tiles[k] = L.tileLayer(l.tile[0], { maxZoom: 18, ...l.tile[1] });
  const groups = { hotspots: L.layerGroup(), perimeters: L.layerGroup(), incidents: L.layerGroup() };
  function applyOverlay(k) {
    const lay = tiles[k] || groups[k];
    OVERLAYS[k].on ? lay.addTo(map) : map.removeLayer(lay);
    if (k === "hotspots") loadHotspots();
  }

  // ------------------------------------------------------------------ layers panel
  function option(type, name, value, label, note, checked, onchange) {
    const id = `opt-${name}-${value}`;
    const i = el("input", { type, name, value, id });
    i.checked = checked;
    i.addEventListener("change", onchange);
    return el("label", { class: "opt", for: id }, i, el("b", { text: label }), note ? el("small", { text: note }) : null);
  }
  function buildPanel() {
    $("#base-list").replaceChildren(...Object.entries(BASES).map(([k, b]) => option("radio", "base", k, b.name, b.note, k === baseKey, () => setBase(k))));
    $("#overlay-list").replaceChildren(...Object.entries(OVERLAYS).map(([k, l]) => option("checkbox", "ov", k, l.name, l.note, l.on, (e) => { l.on = e.target.checked; applyOverlay(k); saveLayers(); renderLegend(); })));
    $("#model-list").replaceChildren(...Object.entries(MODEL).map(([k, l]) => option("checkbox", "md", k, l.name, l.note, l.on, (e) => setModel(k, e.target.checked))));
    const op = $("#pred-opacity"); op.value = predOpacity;
    op.oninput = () => { predOpacity = +op.value; store.set("predOpacity", predOpacity); cellLayer.redraw(); };
  }
  function setModel(k, on) {
    MODEL[k].on = on; saveLayers();
    if (k === "pred") $("#pred-toggle").checked = on;
    const box = document.getElementById(`opt-md-${k}`); if (box) box.checked = on;
    cellLayer.redraw(); renderLegend();
  }
  $("#pred-toggle").addEventListener("change", (e) => setModel("pred", e.target.checked));
  const panel = $("#layers");
  const showPanel = (on) => { panel.hidden = !on; $("#btn-layers").setAttribute("aria-expanded", String(on)); };
  $("#btn-layers").addEventListener("click", () => showPanel(panel.hidden));
  $("#layers-close").addEventListener("click", () => showPanel(false));

  // ------------------------------------------------------------------ state
  const state = { index: null, fires: new Map(), example: null, perims: new Map(), incidents: [], liveAt: null, liveCount: 0, selected: null };

  // ------------------------------------------------------------------ forecast cells (canvas, drawn in the model's own grid)
  function decodeForecast(meta, d) {
    const g = d.grid, w = g.width, h = g.height, n = w * h;
    const F = { meta, g, w, h, prob: b64(d.prob), start: b64(d.start), crown: d.crown ? b64(d.crown) : null, ember: d.ember ? b64(d.ember) : null, actual: d.actual ? b64(d.actual) : null };
    F.buckets = RAMP.map(() => []); F.startIdx = []; F.crownIdx = { 1: [], 2: [] }; F.emberIdx = { 1: [], 2: [], 3: [] }; F.probIdx = [];
    for (let i = 0; i < n; i++) {
      const p = F.prob[i];
      if (p) { F.buckets[Math.min(9, Math.floor(p / 10))].push(i); F.probIdx.push(i); }
      if (F.start[i]) F.startIdx.push(i);
      if (F.crown) { const c = F.crown[i]; if (c >= 50) F.crownIdx[2].push(i); else if (c >= 20) F.crownIdx[1].push(i); }
      if (F.ember && F.ember[i]) F.emberIdx[F.ember[i]].push(i);
    }
    F.startEdges = edges(F.start, w, h); F.actualEdges = F.actual ? edges(F.actual, w, h) : [];
    F.bounds = L.latLngBounds([g.north - h * g.dy, g.west], [g.north, g.west + w * g.dx]);
    // zoom target: the fire and every cell with a 10%+ chance
    let r0 = h, r1 = -1, c0 = w, c1 = -1;
    for (let i = 0; i < n; i++) if (F.start[i] || F.prob[i] >= 10 || (F.actual && F.actual[i])) { const r = (i / w) | 0, c = i - r * w; if (r < r0) r0 = r; if (r > r1) r1 = r; if (c < c0) c0 = c; if (c > c1) c1 = c; }
    F.focus = r1 < 0 ? F.bounds : L.latLngBounds([g.north - (r1 + 1) * g.dy, g.west + c0 * g.dx], [g.north - r0 * g.dy, g.west + (c1 + 1) * g.dx]);
    return F;
  }
  function edges(mask, w, h) {          // cell sides between in-mask and out-of-mask cells: [r, c, side]
    const out = [];
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = r * w + c; if (!mask[i]) continue;
      if (r === 0 || !mask[i - w]) out.push(r, c, 0);
      if (r === h - 1 || !mask[i + w]) out.push(r, c, 2);
      if (c === 0 || !mask[i - 1]) out.push(r, c, 3);
      if (c === w - 1 || !mask[i + 1]) out.push(r, c, 1);
    }
    return out;
  }
  const CellLayer = L.Layer.extend({
    onAdd(m) {
      this._c = L.DomUtil.create("canvas", "cells leaflet-zoom-hide");
      m.getPane("cells").appendChild(this._c);
      m.on("moveend zoomend resize viewreset", this.redraw, this);
      this.redraw();
    },
    onRemove(m) { L.DomUtil.remove(this._c); m.off("moveend zoomend resize viewreset", this.redraw, this); },
    redraw() {
      const m = this._map; if (!m) return;
      const size = m.getSize(), dpr = window.devicePixelRatio || 1, c = this._c;
      c.width = size.x * dpr; c.height = size.y * dpr; c.style.width = size.x + "px"; c.style.height = size.y + "px";
      L.DomUtil.setPosition(c, m.containerPointToLayerPoint([0, 0]));
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.x, size.y);
      for (const F of activeFires()) drawFire(ctx, m, F, size);
    },
  });
  function activeFires() { return state.fires.size ? [...state.fires.values()] : state.example ? [state.example] : []; }
  function drawFire(ctx, m, F, size) {
    const g = F.g;
    if (!m.getBounds().intersects(F.bounds)) return;
    const xs = new Float64Array(F.w + 1), ys = new Float64Array(F.h + 1);
    for (let c = 0; c <= F.w; c++) xs[c] = m.latLngToContainerPoint([g.north, g.west + c * g.dx]).x;
    for (let r = 0; r <= F.h; r++) ys[r] = m.latLngToContainerPoint([g.north - r * g.dy, g.west]).y;
    const px = xs[1] - xs[0];
    let cLo = 0, cHi = F.w - 1, rLo = 0, rHi = F.h - 1;
    while (cLo < F.w - 1 && xs[cLo + 1] < 0) cLo++;
    while (cHi > 0 && xs[cHi] > size.x) cHi--;
    while (rLo < F.h - 1 && ys[rLo + 1] < 0) rLo++;
    while (rHi > 0 && ys[rHi] > size.y) rHi--;
    const pad = px < 3 ? 0.35 : 0.15;             // a hair of overlap so neighbouring cells never show seams
    const fill = (idx) => {
      for (const i of idx) {
        const r = (i / F.w) | 0, cc = i - r * F.w;
        if (r < rLo || r > rHi || cc < cLo || cc > cHi) continue;
        ctx.fillRect(xs[cc], ys[r], xs[cc + 1] - xs[cc] + pad, ys[r + 1] - ys[r] + pad);
      }
    };
    if (MODEL.crown.on) for (const lv of [1, 2]) { ctx.globalAlpha = lv === 2 ? 0.45 : 0.3; ctx.fillStyle = CROWN[lv]; fill(F.crownIdx[lv]); }
    if (MODEL.ember.on) for (const lv of [1, 2, 3]) { ctx.globalAlpha = 0.25 + lv * 0.12; ctx.fillStyle = EMBER[lv]; fill(F.emberIdx[lv]); }
    if (MODEL.pred.on) {
      F.buckets.forEach((idx, k) => { ctx.globalAlpha = predOpacity * (k === 0 ? 0.55 : 0.9); ctx.fillStyle = RAMP[k]; fill(idx); });
      if (MODEL.grid.on && px >= 9) {
        ctx.globalAlpha = 1; ctx.strokeStyle = "rgba(40,20,10,.28)"; ctx.lineWidth = px >= 20 ? 0.8 : 0.5;
        ctx.beginPath();
        for (const i of F.probIdx) { const r = (i / F.w) | 0, cc = i - r * F.w; if (r < rLo || r > rHi || cc < cLo || cc > cHi) continue; ctx.rect(xs[cc] + 0.5, ys[r] + 0.5, xs[cc + 1] - xs[cc] - 1, ys[r + 1] - ys[r] - 1); }
        ctx.stroke();
      }
    }
    if (MODEL.start.on) {
      ctx.globalAlpha = baseKey === "darkGray" ? 0.55 : 0.72; ctx.fillStyle = baseKey === "darkGray" ? "#d9d4cc" : START_FILL; fill(F.startIdx);
      strokeEdges(ctx, F.startEdges, xs, ys, START_EDGE, px >= 6 ? 2 : 1.5);
    }
    if (MODEL.actual.on && F.actualEdges.length) strokeEdges(ctx, F.actualEdges, xs, ys, ACTUAL_EDGE, 2.2);
    ctx.globalAlpha = 1;
  }
  function strokeEdges(ctx, e, xs, ys, color, width) {
    ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "square";
    ctx.beginPath();
    for (let k = 0; k < e.length; k += 3) {
      const r = e[k], c = e[k + 1], s = e[k + 2];
      if (s === 0) { ctx.moveTo(xs[c], ys[r]); ctx.lineTo(xs[c + 1], ys[r]); }
      else if (s === 2) { ctx.moveTo(xs[c], ys[r + 1]); ctx.lineTo(xs[c + 1], ys[r + 1]); }
      else if (s === 3) { ctx.moveTo(xs[c], ys[r]); ctx.lineTo(xs[c], ys[r + 1]); }
      else { ctx.moveTo(xs[c + 1], ys[r]); ctx.lineTo(xs[c + 1], ys[r + 1]); }
    }
    ctx.stroke();
  }
  const cellLayer = new CellLayer().addTo(map);

  // ------------------------------------------------------------------ cell readout (hover on desktop, tap on phones)
  const hi = L.rectangle([[0, 0], [0, 0]], { pane: "hi", color: "#111", weight: 2, fill: false, interactive: false });
  function cellAt(latlng) {
    for (const F of activeFires()) {
      const g = F.g, c = Math.floor((latlng.lng - g.west) / g.dx), r = Math.floor((g.north - latlng.lat) / g.dy);
      if (c < 0 || r < 0 || c >= F.w || r >= F.h) continue;
      const i = r * F.w + c;
      return { F, r, c, i, p: F.prob[i], start: F.start[i], crown: F.crown ? F.crown[i] : 0, ember: F.ember ? F.ember[i] : 0, actual: F.actual ? F.actual[i] : 0 };
    }
    return null;
  }
  function showCell(latlng) {
    const ro = $("#readout"), cell = cellAt(latlng);
    if (!cell || (!cell.p && !cell.start && !cell.actual)) { ro.hidden = true; map.removeLayer(hi); return false; }
    const { F, r, c } = cell, g = F.g;
    hi.setBounds([[g.north - (r + 1) * g.dy, g.west + c * g.dx], [g.north - r * g.dy, g.west + (c + 1) * g.dx]]).addTo(map);
    const hrs = F.meta.example ? F.meta.hours : (state.index?.horizon_hours || 24);
    const parts = [];
    if (cell.start) parts.push(`<b>Fire now</b> <span>burning or burned at forecast start</span>`);
    else parts.push(`<b>${cell.p}%</b> chance of burning <span>in the next ${hrs} h</span>`);
    if (cell.crown >= 50 && !cell.start) parts.push("<span>· crown fire likely</span>");
    if (cell.ember >= 2 && !cell.start) parts.push("<span>· ember landing</span>");
    if (cell.actual) parts.push("<span>· really burned</span>");
    ro.innerHTML = parts.join(" ");
    ro.hidden = false;
    return true;
  }
  map.on("mousemove", (e) => { if (MODEL.pred.on || MODEL.start.on) showCell(e.latlng); });
  map.on("mouseout", () => { $("#readout").hidden = true; map.removeLayer(hi); });
  map.on("click", (e) => { if (showCell(e.latlng)) { clearTimeout(showCell.t); showCell.t = setTimeout(() => { $("#readout").hidden = true; map.removeLayer(hi); }, 6000); } });

  // ------------------------------------------------------------------ live: perimeters
  const perimStyle = (p) => {
    const rx = p.attr_IncidentTypeCategory === "RX";
    return { pane: "perim", color: rx ? "#8a5a2b" : "#c1121f", weight: 2, dashArray: rx ? "5 4" : null, fill: true, fillColor: rx ? "#8a5a2b" : "#c1121f", fillOpacity: 0.06 };
  };
  async function loadPerimeters() {
    try {
      const gj = await getJSON(URLS.perimeters, {
        where: "1=1", outSR: 4326, f: "geojson", geometryPrecision: 5, maxAllowableOffset: 0.0002, resultRecordCount: 2000,
        outFields: "attr_UniqueFireIdentifier,attr_IncidentName,poly_GISAcres,attr_PercentContained,poly_PolygonDateTime,poly_MapMethod,attr_IncidentTypeCategory,attr_POOState",
      });
      groups.perimeters.clearLayers(); state.perims.clear();
      L.geoJSON(gj, {
        style: (f) => perimStyle(f.properties),
        onEachFeature: (f, lay) => { state.perims.set(f.properties.attr_UniqueFireIdentifier, lay); lay.bindPopup(() => perimPopup(f.properties)); },
      }).addTo(groups.perimeters);
      state.liveAt = Date.now(); state.liveCount = gj.features.length; freshLive();
    } catch (e) { $("#fresh-live").textContent = "NIFC unreachable, retrying"; $("#fresh-live").classList.add("stale"); console.warn(e); }
  }
  function freshLive() { if (state.liveAt) { const d = $("#fresh-live"); d.textContent = `${ago(state.liveAt)} · ${state.liveCount} fires`; d.classList.remove("stale"); } }
  const perimPopup = (p) => `<div class="pop"><h5>${esc(p.attr_IncidentName)}</h5><dl>
    <dt>Size</dt><dd>${ac(p.poly_GISAcres)}</dd><dt>Contained</dt><dd>${p.attr_PercentContained ?? "–"}%</dd>
    <dt>Perimeter</dt><dd>${when(p.poly_PolygonDateTime)}</dd><dt>Mapped by</dt><dd>${esc(p.poly_MapMethod || "–")}</dd>
    <dt>Type</dt><dd>${p.attr_IncidentTypeCategory === "RX" ? "Prescribed" : "Wildfire"}</dd></dl></div>`;

  // ------------------------------------------------------------------ live: incidents
  async function loadIncidents() {
    try {
      const gj = await getJSON(URLS.incidents, {
        where: "IncidentTypeCategory='WF' AND IncidentSize>=10", outSR: 4326, f: "geojson", resultRecordCount: 2000,
        outFields: "IncidentName,IncidentSize,PercentContained,FireDiscoveryDateTime,ModifiedOnDateTime_dt,POOState,UniqueFireIdentifier,TotalIncidentPersonnel,FireBehaviorGeneral",
      });
      groups.incidents.clearLayers(); state.incidents = [];
      L.geoJSON(gj, {
        pointToLayer: (f, ll) => L.circleMarker(ll, { pane: "inc", radius: Math.max(4.5, Math.min(15, 3 + Math.sqrt(f.properties.IncidentSize || 0) / 20)), color: "#fff", weight: 1.5,
          fillColor: (f.properties.PercentContained || 0) >= 100 ? "#7d8790" : "#d9480f", fillOpacity: 0.9 }),
        onEachFeature: (f, lay) => {
          const p = f.properties;
          state.incidents.push({ ...p, lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], layer: lay });
          lay.bindPopup(() => `<div class="pop"><h5>${esc(p.IncidentName)}</h5><dl>
            <dt>Size</dt><dd>${ac(p.IncidentSize)}</dd><dt>Contained</dt><dd>${p.PercentContained ?? "–"}%</dd>
            <dt>State</dt><dd>${esc(st(p.POOState))}</dd><dt>Discovered</dt><dd>${when(p.FireDiscoveryDateTime)}</dd>
            ${p.TotalIncidentPersonnel ? `<dt>Personnel</dt><dd>${p.TotalIncidentPersonnel.toLocaleString()}</dd>` : ""}
            <dt>Updated</dt><dd>${when(p.ModifiedOnDateTime_dt)}</dd></dl></div>`);
          lay.on("click", () => select(p.UniqueFireIdentifier, false));
        },
      }).addTo(groups.incidents);
      renderList();
    } catch (e) { console.warn(e); }
  }

  // ------------------------------------------------------------------ live: VIIRS hotspots in view
  let hotT = null, hotSeq = 0;
  const HOT = [[6, "#ffe066"], [12, "#ff922b"], [24, "#d6336c"]];
  function loadHotspots() {
    clearTimeout(hotT);
    hotT = setTimeout(async () => {
      groups.hotspots.clearLayers();
      if (!OVERLAYS.hotspots.on || map.getZoom() < HOT_MIN_ZOOM) return;
      const b = map.getBounds().pad(0.15), seq = ++hotSeq;
      try {
        const gj = await getJSON(URLS.viirs, {
          where: "hours_old<=24", geometry: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(4)).join(","),
          geometryType: "esriGeometryEnvelope", inSR: 4326, spatialRel: "esriSpatialRelIntersects", outSR: 4326, f: "geojson",
          outFields: "esritimeutc,hours_old,frp,confidence,satellite,daynight", resultRecordCount: 8000,
        });
        if (seq !== hotSeq) return;
        const rend = L.canvas({ pane: "hot" });
        L.geoJSON(gj, {
          pointToLayer: (f, ll) => L.circleMarker(ll, { renderer: rend, radius: 4, color: "#3b1d0a", weight: 0.7, fillColor: (HOT.find(([h]) => (f.properties.hours_old ?? 24) <= h) || HOT[2])[1], fillOpacity: 0.95 }),
          onEachFeature: (f, lay) => lay.bindPopup(() => { const p = f.properties; return `<div class="pop"><h5>Satellite hotspot</h5><dl>
            <dt>Detected</dt><dd>${when(p.esritimeutc)}</dd><dt>Age</dt><dd>${p.hours_old ?? "?"} h</dd><dt>Fire power</dt><dd>${p.frp ?? "–"} MW</dd>
            <dt>Satellite</dt><dd>${esc(p.satellite ?? "–")} · ${p.daynight === "N" ? "night" : "day"}</dd></dl><p>375 m pixel: the fire is somewhere inside it.</p></div>`; }),
        }).addTo(groups.hotspots);
      } catch (e) { console.warn(e); }
    }, 350);
  }
  map.on("moveend", loadHotspots);

  // ------------------------------------------------------------------ model forecasts
  async function loadIndex() {
    let idx;
    try { idx = await getJSON(`data/index.json?t=${Date.now()}`); } catch { idx = null; }
    state.index = idx;
    if (!idx) { $("#fresh-model").textContent = "not published yet"; $("#model-note").textContent = "No forecast run has been published yet."; }
    else {
      const n = idx.forecast_count || 0;
      $("#fresh-model").textContent = `${clock(idx.generated)} · ${n} forecast${n === 1 ? "" : "s"}`;
      $("#pred-sub").textContent = `next ${idx.horizon_hours || 24} h, chance each cell burns`;
      $("#model-note").textContent = `Model: ${idx.model?.name || "–"}. ${idx.model?.note || ""}${idx.model?.test_ap ? ` Test score (average precision) ${idx.model.test_ap}.` : ""}`;
      const want = idx.fires.filter((f) => f.forecast);
      for (const id of [...state.fires.keys()]) if (!want.some((f) => f.id === id)) state.fires.delete(id);
      await Promise.all(want.map(async (f) => {
        try { state.fires.set(f.id, decodeForecast(f, await getJSON(`${f.forecast}?t=${encodeURIComponent(idx.generated)}`))); }
        catch (e) { console.warn("forecast", f.id, e); }
      }));
    }
    if (!state.fires.size && !state.example) {
      try { const ex = await getJSON("data/example/index.json"); state.example = decodeForecast(ex, await getJSON(`data/example/${ex.file}`)); }
      catch { /* none shipped */ }
    }
    if (state.example) $("#pred-sub").textContent = state.fires.size ? $("#pred-sub").textContent : `example replay: chance each cell burned in ${state.example.meta.hours} h`;
    cellLayer.redraw(); renderList(); renderLegend();
    if (state.selected) renderDetail(state.selected);
  }

  // ------------------------------------------------------------------ side: list + detail
  function renderList() {
    const q = $("#search").value.trim().toLowerCase();
    const match = (n, s) => !q || (n || "").toLowerCase().includes(q) || st(s).toLowerCase().includes(q);
    const box = $("#fire-list"); box.replaceChildren();
    const fc = [...state.fires.values()].map((F) => F.meta).filter((m) => match(m.name, m.state))
      .sort((a, b) => (b.stats?.expected_new_acres || 0) - (a.stats?.expected_new_acres || 0));
    if (fc.length) {
      box.append(el("p", { class: "group-label", text: `Forecast available (${fc.length})` }));
      fc.forEach((m) => box.append(row(m.id, m.name, m.state, m.acres, m.contained, `perimeter ${ago(m.perimeter_time || m.updated)}`, m.stats)));
    } else if (state.example && match(state.example.meta.name, "")) {
      const m = state.example.meta;
      box.append(el("p", { class: "group-label", text: "No fire is actively spreading · example" }));
      box.append(row("__example", m.name, "", m.acres, null, "replay of a past forecast", m.stats, m.hours));
    }
    const ids = new Set(fc.map((m) => m.id));
    const inc = state.incidents.filter((p) => !ids.has(p.UniqueFireIdentifier) && match(p.IncidentName, p.POOState)).sort((a, b) => (b.IncidentSize || 0) - (a.IncidentSize || 0));
    const open = inc.filter((p) => (p.PercentContained ?? 0) < 100), done = inc.filter((p) => (p.PercentContained ?? 0) >= 100);
    const r = (p) => row(p.UniqueFireIdentifier, p.IncidentName, p.POOState, p.IncidentSize, p.PercentContained, `updated ${ago(p.ModifiedOnDateTime_dt)}`);
    if (open.length) { box.append(el("p", { class: "group-label", text: `Not contained (${open.length})` })); open.slice(0, 200).forEach((p) => box.append(r(p))); }
    if (done.length) {
      const d = el("details", { class: "contained" }, el("summary", { class: "group-label", text: `Contained (${done.length})` }));
      done.slice(0, 200).forEach((p) => d.append(r(p))); if (q) d.open = true; box.append(d);
    }
    if (!box.childElementCount) box.append(el("p", { class: "empty", text: q ? "No fires match." : "No wildfires of 10+ acres reported right now." }));
  }
  function row(id, name, s, acres, contained, sub, stats, hours) {
    const bar = el("span", { class: "bar" }, el("b")); bar.firstChild.style.height = `${Math.max(0, Math.min(100, contained || 0))}%`;
    const hrs = hours || state.index?.horizon_hours || 24;
    return el("button", { class: "row" + (state.selected === id ? " sel" : ""), "data-id": id, onclick: () => select(id, true) },
      bar, el("span", { class: "nm", text: name || "Unnamed fire" }), el("span", { class: "ac", text: ac(acres) }),
      el("span", { class: "sub", text: [st(s), contained !== null && contained !== undefined ? `${contained}% contained` : null, sub].filter(Boolean).join(" · ") }),
      stats && stats.expected_new_acres !== undefined ? el("span", { class: "growth", text: `+${stats.expected_new_acres.toLocaleString()} ac expected · ${hrs} h` }) : null);
  }
  $("#search").addEventListener("input", renderList);

  function renderDetail(id) {
    const box = $("#detail");
    const F = id === "__example" ? state.example : state.fires.get(id);
    const inc = state.incidents.find((p) => p.UniqueFireIdentifier === id);
    const per = state.perims.get(id);
    const m = F ? F.meta : null;
    const name = m?.name || inc?.IncidentName || per?.feature?.properties?.attr_IncidentName;
    if (!name) { box.hidden = true; return; }
    const s = m?.stats || {};
    const hrs = m?.example ? m.hours : (state.index?.horizon_hours || 24);
    const fig = (label, val, hot) => el("div", {}, el("dt", { text: label }), el("dd", { class: hot ? "hot" : "", text: val }));
    const figs = el("dl", { class: "figs" },
      fig("Size now", ac(m?.acres ?? inc?.IncidentSize ?? per?.feature?.properties?.poly_GISAcres)),
      m?.example ? fig("Forecast window", `${m.hours} h`) : fig("Contained", `${m?.contained ?? inc?.PercentContained ?? "–"}%`));
    if (F) figs.append(fig(`Expected growth, ${hrs} h`, ac(s.expected_new_acres), true), fig("Likely (50%+ cells)", ac(s.likely_acres)),
      fig("Possible (10%+ cells)", ac(s.possible_acres)), s.actual_new_acres !== undefined ? fig("Actually burned", ac(s.actual_new_acres)) : fig("Forecast from", when(m.origin)));
    const w = s.wind;
    box.replaceChildren(...[
      el("button", { class: "close", onclick: () => { state.selected = null; box.hidden = true; renderList(); }, text: "Close" }),
      el("h3", { text: name }),
      el("p", { class: "where", text: [st(m?.state || inc?.POOState), m?.example ? "example replay" : null, inc ? `updated ${ago(inc.ModifiedOnDateTime_dt)}` : null].filter(Boolean).join(" · ") }),
      figs,
      w ? el("p", { class: "wind", text: `Forecast wind: from ${w.from_compass} at ${w.mean_mph} mph, gusts to ${w.peak_gust_mph} mph, humidity down to ${w.min_rh_pct}%` }) : null,
      F ? el("p", { class: "note", text: m.example ? m.note : `Forecast from ${when(m.origin)} to ${when(new Date(+new Date(m.origin) + hrs * 3600e3))}. Weather: HRRR run ${when(m.hrrr_run)}.` })
        : el("p", { class: "noforecast", text: "No forecast for this fire: fewer than 3 satellite fire detections near it in the last 48 hours, so it does not look actively spreading. Forecasts refresh every 2 hours." })].filter(Boolean));
    box.hidden = false;
  }
  function select(id, fly) {
    state.selected = id;
    document.querySelectorAll(".row").forEach((b) => b.classList.toggle("sel", b.dataset.id === id));
    renderDetail(id);
    try { history.replaceState(null, "", `#fire=${encodeURIComponent(id)}`); } catch { /* sandboxed */ }
    if (!fly) return;
    const F = id === "__example" ? state.example : state.fires.get(id);
    const per = state.perims.get(id), inc = state.incidents.find((p) => p.UniqueFireIdentifier === id);
    const pad = { paddingTopLeft: [24, 70], paddingBottomRight: [24, innerWidth <= 780 ? innerHeight * 0.4 : 24], maxZoom: 14 };
    if (F) map.flyToBounds(F.focus, pad);
    else if (per) map.flyToBounds(per.getBounds(), pad);
    else if (inc) map.flyTo([inc.lat, inc.lon], 12);
    if (innerWidth <= 780) setSheet("min");
  }

  // ------------------------------------------------------------------ legend
  function renderLegend() {
    const lg = $("#legend"); lg.replaceChildren();
    const fires = activeFires().length;
    if (MODEL.pred.on && fires) {
      const hrs = state.fires.size ? (state.index?.horizon_hours || 24) : state.example?.meta.hours;
      lg.append(el("h4", { text: `Chance each cell burns, next ${hrs} h` }));
      const ramp = el("div", { class: "ramp" }); RAMP.forEach((c) => { const s = el("span"); s.style.background = c; ramp.append(s); });
      lg.append(ramp, el("div", { class: "ticks" }, el("span", { text: "1%" }), el("span", { text: "50%" }), el("span", { text: "100%" })));
    }
    const keys = [];
    if (MODEL.start.on && fires) keys.push([START_FILL, "Fire now", START_EDGE]);
    if (MODEL.actual.on && !state.fires.size && state.example) keys.push(["transparent", "What actually burned", ACTUAL_EDGE]);
    if (MODEL.crown.on && fires) keys.push([CROWN[2], "Crown fire likely"]);
    if (MODEL.ember.on && fires) keys.push([EMBER[2], "Ember landing"]);
    if (OVERLAYS.perimeters.on) keys.push(["transparent", "NIFC perimeter", "#c1121f"]);
    if (OVERLAYS.hotspots.on) keys.push(["#ffe066", "Hotspot under 6 h", null, true], ["#d6336c", "Hotspot 12–24 h", null, true]);
    if (OVERLAYS.incidents.on) keys.push(["#d9480f", "Incident (grey: contained)", null, true]);
    if (keys.length && lg.childElementCount) lg.append(el("div", { class: "sep" }));
    for (const [fillc, label, edge, dot] of keys) {
      const i = el("i", { class: dot ? "dot" : "" }); i.style.background = fillc; if (edge) i.style.boxShadow = `inset 0 0 0 2px ${edge}`;
      lg.append(el("div", { class: "key" }, i, el("span", { text: label })));
    }
    lg.hidden = !lg.childElementCount;
  }

  // ------------------------------------------------------------------ phone sheet, locate
  const side = $("#side");
  function setSheet(mode) {
    side.classList.toggle("open", mode === "open"); side.classList.toggle("min", mode === "min");
    document.body.classList.toggle("sheet-open", mode === "open"); document.body.classList.toggle("sheet-min", mode === "min");
  }
  $("#grip").addEventListener("click", () => setSheet(side.classList.contains("open") ? "min" : side.classList.contains("min") ? "" : "open"));
  $("#search").addEventListener("focus", () => { if (innerWidth <= 780) setSheet("open"); });
  let me = null;
  $("#btn-locate").addEventListener("click", () => map.locate({ setView: true, maxZoom: 12 }));
  map.on("locationfound", (e) => { if (me) map.removeLayer(me); me = L.circleMarker(e.latlng, { pane: "hi", radius: 7, color: "#fff", weight: 2, fillColor: "#1c64f2", fillOpacity: 1 }).addTo(map).bindPopup("You are here"); });
  map.on("locationerror", () => toast("Location not available (permission denied or no signal)"));

  // ------------------------------------------------------------------ start
  setBase(BASES[store.get("base", "usgsTopo")] ? store.get("base", "usgsTopo") : "usgsTopo");
  for (const k of Object.keys(OVERLAYS)) if (OVERLAYS[k].on) applyOverlay(k);
  $("#pred-toggle").checked = MODEL.pred.on;
  buildPanel(); renderLegend();
  (async () => {
    await Promise.all([loadPerimeters(), loadIncidents(), loadIndex()]);
    renderLegend(); loadHotspots();
    const h = decodeURIComponent((location.hash.match(/fire=([^&]+)/) || [])[1] || "");
    if (h) select(h, true);
  })();
  setInterval(() => { loadPerimeters(); loadIncidents(); loadHotspots(); }, LIVE_MS);
  setInterval(loadIndex, INDEX_MS);
  setInterval(freshLive, 30e3);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.liveAt && Date.now() - state.liveAt > LIVE_MS) { loadPerimeters(); loadIncidents(); loadHotspots(); loadIndex(); } });
})();
