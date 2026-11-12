/* GT Fire Prediction (MapLibre): vector basemap with shaded relief; live NIFC perimeters,
   incidents and VIIRS hotspots fetched in the browser; the model's forecast drawn as its own
   150 m cells (data/index.json + data/forecast/*.json, written by live/publish.py). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ sources
  const ARC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services";
  const URLS = {
    perimeters: `${ARC}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query`,
    perimetersYTD: `${ARC}/WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query`,
    incidents: `${ARC}/WFIGS_Incident_Locations_Current/FeatureServer/0/query`,
    viirs: "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/Satellite_VIIRS_Thermal_Hotspots_and_Fire_Activity/FeatureServer/0/query",
  };
  const OFM = "https://tiles.openfreemap.org/styles/";
  const DEM = { type: "raster-dem", tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"], encoding: "terrarium", tileSize: 256, maxzoom: 14, attribution: "Terrain: Mapzen / AWS Open Data" };
  const ESRI = (s) => `https://server.arcgisonline.com/ArcGIS/rest/services/${s}/MapServer/tile/{z}/{y}/{x}`;
  const USGS = (s) => `https://basemap.nationalmap.gov/arcgis/rest/services/${s}/MapServer/tile/{z}/{y}/{x}`;
  const LIVE_MS = 5 * 60e3, INDEX_MS = 10 * 60e3, HOT_MIN_ZOOM = 6;
  const RAMP = ["#fff4c2", "#ffe17a", "#ffc94a", "#ffad33", "#ff8c1f", "#f56a10", "#e2480d", "#c8300f", "#a51d14", "#6e0f12"];
  const START_FILL = "#2a211c", START_EDGE = "#b42318", ACTUAL_EDGE = "#0b7285";
  const CROWN = { 1: [155, 212, 154], 2: [47, 125, 50] }, EMBER = { 1: [214, 200, 255], 2: [159, 132, 242], 3: [106, 63, 212] };
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

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
  const when = (t) => { const d = t ? new Date(t) : null; return d && !isNaN(d) ? d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }) : "–"; };
  const clock = (t) => { const d = new Date(t); return isNaN(d) ? "–" : d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" }); };
  const ago = (t) => {
    if (!t) return "";
    const m = Math.round((Date.now() - +new Date(t)) / 60000);
    if (m < 1) return "just now"; if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60); return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
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
  const FC = (features = []) => ({ type: "FeatureCollection", features });

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  if (store.get("theme", null)) root.dataset.theme = store.get("theme");
  const dark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("#btn-theme").addEventListener("click", () => { root.dataset.theme = dark() ? "light" : "dark"; store.set("theme", root.dataset.theme); });

  // ------------------------------------------------------------------ layer registry
  const yday = new Date(Date.now() - 86400e3).toISOString().slice(0, 10);
  const OVERLAYS = {
    roads: { name: "Roads", note: "Highways, local roads, trails", on: true },
    labels: { name: "Place names", note: "Towns, peaks, lakes", on: true },
    hillshade: { name: "Shaded relief", note: "Terrain shading", on: null },
    viirsImage: { name: `NASA VIIRS image (${yday})`, note: "Yesterday's true-color satellite view: smoke plumes, burn scars", on: false },
    hotspots: { name: "Satellite hotspots, 24 h", note: "NASA VIIRS fire detections (zoom in to load)", on: true },
    perimeters: { name: "NIFC perimeters", note: "Current interagency fire perimeters", on: true },
    incidents: { name: "Incidents", note: "Wildfires of 10+ acres, with names", on: true },
  };
  const MODEL = {
    pred: { name: "Predicted growth", note: "Chance each 150 m cell burns in the forecast window", on: true },
    start: { name: "Fire now", note: "Burning or burned at forecast start (perimeter + 48 h detections)", on: true },
    grid: { name: "Cell outlines", note: "Outline each forecast cell when zoomed in", on: true },
    crown: { name: "Crown fire potential", note: "Where fire would move into the canopy if it arrives", on: false },
    ember: { name: "Ember landing", note: "Where embers would land downwind of intense fire", on: false },
    actual: { name: "What actually burned", note: "Replay example only", on: true },
  };
  const saved = store.get("layers2", {});
  for (const reg of [OVERLAYS, MODEL]) for (const [k, l] of Object.entries(reg)) if (saved[k] !== undefined) l.on = saved[k];
  let predOpacity = store.get("predOpacity2", 0.6);
  const saveLayers = () => store.set("layers2", Object.fromEntries([...Object.entries(OVERLAYS), ...Object.entries(MODEL)].map(([k, l]) => [k, l.on])));

  // ------------------------------------------------------------------ base maps (each is a full style; our layers are re-added on every style load)
  const BASES = {
    terrain: { name: "Terrain", note: "Vector map with shaded relief", vector: "liberty", hill: true },
    light: { name: "Light", note: "Muted vector map with relief", vector: "positron", hill: true },
    satellite: { name: "Satellite", note: "Aerial imagery with roads and names", raster: ESRI("World_Imagery"), max: 19, att: "Imagery: Esri, Maxar, Earthstar Geographics", labels: true },
    usgsTopo: { name: "USGS Topo", note: "Contours, trails (US)", raster: USGS("USGSTopo"), max: 16, att: "USGS The National Map" },
    usgsImagery: { name: "USGS Imagery + Topo", note: "Aerial photos with contours (US)", raster: USGS("USGSImageryTopo"), max: 16, att: "USGS The National Map" },
    dark: { name: "Dark", note: "Low-light canvas", raster: ESRI("Canvas/World_Dark_Gray_Base"), max: 16, att: "Esri, HERE, Garmin, OpenStreetMap contributors", labels: true },
  };
  const vecCache = {};
  async function vectorStyle(name) { if (!vecCache[name]) vecCache[name] = await getJSON(OFM + name); return JSON.parse(JSON.stringify(vecCache[name])); }
  const isRoad = (l) => l["source-layer"] === "transportation" || l["source-layer"] === "transportation_name";
  const isPlace = (l) => l.type === "symbol" && ["place", "water_name", "mountain_peak"].includes(l["source-layer"]);
  async function buildStyle(key) {
    const b = BASES[key];
    let style;
    if (b.vector) style = await vectorStyle(b.vector);
    else {
      const lib = await vectorStyle("liberty");
      style = { version: 8, glyphs: lib.glyphs, sprite: lib.sprite, sources: { base: { type: "raster", tiles: [b.raster], tileSize: 256, maxzoom: b.max, attribution: b.att } }, layers: [{ id: "base", type: "raster", source: "base" }] };
      if (b.labels) {               // roads + names from the vector map on top of the imagery
        style.sources.openmaptiles = lib.sources.openmaptiles;
        const keep = lib.layers.filter((l) => l.source === "openmaptiles" && (isPlace(l) || (isRoad(l) && (l.type === "symbol" || /motorway|trunk|primary|secondary|tertiary|minor/.test(l.id)))));
        for (const l of keep) {
          if (l.type === "symbol") l.paint = { ...(l.paint || {}), "text-color": key === "satellite" ? "#ffffff" : "#e6e6e6", "text-halo-color": "rgba(0,0,0,.75)", "text-halo-width": 1.4 };
          if (l.type === "line") l.paint = { ...(l.paint || {}), "line-opacity": key === "satellite" ? 0.7 : 0.35 };
          style.layers.push(l);
        }
      }
    }
    style.sources.dem = DEM;
    const firstSym = style.layers.findIndex((l) => l.type === "symbol");
    style.layers.splice(firstSym < 0 ? style.layers.length : firstSym, 0, {
      id: "hillshade", type: "hillshade", source: "dem",
      layout: { visibility: (OVERLAYS.hillshade.on ?? b.hill) ? "visible" : "none" },
      paint: { "hillshade-exaggeration": b.vector ? 0.32 : 0.22, "hillshade-shadow-color": "#473b2f", "hillshade-highlight-color": "#ffffff", "hillshade-accent-color": "#473b2f" },
    });
    return style;
  }

  // ------------------------------------------------------------------ state
  const state = { index: null, fires: new Map(), example: null, perims: FC(), incidents: [], incFC: FC(), hot: FC(), liveAt: null, liveCount: 0, selected: null, base: null, fcLayers: 0 };

  // ------------------------------------------------------------------ map
  const v0 = store.get("view", [39.5, -112, 5]);
  const startBase = BASES[store.get("base2", "terrain")] ? store.get("base2", "terrain") : "terrain";
  state.base = startBase;
  const map = new maplibregl.Map({
    container: "map", style: { version: 8, sources: {}, layers: [] }, center: [v0[1], v0[0]], zoom: v0[2], minZoom: 2.5, maxZoom: 17.5,
    attributionControl: { compact: true }, dragRotate: false, pitchWithRotate: false, touchPitch: false,
  });
  map.touchZoomRotate.disableRotation();
  map.scrollZoom.setWheelZoomRate(1 / 220);
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "imperial" }), "bottom-right");
  map.on("moveend", () => { const c = map.getCenter(); store.set("view", [+c.lat.toFixed(4), +c.lng.toFixed(4), +map.getZoom().toFixed(2)]); });

  async function setBase(key) {
    state.base = key; store.set("base2", key);
    document.querySelectorAll('input[name="base"]').forEach((i) => (i.checked = i.value === key));
    const hs = document.getElementById("opt-ov-hillshade"); if (hs && OVERLAYS.hillshade.on === null) hs.checked = BASES[key].hill;
    try { map.setStyle(await buildStyle(key), { diff: false }); }
    catch (e) { console.warn(e); toast("That base map could not load; try another"); }
  }
  map.on("style.load", () => { if (!map.getStyle().glyphs) return; addFireIcons(); addOurLayers(); applyBaseToggles(); });

  function addFireIcons() {
    for (const [name, fill] of [["flame", "#d9480f"], ["flame-done", "#7d8790"]]) {
      if (map.hasImage(name)) continue;
      const s = 48, c = document.createElement("canvas"); c.width = c.height = s;
      const x = c.getContext("2d");
      x.beginPath(); x.arc(s / 2, s / 2, s / 2 - 2, 0, Math.PI * 2); x.fillStyle = fill; x.fill(); x.lineWidth = 3; x.strokeStyle = "#fff"; x.stroke();
      x.fillStyle = "#fff"; x.beginPath();
      x.moveTo(24, 9); x.bezierCurveTo(27, 17, 35, 20, 35, 29); x.bezierCurveTo(35, 36, 30, 40, 24, 40); x.bezierCurveTo(18, 40, 13, 36, 13, 30);
      x.bezierCurveTo(13, 25, 16, 22, 18, 18); x.bezierCurveTo(19, 22, 21, 24, 22, 24); x.bezierCurveTo(22, 18, 23, 13, 24, 9); x.fill();
      map.addImage(name, x.getImageData(0, 0, s, s), { pixelRatio: 2 });
    }
  }

  // ------------------------------------------------------------------ our layers (re-added after every style load)
  function addOurLayers() {
    const firstSym = map.getStyle().layers.find((l) => l.type === "symbol");
    map.addSource("viirs-img", { type: "raster", tiles: [`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_NOAA20_CorrectedReflectance_TrueColor/default/${yday}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`], tileSize: 256, maxzoom: 9, attribution: "NASA GIBS / VIIRS NOAA-20" });
    map.addLayer({ id: "viirs-img", type: "raster", source: "viirs-img", paint: { "raster-opacity": 0.85 } }, firstSym && firstSym.id);
    map.addSource("perims", { type: "geojson", data: state.perims });
    map.addLayer({ id: "perim-fill", type: "fill", source: "perims", paint: { "fill-color": ["case", ["==", ["get", "attr_IncidentTypeCategory"], "RX"], "#8a5a2b", "#d42a20"], "fill-opacity": 0.1 } });
    map.addLayer({ id: "perim-line", type: "line", source: "perims", filter: ["!=", ["get", "attr_IncidentTypeCategory"], "RX"], paint: { "line-color": "#d42a20", "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1, 12, 2.4] } });
    map.addLayer({ id: "perim-line-rx", type: "line", source: "perims", filter: ["==", ["get", "attr_IncidentTypeCategory"], "RX"], paint: { "line-color": "#8a5a2b", "line-width": 1.6, "line-dasharray": [3, 2] } });
    map.addSource("hot", { type: "geojson", data: state.hot });
    map.addLayer({ id: "hot", type: "circle", source: "hot", paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 2.5, 10, 4.5, 14, 7],
      "circle-color": ["step", ["coalesce", ["get", "hours_old"], 24], "#ffe066", 6, "#ff922b", 12, "#d6336c"],
      "circle-stroke-color": "#3b1d0a", "circle-stroke-width": 0.6, "circle-opacity": 0.95 } });
    map.addSource("inc", { type: "geojson", data: state.incFC });
    map.addLayer({ id: "inc", type: "symbol", source: "inc", layout: {
      "icon-image": ["case", [">=", ["coalesce", ["get", "PercentContained"], 0], 100], "flame-done", "flame"],
      "icon-size": ["interpolate", ["linear"], ["coalesce", ["get", "IncidentSize"], 0], 10, 0.7, 1000, 0.9, 50000, 1.2],
      "icon-allow-overlap": true,
      "text-field": ["step", ["zoom"], "", 6, ["get", "IncidentName"]], "text-font": ["Noto Sans Bold"], "text-size": 12,
      "text-offset": [0, 1.25], "text-anchor": "top", "text-optional": true, "symbol-sort-key": ["-", 0, ["coalesce", ["get", "IncidentSize"], 0]] },
      paint: { "text-color": "#3b0d08", "text-halo-color": "rgba(255,255,255,.95)", "text-halo-width": 1.6 } });
    map.addSource("hi", { type: "geojson", data: FC() });
    map.addLayer({ id: "hi", type: "line", source: "hi", paint: { "line-color": "#111", "line-width": 2 } });
    state.fcLayers = 0;
    addForecastLayers();
    applyOverlayVisibility();
  }
  function vis(id, on) { if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); }
  function applyOverlayVisibility() {
    vis("viirs-img", OVERLAYS.viirsImage.on);
    ["perim-fill", "perim-line", "perim-line-rx"].forEach((id) => vis(id, OVERLAYS.perimeters.on));
    vis("hot", OVERLAYS.hotspots.on); vis("inc", OVERLAYS.incidents.on);
  }
  function applyBaseToggles() {
    for (const l of map.getStyle().layers) {
      if (l.source !== "openmaptiles") continue;
      if (isRoad(l)) vis(l.id, OVERLAYS.roads.on);
      else if (isPlace(l)) vis(l.id, OVERLAYS.labels.on);
    }
    vis("hillshade", OVERLAYS.hillshade.on ?? BASES[state.base].hill);
  }

  // ------------------------------------------------------------------ forecast cells: crisp images in the model's own grid
  function decodeForecast(meta, d) {
    const g = d.grid, w = g.width, h = g.height, n = w * h;
    const F = { meta, g, w, h, prob: b64(d.prob), start: b64(d.start), crown: d.crown ? b64(d.crown) : null, ember: d.ember ? b64(d.ember) : null, actual: d.actual ? b64(d.actual) : null };
    F.coords = [[g.west, g.north], [g.west + w * g.dx, g.north], [g.west + w * g.dx, g.north - h * g.dy], [g.west, g.north - h * g.dy]];
    let r0 = h, r1 = -1, c0 = w, c1 = -1;
    for (let i = 0; i < n; i++) if (F.start[i] || F.prob[i] >= 10 || (F.actual && F.actual[i])) { const r = (i / w) | 0, c = i - r * w; if (r < r0) r0 = r; if (r > r1) r1 = r; if (c < c0) c0 = c; if (c > c1) c1 = c; }
    F.focus = r1 < 0 ? [[g.west, g.north - h * g.dy], [g.west + w * g.dx, g.north]] : [[g.west + c0 * g.dx, g.north - (r1 + 1) * g.dy], [g.west + (c1 + 1) * g.dx, g.north - r0 * g.dy]];
    F.startLines = edgeLines(F.start, F);
    F.actualLines = F.actual ? edgeLines(F.actual, F) : FC();
    F.images = renderImages(F);
    F.gridLines = gridLines(F);
    return F;
  }
  function edgeLines(mask, F) {           // boundary between in-mask and out-of-mask cells, as map lines
    const { w, h, g } = F, segs = [];
    const X = (c) => +(g.west + c * g.dx).toFixed(6), Y = (r) => +(g.north - r * g.dy).toFixed(6);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = r * w + c; if (!mask[i]) continue;
      if (r === 0 || !mask[i - w]) segs.push([[X(c), Y(r)], [X(c + 1), Y(r)]]);
      if (r === h - 1 || !mask[i + w]) segs.push([[X(c), Y(r + 1)], [X(c + 1), Y(r + 1)]]);
      if (c === 0 || !mask[i - 1]) segs.push([[X(c), Y(r)], [X(c), Y(r + 1)]]);
      if (c === w - 1 || !mask[i + 1]) segs.push([[X(c + 1), Y(r)], [X(c + 1), Y(r + 1)]]);
    }
    return FC(segs.length ? [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: segs } }] : []);
  }
  function gridLines(F) {                 // thin outline of every cell with a 5%+ chance (drawn from zoom 11.5)
    const { w, h, g } = F, segs = [], on = (i) => F.prob[i] >= 5;
    const X = (c) => +(g.west + c * g.dx).toFixed(6), Y = (r) => +(g.north - r * g.dy).toFixed(6);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = r * w + c;
      if (!on(i)) continue;
      segs.push([[X(c), Y(r)], [X(c + 1), Y(r)]], [[X(c), Y(r)], [X(c), Y(r + 1)]]);
      if (r === h - 1 || !on(i + w)) segs.push([[X(c), Y(r + 1)], [X(c + 1), Y(r + 1)]]);
      if (c === w - 1 || !on(i + 1)) segs.push([[X(c + 1), Y(r)], [X(c + 1), Y(r + 1)]]);
    }
    return FC(segs.length ? [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: segs } }] : []);
  }
  function canvasURL(w, h, paint) {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const x = c.getContext("2d"), im = x.createImageData(w, h);
    paint(im.data); x.putImageData(im, 0, 0);
    return c.toDataURL("image/png");
  }
  function renderImages(F) {
    const { w, h } = F, n = w * h, out = {};
    const rampRGB = RAMP.map(hex);
    out.prob = canvasURL(w, h, (d) => { for (let i = 0; i < n; i++) { const p = F.prob[i]; if (!p) continue; const k = Math.min(9, Math.floor(p / 10)), c = rampRGB[k]; d.set([c[0], c[1], c[2], k === 0 ? 120 : 200 + k * 5], i * 4); } });
    const sf = hex(START_FILL);
    out.start = canvasURL(w, h, (d) => { for (let i = 0; i < n; i++) if (F.start[i]) d.set([sf[0], sf[1], sf[2], 85], i * 4); });
    if (F.crown) out.crown = canvasURL(w, h, (d) => { for (let i = 0; i < n; i++) { const v = F.crown[i]; if (v >= 20) { const c = CROWN[v >= 50 ? 2 : 1]; d.set([c[0], c[1], c[2], v >= 50 ? 120 : 80], i * 4); } } });
    if (F.ember) out.ember = canvasURL(w, h, (d) => { for (let i = 0; i < n; i++) { const v = F.ember[i]; if (v) { const c = EMBER[v]; d.set([c[0], c[1], c[2], 60 + v * 35], i * 4); } } });
    return out;
  }
  function fireList() { return state.fires.size ? [...state.fires.values()] : state.example ? [state.example] : []; }
  function addForecastLayers() {
    if (!map.getLayer("perim-fill")) return;
    const firstSym = map.getStyle().layers.find((l) => l.type === "symbol" && l.source === "openmaptiles");
    const under = firstSym ? firstSym.id : "perim-fill";
    fireList().forEach((F, j) => {
      const id = `fc${j}`;
      const add = (kind, url, paint, minzoom) => {
        if (!url) return;
        map.addSource(`${id}-${kind}`, { type: "image", url, coordinates: F.coords });
        map.addLayer({ id: `${id}-${kind}`, type: "raster", source: `${id}-${kind}`, minzoom: minzoom || 0,
          paint: { "raster-resampling": "nearest", "raster-fade-duration": 0, ...paint } }, under);
      };
      add("crown", F.images.crown, { "raster-opacity": 1 });
      add("ember", F.images.ember, { "raster-opacity": 1 });
      add("prob", F.images.prob, { "raster-opacity": predOpacity });
      map.addSource(`${id}-grid`, { type: "geojson", data: F.gridLines });
      map.addLayer({ id: `${id}-grid`, type: "line", source: `${id}-grid`, minzoom: 11.5,
        paint: { "line-color": "rgba(70,35,15,.38)", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.4, 15, 0.9] } }, "perim-fill");
      add("start", F.images.start, { "raster-opacity": 1 });
      map.addSource(`${id}-startline`, { type: "geojson", data: F.startLines });
      map.addLayer({ id: `${id}-startline`, type: "line", source: `${id}-startline`, paint: { "line-color": START_EDGE, "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1.6, 14, 3.2] } }, "perim-fill");
      map.addSource(`${id}-actual`, { type: "geojson", data: F.actualLines });
      map.addLayer({ id: `${id}-actual`, type: "line", source: `${id}-actual`, paint: { "line-color": ACTUAL_EDGE, "line-width": 2.2 } }, "perim-fill");
    });
    state.fcLayers = fireList().length;
    applyModelVisibility();
  }
  function removeForecastLayers() {
    for (let j = 0; j < state.fcLayers; j++) for (const kind of ["crown", "ember", "prob", "grid", "start", "startline", "actual"]) {
      const id = `fc${j}-${kind}`; if (map.getLayer(id)) map.removeLayer(id); if (map.getSource(id)) map.removeSource(id);
    }
    state.fcLayers = 0;
  }
  function applyModelVisibility() {
    for (let j = 0; j < state.fcLayers; j++) {
      vis(`fc${j}-prob`, MODEL.pred.on); vis(`fc${j}-grid`, MODEL.pred.on && MODEL.grid.on);
      vis(`fc${j}-start`, MODEL.start.on); vis(`fc${j}-startline`, MODEL.start.on);
      vis(`fc${j}-crown`, MODEL.crown.on); vis(`fc${j}-ember`, MODEL.ember.on); vis(`fc${j}-actual`, MODEL.actual.on);
      if (map.getLayer(`fc${j}-prob`)) map.setPaintProperty(`fc${j}-prob`, "raster-opacity", predOpacity);
    }
  }
  function refreshForecastLayers() { if (!map.getLayer("perim-fill")) return; removeForecastLayers(); addForecastLayers(); }

  // ------------------------------------------------------------------ cell readout (hover on desktop, tap on phones)
  function cellAt(ll) {
    for (const F of fireList()) {
      const g = F.g, c = Math.floor((ll.lng - g.west) / g.dx), r = Math.floor((g.north - ll.lat) / g.dy);
      if (c < 0 || r < 0 || c >= F.w || r >= F.h) continue;
      const i = r * F.w + c;
      return { F, r, c, p: F.prob[i], start: F.start[i], crown: F.crown ? F.crown[i] : 0, ember: F.ember ? F.ember[i] : 0, actual: F.actual ? F.actual[i] : 0 };
    }
    return null;
  }
  function hideCell() { $("#readout").hidden = true; const s = map.getSource("hi"); if (s) s.setData(FC()); }
  function showCell(ll) {
    const cell = cellAt(ll);
    if (!cell || !(MODEL.pred.on || MODEL.start.on) || (!cell.p && !cell.start && !cell.actual)) { hideCell(); return false; }
    const { F, r, c } = cell, g = F.g;
    const x0 = g.west + c * g.dx, x1 = x0 + g.dx, y1 = g.north - r * g.dy, y0 = y1 - g.dy;
    const hs = map.getSource("hi"); if (hs) hs.setData(FC([{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] } }]));
    const hrs = F.meta.example ? F.meta.hours : (state.index?.horizon_hours || 24);
    const parts = [cell.start ? `<b>Fire now</b> <span>burning or burned at forecast start</span>` : `<b>${cell.p}%</b> chance of burning <span>in the next ${hrs} h</span>`];
    if (!cell.start && cell.crown >= 50) parts.push("<span>· crown fire likely</span>");
    if (!cell.start && cell.ember >= 2) parts.push("<span>· ember landing</span>");
    if (cell.actual) parts.push("<span>· really burned</span>");
    const ro = $("#readout"); ro.innerHTML = parts.join(" "); ro.hidden = false;
    return true;
  }
  map.on("mousemove", (e) => showCell(e.lngLat));
  map.on("mouseout", hideCell);

  const popup = new maplibregl.Popup({ closeButton: true, maxWidth: "320px", offset: 10 });
  map.on("click", (e) => {
    const layers = ["inc", "hot", "perim-fill"].filter((l) => map.getLayer(l));
    const hits = layers.length ? map.queryRenderedFeatures(e.point, { layers }) : [];
    if (showCell(e.lngLat)) { clearTimeout(showCell.t); showCell.t = setTimeout(hideCell, 6000); }
    if (!hits.length) return;
    const f = hits[0], p = f.properties;
    let html;
    if (f.layer.id === "inc") {
      html = `<div class="pop"><h5>${esc(p.IncidentName)}</h5><dl><dt>Size</dt><dd>${ac(p.IncidentSize)}</dd><dt>Contained</dt><dd>${p.PercentContained ?? "–"}%</dd>
        <dt>State</dt><dd>${esc(st(p.POOState))}</dd><dt>Discovered</dt><dd>${when(p.FireDiscoveryDateTime)}</dd>
        ${p.TotalIncidentPersonnel ? `<dt>Personnel</dt><dd>${Number(p.TotalIncidentPersonnel).toLocaleString()}</dd>` : ""}<dt>Updated</dt><dd>${when(p.ModifiedOnDateTime_dt)}</dd></dl>${links(p.IncidentName, p.POOState)}</div>`;
      select(p.UniqueFireIdentifier, true, 700);
    } else if (f.layer.id === "hot") {
      html = `<div class="pop"><h5>Satellite hotspot</h5><dl><dt>Detected</dt><dd>${when(p.esritimeutc)}</dd><dt>Age</dt><dd>${p.hours_old ?? "?"} h</dd>
        <dt>Fire power</dt><dd>${p.frp ?? "–"} MW</dd><dt>Satellite</dt><dd>${esc(p.satellite ?? "–")} · ${p.daynight === "N" ? "night" : "day"}</dd></dl><p>375 m pixel: the fire is somewhere inside it.</p></div>`;
    } else {
      select(p.attr_UniqueFireIdentifier, true, 700);
      html = `<div class="pop"><h5>${esc(p.attr_IncidentName)}</h5><dl><dt>Size</dt><dd>${ac(p.poly_GISAcres)}</dd><dt>Contained</dt><dd>${p.attr_PercentContained ?? "–"}%</dd>
        <dt>Perimeter</dt><dd>${when(p.poly_PolygonDateTime)}</dd><dt>Mapped by</dt><dd>${esc(p.poly_MapMethod || "–")}</dd><dt>Type</dt><dd>${p.attr_IncidentTypeCategory === "RX" ? "Prescribed" : "Wildfire"}</dd><dt>Source</dt><dd>NIFC ${esc(p.feed || "current")}</dd></dl>${links(p.attr_IncidentName, p.attr_POOState)}</div>`;
    }
    popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
  });
  for (const l of ["inc", "hot", "perim-fill"]) {
    map.on("mouseenter", l, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", l, () => (map.getCanvas().style.cursor = ""));
  }

  // ------------------------------------------------------------------ live data
  // NIFC's "Current" service drops active fires whose incident record went stale (e.g. Ptarmigan, 26k ac,
  // Nov 2026); the Year-to-Date service still has them. We merge both, newest perimeter per fire.
  async function loadPerimeters() {
    try {
      const common = { outSR: 4326, f: "geojson", geometryPrecision: 5, maxAllowableOffset: 0.0002, resultRecordCount: 2000,
        outFields: "attr_UniqueFireIdentifier,attr_IncidentName,poly_GISAcres,attr_PercentContained,poly_PolygonDateTime,poly_MapMethod,attr_IncidentTypeCategory,attr_POOState" };
      const since = new Date(Date.now() - 90 * 86400e3).toISOString().slice(0, 19).replace("T", " ");
      const [cur, ytd] = await Promise.all([
        getJSON(URLS.perimeters, { ...common, where: "1=1" }),
        getJSON(URLS.perimetersYTD, { ...common, where: `attr_IncidentTypeCategory='WF' AND attr_FireOutDateTime IS NULL AND (attr_PercentContained IS NULL OR attr_PercentContained < 100) AND poly_PolygonDateTime >= TIMESTAMP '${since}'` }).catch(() => FC()),
      ]);
      const best = new Map();
      for (const [feed, fc] of [["current", cur], ["year to date", ytd]]) for (const f of fc.features || []) {
        const id = f.properties.attr_UniqueFireIdentifier, old = best.get(id);
        f.properties.feed = feed;
        if (!old || (f.properties.poly_PolygonDateTime || 0) > (old.properties.poly_PolygonDateTime || 0)) best.set(id, f);
      }
      state.perims = FC([...best.values()]);
      const s = map.getSource("perims"); if (s) s.setData(state.perims);
      state.liveAt = Date.now(); state.liveCount = state.perims.features.length; freshLive();
    } catch (e) { $("#fresh-live").textContent = "NIFC unreachable, retrying"; $("#fresh-live").classList.add("stale"); console.warn(e); }
  }
  function freshLive() { if (state.liveAt) { const d = $("#fresh-live"); d.textContent = `${ago(state.liveAt)} · ${state.liveCount} perimeters`; d.classList.remove("stale"); } }
  async function loadIncidents() {
    try {
      state.incFC = await getJSON(URLS.incidents, {
        where: "IncidentTypeCategory='WF' AND IncidentSize>=10", outSR: 4326, f: "geojson", resultRecordCount: 2000,
        outFields: "IncidentName,IncidentSize,PercentContained,FireDiscoveryDateTime,ModifiedOnDateTime_dt,POOState,UniqueFireIdentifier,TotalIncidentPersonnel",
      });
      state.incidents = state.incFC.features.map((f) => ({ ...f.properties, lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1] }));
      const s = map.getSource("inc"); if (s) s.setData(state.incFC);
      renderList();
    } catch (e) { console.warn(e); }
  }
  let hotT = null, hotSeq = 0;
  function loadHotspots() {
    clearTimeout(hotT);
    hotT = setTimeout(async () => {
      if (!OVERLAYS.hotspots.on || map.getZoom() < HOT_MIN_ZOOM) { state.hot = FC(); const s = map.getSource("hot"); if (s) s.setData(state.hot); return; }
      const b = map.getBounds(), seq = ++hotSeq;
      try {
        const gj = await getJSON(URLS.viirs, {
          where: "hours_old<=24", geometry: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(4)).join(","),
          geometryType: "esriGeometryEnvelope", inSR: 4326, spatialRel: "esriSpatialRelIntersects", outSR: 4326, f: "geojson",
          outFields: "esritimeutc,hours_old,frp,confidence,satellite,daynight", resultRecordCount: 8000,
        });
        if (seq !== hotSeq) return;
        state.hot = gj; const s = map.getSource("hot"); if (s) s.setData(gj);
      } catch (e) { console.warn(e); }
    }, 350);
  }
  map.on("moveend", loadHotspots);

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
      const want = idx.fires.filter((f) => f.forecast), next = new Map();
      await Promise.all(want.map(async (f) => {
        try { next.set(f.id, decodeForecast(f, await getJSON(`${f.forecast}?t=${encodeURIComponent(idx.generated)}`))); }
        catch (e) { console.warn("forecast", f.id, e); }
      }));
      state.fires = next;
    }
    if (!state.fires.size && !state.example) {
      try { const ex = await getJSON("data/example/index.json"); state.example = decodeForecast(ex, await getJSON(`data/example/${ex.file}`)); }
      catch { /* none shipped */ }
    }
    if (!state.fires.size && state.example) $("#pred-sub").textContent = `example replay: chance each cell burned in ${state.example.meta.hours} h`;
    refreshForecastLayers(); renderList(); renderLegend();
    if (state.selected) renderDetail(state.selected);
  }

  // ------------------------------------------------------------------ layers panel
  function option(type, name, value, label, note, checked, onchange) {
    const id = `opt-${name}-${value}`, i = el("input", { type, name, value, id });
    i.checked = checked; i.addEventListener("change", onchange);
    return el("label", { class: "opt", for: id }, i, el("b", { text: label }), note ? el("small", { text: note }) : null);
  }
  function buildPanel() {
    $("#base-list").replaceChildren(...Object.entries(BASES).map(([k, b]) => option("radio", "base", k, b.name, b.note, k === state.base, () => setBase(k))));
    $("#overlay-list").replaceChildren(...Object.entries(OVERLAYS).map(([k, l]) => option("checkbox", "ov", k, l.name, l.note, l.on ?? BASES[state.base].hill, (e) => {
      l.on = e.target.checked; saveLayers();
      if (["roads", "labels", "hillshade"].includes(k)) applyBaseToggles(); else applyOverlayVisibility();
      if (k === "hotspots") loadHotspots();
      renderLegend();
    })));
    $("#model-list").replaceChildren(...Object.entries(MODEL).map(([k, l]) => option("checkbox", "md", k, l.name, l.note, l.on, (e) => setModel(k, e.target.checked))));
    const op = $("#pred-opacity"); op.value = predOpacity;
    op.oninput = () => { predOpacity = +op.value; store.set("predOpacity2", predOpacity); applyModelVisibility(); };
  }
  function setModel(k, on) {
    MODEL[k].on = on; saveLayers();
    if (k === "pred") $("#pred-toggle").checked = on;
    const box = document.getElementById(`opt-md-${k}`); if (box) box.checked = on;
    applyModelVisibility(); renderLegend(); if (!on) hideCell();
  }
  $("#pred-toggle").addEventListener("change", (e) => setModel("pred", e.target.checked));
  const panel = $("#layers");
  const showPanel = (on) => { panel.hidden = !on; $("#btn-layers").setAttribute("aria-expanded", String(on)); };
  $("#btn-layers").addEventListener("click", () => showPanel(panel.hidden));
  $("#layers-close").addEventListener("click", () => showPanel(false));

  // ------------------------------------------------------------------ side: list + detail
  function renderList() {
    const q = $("#search").value.trim().toLowerCase();
    const match = (n, s) => !q || (n || "").toLowerCase().includes(q) || st(s).toLowerCase().includes(q);
    const box = $("#fire-list"); box.replaceChildren();
    const fc = [...state.fires.values()].map((F) => F.meta).filter((m) => match(m.name, m.state)).sort((a, b) => (b.stats?.expected_new_acres || 0) - (a.stats?.expected_new_acres || 0));
    if (fc.length) {
      box.append(el("p", { class: "group-label", text: `Forecast available (${fc.length})` }));
      fc.forEach((m) => box.append(row(m.id, m.no_heat ? `${m.name} *` : m.name, m.state, m.acres, m.contained,
        [m.source === "incident" ? "no perimeter yet" : `perimeter ${ago(m.perimeter_time || m.updated)}`, m.no_heat ? "no satellite heat" : null].filter(Boolean).join(" · "), m.stats)));
      if (fc.some((m) => m.no_heat)) box.append(el("p", { class: "footnote", text: NO_HEAT }));
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
    if (done.length) { const d = el("details", { class: "contained" }, el("summary", { class: "group-label", text: `Contained (${done.length})` })); done.slice(0, 200).forEach((p) => d.append(r(p))); if (q) d.open = true; box.append(d); }
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

  const NO_HEAT = "* No satellite fire detections in the last 48 h (NOAA-20, NOAA-21). The forecast starts from the reported size of a newly reported fire; the fire may be smaller than reported or already out.";
  function perimFor(id) { return state.perims.features.find((f) => f.properties.attr_UniqueFireIdentifier === id); }
  function renderDetail(id) {
    const box = $("#detail");
    const F = id === "__example" ? state.example : state.fires.get(id);
    const inc = state.incidents.find((p) => p.UniqueFireIdentifier === id), per = perimFor(id);
    const m = F ? F.meta : null;
    const name = m?.name || inc?.IncidentName || per?.properties.attr_IncidentName;
    if (!name) { box.hidden = true; return; }
    const s = m?.stats || {}, hrs = m?.example ? m.hours : (state.index?.horizon_hours || 24);
    const fig = (label, val, hot) => el("div", {}, el("dt", { text: label }), el("dd", { class: hot ? "hot" : "", text: val }));
    const figs = el("dl", { class: "figs" }, fig("Size now", ac(m?.acres ?? inc?.IncidentSize ?? per?.properties.poly_GISAcres)),
      m?.example ? fig("Forecast window", `${m.hours} h`) : fig("Contained", (m?.contained ?? inc?.PercentContained) === null || (m?.contained ?? inc?.PercentContained) === undefined ? "not reported" : `${m?.contained ?? inc?.PercentContained}%`));
    if (F) figs.append(fig(`Growth, next ${hrs} h`, ac(s.expected_new_acres), true), fig("Likely, 50%+", ac(s.likely_acres)),
      fig("Possible, 10%+", ac(s.possible_acres)), s.actual_new_acres !== undefined ? fig("Actually burned", ac(s.actual_new_acres)) : fig("Forecast from", when(m.origin)));
    const w = s.wind;
    box.replaceChildren(...[
      el("button", { class: "close", onclick: () => { state.selected = null; box.hidden = true; renderList(); }, text: "Close" }),
      el("h3", { text: m?.no_heat ? `${name} *` : name }),
      el("p", { class: "where", text: [st(m?.state || inc?.POOState), m?.example ? "example replay" : null, m?.source === "incident" ? "no NIFC perimeter yet" : null, inc ? `updated ${ago(inc.ModifiedOnDateTime_dt)}` : null].filter(Boolean).join(" · ") }),
      figs,
      m?.no_heat ? el("p", { class: "footnote", text: NO_HEAT }) : null,
      w ? el("p", { class: "wind", text: `Forecast wind: from ${w.from_compass} at ${w.mean_mph} mph, gusts to ${w.peak_gust_mph} mph, humidity down to ${w.min_rh_pct}%` }) : null,
      F ? el("p", { class: "note", text: m.example ? m.note : `Forecast from ${when(m.origin)} to ${when(new Date(+new Date(m.origin) + hrs * 3600e3))}. Weather: HRRR run ${when(m.hrrr_run)}.${m.source === "incident" ? " Starting fire estimated from satellite detections (no perimeter yet)." : ""}` })
        : el("p", { class: "noforecast", text: "No forecast for this fire: fewer than 3 satellite fire detections near it in the last 48 hours, so it does not look actively spreading. Forecasts refresh every 2 hours." }),
      m?.example ? null : el("p", { class: "links" },
        el("a", { href: wdURL(name, m?.state || inc?.POOState || per?.properties.attr_POOState), target: "_blank", rel: "noopener", text: "Find on Watch Duty ↗" }), " · ",
        el("a", { href: iwURL(name, m?.state || inc?.POOState || per?.properties.attr_POOState), target: "_blank", rel: "noopener", text: "InciWeb ↗" })),
    ].filter(Boolean));
    box.hidden = false;
  }
  function wdURL(name, state) { return `https://www.google.com/search?q=${encodeURIComponent(`site:app.watchduty.org "${String(name || "").replace(/\s*fire$/i, "")} Fire" ${st(state)}`)}`; }
  function iwURL(name, state) { return `https://www.google.com/search?q=${encodeURIComponent(`site:inciweb.wildfire.gov ${name} ${st(state)}`)}`; }
  function links(name, state) {
    return `<p class="links"><a href="${wdURL(name, state)}" target="_blank" rel="noopener">Find on Watch Duty ↗</a> · <a href="${iwURL(name, state)}" target="_blank" rel="noopener">InciWeb ↗</a></p>`;
  }
  function select(id, fly, duration = 1200) {
    state.selected = id;
    document.querySelectorAll(".row").forEach((b) => b.classList.toggle("sel", b.dataset.id === id));
    renderDetail(id);
    try { history.replaceState(null, "", `#fire=${encodeURIComponent(id)}`); } catch { /* sandboxed */ }
    if (!fly) return;
    const F = id === "__example" ? state.example : state.fires.get(id);
    const per = perimFor(id), inc = state.incidents.find((p) => p.UniqueFireIdentifier === id);
    const padding = { top: 80, left: 40, right: 40, bottom: innerWidth <= 780 ? Math.round(innerHeight * 0.4) : 40 };
    if (F) map.fitBounds(F.focus, { padding, maxZoom: 14, duration });
    else if (per) map.fitBounds(bbox(per.geometry), { padding, maxZoom: 14, duration });
    else if (inc) map.flyTo({ center: [inc.lon, inc.lat], zoom: 12, duration });
    if (innerWidth <= 780) setSheet("min");
  }
  function bbox(geom) {
    let w = 180, s = 90, e = -180, n = -90;
    const walk = (c) => { if (typeof c[0] === "number") { w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]); } else c.forEach(walk); };
    walk(geom.coordinates); return [[w, s], [e, n]];
  }

  // ------------------------------------------------------------------ legend
  function renderLegend() {
    const lg = $("#legend"); lg.replaceChildren();
    const fires = fireList().length;
    if (MODEL.pred.on && fires) {
      const hrs = state.fires.size ? (state.index?.horizon_hours || 24) : state.example?.meta.hours;
      lg.append(el("h4", { text: `Chance each cell burns, next ${hrs} h` }));
      const ramp = el("div", { class: "ramp" }); RAMP.forEach((c) => { const sp = el("span"); sp.style.background = c; ramp.append(sp); });
      lg.append(ramp, el("div", { class: "ticks" }, el("span", { text: "1%" }), el("span", { text: "50%" }), el("span", { text: "100%" })));
    }
    const keys = [];
    if (MODEL.start.on && fires) keys.push([START_FILL, "Fire now", START_EDGE]);
    if (MODEL.actual.on && !state.fires.size && state.example) keys.push(["transparent", "What actually burned", ACTUAL_EDGE]);
    if (MODEL.crown.on && fires) keys.push(["#2f7d32", "Crown fire likely"]);
    if (MODEL.ember.on && fires) keys.push(["#9f84f2", "Ember landing"]);
    if (OVERLAYS.perimeters.on) keys.push(["transparent", "NIFC perimeter", "#d42a20"]);
    if (OVERLAYS.hotspots.on) keys.push(["#ffe066", "Hotspot under 6 h", null, true], ["#d6336c", "Hotspot 12–24 h", null, true]);
    if (OVERLAYS.incidents.on) keys.push(["#d9480f", "Incident (grey: contained)", null, true]);
    if (keys.length && lg.childElementCount) lg.append(el("div", { class: "sep" }));
    for (const [f, label, edge, dot] of keys) {
      const i = el("i", { class: dot ? "dot" : "" }); i.style.background = f; if (edge) i.style.boxShadow = `inset 0 0 0 2px ${edge}`;
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
  $("#btn-locate").addEventListener("click", () => {
    if (!navigator.geolocation) return toast("Location not available on this device");
    navigator.geolocation.getCurrentPosition((p) => {
      const ll = [p.coords.longitude, p.coords.latitude];
      if (me) me.remove();
      const dot = el("div"); dot.style.cssText = "width:14px;height:14px;border-radius:50%;background:#1c64f2;border:2px solid #fff;box-shadow:0 0 0 2px rgba(28,100,242,.3)";
      me = new maplibregl.Marker({ element: dot }).setLngLat(ll).addTo(map);
      map.flyTo({ center: ll, zoom: Math.max(map.getZoom(), 11) });
    }, () => toast("Location not available (permission denied or no signal)"), { enableHighAccuracy: true, timeout: 10000 });
  });

  // ------------------------------------------------------------------ start
  $("#pred-toggle").checked = MODEL.pred.on;
  buildPanel(); renderLegend();
  setBase(startBase);
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
