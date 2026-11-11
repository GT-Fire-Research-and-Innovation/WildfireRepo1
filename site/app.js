/* SpreadWatch: live NIFC perimeters + VIIRS hotspots (fetched in the browser) and the model's
   spread forecasts (data/index.json + data/forecast/*.geojson, published by live/publish.py). */
(() => {
  "use strict";

  const ARC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services";
  const URLS = {
    perimeters: `${ARC}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query`,
    incidents: `${ARC}/WFIGS_Incident_Locations_Current/FeatureServer/0/query`,
    viirs: "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/Satellite_VIIRS_Thermal_Hotspots_and_Fire_Activity/FeatureServer/0/query",
  };
  const LIVE_MS = 5 * 60 * 1000, INDEX_MS = 10 * 60 * 1000, HOTSPOT_MIN_ZOOM = 6;
  const BAND_COLORS = { 0.1: "#ffd43b", 0.3: "#ff922b", 0.5: "#f03e3e", 0.7: "#a61e4d" };
  const CROWN_COLORS = { 1: "#8ce99a", 2: "#2b8a3e" };
  const EMBER_COLORS = { 1: "#d0bfff", 2: "#9775fa", 3: "#6741d9" };
  const HOT_COLORS = [[6, "#ffe066"], [12, "#ff922b"], [24, "#e03131"], [48, "#862e9c"]];

  // ------------------------------------------------------------------ small helpers
  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    for (const k of kids) if (k !== null && k !== undefined) n.append(k);
    return n;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem("sw:" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("sw:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const fmtAc = (a) => (a === null || a === undefined || isNaN(a)) ? "–" : Math.round(a).toLocaleString() + " ac";
  const fmtTime = (t) => {
    if (!t) return "–";
    const d = typeof t === "number" ? new Date(t) : new Date(t);
    if (isNaN(d)) return "–";
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
  };
  const ago = (d) => {
    if (!d) return "";
    const m = Math.round((Date.now() - d) / 60000);
    if (m < 1) return "just now";
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  async function getJSON(url, params, ms = 30000) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), ms);
    try {
      const u = params ? url + "?" + new URLSearchParams(params) : url;
      const r = await fetch(u, { signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally { clearTimeout(t); }
  }
  function toast(msg, ms = 3500) {
    const t = $("#toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
  }
  function pill(id, text, state) {
    const p = $(id);
    p.className = "pill" + (state ? " " + state : "");
    p.lastChild.textContent = text;
    p.title = text;
  }

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  const savedTheme = store.get("theme", null);
  if (savedTheme) root.dataset.theme = savedTheme;
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("#btn-theme").addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    store.set("theme", root.dataset.theme);
  });

  // ------------------------------------------------------------------ map + basemaps
  L.Popup.mergeOptions({ autoPanPadding: [16, 16], maxWidth: 320 });
  const map = L.map("map", { zoomControl: false, worldCopyJump: true, minZoom: 3 }).setView(store.get("view", [39.5, -112, 5]).slice(0, 2), store.get("view", [39.5, -112, 5])[2]);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.control.scale({ position: "bottomright", imperial: true, metric: true }).addTo(map);
  map.on("moveend", () => { const c = map.getCenter(); store.set("view", [+c.lat.toFixed(4), +c.lng.toFixed(4), map.getZoom()]); });

  const esri = (svc) => `https://server.arcgisonline.com/ArcGIS/rest/services/${svc}/MapServer/tile/{z}/{y}/{x}`;
  const BASEMAPS = {
    topo: { name: "Topo", layers: [L.tileLayer(esri("World_Topo_Map"), { maxZoom: 19, attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS" })] },
    imagery: { name: "Satellite", layers: [
      L.tileLayer(esri("World_Imagery"), { maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" }),
      L.tileLayer(esri("Reference/World_Boundaries_and_Places"), { maxZoom: 19 }),
    ] },
    terrain: { name: "Terrain", layers: [L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", { maxZoom: 17, attribution: "&copy; OpenStreetMap contributors, SRTM | &copy; OpenTopoMap (CC-BY-SA)" })] },
    dark: { name: "Dark", layers: [
      L.tileLayer(esri("Canvas/World_Dark_Gray_Base"), { maxZoom: 16, attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, OpenStreetMap contributors" }),
      L.tileLayer(esri("Canvas/World_Dark_Gray_Reference"), { maxZoom: 16 }),
    ] },
  };
  let currentBase = null;
  function setBase(key) {
    if (currentBase) BASEMAPS[currentBase].layers.forEach((l) => map.removeLayer(l));
    BASEMAPS[key].layers.forEach((l) => l.addTo(map));
    currentBase = key; store.set("base", key);
    document.querySelectorAll("#basemaps button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.key === key)));
  }
  for (const [k, b] of Object.entries(BASEMAPS)) {
    $("#basemaps").append(el("button", { role: "radio", "data-key": k, "aria-checked": "false", onclick: () => setBase(k), text: b.name }));
  }
  setBase(store.get("base", isDark() ? "dark" : "topo"));

  // panes keep the stacking order no matter which layer loads first
  [["ember", 404], ["crown", 405], ["forecast", 410], ["start", 415], ["perim", 420], ["hot", 430], ["inc", 440]]
    .forEach(([n, z]) => { map.createPane(n); map.getPane(n).style.zIndex = z; });
  const hotRenderer = L.canvas({ pane: "hot", padding: 0.3 });

  // ------------------------------------------------------------------ layer registry (panel + legend)
  const LAYERS = {
    forecast: { group: "model", name: "Spread forecast", desc: "Chance each area burns within the forecast window", color: "#ff922b", on: true, opacity: 0.55 },
    start: { group: "model", name: "Fire at forecast start", desc: "Latest perimeter plus satellite detections (48 h)", color: "#495057", on: true, opacity: 0.45 },
    crown: { group: "model", name: "Crown fire potential", desc: "Where fire would likely move into the canopy if it arrives", color: "#2b8a3e", on: false, opacity: 0.35 },
    ember: { group: "model", name: "Ember spotting", desc: "Where embers would land, downwind of intense fire", color: "#9775fa", on: false, opacity: 0.4 },
    actual: { group: "model", name: "What actually burned", desc: "Replay example only: the real next perimeter", color: "#15aabf", on: true, opacity: 1 },
    perimeters: { group: "live", name: "NIFC perimeters", desc: "Current interagency perimeters (WFIGS)", color: "#c92a2a", on: true, opacity: 0.12 },
    incidents: { group: "live", name: "Incident locations", desc: "Wildfires 10+ acres, sized by acres", color: "#e8590c", on: true, opacity: 0.85 },
    hotspots: { group: "live", name: "Satellite hotspots", desc: "VIIRS detections, last 24 h (zoom in to load)", color: "#ffd43b", on: true, opacity: 0.9 },
  };
  const savedLayers = store.get("layers", {});
  for (const [k, v] of Object.entries(savedLayers)) if (LAYERS[k]) Object.assign(LAYERS[k], v);
  const groups = Object.fromEntries(Object.keys(LAYERS).map((k) => [k, L.layerGroup()]));
  for (const [k, l] of Object.entries(LAYERS)) if (l.on) groups[k].addTo(map);

  function saveLayers() { store.set("layers", Object.fromEntries(Object.entries(LAYERS).map(([k, v]) => [k, { on: v.on, opacity: v.opacity }]))); }
  function buildPanel() {
    for (const g of ["live", "model"]) {
      const box = $(g === "live" ? "#live-layers" : "#model-layers");
      box.replaceChildren();
      for (const [k, l] of Object.entries(LAYERS)) {
        if (l.group !== g) continue;
        const id = "lyr-" + k;
        const cb = el("input", { type: "checkbox", id });
        cb.checked = l.on;
        cb.addEventListener("change", () => { l.on = cb.checked; l.on ? groups[k].addTo(map) : map.removeLayer(groups[k]); saveLayers(); renderLegend(); if (k === "hotspots") loadHotspots(); });
        const rng = el("input", { type: "range", min: "0.05", max: "1", step: "0.05", id: id + "-op", "aria-label": l.name + " opacity" });
        rng.value = l.opacity;
        rng.addEventListener("input", () => { l.opacity = +rng.value; restyle(k); saveLayers(); });
        const sw = el("span", { class: "sw" }); sw.style.background = l.color;
        box.append(el("div", { class: "lyr" }, cb, el("label", { for: id }, sw, l.name), el("span", { class: "desc", text: l.desc }), rng));
      }
    }
  }
  function restyle(k) {
    groups[k].eachLayer((lay) => {
      if (lay.setStyle && lay.eachLayer) lay.eachLayer((f) => f.setStyle && f.setStyle(styleFor(k, f.feature)));
      else if (lay.setStyle) lay.setStyle(styleFor(k, lay.feature));
    });
  }
  function styleFor(k, f) {
    const p = (f && f.properties) || {};
    const op = LAYERS[k].opacity;
    if (k === "forecast") return { pane: "forecast", color: BAND_COLORS[p.min_prob], weight: 0.6, opacity: Math.min(1, op + 0.2), fillColor: BAND_COLORS[p.min_prob], fillOpacity: op * (0.45 + p.min_prob * 0.8) };
    if (k === "actual") return { pane: "start", color: "#15aabf", weight: 2.5, opacity: op, fill: false };
    if (k === "start") return { pane: "start", color: "#212529", weight: 1.5, fillColor: "#495057", fillOpacity: op };
    if (k === "crown") return { pane: "crown", color: CROWN_COLORS[p.level], weight: 0, fillColor: CROWN_COLORS[p.level], fillOpacity: op * (p.level === 2 ? 1.2 : 0.8) };
    if (k === "ember") return { pane: "ember", color: EMBER_COLORS[p.level], weight: 0, fillColor: EMBER_COLORS[p.level], fillOpacity: op * (0.5 + p.level * 0.3) };
    if (k === "perimeters") {
      const rx = p.attr_IncidentTypeCategory === "RX";
      return { pane: "perim", color: rx ? "#8f5b2e" : "#c92a2a", weight: 2, dashArray: rx ? "5 4" : null, fillColor: rx ? "#8f5b2e" : "#c92a2a", fillOpacity: op };
    }
    if (k === "incidents") return { pane: "inc", radius: Math.max(5, Math.min(18, 3 + Math.sqrt(p.IncidentSize || 0) / 18)), color: "#fff", weight: 1.5, fillColor: (p.PercentContained || 0) >= 100 ? "#868e96" : "#e8590c", fillOpacity: op };
    if (k === "hotspots") {
      const h = p.hours_old ?? 24;
      const c = (HOT_COLORS.find(([lim]) => h <= lim) || HOT_COLORS[3])[1];
      return { renderer: hotRenderer, radius: 4, color: "#00000066", weight: 0.6, fillColor: c, fillOpacity: op };
    }
    return {};
  }

  function renderLegend() {
    const lg = $("#legend");
    lg.replaceChildren();
    if (LAYERS.forecast.on && state.index && state.forecastLayers.size) {
      lg.append(el("h4", { text: `Chance of burning, next ${state.index.horizon_hours || 24} h` }));
      const ramp = el("div", { class: "ramp" }), lbl = el("div", { class: "ramp-lbl" });
      for (const b of [0.1, 0.3, 0.5, 0.7]) { const s = el("span"); s.style.background = BAND_COLORS[b]; ramp.append(s); lbl.append(el("span", { text: `${b * 100}%+` })); }
      lg.append(ramp, lbl);
    }
    const rows = [];
    if (LAYERS.perimeters.on) rows.push(["#c92a2a", "NIFC perimeter (dashed: prescribed)"]);
    if (LAYERS.start.on && state.forecastLayers.size) rows.push(["#495057", "Fire at forecast start"]);
    if (LAYERS.crown.on && state.forecastLayers.size) rows.push(["#2b8a3e", "Crown fire likely if reached"]);
    if (LAYERS.ember.on && state.forecastLayers.size) rows.push(["#9775fa", "Ember landing"]);
    if (LAYERS.actual.on && state.forecastLayers.has("__example")) rows.push(["#15aabf", "What actually burned (example)"]);
    if (LAYERS.hotspots.on) rows.push(["#ffe066", "Hotspot < 6 h", true], ["#e03131", "Hotspot 12–24 h", true]);
    if (LAYERS.incidents.on) rows.push(["#e8590c", "Incident (grey: 100% contained)", true]);
    if (rows.length) {
      if (lg.childElementCount) lg.append(el("div", { style: "height:6px" }));
      for (const [c, t, dot] of rows) { const i = el("i", { class: dot ? "dot" : "" }); i.style.background = c; lg.append(el("div", { class: "row" }, i, el("span", { text: t }))); }
    }
    lg.hidden = !lg.childElementCount;
  }

  // ------------------------------------------------------------------ state
  const state = {
    perimeters: new Map(),     // fire id -> leaflet layer
    incidents: [],             // live incident properties + latlng
    index: null,               // data/index.json
    forecastLayers: new Map(), // fire id -> {forecast, start, crown, ember, bounds}
    selected: null,
    liveAt: null,
  };

  // ------------------------------------------------------------------ live: perimeters
  async function loadPerimeters() {
    try {
      const gj = await getJSON(URLS.perimeters, {
        where: "1=1", outSR: 4326, f: "geojson", geometryPrecision: 5, maxAllowableOffset: 0.0002, resultRecordCount: 2000,
        outFields: "attr_UniqueFireIdentifier,attr_IncidentName,poly_GISAcres,attr_PercentContained,poly_PolygonDateTime,poly_MapMethod,attr_IncidentTypeCategory,attr_POOState",
      });
      groups.perimeters.clearLayers(); state.perimeters.clear();
      L.geoJSON(gj, {
        style: (f) => styleFor("perimeters", f),
        onEachFeature: (f, lay) => {
          const p = f.properties;
          state.perimeters.set(p.attr_UniqueFireIdentifier, lay);
          lay.bindPopup(() => perimeterPopup(p));
        },
      }).addTo(groups.perimeters);
      state.liveAt = Date.now();
      updateLivePill(gj.features.length);
    } catch (e) {
      pill("#st-live", "Perimeters: NIFC unreachable, retrying", "warn");
      console.warn("perimeters", e);
    }
  }
  function updateLivePill(n) {
    if (n !== undefined) updateLivePill.n = n;
    if (state.liveAt) pill("#st-live", `Perimeters: live · ${updateLivePill.n} fires · ${ago(state.liveAt)}`, "ok");
  }
  function perimeterPopup(p) {
    const fc = state.index && state.index.fires.find((f) => f.id === p.attr_UniqueFireIdentifier && f.forecast);
    return `<div class="pop"><h5>${esc(p.attr_IncidentName)}</h5><dl class="kv">
      <dt>Size</dt><dd>${fmtAc(p.poly_GISAcres)}</dd>
      <dt>Contained</dt><dd>${p.attr_PercentContained ?? "–"}%</dd>
      <dt>Perimeter</dt><dd>${fmtTime(p.poly_PolygonDateTime)}</dd>
      <dt>Mapped by</dt><dd>${esc(p.poly_MapMethod || "–")}</dd>
      <dt>Type</dt><dd>${p.attr_IncidentTypeCategory === "RX" ? "Prescribed fire" : "Wildfire"}</dd></dl>
      ${fc ? `<p class="note">Spread forecast available: see the fire list.</p>` : ""}</div>`;
  }

  // ------------------------------------------------------------------ live: incidents
  async function loadIncidents() {
    try {
      const gj = await getJSON(URLS.incidents, {
        where: "IncidentTypeCategory='WF' AND IncidentSize>=10", outSR: 4326, f: "geojson", resultRecordCount: 2000,
        outFields: "IncidentName,IncidentSize,PercentContained,FireDiscoveryDateTime,ModifiedOnDateTime_dt,POOState,UniqueFireIdentifier,TotalIncidentPersonnel,FireBehaviorGeneral,IncidentShortDescription",
      });
      groups.incidents.clearLayers();
      state.incidents = [];
      L.geoJSON(gj, {
        pointToLayer: (f, ll) => L.circleMarker(ll, styleFor("incidents", f)),
        onEachFeature: (f, lay) => {
          const p = f.properties;
          state.incidents.push({ ...p, lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], layer: lay });
          lay.bindPopup(() => incidentPopup(p));
          lay.on("click", () => select(p.UniqueFireIdentifier, false));
        },
      }).addTo(groups.incidents);
      renderList();
    } catch (e) { console.warn("incidents", e); }
  }
  function incidentPopup(p) {
    return `<div class="pop"><h5>${esc(p.IncidentName)}</h5><dl class="kv">
      <dt>Size</dt><dd>${fmtAc(p.IncidentSize)}</dd>
      <dt>Contained</dt><dd>${p.PercentContained ?? "–"}%</dd>
      <dt>State</dt><dd>${esc((p.POOState || "").replace("US-", ""))}</dd>
      <dt>Discovered</dt><dd>${fmtTime(p.FireDiscoveryDateTime)}</dd>
      ${p.TotalIncidentPersonnel ? `<dt>Personnel</dt><dd>${p.TotalIncidentPersonnel.toLocaleString()}</dd>` : ""}
      ${p.FireBehaviorGeneral ? `<dt>Behavior</dt><dd>${esc(p.FireBehaviorGeneral)}</dd>` : ""}
      <dt>Updated</dt><dd>${fmtTime(p.ModifiedOnDateTime_dt)}</dd></dl></div>`;
  }

  // ------------------------------------------------------------------ live: VIIRS hotspots (current view only)
  let hotTimer = null, hotSeq = 0;
  function loadHotspots() {
    clearTimeout(hotTimer);
    hotTimer = setTimeout(async () => {
      groups.hotspots.clearLayers();
      if (!LAYERS.hotspots.on || map.getZoom() < HOTSPOT_MIN_ZOOM) return;
      const b = map.getBounds().pad(0.15), seq = ++hotSeq;
      try {
        const gj = await getJSON(URLS.viirs, {
          where: "hours_old<=24", geometry: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(4)).join(","),
          geometryType: "esriGeometryEnvelope", inSR: 4326, spatialRel: "esriSpatialRelIntersects", outSR: 4326, f: "geojson",
          outFields: "acq_date,acq_time,esritimeutc,hours_old,frp,confidence,satellite,daynight", resultRecordCount: 8000,
        });
        if (seq !== hotSeq) return;
        L.geoJSON(gj, {
          pointToLayer: (f, ll) => L.circleMarker(ll, { ...styleFor("hotspots", f), pane: "hot" }),
          onEachFeature: (f, lay) => lay.bindPopup(() => {
            const p = f.properties;
            return `<div class="pop"><h5>Satellite hotspot</h5><dl class="kv">
              <dt>Detected</dt><dd>${fmtTime(p.esritimeutc)} (${p.hours_old ?? "?"} h ago)</dd>
              <dt>Fire power</dt><dd>${p.frp ?? "–"} MW</dd><dt>Confidence</dt><dd>${esc(p.confidence ?? "–")}</dd>
              <dt>Satellite</dt><dd>${esc(p.satellite ?? "–")} · ${p.daynight === "N" ? "night" : "day"}</dd></dl>
              <p class="note">375 m pixel: the fire is somewhere inside it.</p></div>`;
          }),
        }).addTo(groups.hotspots);
      } catch (e) { console.warn("hotspots", e); }
    }, 350);
  }
  map.on("moveend", loadHotspots);
  map.on("zoomend", () => { if (map.getZoom() < HOTSPOT_MIN_ZOOM && LAYERS.hotspots.on && !loadHotspots._told) { loadHotspots._told = true; toast("Zoom in to load satellite hotspots"); } });

  // ------------------------------------------------------------------ model forecasts
  async function loadIndex() {
    let idx;
    try { idx = await getJSON("data/index.json?t=" + Date.now()); }
    catch (e) {
      pill("#st-fc", "Forecast: not published yet", "warn");
      $("#model-note").textContent = "No forecast run has been published yet.";
      await loadExample(); renderList(); return;
    }
    state.index = idx;
    const n = idx.forecast_count || 0;
    const when = fmtTime(idx.generated);
    $("#fc-horizon").textContent = `· next ${idx.horizon_hours || 24} h`;
    $("#model-note").textContent = `${idx.model?.name || "Model"}. ${idx.model?.note || ""}${idx.model?.test_ap ? ` Test score (AP) ${idx.model.test_ap}.` : ""} Run ${when}.`;
    pill("#st-fc", n ? `Forecast: ${n} fire${n > 1 ? "s" : ""} · run ${ago(new Date(idx.generated))}` : `Forecast: no actively burning fires · run ${ago(new Date(idx.generated))}`, n ? "ok" : "");
    const wanted = new Set(idx.fires.filter((f) => f.forecast).map((f) => f.id));
    for (const id of [...state.forecastLayers.keys()]) if (!wanted.has(id) && id !== "__example") dropForecast(id);
    await Promise.all(idx.fires.filter((f) => f.forecast).map((f) => loadForecast(f.id, f.forecast + "?t=" + encodeURIComponent(idx.generated), f)));
    if (!n) await loadExample(); else dropForecast("__example");
    renderList(); renderLegend();
  }
  async function loadExample() {
    if (state.forecastLayers.has("__example")) return;
    try {
      const ex = await getJSON("data/example/index.json");
      state.example = ex;
      await loadForecast("__example", "data/example/" + ex.file, ex);
    } catch { /* no example shipped */ }
  }
  function dropForecast(id) {
    const fl = state.forecastLayers.get(id);
    if (!fl) return;
    for (const k of ["forecast", "start", "crown", "ember", "actual"]) if (fl[k]) groups[k].removeLayer(fl[k]);
    state.forecastLayers.delete(id);
  }
  async function loadForecast(id, url, meta) {
    let gj;
    try { gj = await getJSON(url); } catch (e) { console.warn("forecast", id, e); return; }
    dropForecast(id);
    const fl = { meta };
    for (const k of ["ember", "crown", "forecast", "start", "actual"]) {
      const feats = gj.features.filter((f) => f.properties.layer === k);
      if (k === "forecast") feats.sort((a, b) => a.properties.min_prob - b.properties.min_prob);
      if (!feats.length) continue;
      fl[k] = L.geoJSON({ type: "FeatureCollection", features: feats }, {
        style: (f) => styleFor(k, f),
        onEachFeature: (f, lay) => lay.bindPopup(() => forecastPopup(meta, f.properties)),
      }).addTo(groups[k]);
    }
    // zoom to the fire and its forecast, not to the crown/ember zones that cover the whole grid
    const core = gj.features.filter((f) => ["start", "forecast", "actual"].includes(f.properties.layer));
    fl.bounds = L.geoJSON({ type: "FeatureCollection", features: core.length ? core : gj.features }).getBounds();
    state.forecastLayers.set(id, fl);
  }
  function windText(w) {
    return w ? `From ${w.from_compass} at ${w.mean_mph} mph, gusts to ${w.peak_gust_mph} mph · min RH ${w.min_rh_pct}%` : null;
  }
  function forecastPopup(meta, p) {
    const s = meta.stats || {};
    const w = windText(s.wind);
    return `<div class="pop"><h5>${esc(meta.name)}${meta.example ? " (example)" : ""}</h5>
      <p style="margin:0 0 6px;font-weight:600">${esc(p.label || "")}</p><dl class="kv">
      <dt>Expected growth</dt><dd>${fmtAc(s.expected_new_acres)}</dd>
      <dt>Likely (50%+)</dt><dd>${fmtAc(s.likely_acres)}</dd>
      <dt>Possible (10%+)</dt><dd>${fmtAc(s.possible_acres)}</dd>
      ${s.actual_new_acres !== undefined ? `<dt>Actually burned</dt><dd>${fmtAc(s.actual_new_acres)}</dd>` : ""}
      <dt>Forecast from</dt><dd>${fmtTime(meta.origin)}</dd>
      <dt>Window</dt><dd>${meta.example ? meta.hours : (state.index?.horizon_hours || meta.hours || 24)} h</dd>
      ${w ? `<dt>Wind</dt><dd>${esc(w)}</dd>` : ""}</dl>
      <p class="note">${meta.example ? esc(meta.note || "") : "Model output, not an official forecast. Weather: HRRR forecast run " + esc(fmtTime(meta.hrrr_run))}</p></div>`;
  }

  // ------------------------------------------------------------------ fire list
  function renderList() {
    const q = $("#search").value.trim().toLowerCase();
    const match = (name, st) => !q || (name || "").toLowerCase().includes(q) || (st || "").toLowerCase().includes(q);
    const box = $("#fire-list");
    box.replaceChildren();
    const fc = (state.index?.fires || []).filter((f) => f.forecast && match(f.name, f.state));
    if (fc.length) {
      box.append(el("p", { class: "section-label", text: "Spread forecast available" }));
      fc.sort((a, b) => (b.stats?.expected_new_acres || 0) - (a.stats?.expected_new_acres || 0));
      for (const f of fc) box.append(fireRow(f.id, f.name, f.state, f.acres, f.contained,
        `Perimeter ${fmtTime(f.perimeter_time || f.updated)}`, forecastTags(f)));
    }
    if (state.example && !fc.length && match(state.example.name, "")) {
      box.append(el("p", { class: "section-label", text: "Example forecast (replay)" }));
      box.append(fireRow("__example", state.example.name, state.example.state, state.example.acres, null, state.example.note, forecastTags(state.example)));
    }
    const fcIds = new Set(fc.map((f) => f.id));
    const inc = state.incidents.filter((p) => !fcIds.has(p.UniqueFireIdentifier) && match(p.IncidentName, p.POOState))
      .sort((a, b) => (b.IncidentSize || 0) - (a.IncidentSize || 0));
    const open = inc.filter((p) => (p.PercentContained ?? 0) < 100), done = inc.filter((p) => (p.PercentContained ?? 0) >= 100);
    const row = (p) => fireRow(p.UniqueFireIdentifier, p.IncidentName, p.POOState, p.IncidentSize, p.PercentContained, `Updated ${ago(p.ModifiedOnDateTime_dt)}`, null);
    if (open.length) {
      box.append(el("p", { class: "section-label", text: `Not yet contained (${open.length})` }));
      open.slice(0, 150).forEach((p) => box.append(row(p)));
    }
    if (done.length) {
      const det = el("details", { class: "done" }, el("summary", { class: "section-label", text: `100% contained (${done.length})` }));
      done.slice(0, 150).forEach((p) => det.append(row(p)));
      if (q) det.open = true;
      box.append(det);
    }
    if (!box.childElementCount) box.append(el("p", { class: "empty", text: q ? "No fires match that search." : "No active wildfires of 10+ acres reported right now." }));
  }
  function forecastTags(f) {
    const s = f.stats || {};
    const tags = [];
    if (s.expected_new_acres !== undefined) tags.push(el("span", { class: "tag hot", text: `+${s.expected_new_acres.toLocaleString()} ac expected in ${f.example ? f.hours : (state.index?.horizon_hours || f.hours || 24)} h` }));
    if (s.wind) tags.push(el("span", { class: "tag", text: `Wind ${s.wind.from_compass} ${s.wind.mean_mph} mph` }));
    return tags;
  }
  function fireRow(id, name, st, acres, contained, meta, tags) {
    const c = el("span", { class: "contain", title: `${contained ?? 0}% contained` }, el("b"));
    c.firstChild.style.height = `${Math.max(0, Math.min(100, contained || 0))}%`;
    const b = el("button", { class: "fire" + (state.selected === id ? " sel" : ""), "data-id": id, onclick: () => select(id, true) },
      c, el("span", { class: "nm", text: name || "Unnamed fire" }), el("span", { class: "ac", text: fmtAc(acres) }),
      el("span", { class: "meta", text: [(st || "").replace("US-", ""), contained !== null && contained !== undefined ? `${contained}% contained` : null, meta].filter(Boolean).join(" · ") }),
      tags && tags.length ? el("span", { class: "fc" }, ...tags) : null);
    return b;
  }
  $("#search").addEventListener("input", renderList);

  function select(id, fly) {
    state.selected = id;
    document.querySelectorAll(".fire").forEach((b) => b.classList.toggle("sel", b.dataset.id === id));
    try { history.replaceState(null, "", id ? "#fire=" + encodeURIComponent(id) : location.pathname); } catch { /* sandboxed */ }
    if (!fly) return;
    const fl = state.forecastLayers.get(id);
    const per = state.perimeters.get(id);
    const inc = state.incidents.find((p) => p.UniqueFireIdentifier === id);
    const pad = { paddingTopLeft: [20, 20], paddingBottomRight: [20, innerWidth <= 760 ? innerHeight * 0.4 : 20], maxZoom: 13 };
    const after = (fn) => map.once("moveend", () => setTimeout(fn, 50));
    if (fl && fl.bounds && fl.bounds.isValid()) {
      const lay = fl.start || fl.forecast;
      if (lay) after(() => { const first = lay.getLayers()[0]; if (first) first.openPopup(fl.bounds.getCenter()); });
      map.flyToBounds(fl.bounds, pad);
    } else if (per) { after(() => per.openPopup()); map.flyToBounds(per.getBounds(), pad); }
    else if (inc) { after(() => inc.layer.openPopup()); map.flyTo([inc.lat, inc.lon], 11); }
    if (innerWidth <= 760) setSheet("min");
  }

  // ------------------------------------------------------------------ phone bottom sheet
  const side = $("#side");
  function setSheet(mode) {
    side.classList.toggle("open", mode === "open");
    side.classList.toggle("min", mode === "min");
    document.body.classList.toggle("sheet-open", mode === "open");
    document.body.classList.toggle("sheet-min", mode === "min");
    $("#sheet-handle").setAttribute("aria-label", mode === "open" ? "Shrink fire list" : "Expand fire list");
  }
  $("#sheet-handle").addEventListener("click", () => setSheet(side.classList.contains("open") ? "min" : side.classList.contains("min") ? "" : "open"));
  $("#search").addEventListener("focus", () => { if (innerWidth <= 760) setSheet("open"); });

  // ------------------------------------------------------------------ panel, locate
  const panel = $("#panel");
  const togglePanel = (show) => { panel.hidden = !show; $("#btn-layers").setAttribute("aria-expanded", String(show)); };
  $("#btn-layers").addEventListener("click", () => togglePanel(panel.hidden));
  $("#panel-close").addEventListener("click", () => togglePanel(false));
  let me = null;
  $("#btn-locate").addEventListener("click", () => map.locate({ setView: true, maxZoom: 11 }));
  map.on("locationfound", (e) => { if (me) map.removeLayer(me); me = L.circleMarker(e.latlng, { radius: 7, color: "#fff", weight: 2, fillColor: "#1c7ed6", fillOpacity: 1 }).addTo(map).bindPopup("You are here"); });
  map.on("locationerror", () => toast("Location is not available (permission denied or no signal)"));

  // ------------------------------------------------------------------ start
  buildPanel(); renderLegend();
  if (!store.get("seenPanel", false) && innerWidth > 760) { togglePanel(true); store.set("seenPanel", true); }
  (async () => {
    await Promise.all([loadPerimeters(), loadIncidents(), loadIndex()]);
    renderLegend(); loadHotspots();
    const h = decodeURIComponent((location.hash.match(/fire=([^&]+)/) || [])[1] || "");
    if (h) select(h, true);
  })();
  setInterval(() => { loadPerimeters(); loadIncidents(); loadHotspots(); }, LIVE_MS);
  setInterval(loadIndex, INDEX_MS);
  setInterval(() => updateLivePill(), 30000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.liveAt && Date.now() - state.liveAt > LIVE_MS) { loadPerimeters(); loadIncidents(); loadHotspots(); loadIndex(); } });
})();
