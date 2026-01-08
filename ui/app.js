const CSV_DIR = "../data/feat";
const VIDEO_DIR = "../data/converted_videos";
const PLAY_RATE = 0.5;
const MAX_POINTS = 360;
const LANE_HALF_WIDTH_M = 1.8;
const LANE_OFFSET_TAU = 0.35;
const LANE_OFFSET_HOLD_S = 0.6;

const chartConfigs = {
  polar: {
    dataset: "polar",
    timeKey: "t_seconds",
    defaultMode: "hrv",
    modes: {
      hrv: {
        label: "HRV",
        series: [
          { key: "polar_rmssd", label: "RMSSD", color: "#0a84ff" },
          { key: "polar_sdnn", label: "SDNN", color: "#34c759" }
        ]
      },
      quality: {
        label: "Quality",
        series: [
          { key: "polar_quality_pct", label: "Quality", color: "#0a84ff" },
          { key: "polar_reliable", label: "Reliable", color: "#34c759" }
        ]
      }
    }
  },
  muse: {
    dataset: "muse",
    timeKey: "t_seconds",
    defaultMode: "bands",
    modes: {
      bands: {
        label: "Bands",
        series: [
          { key: "muse_alpha_mean", label: "Alpha", color: "#0a84ff" },
          { key: "muse_beta_mean", label: "Beta", color: "#ff9500" },
          { key: "muse_theta_mean", label: "Theta", color: "#34c759" },
          { key: "muse_gamma_mean", label: "Gamma", color: "#af52de" }
        ]
      },
      ratios: {
        label: "Ratios",
        series: [
          { key: "muse_beta_alpha_ratio", label: "Beta/Alpha", color: "#ff9500" }
        ]
      }
    }
  },
  obd: {
    dataset: "obd",
    timeKey: "t_seconds",
    defaultMode: "vehicle",
    modes: {
      vehicle: {
        label: "Vehicle",
        series: [
          { key: "obd_Vehicle Speed Sensor [km/h]", label: "Speed", color: "#0a84ff" },
          { key: "obd_Engine RPM [RPM]", label: "RPM", color: "#ff9500" }
        ]
      },
      throttle: {
        label: "Throttle",
        series: [
          { key: "obd_Absolute Throttle Position [%]", label: "Throttle", color: "#0a84ff" },
          { key: "obd_Engine torque [Nm]", label: "Torque", color: "#34c759" }
        ]
      }
    }
  },
  phyphox: {
    dataset: "phyphox",
    timeKey: "t_seconds",
    defaultMode: "motion",
    modes: {
      motion: {
        label: "Acceleration",
        series: [
          { key: "phy_X (m/s^2)", label: "Accel X", color: "#0a84ff" },
          { key: "phy_Y (m/s^2)", label: "Accel Y", color: "#34c759" },
          { key: "phy_Z (m/s^2)", label: "Accel Z", color: "#ff9500" }
        ]
      },
      rotation: {
        label: "Gyro",
        series: [
          { key: "phy_X (rad/s)", label: "Gyro X", color: "#0a84ff" },
          { key: "phy_Y (rad/s)", label: "Gyro Y", color: "#34c759" },
          { key: "phy_Z (rad/s)", label: "Gyro Z", color: "#ff9500" }
        ]
      }
    }
  }
};

const datasetTimeKeys = {
  fused: "t_seconds",
  polar: "t_seconds",
  muse: "t_seconds",
  obd: "t_seconds",
  phyphox: "t_seconds"
};

const state = {
  datasets: {},
  events: [],
  charts: {},
  chartModes: {},
  metricElements: {},
  godMode: true,
  loaded: false,
  loading: false,
  playing: false,
  dataStart: 0,
  timelineStart: 0,
  duration: 0,
  dataDuration: 0,
  currentTime: 0,
  lastTick: 0,
  lastRoadSync: 0,
  lastDriverSync: 0,
  lastRoadCheck: 0,
  lastDriverCheck: 0,
  lastRoadTime: null,
  lastDriverTime: null,
  anchorTime: null,
  videoOffsets: {
    road: 0,
    driver: 0
  },
  laneOffset: {
    current: null,
    lastValid: null,
    lastValidTime: null,
    lastUpdateTime: null
  },
  logMetrics: {
    capacity: 5,
    itemHeight: 0,
    gap: 0,
    listHeight: 0,
    needsMeasure: true,
    observer: null
  }
};

const ui = {};

document.addEventListener("DOMContentLoaded", () => {
  ui.loadButton = document.getElementById("loadButton");
  ui.loadStatus = document.getElementById("loadStatus");
  ui.playButton = document.getElementById("playButton");
  ui.roadVideo = document.getElementById("roadVideo");
  ui.driverVideo = document.getElementById("driverVideo");
  ui.roadStatusChip = document.getElementById("roadStatusChip");
  ui.driverStatusChip = document.getElementById("driverStatusChip");
  ui.roadFallback = document.getElementById("roadFallback");
  ui.driverFallback = document.getElementById("driverFallback");
  ui.currentTime = document.getElementById("currentTime");
  ui.totalTime = document.getElementById("totalTime");
  ui.transportProgress = document.getElementById("transportProgress");
  ui.transportRate = document.getElementById("transportRate");
  ui.playRateBadge = document.getElementById("playRateBadge");
  ui.sessionTitle = document.getElementById("sessionTitle");
  ui.sessionSubtitle = document.getElementById("sessionSubtitle");
  ui.decisionStatus = document.getElementById("decisionStatus");
  ui.decisionConfidence = document.getElementById("decisionConfidence");
  ui.decisionReasons = document.getElementById("decisionReasons");
  ui.decisionStart = document.getElementById("decisionStart");
  ui.decisionEnd = document.getElementById("decisionEnd");
  ui.decisionDuration = document.getElementById("decisionDuration");
  ui.logList = document.getElementById("logList");
  ui.leftLaneStatus = document.getElementById("leftLaneStatus");
  ui.rightLaneStatus = document.getElementById("rightLaneStatus");
  ui.leftLaneDist = document.getElementById("leftLaneDist");
  ui.rightLaneDist = document.getElementById("rightLaneDist");
  ui.leftLaneTlc = document.getElementById("leftLaneTlc");
  ui.rightLaneTlc = document.getElementById("rightLaneTlc");
  ui.laneTracks = Array.from(document.querySelectorAll(".lane-track"));

  ui.loadButton.addEventListener("click", loadAllData);
  ui.playButton.addEventListener("click", togglePlay);
  document.querySelectorAll("[data-skip]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const delta = Number(btn.dataset.skip || 0);
      seekTo(state.currentTime + delta);
    });
  });

  ui.playButton.disabled = true;
  ui.transportRate.textContent = `${PLAY_RATE.toFixed(1)}x`;
  if (ui.playRateBadge) {
    ui.playRateBadge.textContent = `Play ${PLAY_RATE.toFixed(1)}x`;
  }

  state.metricElements = collectMetricElements();
  initFlagGroups();
  initChartModes();
  initLogMetrics();

  if (window.location.protocol === "file:") {
    setLoadStatus("Apri con un server locale per caricare i CSV.", true);
  }

  updateLoadButtonAvailability();
});

function initFlagGroups() {
  document.querySelectorAll("[data-flag-group]").forEach((master) => {
    const group = master.dataset.flagGroup;
    const items = Array.from(document.querySelectorAll(`[data-flag-item="${group}"]`));

    const syncMaster = () => {
      const allOn = items.every((item) => item.checked);
      const allOff = items.every((item) => !item.checked);
      master.checked = allOn;
      master.indeterminate = !allOn && !allOff;
    };

    master.addEventListener("change", () => {
      items.forEach((item) => {
        item.checked = master.checked;
      });
      master.indeterminate = false;
    });

    items.forEach((item) => {
      item.addEventListener("change", syncMaster);
    });

    syncMaster();
  });
}

function initChartModes() {
  document.querySelectorAll("[data-chart-mode-group]").forEach((group) => {
    const chartId = group.dataset.chartModeGroup;
    const buttons = Array.from(group.querySelectorAll("[data-chart-mode]"));
    if (!chartId || !buttons.length) {
      return;
    }

    const config = chartConfigs[chartId];
    const defaultMode = group.dataset.defaultMode
      || config?.defaultMode
      || buttons[0].dataset.chartMode;
    setChartMode(chartId, defaultMode, true);

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        setChartMode(chartId, button.dataset.chartMode);
      });
    });
  });
}

function setChartMode(chartId, mode, silent = false) {
  const config = chartConfigs[chartId];
  if (!config) {
    return;
  }
  const modes = config.modes ? Object.keys(config.modes) : [];
  const resolvedMode = modes.includes(mode)
    ? mode
    : (config.defaultMode || modes[0] || mode);

  state.chartModes[chartId] = resolvedMode;
  updateChartModeUi(chartId, resolvedMode);

  if (!silent) {
    buildCharts();
    updateCharts(state.timelineStart + state.currentTime);
  }
}

function updateChartModeUi(chartId, mode) {
  const group = document.querySelector(`[data-chart-mode-group="${chartId}"]`);
  if (group) {
    group.querySelectorAll("[data-chart-mode]").forEach((button) => {
      const isActive = button.dataset.chartMode === mode;
      button.classList.toggle("chip--accent", isActive);
      button.classList.toggle("chip--muted", !isActive);
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  document.querySelectorAll(`[data-widget-group="${chartId}"]`).forEach((stack) => {
    const isActive = stack.dataset.widgetMode === mode;
    stack.classList.toggle("is-active", isActive);
  });
}

function initLogMetrics() {
  if (!ui.logList) {
    return;
  }

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      state.logMetrics.needsMeasure = true;
    });
    observer.observe(ui.logList);
    state.logMetrics.observer = observer;
    return;
  }

  window.addEventListener("resize", () => {
    state.logMetrics.needsMeasure = true;
  });
}

function isGodModeEnabled() {
  return state.godMode;
}

function updateLoadButtonAvailability() {
  if (!ui.loadButton) {
    return;
  }
  const disableStart = !state.loaded && !isGodModeEnabled();
  ui.loadButton.disabled = state.loading || disableStart;
}

function collectMetricElements() {
  const elements = Array.from(document.querySelectorAll("[data-source][data-field]"));
  const map = {};

  elements.forEach((el) => {
    const source = el.dataset.source;
    if (!map[source]) {
      map[source] = [];
    }
    map[source].push({
      el,
      field: el.dataset.field,
      precision: Number(el.dataset.precision || 2)
    });
  });

  return map;
}

async function loadAllData() {
  if (state.loading) {
    return;
  }
  if (!state.loaded && !isGodModeEnabled()) {
    return;
  }
  if (state.loaded) {
    const confirmed = window.confirm("Reset session? The page will reload.");
    if (!confirmed) {
      return;
    }
    window.location.reload();
    return;
  }

  state.loading = true;
  updateLoadButtonAvailability();
  ui.loadButton.classList.remove("action-btn--reset");
  ui.loadButton.textContent = "Caricamento...";

  try {
    const [fused, polar, muse, obd, phyphox, eventsData] = await Promise.all([
      loadCsvFile("fused_10hz.csv"),
      loadCsvFile("polar_features.csv"),
      loadCsvFile("muse_features.csv"),
      loadCsvFile("obd_features.csv"),
      loadCsvFile("phyphox_features.csv"),
      loadCsvFile("lane_events.csv")
    ]);

    state.datasets = { fused, polar, muse, obd, phyphox };
    state.events = buildEvents(eventsData);
    state.dataStart = getTimelineStart(fused);
    state.timelineStart = state.dataStart;
    state.dataDuration = getTimelineDuration(fused);
    state.duration = state.dataDuration;
    state.currentTime = 0;
    state.laneOffset = {
      current: null,
      lastValid: null,
      lastValidTime: null,
      lastUpdateTime: null
    };
    const sessionConfig = await loadSessionConfig();
    state.videoOffsets = { road: sessionConfig.road, driver: sessionConfig.driver };
    state.anchorTime = await loadAnchorTime(sessionConfig);

    buildCharts();
    await updateVideoSources();
    updateUi(0);

    ui.playButton.disabled = false;
    setLoadButtonState("reset");
    setLoadStatus("Database in use: data/feat, /video_converted");
    state.loaded = true;
  } catch (error) {
    setLoadStatus("Errore nel caricamento. Avvia un server locale.", true);
    setLoadButtonState(state.loaded ? "reset" : "start");
  } finally {
    state.loading = false;
    updateLoadButtonAvailability();
  }
}

function loadCsvFile(fileName) {
  return fetch(`${CSV_DIR}/${fileName}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load ${fileName}`);
      }
      return response.text();
    })
    .then((text) => parseCsv(text));
}

async function loadSessionConfig() {
  const config = { road: 0, driver: 0, sessionDate: "", anchorObdTime: "" };

  try {
    const response = await fetch("../config/session.yaml");
    if (!response.ok) {
      return config;
    }
    const text = await response.text();
    config.road = parseYamlNumber(text, "anchor_road_video_s");
    config.driver = parseYamlNumber(text, "anchor_driver_video_s");
    config.sessionDate = parseYamlString(text, "session_date");
    config.anchorObdTime = parseYamlString(text, "anchor_obd_time");
  } catch (error) {
    return config;
  }

  return config;
}

function parseYamlNumber(text, key) {
  const regex = new RegExp(`^${key}:\\s*([^\\n#]+)`, "m");
  const match = text.match(regex);
  if (!match) {
    return 0;
  }
  const value = Number(String(match[1]).trim());
  return Number.isFinite(value) ? value : 0;
}

function parseYamlString(text, key) {
  const regex = new RegExp(`^${key}:\\s*([^\\n#]+)`, "m");
  const match = text.match(regex);
  if (!match) {
    return "";
  }
  const raw = String(match[1]).trim();
  return raw.replace(/^["']|["']$/g, "");
}

async function loadAnchorTime(config) {
  if (!config || !config.sessionDate || !config.anchorObdTime) {
    return null;
  }
  const anchorMs = parseDateTimeValue(`${config.sessionDate} ${config.anchorObdTime}`, config.sessionDate);
  if (!Number.isFinite(anchorMs)) {
    return null;
  }
  const rawStart = await loadRawStartTime(config.sessionDate);
  if (!Number.isFinite(rawStart)) {
    return null;
  }

  let globalStart = rawStart;
  const roadStart = anchorMs - (Number.isFinite(config.road) ? config.road : 0) * 1000;
  const driverStart = anchorMs - (Number.isFinite(config.driver) ? config.driver : 0) * 1000;

  if (Number.isFinite(roadStart)) {
    globalStart = Math.min(globalStart, roadStart);
  }
  if (Number.isFinite(driverStart)) {
    globalStart = Math.min(globalStart, driverStart);
  }

  return (anchorMs - globalStart) / 1000;
}

async function loadRawStartTime(sessionDate) {
  const files = await listRawCsvFiles(sessionDate);
  if (!files.length) {
    return null;
  }
  const starts = await Promise.all(files.map((file) => loadCsvStartTime(file, sessionDate)));
  const valid = starts.filter((value) => Number.isFinite(value));
  return valid.length ? Math.min(...valid) : null;
}

async function listRawCsvFiles(sessionDate) {
  try {
    const response = await fetch("../data/raw/");
    if (!response.ok) {
      return [];
    }
    const text = await response.text();
    const doc = new DOMParser().parseFromString(text, "text/html");
    const links = Array.from(doc.querySelectorAll("a"));
    const names = links
      .map((link) => link.getAttribute("href") || "")
      .filter((name) => name.endsWith(".csv") && !name.includes("baseline"));

    const sensorNames = names.filter((name) => {
      const lower = name.toLowerCase();
      return ["muse", "polar", "phyphox", "obd"].some((token) => lower.includes(token));
    });

    const filtered = sessionDate
      ? sensorNames.filter((name) => name.includes(sessionDate))
      : sensorNames;
    const selected = filtered.length ? filtered : (sensorNames.length ? sensorNames : names);

    return selected.map((name) => `../data/raw/${name.replace(/^[.\\/]+/, "")}`);
  } catch (error) {
    return [];
  }
}

async function loadCsvStartTime(url, sessionDate) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const text = await response.text();
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length);
    if (lines.length < 2) {
      return null;
    }
    const headers = splitCsvLine(lines[0]);
    const timeColumn = findTimeColumn(headers);
    if (!timeColumn) {
      return null;
    }
    const row = splitCsvLine(lines[1]);
    const index = headers.indexOf(timeColumn);
    return parseDateTimeValue(row[index], sessionDate);
  } catch (error) {
    return null;
  }
}

function findTimeColumn(headers) {
  if (!headers || !headers.length) {
    return "";
  }
  const lower = headers.map((header) => header.trim().toLowerCase());
  const timestampIndex = lower.findIndex((header) => header.includes("timestamp"));
  if (timestampIndex >= 0) {
    return headers[timestampIndex];
  }
  const datetimeIndex = lower.findIndex((header) => header.includes("datetime") || header.includes("date_time"));
  if (datetimeIndex >= 0) {
    return headers[datetimeIndex];
  }
  const timeIndex = lower.findIndex((header) => header === "time" || header.endsWith("_time"));
  if (timeIndex >= 0) {
    return headers[timeIndex];
  }
  return "";
}

function parseDateTimeValue(value, sessionDate) {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return null;
  }
  const match = trimmed.match(/^(\d{4}-\d{2}-\d{2})[ T](.+)$/);
  if (match) {
    return toUtcMillis(match[1], match[2]);
  }
  if (sessionDate) {
    return toUtcMillis(sessionDate, trimmed);
  }
  return null;
}

function toUtcMillis(dateStr, timeStr) {
  if (!dateStr || !timeStr) {
    return null;
  }
  const dateParts = dateStr.split("-").map((part) => Number(part));
  if (dateParts.length !== 3 || dateParts.some((part) => Number.isNaN(part))) {
    return null;
  }
  const [year, month, day] = dateParts;
  const cleanedTime = String(timeStr).trim().replace(/Z$/i, "");
  const timeParts = cleanedTime.split(":");
  const hour = Number(timeParts[0] || 0);
  const minute = Number(timeParts[1] || 0);
  let second = 0;
  let ms = 0;
  if (timeParts.length > 2) {
    const [secPart, fracPart] = String(timeParts[2]).split(/[.,]/);
    second = Number(secPart || 0);
    if (fracPart) {
      ms = Number(fracPart.padEnd(3, "0").slice(0, 3));
    }
  }
  if ([year, month, day, hour, minute, second, ms].some((value) => Number.isNaN(value))) {
    return null;
  }
  return Date.UTC(year, month - 1, day, hour, minute, second, ms);
}

function parseCsv(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    return { headers: [], columns: {}, length: 0 };
  }

  const lines = trimmed.split(/\r?\n/);
  const headers = splitCsvLine(lines.shift());
  const columns = {};

  headers.forEach((header) => {
    columns[header] = [];
  });

  lines.forEach((line) => {
    if (!line) {
      return;
    }
    const cells = splitCsvLine(line);
    headers.forEach((header, index) => {
      columns[header].push(parseValue(cells[index] || ""));
    });
  });

  return { headers, columns, length: columns[headers[0]]?.length || 0 };
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

function parseValue(rawValue) {
  const value = rawValue.trim();
  if (value === "") {
    return null;
  }
  if (value === "inf" || value === "Infinity") {
    return Infinity;
  }
  const numberValue = Number(value);
  if (!Number.isNaN(numberValue)) {
    return numberValue;
  }
  return value;
}

function buildEvents(data) {
  if (!data || !data.columns || !data.length) {
    return [];
  }

  const events = [];
  for (let i = 0; i < data.length; i += 1) {
    events.push({
      start: toNumber(data.columns.start_t?.[i]),
      end: toNumber(data.columns.end_t?.[i]),
      reason: String(data.columns.reason_code?.[i] || "").trim(),
      shouldWarn: Number(data.columns.should_warn?.[i] || 0),
      warningStrength: String(data.columns.warning_strength?.[i] || "").trim(),
      confidence: data.columns.confidence?.[i]
    });
  }

  return events.filter((event) => event.start !== null);
}

function toNumber(value) {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function getTimelineStart(dataset) {
  const times = dataset?.columns?.t_seconds;
  if (!times || !times.length) {
    return 0;
  }
  return typeof times[0] === "number" ? times[0] : 0;
}

function getTimelineDuration(dataset) {
  const times = dataset?.columns?.t_seconds;
  if (!times || !times.length) {
    return 0;
  }
  const start = typeof times[0] === "number" ? times[0] : 0;
  const end = typeof times[times.length - 1] === "number" ? times[times.length - 1] : 0;
  return Math.max(0, end - start);
}

function getWindowBounds(time) {
  if (!time || !time.length) {
    return null;
  }
  const start = Number.isFinite(state.timelineStart) ? state.timelineStart : time[0];
  let end = Number.isFinite(state.duration) ? start + state.duration : time[time.length - 1];
  if (!Number.isFinite(end) || end <= start) {
    end = time[time.length - 1];
  }
  return { start, end };
}

function getWindowIndices(times, windowStart, windowEnd) {
  if (!times || !times.length) {
    return { startIndex: 0, endIndex: -1 };
  }
  let startIndex = findIndexByTime(times, windowStart);
  if (times[startIndex] < windowStart && startIndex < times.length - 1) {
    startIndex += 1;
  }
  let endIndex = findIndexByTime(times, windowEnd);
  if (endIndex < startIndex) {
    endIndex = startIndex;
  }
  return { startIndex, endIndex };
}

function buildCharts() {
  Object.entries(chartConfigs).forEach(([chartId, config]) => {
    const dataset = state.datasets[config.dataset];
    const chartEl = document.querySelector(`[data-chart="${chartId}"]`);
    const legendEl = document.querySelector(`[data-legend="${chartId}"]`);

    if (!chartEl || !legendEl) {
      return;
    }

    const svg = chartEl.querySelector(".chart-svg");
    const linesGroup = chartEl.querySelector(".chart-lines");
    const cursor = chartEl.querySelector(".chart-cursor");
    const emptyEl = chartEl.querySelector(".chart-empty");

    if (!dataset || !dataset.columns) {
      if (emptyEl) {
        emptyEl.style.display = "flex";
      }
      return;
    }

    const availableModes = config.modes ? Object.keys(config.modes) : [];
    const savedMode = state.chartModes[chartId];
    const modeKey = config.modes
      ? (availableModes.includes(savedMode) ? savedMode : (config.defaultMode || availableModes[0]))
      : null;
    const mode = config.modes ? config.modes[modeKey] : null;
    const seriesConfig = mode?.series || config.series || [];
    const time = dataset.columns[config.timeKey] || [];
    const series = seriesConfig.filter((entry) => dataset.columns[entry.key]);
    const emptyLabel = mode?.emptyText || "Load data";

    if (!time.length || !series.length) {
      if (emptyEl) {
        emptyEl.style.display = "flex";
        emptyEl.textContent = emptyLabel;
      }
      return;
    }

    const window = getWindowBounds(time);
    const { startIndex, endIndex } = window
      ? getWindowIndices(time, window.start, window.end)
      : { startIndex: 0, endIndex: time.length - 1 };

    if (endIndex < startIndex) {
      if (emptyEl) {
        emptyEl.style.display = "flex";
      }
      return;
    }

    const timeWindow = time.slice(startIndex, endIndex + 1);
    if (!timeWindow.length) {
      if (emptyEl) {
        emptyEl.style.display = "flex";
        emptyEl.textContent = emptyLabel;
      }
      return;
    }

    if (emptyEl) {
      emptyEl.style.display = "none";
    }

    linesGroup.innerHTML = "";
    legendEl.innerHTML = "";

    const seriesWindow = series.map((entry) => dataset.columns[entry.key].slice(startIndex, endIndex + 1));
    const allValues = seriesWindow;
    const { min, max } = getMinMax(allValues);

    series.forEach((entry, index) => {
      const values = seriesWindow[index] || [];
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "chart-line");
      path.setAttribute("stroke", entry.color);
      path.setAttribute("d", buildPath(values, min, max));
      linesGroup.appendChild(path);

      const legendItem = document.createElement("div");
      legendItem.className = "legend-item";

      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = entry.color;

      const label = document.createElement("span");
      label.textContent = entry.label;

      legendItem.appendChild(swatch);
      legendItem.appendChild(label);
      legendEl.appendChild(legendItem);
    });

    state.charts[chartId] = {
      cursor,
      start: typeof timeWindow[0] === "number" ? timeWindow[0] : 0,
      end: typeof timeWindow[timeWindow.length - 1] === "number"
        ? timeWindow[timeWindow.length - 1]
        : 0
    };
  });
}

function getMinMax(seriesList) {
  let min = Infinity;
  let max = -Infinity;

  seriesList.forEach((series) => {
    series.forEach((value) => {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        return;
      }
      if (value < min) {
        min = value;
      }
      if (value > max) {
        max = value;
      }
    });
  });

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }

  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }

  return { min, max };
}

function buildPath(values, min, max) {
  const len = values.length;
  if (!len) {
    return "M0,20";
  }

  if (len < 2) {
    const value = values[0];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return "M0,20";
    }
    const y = 40 - ((value - min) / (max - min)) * 40;
    return `M0,${y.toFixed(2)}`;
  }

  const step = Math.max(1, Math.floor(len / MAX_POINTS));
  const range = max - min;
  let d = "";
  let started = false;

  for (let i = 0; i < len; i += step) {
    const value = values[i];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      started = false;
      continue;
    }
    const x = (i / (len - 1)) * 100;
    const y = 40 - ((value - min) / range) * 40;
    if (!started) {
      d += `M${x.toFixed(2)},${y.toFixed(2)}`;
      started = true;
    } else {
      d += ` L${x.toFixed(2)},${y.toFixed(2)}`;
    }
  }

  return d || "M0,20";
}

async function updateVideoSources() {
  const roadOffset = state.videoOffsets.road || 0;
  const driverOffset = state.videoOffsets.driver || 0;

  toggleVideoFallback(ui.roadFallback, true, "Caricamento video...");
  toggleVideoFallback(ui.driverFallback, true, "Caricamento video...");

  ui.roadVideo.src = await resolveVideoSource("road_annotated.mp4");
  ui.driverVideo.src = await resolveVideoSource("driver_annotated.mp4");
  ui.roadVideo.muted = true;
  ui.driverVideo.muted = true;
  applyPlaybackRate(ui.roadVideo);
  applyPlaybackRate(ui.driverVideo);

  ui.roadVideo.load();
  ui.driverVideo.load();

  ui.roadVideo.onloadedmetadata = () => {
    applyPlaybackRate(ui.roadVideo);
    if (ui.roadMeta) {
      ui.roadMeta.textContent = `Dur ${formatTime(ui.roadVideo.duration)}`;
    }
    seekVideoToOffset(ui.roadVideo, roadOffset);
    updateDurationFromVideos();
  };

  ui.driverVideo.onloadedmetadata = () => {
    applyPlaybackRate(ui.driverVideo);
    if (ui.driverMeta) {
      ui.driverMeta.textContent = `Dur ${formatTime(ui.driverVideo.duration)}`;
    }
    seekVideoToOffset(ui.driverVideo, driverOffset);
    updateDurationFromVideos();
  };

  ui.roadVideo.onloadeddata = () => {
    toggleVideoFallback(ui.roadFallback, false);
    if (state.playing) {
      attemptVideoPlay(ui.roadVideo, state.videoOffsets.road || 0);
    }
  };

  ui.driverVideo.onloadeddata = () => {
    toggleVideoFallback(ui.driverFallback, false);
    if (state.playing) {
      attemptVideoPlay(ui.driverVideo, state.videoOffsets.driver || 0);
    }
  };

  ui.roadVideo.onerror = () => {
    toggleVideoFallback(ui.roadFallback, true, "Video road non caricato");
  };

  ui.driverVideo.onerror = () => {
    toggleVideoFallback(ui.driverFallback, true, "Video driver non caricato");
  };
}

function applyPlaybackRate(video) {
  if (!video) {
    return;
  }
  video.defaultPlaybackRate = PLAY_RATE;
  video.playbackRate = PLAY_RATE;
}

function isVideoReady(video) {
  return video && video.readyState >= 2 && !video.error;
}

function getPlaybackMaster() {
  const roadReady = isVideoReady(ui.roadVideo);
  const driverReady = isVideoReady(ui.driverVideo);

  const road = roadReady
    ? { video: ui.roadVideo, offset: state.videoOffsets.road || 0, label: "Road" }
    : null;
  const driver = driverReady
    ? { video: ui.driverVideo, offset: state.videoOffsets.driver || 0, label: "Driver" }
    : null;

  const candidates = [road, driver].filter(Boolean);
  if (!candidates.length) {
    return null;
  }

  const playing = candidates.find((item) => !item.video.paused && !item.video.ended);
  return playing || candidates[0];
}

function getAlignedVideoTime(timelineTime, offset) {
  const safeOffset = Number.isFinite(offset) ? offset : 0;
  if (Number.isFinite(state.anchorTime)) {
    const globalTime = state.timelineStart + timelineTime;
    return globalTime - state.anchorTime + safeOffset;
  }
  return timelineTime + safeOffset;
}

function getTimelineTimeFromVideo(videoTime, offset) {
  const safeOffset = Number.isFinite(offset) ? offset : 0;
  if (Number.isFinite(state.anchorTime)) {
    const globalTime = videoTime + state.anchorTime - safeOffset;
    return globalTime - state.timelineStart;
  }
  return videoTime - safeOffset;
}

function getVideoWindowRange(video, offset) {
  const duration = getValidVideoDuration(video);
  if (!duration || !Number.isFinite(state.anchorTime)) {
    return null;
  }
  const safeOffset = Number.isFinite(offset) ? offset : 0;
  const start = state.anchorTime - safeOffset;
  const end = start + duration;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null;
  }
  return { start, end };
}

function getValidVideoDuration(video) {
  if (!video || !Number.isFinite(video.duration)) {
    return null;
  }
  if (video.duration > 0 && video.duration < 5 && state.dataDuration > 5) {
    return null;
  }
  return video.duration;
}

function applyTimelineWindow(windowStart, windowEnd) {
  if (!Number.isFinite(windowStart) || !Number.isFinite(windowEnd)) {
    return;
  }
  const prevDataTime = state.timelineStart + state.currentTime;
  state.timelineStart = windowStart;
  state.duration = Math.max(0, windowEnd - windowStart);
  const nextCurrent = prevDataTime - state.timelineStart;
  state.currentTime = Math.max(0, Math.min(state.duration, Number.isFinite(nextCurrent) ? nextCurrent : 0));
}

async function resolveVideoSource(fileName) {
  const h264Name = fileName.replace(".mp4", "_h264.mp4");
  const candidates = [
    `${VIDEO_DIR}/${fileName}`,
    `${VIDEO_DIR}/${h264Name}`,
    `${CSV_DIR}/${h264Name}`,
    `${CSV_DIR}/${fileName}`
  ];

  for (const url of candidates) {
    if (await checkUrlExists(url)) {
      return url;
    }
  }

  return `${CSV_DIR}/${fileName}`;
}

async function checkUrlExists(url) {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return response.ok;
  } catch (error) {
    return false;
  }
}

function seekVideoToOffset(video, offset) {
  if (!video) {
    return;
  }
  const alignedTime = getAlignedVideoTime(0, offset);
  applyVideoTime(video, alignedTime);
}

function applyVideoTime(video, targetTime) {
  if (!video || !Number.isFinite(targetTime)) {
    return;
  }
  if (Number.isFinite(video.duration)) {
    const maxTime = Math.max(0, video.duration - 0.05);
    video.currentTime = Math.min(Math.max(0, targetTime), maxTime);
  } else {
    video.currentTime = Math.max(0, targetTime);
  }
}

function attemptVideoPlay(video, offset = null) {
  if (!state.playing || !video) {
    return;
  }
  if (video.readyState < 2 || !video.paused) {
    return;
  }
  if (Number.isFinite(offset) && Number.isFinite(state.anchorTime)) {
    const globalTime = state.timelineStart + state.currentTime;
    const videoStart = state.anchorTime - offset;
    if (globalTime < videoStart - 0.05) {
      video.pause();
      return;
    }
  }
  const now = performance.now();
  const lastAttempt = Number(video.dataset.playAttempt || 0);
  if (now - lastAttempt < 800) {
    return;
  }
  video.dataset.playAttempt = String(now);
  applyPlaybackRate(video);
  const attempt = video.play();
  if (attempt && typeof attempt.catch === "function") {
    attempt.catch(() => {
      // Ignore autoplay errors; fallback to timeline sync.
    });
  }
}

function syncVideo(video, offset, label) {
  if (!video || video.readyState < 2) {
    return;
  }
  const target = getAlignedVideoTime(state.currentTime, offset);
  if (!Number.isFinite(target)) {
    return;
  }
  const now = performance.now();
  const key = `last${label}Sync`;
  const lastSync = state[key] || 0;
  const minInterval = 1000;
  const canAdjust = now - lastSync > minInterval;
  if (!canAdjust || video.seeking) {
    return;
  }
  const diff = Math.abs(video.currentTime - target);
  if (diff > 0.5) {
    applyVideoTime(video, target);
    state[key] = now;
  }
  if (state.playing) {
    attemptVideoPlay(video, offset);
  }
}

function monitorRoadPlayback() {
  monitorVideoPlayback(ui.roadVideo, state.videoOffsets.road || 0, "Road");
}

function monitorDriverPlayback() {
  monitorVideoPlayback(ui.driverVideo, state.videoOffsets.driver || 0, "Driver");
}

function monitorVideoPlayback(video, offset, label) {
  if (!state.playing || !video || video.readyState < 2) {
    return;
  }

  const now = performance.now();
  const checkKey = `last${label}Check`;
  const timeKey = `last${label}Time`;
  const lastCheck = state[checkKey] || 0;
  if (now - lastCheck < 1200) {
    return;
  }

  const current = video.currentTime;
  const last = state[timeKey];
  if (video.paused || (last !== null && Math.abs(current - last) < 0.01)) {
    attemptVideoPlay(video, offset);
  }

  state[timeKey] = current;
  state[checkKey] = now;
}

function updateDurationFromVideos() {
  const dataStart = Number.isFinite(state.dataStart) ? state.dataStart : 0;
  const dataDuration = Number.isFinite(state.dataDuration) ? state.dataDuration : 0;
  let windowStart = dataStart;
  let windowEnd = dataDuration > 0 ? dataStart + dataDuration : Number.POSITIVE_INFINITY;

  const roadWindow = getVideoWindowRange(ui.roadVideo, state.videoOffsets.road);
  const driverWindow = getVideoWindowRange(ui.driverVideo, state.videoOffsets.driver);

  if (roadWindow || driverWindow) {
    if (roadWindow) {
      windowStart = Math.max(windowStart, roadWindow.start);
      windowEnd = Math.min(windowEnd, roadWindow.end);
    }
    if (driverWindow) {
      windowStart = Math.max(windowStart, driverWindow.start);
      windowEnd = Math.min(windowEnd, driverWindow.end);
    }
  } else {
    const roadDuration = getValidVideoDuration(ui.roadVideo);
    const driverDuration = getValidVideoDuration(ui.driverVideo);
    const durations = [roadDuration, driverDuration].filter((value) => Number.isFinite(value));
    if (durations.length) {
      const minDuration = Math.min(...durations);
      windowEnd = Math.min(windowEnd, windowStart + minDuration);
    }
  }

  if (!Number.isFinite(windowEnd)) {
    windowEnd = dataStart + Math.max(0, dataDuration);
  }
  if (!Number.isFinite(windowStart) || windowEnd < windowStart) {
    windowStart = dataStart;
    windowEnd = dataStart + Math.max(0, dataDuration);
  }

  applyTimelineWindow(windowStart, windowEnd);
  buildCharts();
  seekTo(state.currentTime, false);
}

function toggleVideoFallback(element, show, message = "") {
  if (!element) {
    return;
  }
  if (show) {
    element.textContent = message || "Video not loaded";
    element.classList.remove("is-hidden");
  } else {
    element.classList.add("is-hidden");
  }
}

function togglePlay() {
  if (!state.loaded) {
    return;
  }

  if (state.playing) {
    pausePlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (state.currentTime >= state.duration) {
    seekTo(0);
  }

  state.playing = true;
  ui.playButton.classList.add("is-playing");
  seekTo(state.currentTime, false);
  attemptVideoPlay(ui.roadVideo, state.videoOffsets.road || 0);
  attemptVideoPlay(ui.driverVideo, state.videoOffsets.driver || 0);

  state.lastTick = performance.now();
  requestAnimationFrame(tick);
}

function pausePlayback() {
  state.playing = false;
  ui.playButton.classList.remove("is-playing");
  ui.roadVideo.pause();
  ui.driverVideo.pause();
}

function tick(now) {
  if (!state.playing) {
    return;
  }

  const master = getPlaybackMaster();
  let time = state.currentTime;

  if (master) {
    const masterTime = getTimelineTimeFromVideo(master.video.currentTime, master.offset);
    if (Number.isFinite(masterTime)) {
      time = masterTime;
    }
  } else {
    const delta = (now - state.lastTick) / 1000;
    time = state.currentTime + delta * PLAY_RATE;
  }

  state.lastTick = now;

  if (!Number.isFinite(time)) {
    time = 0;
  }

  const clamped = state.duration > 0
    ? Math.max(0, Math.min(state.duration, time))
    : Math.max(0, time);

  state.currentTime = clamped;

  if (state.duration > 0 && clamped >= state.duration) {
    updateUi(clamped);
    pausePlayback();
    return;
  }

  if (master) {
    if (master.label === "Road") {
      syncVideo(ui.driverVideo, state.videoOffsets.driver || 0, "Driver");
    } else {
      syncVideo(ui.roadVideo, state.videoOffsets.road || 0, "Road");
    }
  } else {
    syncRoadVideo();
    syncDriverVideo();
  }

  updateUi(clamped);

  attemptVideoPlay(ui.roadVideo, state.videoOffsets.road || 0);
  attemptVideoPlay(ui.driverVideo, state.videoOffsets.driver || 0);
  monitorRoadPlayback();
  monitorDriverPlayback();

  if (state.playing) {
    requestAnimationFrame(tick);
  }
}

function seekTo(time, skipVideoSync = false) {
  const clamped = Math.max(0, Math.min(state.duration, time));
  state.currentTime = clamped;

  if (!skipVideoSync) {
    applyVideoTime(ui.roadVideo, getAlignedVideoTime(clamped, state.videoOffsets.road || 0));
    applyVideoTime(ui.driverVideo, getAlignedVideoTime(clamped, state.videoOffsets.driver || 0));
  } else {
    syncRoadVideo();
    syncDriverVideo();
  }

  updateUi(clamped);
}

function syncRoadVideo() {
  syncVideo(ui.roadVideo, state.videoOffsets.road || 0, "Road");
}

function syncDriverVideo() {
  syncVideo(ui.driverVideo, state.videoOffsets.driver || 0, "Driver");
}

function updateUi(time) {
  const dataTime = time + state.timelineStart;

  updateTransport(time);
  updateMetrics(dataTime);
  updateLanePanels(dataTime);
  updateStatusChips(dataTime);
  updateCharts(dataTime);
  updateDecision(dataTime);
  updateLog(dataTime);
}

function updateTransport(time) {
  ui.currentTime.textContent = formatTime(time);
  ui.totalTime.textContent = formatTime(state.duration);
  const percent = state.duration ? (time / state.duration) * 100 : 0;
  ui.transportProgress.style.width = `${percent.toFixed(2)}%`;
}

function updateMetrics(dataTime) {
  Object.entries(state.metricElements).forEach(([source, items]) => {
    const dataset = state.datasets[source];
    if (!dataset) {
      return;
    }

    const timeKey = datasetTimeKeys[source] || "t_seconds";
    const times = dataset.columns[timeKey] || [];
    const index = findIndexByTime(times, dataTime);

    items.forEach(({ el, field, precision }) => {
      const value = dataset.columns[field]?.[index];
      el.textContent = formatValue(value, precision);
    });
  });
}

function updateLanePanels(dataTime) {
  const fused = state.datasets.fused;
  if (!fused) {
    return;
  }

  const times = fused.columns.t_seconds || [];
  const index = findIndexByTime(times, dataTime);
  const distRaw = fused.columns.road_lane_dist_to_edge_m?.[index];
  const tlc = fused.columns.road_lane_tlc_s?.[index];
  const dist = Number.isFinite(distRaw) ? distRaw : null;
  const offset = fused.columns.road_lane_lane_offset_m?.[index];

  const leftDist = dist !== null && dist < 0 ? Math.abs(dist) : null;
  const rightDist = dist !== null && dist > 0 ? dist : null;

  updateLaneSide("left", leftDist, tlc);
  updateLaneSide("right", rightDist, tlc);
  updateLaneOffset(offset, dataTime);
}

function updateLaneOffset(offset, dataTime) {
  if (!ui.laneTracks?.length) {
    return;
  }
  const now = Number.isFinite(dataTime) ? dataTime : 0;
  let target = null;

  if (Number.isFinite(offset)) {
    target = offset;
    state.laneOffset.lastValid = offset;
    state.laneOffset.lastValidTime = now;
  } else if (Number.isFinite(state.laneOffset.lastValid)) {
    target = state.laneOffset.lastValid;
  }

  const lastUpdate = state.laneOffset.lastUpdateTime;
  const dt = Number.isFinite(lastUpdate) ? Math.max(0, now - lastUpdate) : 0;
  state.laneOffset.lastUpdateTime = now;

  if (!Number.isFinite(target) || !Number.isFinite(LANE_HALF_WIDTH_M) || LANE_HALF_WIDTH_M <= 0) {
    ui.laneTracks.forEach((track) => {
      track.style.setProperty("--lane-fill", "50%");
      track.classList.add("is-muted");
    });
    state.laneOffset.current = null;
    return;
  }

  if (!Number.isFinite(state.laneOffset.current) || dt <= 0) {
    state.laneOffset.current = target;
  } else {
    const alpha = 1 - Math.exp(-dt / LANE_OFFSET_TAU);
    state.laneOffset.current += (target - state.laneOffset.current) * alpha;
  }

  const laneWidth = LANE_HALF_WIDTH_M * 2;
  const leftFill = Math.max(0, Math.min(1, (LANE_HALF_WIDTH_M + state.laneOffset.current) / laneWidth));
  const rightFill = Math.max(0, Math.min(1, (LANE_HALF_WIDTH_M - state.laneOffset.current) / laneWidth));
  const isMuted = !Number.isFinite(offset) && Number.isFinite(state.laneOffset.lastValidTime)
    ? now - state.laneOffset.lastValidTime > LANE_OFFSET_HOLD_S
    : false;
  ui.laneTracks.forEach((track) => {
    const ratio = track.dataset.side === "right" ? rightFill : leftFill;
    track.style.setProperty("--lane-fill", `${(ratio * 100).toFixed(1)}%`);
    track.classList.toggle("is-muted", isMuted);
  });
}

function updateLaneSide(side, distance, tlc) {
  const isLeft = side === "left";
  const distEl = isLeft ? ui.leftLaneDist : ui.rightLaneDist;
  const tlcEl = isLeft ? ui.leftLaneTlc : ui.rightLaneTlc;
  const statusEl = isLeft ? ui.leftLaneStatus : ui.rightLaneStatus;

  distEl.textContent = formatValue(distance, 2);
  tlcEl.textContent = formatValue(tlc, 2);

  if (distance === null || distance === undefined) {
    statusEl.textContent = "N/A";
    setChipStatus(statusEl, "muted");
    return;
  }

  if (distance < 0.3) {
    statusEl.textContent = "CLOSE";
    setChipStatus(statusEl, "warn");
  } else {
    statusEl.textContent = "SAFE";
    setChipStatus(statusEl, "ok");
  }
}

function updateStatusChips(dataTime) {
  const fused = state.datasets.fused;
  if (!fused) {
    return;
  }

  const times = fused.columns.t_seconds || [];
  const index = findIndexByTime(times, dataTime);

  const tlc = fused.columns.road_lane_tlc_s?.[index];
  const gaze = fused.columns.driver_gaze_on_road_prob?.[index];

  if (typeof tlc === "number" && tlc < 1) {
    ui.roadStatusChip.textContent = "ROAD WARNING";
    setChipStatus(ui.roadStatusChip, "warn");
  } else {
    ui.roadStatusChip.textContent = "ROAD OK";
    setChipStatus(ui.roadStatusChip, "ok");
  }

  if (typeof gaze === "number" && gaze < 0.6) {
    ui.driverStatusChip.textContent = "GAZE OFF";
    setChipStatus(ui.driverStatusChip, "warn");
  } else {
    ui.driverStatusChip.textContent = "GAZE OK";
    setChipStatus(ui.driverStatusChip, "ok");
  }
}

function updateCharts(dataTime) {
  Object.values(state.charts).forEach((chart) => {
    if (!chart.cursor || chart.end === chart.start) {
      return;
    }
    const progress = (dataTime - chart.start) / (chart.end - chart.start);
    const clamped = Math.max(0, Math.min(1, progress));
    const x = (clamped * 100).toFixed(2);
    chart.cursor.setAttribute("x1", x);
    chart.cursor.setAttribute("x2", x);
  });
}

function updateDecision(dataTime) {
  const event = getLatestEvent(dataTime);

  if (!event) {
    ui.decisionStatus.textContent = "N/A";
    setStatusPill(ui.decisionStatus, "muted");
    ui.decisionConfidence.textContent = "--";
    ui.decisionStart.textContent = "--";
    ui.decisionEnd.textContent = "--";
    ui.decisionDuration.textContent = "--";
    ui.decisionReasons.innerHTML = [
      "<div class=\"reason-item\"><span class=\"reason-dot\"></span>--</div>",
      "<div class=\"reason-item\"><span class=\"reason-dot\"></span>--</div>"
    ].join("");
    return;
  }

  const shouldWarn = Number(event.shouldWarn) === 1;
  const warnLabel = event.warningStrength ? `WARNING ${event.warningStrength}` : "WARNING";
  ui.decisionStatus.textContent = shouldWarn ? warnLabel : "SUPPRESS";
  setStatusPill(ui.decisionStatus, shouldWarn ? "warning" : "ok");

  ui.decisionConfidence.textContent = formatValue(event.confidence, 2);
  const hasStart = Number.isFinite(event.start);
  const hasEnd = Number.isFinite(event.end);
  ui.decisionStart.textContent = hasStart ? formatTime(event.start) : "--";
  ui.decisionEnd.textContent = hasEnd ? formatTime(event.end) : (hasStart ? "In corso" : "--");
  if (hasStart) {
    const duration = Math.max(0, (hasEnd ? event.end : dataTime) - event.start);
    ui.decisionDuration.textContent = `${formatValue(duration, 2)} s`;
  } else {
    ui.decisionDuration.textContent = "--";
  }

  const reasons = event.reason ? event.reason.split("+") : [];
  ui.decisionReasons.innerHTML = "";

  if (!reasons.length) {
    ui.decisionReasons.innerHTML = [
      "<div class=\"reason-item\"><span class=\"reason-dot\"></span>--</div>",
      "<div class=\"reason-item\"><span class=\"reason-dot\"></span>--</div>"
    ].join("");
    return;
  }

  reasons.forEach((reason, index) => {
    const item = document.createElement("div");
    item.className = "reason-item";
    const dot = document.createElement("span");
    dot.className = index === 0 && shouldWarn ? "reason-dot reason-dot--warn" : "reason-dot";
    const text = document.createElement("span");
    text.textContent = formatReason(reason);
    item.appendChild(dot);
    item.appendChild(text);
    ui.decisionReasons.appendChild(item);
  });
}

function updateLog(dataTime) {
  if (!ui.logList) {
    return;
  }
  if (state.logMetrics.needsMeasure) {
    updateLogMetrics();
  }

  if (!state.events.length) {
    ui.logList.innerHTML = "<div class=\"log-item\"><span class=\"log-time\">--</span><span class=\"log-msg\">No events</span></div>";
    return;
  }

  const windowStart = Number.isFinite(state.timelineStart) ? state.timelineStart : 0;
  const windowEnd = Number.isFinite(state.duration) ? windowStart + state.duration : Number.POSITIVE_INFINITY;
  const visible = state.events.filter((event) => event.start !== null
    && event.start <= dataTime
    && event.start >= windowStart
    && event.start <= windowEnd);
  const capacity = Math.max(1, state.logMetrics.capacity || 5);
  const latest = visible.slice(-capacity).reverse();

  if (!latest.length) {
    ui.logList.innerHTML = "<div class=\"log-item\"><span class=\"log-time\">--</span><span class=\"log-msg\">Waiting for events</span></div>";
    return;
  }

  ui.logList.innerHTML = "";

  latest.forEach((event) => {
    const item = document.createElement("div");
    item.className = "log-item";

    const time = document.createElement("span");
    time.className = "log-time";
    const eventTime = Number.isFinite(event.start) ? Math.max(0, event.start - windowStart) : 0;
    time.textContent = formatTime(eventTime);

    const msg = document.createElement("span");
    msg.className = "log-msg";
    msg.textContent = event.reason ? formatReason(event.reason) : "--";

    const tag = document.createElement("span");
    tag.className = event.shouldWarn ? "log-tag log-tag--warn" : "log-tag";
    tag.textContent = event.shouldWarn ? "WARNING" : "OK";

    item.appendChild(time);
    item.appendChild(msg);
    item.appendChild(tag);
    ui.logList.appendChild(item);
  });
}

function updateLogMetrics() {
  if (!ui.logList) {
    return;
  }

  const listHeight = ui.logList.clientHeight;
  if (!listHeight) {
    return;
  }

  const styles = window.getComputedStyle(ui.logList);
  const gap = parseFloat(styles.rowGap || styles.gap || "0") || 0;
  const sample = ui.logList.querySelector(".log-item");
  const itemHeight = sample ? sample.getBoundingClientRect().height : measureLogItemHeight();

  if (!itemHeight) {
    return;
  }

  const capacity = Math.max(1, Math.floor((listHeight + gap) / (itemHeight + gap)));
  state.logMetrics.capacity = capacity;
  state.logMetrics.itemHeight = itemHeight;
  state.logMetrics.gap = gap;
  state.logMetrics.listHeight = listHeight;
  state.logMetrics.needsMeasure = false;
}

function measureLogItemHeight() {
  if (!ui.logList) {
    return 0;
  }

  const probe = document.createElement("div");
  probe.className = "log-item";
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.pointerEvents = "none";
  probe.style.left = "0";
  probe.style.top = "0";
  probe.style.width = `${ui.logList.clientWidth}px`;
  probe.innerHTML = "<span class=\"log-time\">00:00</span><span class=\"log-msg\">Evento</span><span class=\"log-tag\">OK</span>";
  ui.logList.appendChild(probe);
  const height = probe.getBoundingClientRect().height;
  ui.logList.removeChild(probe);
  return height;
}

function getLatestEvent(dataTime) {
  if (!state.events.length) {
    return null;
  }
  const candidates = state.events.filter((event) => event.start !== null && event.start <= dataTime);
  return candidates.length ? candidates[candidates.length - 1] : null;
}

function setLoadStatus(message, isError = false) {
  ui.loadStatus.textContent = message;
  ui.loadStatus.classList.toggle("load-status--error", isError);
}

function setLoadButtonState(mode) {
  if (!ui.loadButton) {
    return;
  }
  if (mode === "reset") {
    ui.loadButton.textContent = "RESET";
    ui.loadButton.setAttribute("aria-label", "Reset session");
    ui.loadButton.classList.add("action-btn--reset");
    if (ui.sessionTitle) {
      ui.sessionTitle.textContent = "Session in progress";
    }
    if (ui.sessionSubtitle) {
      ui.sessionSubtitle.textContent = "Press RESET to end";
    }
  } else {
    ui.loadButton.textContent = "START";
    ui.loadButton.setAttribute("aria-label", "Start processing");
    ui.loadButton.classList.remove("action-btn--reset");
    if (ui.sessionTitle) {
      ui.sessionTitle.textContent = "Start session";
    }
    if (ui.sessionSubtitle) {
      ui.sessionSubtitle.textContent = "Press START to begin";
    }
  }
}

function setChipStatus(element, status) {
  element.classList.remove("chip--ok", "chip--warn", "chip--muted", "chip--live");
  if (status === "ok") {
    element.classList.add("chip--ok");
  } else if (status === "warn") {
    element.classList.add("chip--warn");
  } else {
    element.classList.add("chip--muted");
  }
}

function setStatusPill(element, status) {
  element.classList.remove("status-pill--warning", "status-pill--ok", "status-pill--muted");
  if (status === "warning") {
    element.classList.add("status-pill--warning");
  } else if (status === "ok") {
    element.classList.add("status-pill--ok");
  } else {
    element.classList.add("status-pill--muted");
  }
}

function findIndexByTime(times, target) {
  if (!times || !times.length) {
    return 0;
  }

  if (target <= times[0]) {
    return 0;
  }

  if (target >= times[times.length - 1]) {
    return times.length - 1;
  }

  let low = 0;
  let high = times.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const value = times[mid];
    if (value === target) {
      return mid;
    }
    if (value < target) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return Math.max(0, high);
}

function formatValue(value, precision) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "inf";
    }
    return value.toFixed(precision);
  }
  const text = String(value).trim();
  return text ? text : "--";
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) {
    return "00:00:00";
  }
  const total = Math.max(0, Math.floor(seconds));
  const hrs = String(Math.floor(total / 3600)).padStart(2, "0");
  const mins = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${hrs}:${mins}:${secs}`;
}

function formatReason(reason) {
  if (!reason) {
    return "--";
  }
  return reason.replace(/_/g, " ").replace(/\+/g, " + ").trim();
}
