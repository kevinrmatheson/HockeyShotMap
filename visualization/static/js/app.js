const appShell = document.querySelector(".app-shell");
const latestSeason = appShell.dataset.latestSeason || "";
const canvas = document.getElementById("heatmap-canvas");
const stage = document.getElementById("rink-stage");
const tooltip = document.getElementById("heatmap-tooltip");

const controls = {
   season: document.getElementById("season-filter"),
   team: document.getElementById("team-filter"),
   player: document.getElementById("player-filter"),
   playerOptions: document.getElementById("player-options"),
   shotResult: document.getElementById("shot-result-filter"),
   homeAway: document.getElementById("home-away-filter"),
   period: document.getElementById("period-filter"),
   densityMode: document.getElementById("density-mode"),
   goalPctMode: document.getElementById("goal-pct-mode"),
   copyLink: document.getElementById("copy-link-button"),
   reset: document.getElementById("reset-button"),
   sensitivity: document.getElementById("sensitivity-slider"),
   sensValue: document.getElementById("sens-value"),
};

const summaryEls = {
   shotCount: document.getElementById("shot-count"),
   goalCount: document.getElementById("goal-count"),
   goalPct: document.getElementById("goal-pct"),
   binCount: document.getElementById("bin-count"),
   vizTitle: document.getElementById("viz-title"),
};

const state = {
   season: latestSeason,
   team: "all",
   player: "",
   shot_result: "all",
   home_away: "all",
   period: "all",
   mode: "density",
   bin_size: 5,
   sensitivity: 1.0,
};

let currentBins = [];
let currentSummary = { shot_count: 0, goal_count: 0, goal_pct: 0 };
let canvasContext = null;
let canvasSize = { width: 0, height: 0 };

function parseStateFromUrl() {
   const params = new URLSearchParams(window.location.search);
   if (params.get("season")) state.season = params.get("season");
   if (params.get("team")) state.team = params.get("team");
   if (params.get("player")) state.player = params.get("player");
   if (params.get("shot_result")) state.shot_result = params.get("shot_result");
   if (params.get("home_away")) state.home_away = params.get("home_away");
   if (params.get("period")) state.period = params.get("period");
   if (params.get("mode") === "goal_pct") state.mode = "goal_pct";
   if (params.get("bin_size")) {
      const parsed = Number(params.get("bin_size"));
      if (!Number.isNaN(parsed) && parsed > 0) state.bin_size = parsed;
   }
   if (params.get("sens")) {
      const parsed = Number(params.get("sens"));
      if (!Number.isNaN(parsed) && parsed > 0) state.sensitivity = parsed;
   }
}

function syncUrl() {
   const params = new URLSearchParams();
   if (state.season) params.set("season", state.season);
   if (state.team !== "all") params.set("team", state.team);
   if (state.player) params.set("player", state.player);
   if (state.shot_result !== "all") params.set("shot_result", state.shot_result);
   if (state.home_away !== "all") params.set("home_away", state.home_away);
   if (state.period !== "all") params.set("period", state.period);
   if (state.mode !== "density") params.set("mode", state.mode);
   if (state.bin_size !== 5) params.set("bin_size", String(state.bin_size));
   if (state.sensitivity !== 1.0) params.set("sens", String(state.sensitivity));

   const nextUrl = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
   window.history.replaceState({}, "", nextUrl);
}

function populateSelect(select, values, allLabel = "All") {
   select.innerHTML = "";
   const defaultOption = document.createElement("option");
   defaultOption.value = "all";
   defaultOption.textContent = allLabel;
   select.appendChild(defaultOption);

   values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
   });
}

function populatePlayers(values) {
   controls.playerOptions.innerHTML = "";
   values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      controls.playerOptions.appendChild(option);
   });
}

function syncControlsFromState() {
   controls.season.value = state.season || latestSeason || "all";
   controls.team.value = state.team;
   controls.player.value = state.player;
   controls.shotResult.value = state.shot_result;
   controls.homeAway.value = state.home_away;
   controls.period.value = state.period;
   controls.sensitivity.value = String(state.sensitivity);
   controls.sensValue.textContent = state.sensitivity.toFixed(2);
   setMode(state.mode);
}

function readControlsIntoState() {
   state.season = controls.season.value === "all" ? "" : controls.season.value;
   state.team = controls.team.value;
   state.player = controls.player.value.trim();
   state.shot_result = controls.shotResult.value;
   state.home_away = controls.homeAway.value;
   state.period = controls.period.value;
}

function setMode(mode) {
   state.mode = mode;
   controls.densityMode.classList.toggle("is-active", mode === "density");
   controls.goalPctMode.classList.toggle("is-active", mode === "goal_pct");
   summaryEls.vizTitle.textContent = mode === "density" ? "Shot density" : "Goal percentage";
}

function buildQueryParams() {
   const params = new URLSearchParams();
   if (state.season) params.set("season", state.season);
   if (state.team !== "all") params.set("team", state.team);
   if (state.player) params.set("player", state.player);
   if (state.shot_result !== "all") params.set("shot_result", state.shot_result);
   if (state.home_away !== "all") params.set("home_away", state.home_away);
   if (state.period !== "all") params.set("period", state.period);
   params.set("bin_size", String(state.bin_size));
   return params;
}

async function fetchJson(url) {
   const response = await fetch(url, { headers: { Accept: "application/json" } });
   if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
   }
   return response.json();
}

function resizeCanvas() {
   const rect = stage.getBoundingClientRect();
   const devicePixelRatio = window.devicePixelRatio || 1;
   const width = Math.max(1, Math.round(rect.width * devicePixelRatio));
   const height = Math.max(1, Math.round(rect.height * devicePixelRatio));

   if (canvas.width === width && canvas.height === height) {
      return;
   }

   canvas.width = width;
   canvas.height = height;
   canvas.style.width = `${rect.width}px`;
   canvas.style.height = `${rect.height}px`;
   canvasContext = canvas.getContext("2d");
   canvasSize = { width, height };
   renderHeatmap();
}

function normalizeCanvasPoint(pointX, pointY) {
   const x = (pointX / 100) * canvasSize.width;
   const y = ((42.5 - pointY) / 85) * canvasSize.height;
   return { x, y };
}

function heatColor(value, maxValue, mode, shotCount) {
   const safeMax = Math.max(1, maxValue);
   const rawNormalized = Math.min(1, value / safeMax);
   // Apply sensitivity as a power curve: <1 = ramp faster, >1 = ramp slower
   const power = 1.0 / Math.max(0.1, state.sensitivity);
   const normalized = Math.pow(rawNormalized, power);
   const intensity = Math.min(1, Math.max(0.15, shotCount / Math.max(1, currentSummary.shot_count || 1)));

   if (mode === "goal_pct") {
      const r = Math.round(255 * normalized);
      const g = Math.round(180 * (1 - normalized * 0.7));
      const b = Math.round(90 * (1 - normalized));
      return `rgba(${r}, ${g}, ${b}, ${0.2 + intensity * 0.62})`;
   }

   const r = Math.round(255 * normalized);
   const g = Math.round(152 + 60 * (1 - normalized));
   const b = Math.round(66 * (1 - normalized));
   return `rgba(${r}, ${g}, ${b}, ${0.16 + intensity * 0.68})`;
}

function renderHeatmap() {
   if (!canvasContext || !canvasSize.width || !canvasSize.height) {
      return;
   }

   canvasContext.clearRect(0, 0, canvasSize.width, canvasSize.height);
   currentBins.forEach((bin) => {
      const position = normalizeCanvasPoint(bin.x, bin.y);
      const binWidth = (state.bin_size / 100) * canvasSize.width;
      const binHeight = (state.bin_size / 85) * canvasSize.height;
      const value = state.mode === "goal_pct" ? bin.goal_pct : bin.shot_count;
      const maxValue = state.mode === "goal_pct"
         ? Math.max(100, ...currentBins.map((item) => item.goal_pct || 0))
         : Math.max(1, ...currentBins.map((item) => item.shot_count || 0));

      canvasContext.fillStyle = heatColor(value, maxValue, state.mode, bin.shot_count);
      canvasContext.fillRect(position.x - binWidth / 2, position.y - binHeight / 2, binWidth, binHeight);
   });
}

function updateTooltip(event) {
   if (!currentBins.length || !canvasSize.width || !canvasSize.height) {
      tooltip.hidden = true;
      return;
   }

   const rect = canvas.getBoundingClientRect();
   const x = (event.clientX - rect.left) * (canvasSize.width / rect.width);
   const y = (event.clientY - rect.top) * (canvasSize.height / rect.height);

   const binWidth = (state.bin_size / 100) * canvasSize.width;
   const binHeight = (state.bin_size / 85) * canvasSize.height;
   const hovered = currentBins.find((bin) => {
      const position = normalizeCanvasPoint(bin.x, bin.y);
      return (
         x >= position.x - binWidth / 2 &&
         x <= position.x + binWidth / 2 &&
         y >= position.y - binHeight / 2 &&
         y <= position.y + binHeight / 2
      );
   });

   if (!hovered) {
      tooltip.hidden = true;
      return;
   }

   tooltip.innerHTML = `
      <strong>${state.mode === "goal_pct" ? `${hovered.goal_pct.toFixed(1)}% goal rate` : `${hovered.shot_count} shots`}</strong>
      <span class="value">Goals: ${hovered.goal_count}</span>
      <span class="muted">Shot count: ${hovered.shot_count}</span>
      <span class="muted">Bin center: (${hovered.x.toFixed(1)}, ${hovered.y.toFixed(1)})</span>
   `;
   tooltip.hidden = false;
   tooltip.style.left = `${event.offsetX}px`;
   tooltip.style.top = `${event.offsetY}px`;
}

function renderSummary(summary) {
   summaryEls.shotCount.textContent = String(summary.shot_count ?? 0);
   summaryEls.goalCount.textContent = String(summary.goal_count ?? 0);
   summaryEls.goalPct.textContent = `${Number(summary.goal_pct ?? 0).toFixed(1)}%`;
   summaryEls.binCount.textContent = String(currentBins.length);
}

async function refreshDashboard() {
   syncUrl();
   const params = buildQueryParams();
   const data = await fetchJson(`/api/dashboard?${params.toString()}`);
   currentBins = data.bins || [];
   currentSummary = data.summary || { shot_count: 0, goal_count: 0, goal_pct: 0 };
   renderSummary(currentSummary);
   resizeCanvas();
   renderHeatmap();
}

function setModeAndRefresh(mode) {
   if (state.mode === mode) return;
   setMode(mode);
   syncUrl();
   renderHeatmap();
}

async function initialize() {
   parseStateFromUrl();
   const options = await fetchJson("/api/options");

   populateSelect(controls.season, options.seasons || [], "All seasons");
   populateSelect(controls.team, options.teams || [], "All teams");
   populatePlayers(options.players || []);
   populateSelect(controls.shotResult, options.shot_results || [], "All shot results");
   populateSelect(controls.homeAway, options.home_away || [], "All locations");
   populateSelect(controls.period, options.periods || [], "All periods");

   if (!state.season) {
      state.season = latestSeason || options.seasons?.[0] || "";
   }

   syncControlsFromState();

   controls.season.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.team.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.player.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.shotResult.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.homeAway.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.period.addEventListener("change", () => {
      readControlsIntoState();
      refreshDashboard().catch(console.error);
   });
   controls.densityMode.addEventListener("click", () => setModeAndRefresh("density"));
   controls.goalPctMode.addEventListener("click", () => setModeAndRefresh("goal_pct"));
   controls.sensitivity.addEventListener("input", () => {
      state.sensitivity = Number(controls.sensitivity.value);
      controls.sensValue.textContent = state.sensitivity.toFixed(2);
      renderHeatmap();
   });
   controls.copyLink.addEventListener("click", async () => {
      await navigator.clipboard.writeText(window.location.href);
      controls.copyLink.textContent = "Copied";
      window.setTimeout(() => {
         controls.copyLink.textContent = "Copy share link";
      }, 1200);
   });
   controls.reset.addEventListener("click", () => {
      state.season = latestSeason || state.season;
      state.team = "all";
      state.player = "";
      state.shot_result = "all";
      state.home_away = "all";
      state.period = "all";
      state.mode = "density";
      state.sensitivity = 1.0;
      controls.sensitivity.value = "1.0";
      controls.sensValue.textContent = "1.00";
      syncControlsFromState();
      refreshDashboard().catch(console.error);
   });

   canvas.addEventListener("mousemove", updateTooltip);
   canvas.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
   });

   window.addEventListener("resize", () => {
      resizeCanvas();
   });

   await refreshDashboard();
}

initialize().catch((error) => {
   console.error(error);
   summaryEls.vizTitle.textContent = "Unable to load dashboard";
});
