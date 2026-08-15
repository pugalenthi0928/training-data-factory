const STAGES = [
  "ingest",
  "source_governance",
  "generate",
  "quality",
  "record_governance",
  "dedupe",
  "judge",
  "contamination",
  "difficulty",
  "select",
  "split",
  "profile",
];

const state = {
  presets: [],
  selectedPreset: "release-controls",
  custom: false,
  running: false,
  jobId: null,
  pollTimer: null,
};

const elements = {
  presetList: document.querySelector("#preset-list"),
  customToggle: document.querySelector("#custom-toggle"),
  customInputs: document.querySelector("#custom-inputs"),
  titleOne: document.querySelector("#title-one"),
  titleTwo: document.querySelector("#title-two"),
  documentOne: document.querySelector("#document-one"),
  documentTwo: document.querySelector("#document-two"),
  countOne: document.querySelector("#count-one"),
  countTwo: document.querySelector("#count-two"),
  runButton: document.querySelector("#run-button"),
  runMessage: document.querySelector("#run-message"),
  stageList: document.querySelector("#stage-list"),
  completedCount: document.querySelector("#completed-count"),
  progressBar: document.querySelector("#progress-bar"),
  resultSection: document.querySelector("#result-section"),
  releaseId: document.querySelector("#release-id"),
  runDuration: document.querySelector("#run-duration"),
  metricRecords: document.querySelector("#metric-records"),
  metricGates: document.querySelector("#metric-gates"),
  metricOverlap: document.querySelector("#metric-overlap"),
  metricContamination: document.querySelector("#metric-contamination"),
  artifactCount: document.querySelector("#artifact-count"),
  artifactList: document.querySelector("#artifact-list"),
  downloadButton: document.querySelector("#download-button"),
  viewerFilename: document.querySelector("#viewer-filename"),
  viewerSize: document.querySelector("#viewer-size"),
  artifactContent: document.querySelector("#artifact-content"),
};

function stageLabel(name) {
  return name.split("_").map((part) => part[0].toUpperCase() + part.slice(1)).join(" ");
}

function metricSummary(metrics) {
  if (!metrics || typeof metrics !== "object") return "waiting for execution";
  const entries = Object.entries(metrics).filter(([, value]) =>
    ["string", "number", "boolean"].includes(typeof value)
  );
  if (!entries.length) return "evidence recorded";
  return entries.slice(0, 2).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ");
}

function renderStages(stages = null) {
  const rows = stages || STAGES.map((name) => ({ name, label: stageLabel(name), status: "pending", metrics: {} }));
  elements.stageList.replaceChildren();
  rows.forEach((stage, index) => {
    const item = document.createElement("li");
    item.className = `stage-item ${stage.status || "pending"}`;

    const number = document.createElement("span");
    number.className = "stage-index";
    number.textContent = String(index + 1).padStart(2, "0");

    const name = document.createElement("span");
    name.className = "stage-name";
    name.textContent = stage.label || stageLabel(stage.name);

    const metric = document.createElement("span");
    metric.className = "stage-metric";
    metric.textContent = metricSummary(stage.metrics);

    const status = document.createElement("span");
    status.className = "stage-status";
    status.setAttribute("aria-label", stage.status || "pending");

    item.append(number, name, metric, status);
    elements.stageList.append(item);
  });
}

function renderPresets() {
  elements.presetList.replaceChildren();
  state.presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-option";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(!state.custom && preset.slug === state.selectedPreset));
    button.dataset.slug = preset.slug;

    const radio = document.createElement("span");
    radio.className = "preset-radio";

    const copy = document.createElement("span");
    copy.className = "preset-copy";
    const eyebrow = document.createElement("span");
    eyebrow.textContent = preset.eyebrow;
    const label = document.createElement("strong");
    label.textContent = preset.label;
    copy.append(eyebrow, label);

    const description = document.createElement("small");
    description.textContent = preset.description;
    button.append(radio, copy, description);
    button.addEventListener("click", () => {
      state.selectedPreset = preset.slug;
      state.custom = false;
      elements.customToggle.checked = false;
      elements.customInputs.hidden = true;
      renderPresets();
      updateRunAvailability();
    });
    elements.presetList.append(button);
  });
}

function countText(textarea, counter) {
  const length = textarea.value.trim().replace(/\s+/g, " ").length;
  counter.textContent = `${length} / 320 minimum`;
  counter.classList.toggle("valid", length >= 320);
  return length;
}

function updateRunAvailability() {
  const one = countText(elements.documentOne, elements.countOne);
  const two = countText(elements.documentTwo, elements.countTwo);
  const customValid = !state.custom || (
    one >= 320 && two >= 320 && one <= 12000 && two <= 12000 && one + two <= 24000
  );
  elements.runButton.disabled = state.running || !customValid || !state.presets.length;
}

function setMessage(message, isError = false) {
  elements.runMessage.textContent = message;
  elements.runMessage.classList.toggle("error", isError);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) {
      // The fallback message is enough for non-JSON failures.
    }
    throw new Error(detail);
  }
  return response;
}

function requestPayload() {
  if (!state.custom) return { preset: state.selectedPreset };
  return {
    preset: null,
    documents: [
      { title: elements.titleOne.value.trim(), text: elements.documentOne.value.trim() },
      { title: elements.titleTwo.value.trim(), text: elements.documentTwo.value.trim() },
    ],
  };
}

async function startRun() {
  if (state.running) return;
  state.running = true;
  updateRunAvailability();
  elements.resultSection.hidden = true;
  elements.artifactList.replaceChildren();
  renderStages();
  elements.completedCount.textContent = "0";
  elements.progressBar.className = "";
  setMessage("Submitting a bounded run to the real Forge worker.");

  try {
    const response = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload()),
    });
    const body = await response.json();
    state.jobId = body.job_id;
    sessionStorage.setItem("forgeJobId", state.jobId);
    setMessage(body.reused ? "Reopening the matching content-addressed run." : "Run accepted. Reading stage events now.");
    await pollRun();
  } catch (error) {
    state.running = false;
    updateRunAvailability();
    setMessage(error.message, true);
  }
}

function updateStatus(status) {
  renderStages(status.stages);
  elements.completedCount.textContent = String(status.completed_stages);
  elements.progressBar.className = `progress-${status.completed_stages}`;
  if (status.status === "queued") {
    setMessage("Queued behind the current bounded run.");
  } else if (status.status === "running") {
    const stage = status.current_stage ? stageLabel(status.current_stage) : "pipeline";
    setMessage(`Running ${stage}. ${status.completed_stages} of ${status.total_stages} stages complete.`);
  }
}

async function pollRun() {
  if (!state.jobId) return;
  try {
    const response = await api(`/api/runs/${state.jobId}`);
    const status = await response.json();
    updateStatus(status);
    if (status.status === "succeeded") {
      state.running = false;
      updateRunAvailability();
      setMessage("Verified release complete. The full evidence bundle is ready.");
      await showResult(status);
      return;
    }
    if (status.status === "failed") {
      state.running = false;
      updateRunAvailability();
      setMessage(status.error || "The pipeline did not complete.", true);
      return;
    }
    state.pollTimer = window.setTimeout(pollRun, 450);
  } catch (error) {
    state.running = false;
    updateRunAvailability();
    setMessage(error.message, true);
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

async function showResult(status) {
  const summary = status.summary;
  elements.releaseId.textContent = summary.release_id;
  elements.runDuration.textContent = `Completed in ${status.elapsed_seconds.toFixed(2)} seconds`;
  elements.metricRecords.textContent = String(summary.records);
  elements.metricGates.textContent = String(summary.gates_passed);
  elements.metricOverlap.textContent = String(summary.source_overlap);
  elements.metricContamination.textContent = String(summary.contamination_flags);
  elements.downloadButton.href = `/api/runs/${state.jobId}/download`;

  const response = await api(`/api/runs/${state.jobId}/artifacts`);
  const body = await response.json();
  renderArtifacts(body.artifacts);
  elements.resultSection.hidden = false;
  window.setTimeout(() => elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
  if (body.artifacts.length) await loadArtifact(body.artifacts[0], null);
}

function renderArtifacts(artifacts) {
  elements.artifactList.replaceChildren();
  elements.artifactCount.textContent = `${artifacts.length} FILES`;
  artifacts.forEach((artifact, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "artifact-button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(index === 0));
    button.dataset.key = artifact.key;

    const glyph = document.createElement("i");
    const label = document.createElement("span");
    label.textContent = artifact.label;
    const size = document.createElement("small");
    size.textContent = formatBytes(artifact.bytes);
    button.append(glyph, label, size);
    button.addEventListener("click", () => loadArtifact(artifact, button));
    elements.artifactList.append(button);
  });
}

async function loadArtifact(artifact, selectedButton) {
  document.querySelectorAll(".artifact-button").forEach((button) => {
    button.setAttribute("aria-selected", String(button === selectedButton || (!selectedButton && button.dataset.key === artifact.key)));
  });
  elements.viewerFilename.textContent = artifact.filename;
  elements.viewerSize.textContent = formatBytes(artifact.bytes);
  elements.artifactContent.textContent = "Loading verified artifact...";
  try {
    const response = await api(`/api/runs/${state.jobId}/artifacts/${artifact.key}`);
    const raw = await response.text();
    if (artifact.media_type === "application/json") {
      elements.artifactContent.textContent = JSON.stringify(JSON.parse(raw), null, 2);
    } else if (artifact.media_type === "application/x-ndjson") {
      const rows = raw.split("\n").filter(Boolean).map((line) => JSON.parse(line));
      elements.artifactContent.textContent = rows.length ? rows.map((row) => JSON.stringify(row, null, 2)).join("\n\n") : "No quarantined records.";
    } else {
      elements.artifactContent.textContent = raw;
    }
  } catch (error) {
    elements.artifactContent.textContent = error.message;
  }
}

async function restoreRun() {
  const jobId = sessionStorage.getItem("forgeJobId");
  if (!jobId || !/^[a-f0-9]{16}$/.test(jobId)) return;
  state.jobId = jobId;
  try {
    const response = await api(`/api/runs/${jobId}`);
    const status = await response.json();
    updateStatus(status);
    if (status.status === "succeeded") {
      await showResult(status);
    } else if (status.status === "queued" || status.status === "running") {
      state.running = true;
      updateRunAvailability();
      await pollRun();
    }
  } catch (_) {
    sessionStorage.removeItem("forgeJobId");
    state.jobId = null;
  }
}

async function initialise() {
  renderStages();
  elements.customToggle.addEventListener("change", () => {
    state.custom = elements.customToggle.checked;
    elements.customInputs.hidden = !state.custom;
    renderPresets();
    updateRunAvailability();
  });
  elements.documentOne.addEventListener("input", updateRunAvailability);
  elements.documentTwo.addEventListener("input", updateRunAvailability);
  elements.runButton.addEventListener("click", startRun);

  try {
    const response = await api("/api/presets");
    const body = await response.json();
    state.presets = body.presets;
    renderPresets();
    updateRunAvailability();
    await restoreRun();
  } catch (error) {
    setMessage(`Could not load the demonstration contract. ${error.message}`, true);
  }
}

initialise();
