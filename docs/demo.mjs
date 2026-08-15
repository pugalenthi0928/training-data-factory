import { runBrowserDemo } from "./demo-core.mjs";

const cleanBenchmark =
  "A solar observatory records magnetic activity across several wavelengths. Calibration notes explain how exposure settings influence measurements and how instruments are checked before each observation window.";

const sourceInputs = [...document.querySelectorAll("[data-source]")];
const benchmarkInput = document.querySelector("#benchmark");
const scenarioInput = document.querySelector("#scenario");
const runButton = document.querySelector("#run-demo");
const resetButton = document.querySelector("#reset-demo");
const errorMessage = document.querySelector("#run-error");
const resultPanel = document.querySelector("#result-panel");
const artifactTabs = document.querySelector("#artifact-tabs");
const artifactOutput = document.querySelector("#artifact-output");
const downloadButton = document.querySelector("#download-artifact");
const copyButton = document.querySelector("#copy-artifact");

let currentArtifacts = {};
let selectedArtifact = null;

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function currentSources() {
  return sourceInputs.map((input, index) => ({
    name: input.dataset.name || `source-${index + 1}.txt`,
    content: input.value,
  }));
}

function setScenario() {
  if (scenarioInput.value === "contaminated") {
    benchmarkInput.value = sourceInputs[0].value;
  } else {
    benchmarkInput.value = cleanBenchmark;
  }
}

function resetStages() {
  for (const stage of document.querySelectorAll("[data-stage]")) {
    stage.dataset.state = "waiting";
    stage.querySelector(".stage-status").textContent = "Waiting";
  }
}

function updateStage(index, state, label) {
  const stage = document.querySelector(`[data-stage="${index}"]`);
  stage.dataset.state = state;
  stage.querySelector(".stage-status").textContent = label;
}

function setMetric(name, value) {
  document.querySelector(`[data-metric="${name}"]`).textContent = value;
}

function selectArtifact(name) {
  selectedArtifact = name;
  for (const button of artifactTabs.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.artifact === name);
  }
  artifactOutput.textContent = JSON.stringify(currentArtifacts[name], null, 2);
  downloadButton.disabled = false;
  copyButton.disabled = false;
}

function renderArtifacts(artifacts) {
  currentArtifacts = artifacts;
  artifactTabs.replaceChildren();

  for (const name of Object.keys(artifacts)) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.artifact = name;
    button.textContent = name;
    button.addEventListener("click", () => selectArtifact(name));
    artifactTabs.append(button);
  }

  selectArtifact(Object.keys(artifacts)[0]);
}

function renderResult(result) {
  setMetric("sources", result.sourceRecords.length);
  setMetric("examples", result.examples.length);
  setMetric(
    "split",
    result.split
      ? `${result.split.train.length} / ${result.split.test.length}`
      : "Blocked",
  );
  setMetric(
    "overlap",
    result.split ? result.split.manifest.overlap.document_ids.length : "N/A",
  );

  const gate = document.querySelector("#integrity-gate");
  gate.dataset.status = result.status;
  if (result.status === "passed") {
    gate.innerHTML =
      "<strong>Integrity gate passed.</strong> No source or chunk identifier crosses the train and test boundary.";
  } else {
    gate.innerHTML = `<strong>Run blocked.</strong> The benchmark screen flagged ${result.contamination.flagged_count} generated example${result.contamination.flagged_count === 1 ? "" : "s"}. No split was produced.`;
  }

  renderArtifacts(result.artifacts);
  resultPanel.hidden = false;
}

async function animateResult(result) {
  const finalStage = result.status === "passed" ? 5 : 3;
  for (let index = 1; index <= finalStage; index += 1) {
    updateStage(index, "running", "Running");
    await wait(180);
    updateStage(
      index,
      index === 3 && result.status === "blocked" ? "blocked" : "passed",
      index === 3 && result.status === "blocked" ? "Blocked" : "Passed",
    );
  }

  if (result.status === "blocked") {
    updateStage(4, "skipped", "Not run");
    updateStage(5, "skipped", "Not run");
  }
}

async function runDemo() {
  runButton.disabled = true;
  resetButton.disabled = true;
  errorMessage.hidden = true;
  resultPanel.hidden = true;
  resetStages();

  try {
    const result = await runBrowserDemo({
      sources: currentSources(),
      benchmarkText: benchmarkInput.value,
      testFraction: 0.33,
      seed: 42,
    });
    await animateResult(result);
    renderResult(result);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    runButton.disabled = false;
    resetButton.disabled = false;
  }
}

function resetDemo() {
  for (const input of sourceInputs) input.value = input.defaultValue;
  scenarioInput.value = "clean";
  setScenario();
  resetStages();
  resultPanel.hidden = true;
  errorMessage.hidden = true;
  currentArtifacts = {};
  selectedArtifact = null;
}

downloadButton.addEventListener("click", () => {
  if (!selectedArtifact) return;
  const blob = new Blob(
    [JSON.stringify(currentArtifacts[selectedArtifact], null, 2)],
    { type: "application/json" },
  );
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = selectedArtifact;
  link.click();
  URL.revokeObjectURL(link.href);
});

copyButton.addEventListener("click", async () => {
  if (!selectedArtifact) return;
  await navigator.clipboard.writeText(
    JSON.stringify(currentArtifacts[selectedArtifact], null, 2),
  );
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy JSON";
  }, 1200);
});

scenarioInput.addEventListener("change", setScenario);
runButton.addEventListener("click", runDemo);
resetButton.addEventListener("click", resetDemo);
setScenario();
