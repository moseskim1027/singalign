import "./styles.css";
import type {
  ComparisonExample,
  ComparisonManifest,
  ComparisonSummary,
  LatestComparison,
  MultiConditionReport,
} from "./types";

const app = document.querySelector<HTMLElement>("#app");

if (!app) {
  throw new Error("application root is missing");
}

app.innerHTML = `
  <header class="hero">
    <p class="eyebrow">Research inspection tool</p>
    <h1>SingAlign model comparison</h1>
    <p class="intro">Review paired objective metrics and local listening artifacts from one tracked comparison run.</p>
  </header>
  <section id="landing" class="landing" aria-labelledby="landing-title">
    <p class="section-kicker">Choose a research workflow</p>
    <h2 id="landing-title">What would you like to investigate?</h2>
    <div class="study-choices">
      <label class="study-choice"><input type="radio" name="landing-study" value="study1" checked /><span><strong>Study 1</strong><small>Same-singer, score-conditioned synthesis</small></span></label>
      <label class="study-choice"><input type="radio" name="landing-study" value="study2" /><span><strong>Study 2</strong><small>Content-and-melody transfer to a new instrumental</small></span></label>
    </div>
    <p id="landing-footnote" class="landing-footnote"></p>
    <button id="continue-study" type="button">Continue</button>
  </section>
  <nav id="workflow-nav" class="tabs" aria-label="Experiment workflow" hidden>
    <button type="button" class="tab active" data-tab="training">Study 1</button>
    <button type="button" class="tab" data-tab="evaluation">Evaluation</button>
    <button type="button" class="tab" data-tab="comparison">Comparison</button>
    <button type="button" class="tab" data-tab="studies">Study 2</button>
  </nav>
  <section class="tab-panel" data-panel="training" hidden>
    <section class="loader" aria-labelledby="training-title">
      <div>
        <p class="section-kicker">Study 1 · same-singer synthesis</p>
        <h2 id="training-title">Score-conditioned singing synthesis</h2>
        <p>Train and evaluate the PJS vocalist from score, phoneme, timing, and observed-F0 conditioning.</p>
      </div>
      <form id="training-form">
        <input id="training-experiment" type="hidden" value="study1" />
        <p id="experiment-context" class="status" role="status"></p>
        <div id="training-fields" class="training-fields"></div>
        <button type="submit">Generate command</button>
      </form>
      <pre id="training-command" class="command" hidden></pre>
    </section>
  </section>
  <section class="tab-panel" data-panel="evaluation" hidden>
  <section class="loader" aria-labelledby="evaluation-title">
    <div>
      <h2 id="evaluation-title">Evaluation interface</h2>
      <p>Evaluate the checkpoint produced by Training using its registered protocol.</p>
    </div>
    <form id="evaluation-form">
      <label for="evaluation-experiment">Experiment</label>
      <select id="evaluation-experiment"></select>
      <label for="evaluation-checkpoint">Checkpoint</label>
      <input id="evaluation-checkpoint" value="checkpoints/baseline/best.pt" required />
      <button type="submit">Generate evaluation command</button>
    </form>
    <pre id="evaluation-command" class="command" hidden></pre>
    <p class="evaluation-footnote"><strong>Interpretation boundary:</strong> this is an objective inspection workflow, not a blinded listening study. Generated examples use approximate Griffin-Lim reconstruction.</p>
  </section>
  <section class="loader" aria-labelledby="loader-title">
    <div>
      <h2 id="loader-title">Load a comparison run</h2>
      <p>Paste the MLflow run ID printed by <code>singalign-compare</code>.</p>
    </div>
    <form id="run-form">
      <label for="run-id">Run ID</label>
      <div class="input-row">
        <input id="run-id" name="run-id" autocomplete="off" spellcheck="false" placeholder="Latest comparison loads automatically" required />
        <button type="submit">Load comparison</button>
      </div>
    </form>
    <p id="status" class="status" role="status"></p>
  </section>
  <section id="results" class="results" hidden></section>
  </section>
  <section class="tab-panel" data-panel="comparison" hidden>
  <section class="loader" aria-labelledby="multi-title">
    <div>
      <h2 id="multi-title">Load multi-condition report</h2>
      <p>Enter the report path relative to <code>/reports/</code>.</p>
    </div>
    <form id="multi-form">
      <label for="multi-path">Report path</label>
      <div class="input-row">
        <input id="multi-path" name="multi-path" placeholder="multi-condition/example.json" required />
        <button type="submit">Load conditions</button>
      </div>
    </form>
    <p id="multi-status" class="status" role="status"></p>
  </section>
  <section id="multi-results" class="results" hidden></section>
  </section>
  <section class="tab-panel" data-panel="studies" hidden>
    <section class="loader" aria-labelledby="studies-title">
      <div>
        <p class="section-kicker">Study 2 · content-and-melody transfer</p>
        <h2 id="studies-title">Transfer vocal content to a new instrumental</h2>
        <p>Run a controlled source/target remix with explicit pairing, alignment, and output paths.</p>
      </div>
      <form id="studies-form">
        <input id="study-kind" type="hidden" value="study-2" />
        <label for="study-manifest">Pair manifest (Study 2)</label>
        <input id="study-manifest" value="experiments/study-2-pairs.example.json" />
        <label for="study-source">Source vocal (Study 2)</label>
        <input id="study-source" value="data/interim/source.wav" />
        <label for="study-target">Target instrumental (Study 2)</label>
        <input id="study-target" value="data/interim/target.wav" />
        <button type="submit">Generate Docker command</button>
      </form>
      <pre id="study-command" class="command" hidden></pre>
      <p class="status">Study 1 uses score/phoneme conditioning and observed F0. Study 2 keeps source/target pairing, tempo, transposition, and MusicGen conditions explicit.</p>
    </section>
  </section>
  </section>
  <footer>SingAlign research inspection tool · PJS study workflows</footer>
`;

const form = document.querySelector<HTMLFormElement>("#run-form");
const input = document.querySelector<HTMLInputElement>("#run-id");
const status = document.querySelector<HTMLElement>("#status");
const results = document.querySelector<HTMLElement>("#results");
const multiForm = document.querySelector<HTMLFormElement>("#multi-form");
const multiInput = document.querySelector<HTMLInputElement>("#multi-path");
const multiStatus = document.querySelector<HTMLElement>("#multi-status");
const multiResults = document.querySelector<HTMLElement>("#multi-results");
const trainingForm = document.querySelector<HTMLFormElement>("#training-form");
const trainingExperiment = document.querySelector<HTMLSelectElement>("#training-experiment");
const trainingFields = document.querySelector<HTMLElement>("#training-fields");
const trainingCommand = document.querySelector<HTMLElement>("#training-command");
const experimentContext = document.querySelector<HTMLElement>("#experiment-context");
const evaluationForm = document.querySelector<HTMLFormElement>("#evaluation-form");
const evaluationExperiment = document.querySelector<HTMLSelectElement>("#evaluation-experiment");
const evaluationCheckpoint = document.querySelector<HTMLInputElement>("#evaluation-checkpoint");
const evaluationCommand = document.querySelector<HTMLElement>("#evaluation-command");
const studiesForm = document.querySelector<HTMLFormElement>("#studies-form");
const studyKind = document.querySelector<HTMLSelectElement>("#study-kind");
const studyManifest = document.querySelector<HTMLInputElement>("#study-manifest");
const studySource = document.querySelector<HTMLInputElement>("#study-source");
const studyTarget = document.querySelector<HTMLInputElement>("#study-target");
const studyCommand = document.querySelector<HTMLElement>("#study-command");
const landing = document.querySelector<HTMLElement>("#landing");
const landingFootnote = document.querySelector<HTMLElement>("#landing-footnote");
const continueStudy = document.querySelector<HTMLButtonElement>("#continue-study");
const workflowNav = document.querySelector<HTMLElement>("#workflow-nav");
const tabs = [...document.querySelectorAll<HTMLButtonElement>(".tab")];
const panels = [...document.querySelectorAll<HTMLElement>(".tab-panel")];
const workflowReady = { evaluation: false, comparison: false };

if (!form || !input || !status || !results || !multiForm || !multiInput || !multiStatus || !multiResults || !trainingForm || !trainingExperiment || !trainingFields || !trainingCommand || !experimentContext || !evaluationForm || !evaluationExperiment || !evaluationCheckpoint || !evaluationCommand || !studiesForm || !studyKind || !studyManifest || !studySource || !studyTarget || !studyCommand || !landing || !landingFootnote || !continueStudy || !workflowNav) {
  throw new Error("comparison controls are missing");
}

const landingNotes = {
  study1: "Study 1 tests whether the PJS vocalist can be reconstructed from lyrics, phonemes, score timing, symbolic pitch, and observed F0. It is a same-singer diagnostic, not an unseen-singer evaluation.",
  study2: "Study 2 tests whether source vocal content, phoneme timing, and melody are preserved over a different instrumental. It uses explicit source/target pairs and does not make participant or identity-generalization claims.",
};
const selectedLandingStudy = (): "study1" | "study2" => (document.querySelector<HTMLInputElement>('input[name="landing-study"]:checked')?.value === "study2" ? "study2" : "study1");
const updateLandingNote = (): void => { landingFootnote.textContent = landingNotes[selectedLandingStudy()]; };
document.querySelectorAll<HTMLInputElement>('input[name="landing-study"]').forEach((radio) => radio.addEventListener("change", updateLandingNote));
updateLandingNote();
continueStudy.addEventListener("click", () => {
  landing.hidden = true;
  workflowNav.hidden = false;
  setTab(selectedLandingStudy() === "study1" ? "training" : "studies");
});

studiesForm.addEventListener("submit", (event) => {
  event.preventDefault();
  studyCommand.hidden = false;
  if (studyKind.value === "study-1") {
    studyCommand.textContent = "docker compose run --rm research \\\n+  singalign-conditioned-train --config configs/training/conditioned.yaml --index data/interim/pjs/index.jsonl --splits data/interim/pjs/splits.json";
    return;
  }
  studyCommand.textContent = `docker compose run --rm research \\\n+  singalign-transfer --source ${studySource.value} --target ${studyTarget.value} --output reports/study-2/transfer.wav\n\nManifest: ${studyManifest.value}`;
  fetch("http://localhost:8000/training", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment: "study2", source: studySource.value, target: studyTarget.value, output: "reports/study-2/transfer.wav" }),
  }).then(async (response) => {
    if (!response.ok) throw new Error((await response.json()).detail ?? "Study 2 launch failed.");
    const result = (await response.json()) as { job_id: string };
    studyCommand.textContent += `\nJob started: ${result.job_id}`;
  }).catch((error: unknown) => {
    studyCommand.textContent += `\nAPI unavailable: ${error instanceof Error ? error.message : "launch failed"}`;
  });
});

const number = (value: number): string =>
  new Intl.NumberFormat("en", {
    maximumSignificantDigits: 5,
    signDisplay: "exceptZero",
  }).format(value);

const duration = (value: number): string =>
  new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(value);

const label = (name: string): string => name.replaceAll("_", " ");

const trainingExperiments = {
  study1: { label: "Study 1 · Same-singer synthesis", config: "configs/training/conditioned.yaml", evaluation: "configs/evaluation/baseline.yaml", prerequisite: "none", compatible: "score/phoneme/F0 conditioning", args: { epochs: 10, segment_seconds: 3, frame_rate: 100 } },
} as const;

Object.entries(trainingExperiments).forEach(([key, experiment]) => {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = experiment.label;
  trainingExperiment.append(option);
  evaluationExperiment.append(option.cloneNode(true));
});

const renderTrainingFields = (): void => {
  trainingFields.replaceChildren();
  const experiment = trainingExperiments[trainingExperiment.value as keyof typeof trainingExperiments];
  experimentContext.textContent = `Study config: ${experiment.config} · ${experiment.compatible}`;
  Object.entries(experiment.args).forEach(([name, value]) => {
    const field = document.createElement("label");
    field.textContent = label(name);
    const input = document.createElement("input");
    input.name = name;
    input.type = "number";
    input.step = "any";
    input.value = String(value);
    field.append(input);
    trainingFields.append(field);
  });
};
renderTrainingFields();
trainingExperiment.addEventListener("change", renderTrainingFields);
evaluationExperiment.addEventListener("change", () => {
  const experiment = trainingExperiments[evaluationExperiment.value as keyof typeof trainingExperiments];
  evaluationCheckpoint.value = experiment.prerequisite === "none" ? `${experiment.config.replace("configs/training/", "checkpoints/").replace(".yaml", "/best.pt")}` : `checkpoints/${evaluationExperiment.value}/best.pt`;
});
evaluationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const experiment = trainingExperiments[evaluationExperiment.value as keyof typeof trainingExperiments];
  const commandName = evaluationExperiment.value === "study2" ? "study" : "evaluate";
  evaluationCommand.hidden = false;
  evaluationCommand.textContent = `docker compose run --rm research \\\n+  singalign-${commandName} \\\n+  --config ${experiment.evaluation} \\\n+  --checkpoint ${evaluationCheckpoint.value} \\\n+  --splits data/interim/pjs/splits.json`;
  workflowReady.comparison = true;
  updateWorkflowAccess();
});

const setTab = (name: string): void => {
  tabs.forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  panels.forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
};
const updateWorkflowAccess = (): void => {
  tabs.forEach((tab) => {
    const name = tab.dataset.tab;
    tab.disabled = name === "evaluation" ? !workflowReady.evaluation : name === "comparison" ? !workflowReady.comparison : false;
    if (tab.disabled && tab.classList.contains("active")) setTab("training");
  });
};
tabs.forEach((tab) => tab.addEventListener("click", () => setTab(tab.dataset.tab ?? "training")));
updateWorkflowAccess();

const safePath = (path: string): string =>
  path
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");

const audioCard = (
  title: string,
  runId: string,
  path: string,
  generated: boolean,
): HTMLElement => {
  const card = document.createElement("article");
  card.className = "audio-card";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "metadata";
  audio.src = `/reports/${encodeURIComponent(runId)}/${safePath(path)}`;
  card.append(heading, audio);
  if (generated) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = "Approximate reconstruction";
    card.append(badge);
  }
  return card;
};

const renderMulti = (report: MultiConditionReport): void => {
  multiResults.replaceChildren();
  multiResults.hidden = false;
  const title = document.createElement("h2");
  title.textContent = `${report.condition_count} conditions`;
  const filter = document.createElement("select");
  filter.setAttribute("aria-label", "Condition filter");
  filter.innerHTML = `<option value="all">All conditions</option>${report.conditions.map((condition) => `<option value="${condition.name}">${condition.name}</option>`).join("")}`;
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Condition</th><th>Method</th><th>Mel MSE</th><th>Mel MAE</th><th>Spectral convergence</th></tr></thead><tbody></tbody>";
  const body = table.querySelector("tbody");
  if (!body) throw new Error("multi-condition table body is missing");
  const renderRows = (selected: string): void => {
    body.replaceChildren();
    report.conditions.filter((condition) => selected === "all" || condition.name === selected).forEach((condition) => {
    const row = document.createElement("tr");
    const metrics = condition.metrics as Record<string, number | { mean: number }>;
    [condition.name, condition.method, metrics.log_mel_mse, metrics.log_mel_mae, metrics.spectral_convergence].forEach((value, index) => {
      const cell = document.createElement(index < 2 ? "th" : "td");
      cell.textContent = typeof value === "number" ? number(value) : typeof value === "object" && value !== null && "mean" in value ? number(value.mean) : "—";
      row.append(cell);
    });
    body.append(row);
    if (condition.candidates?.length || condition.conditioning) {
      const detail = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      const candidates = condition.candidates?.map((candidate) => `${candidate.rank}. ${candidate.id} (${number(candidate.score)})`).join(" · ") ?? "No candidate rankings";
      const conditioning = condition.conditioning ? `Pitch: ${condition.conditioning.pitch_range?.join("–") ?? "—"} MIDI · Phonemes: ${condition.conditioning.phoneme_count ?? "—"} · ${condition.conditioning.frame_rate ?? "—"} Hz` : "No conditioning metadata";
      cell.textContent = `${candidates} | ${conditioning}`;
      detail.append(cell);
      body.append(detail);
    }
    });
  };
  renderRows("all");
  filter.addEventListener("change", () => renderRows(filter.value));
  const note = document.createElement("p");
  note.className = "status";
  note.textContent = report.anchor ? `Paired anchor: ${report.anchor} · ${report.example_count ?? "?"} examples · intervals and effect sizes are available in the JSON report.` : "Legacy single-example report; aggregate uncertainty fields are not available.";
  multiResults.append(title, filter, note, table);
};

const renderExample = (
  container: HTMLElement,
  runId: string,
  example: ComparisonExample,
  index: number,
  total: number,
): void => {
  container.replaceChildren();
  const header = document.createElement("div");
  header.className = "example-header";
  const title = document.createElement("h3");
  title.textContent = `Example ${index + 1} of ${total}: ${example.id}`;
  const disclosure = document.createElement("p");
  disclosure.textContent = `${example.disclosure} Selected reference offset: ${duration(example.offset_seconds)} seconds.`;
  header.append(title, disclosure);

  const grid = document.createElement("div");
  grid.className = "audio-grid";
  grid.append(
    audioCard("Reference", runId, example.reference, false),
    audioCard("Baseline", runId, example.baseline, true),
    audioCard("Aligned", runId, example.aligned, true),
  );
  container.append(header, grid);
};

const render = (
  runId: string,
  manifest: ComparisonManifest,
  summary: ComparisonSummary,
): void => {
  results.replaceChildren();
  results.hidden = false;

  const heading = document.createElement("div");
  heading.className = "results-heading";
  const title = document.createElement("h2");
  title.textContent = manifest.title;
  const metadata = document.createElement("p");
  metadata.textContent = `${summary.split} split · ${summary.examples} paired examples · lower is better`;
  heading.append(title, metadata);
  if (manifest.mlflow_experiment_id && manifest.mlflow_run_id) {
    const mlflowButton = document.createElement("button");
    mlflowButton.type = "button";
    mlflowButton.textContent = "MLflow run";
    mlflowButton.addEventListener("click", () => {
      const url = `${location.protocol}//${location.hostname}:5001/#/experiments/${encodeURIComponent(manifest.mlflow_experiment_id!)}/runs/${encodeURIComponent(manifest.mlflow_run_id!)}`;
      window.open(url, "_blank", "noopener,noreferrer");
    });
    heading.append(mlflowButton);
  }

  const durationsMatch =
    Math.abs(summary.training_segment_seconds - summary.comparison_segment_seconds) <
    Number.EPSILON;
  const durationSummary = document.createElement("div");
  durationSummary.className = `duration-summary ${durationsMatch ? "matched" : "warning"}`;
  const durationStatus = document.createElement("strong");
  durationStatus.textContent = durationsMatch
    ? "Matched duration"
    : "Out-of-training-window";
  const durationDetails = document.createElement("span");
  durationDetails.textContent = `Training window: ${duration(summary.training_segment_seconds)} seconds · Listening window: ${duration(summary.comparison_segment_seconds)} seconds`;
  durationSummary.append(durationStatus, durationDetails);

  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap";
  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr><th>Metric</th><th>Baseline</th><th>Aligned</th><th>Delta</th><th>Confidence interval</th><th>W / T / L</th></tr></thead>
    <tbody></tbody>
  `;
  const body = table.querySelector("tbody");
  if (!body) throw new Error("metrics table body is missing");
  Object.entries(summary.metrics).forEach(([name, metric]) => {
    const row = document.createElement("tr");
    const values = [
      label(name),
      number(metric.baseline_mean),
      number(metric.aligned_mean),
      number(metric.delta.mean),
      `[${number(metric.delta.lower)}, ${number(metric.delta.upper)}]`,
      `${metric.wins} / ${metric.ties} / ${metric.losses}`,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  });
  tableWrap.append(table);

  const listening = document.createElement("section");
  listening.className = "listening";
  const example = document.createElement("div");
  const controls = document.createElement("div");
  controls.className = "navigation";
  const previous = document.createElement("button");
  previous.textContent = "Previous";
  const next = document.createElement("button");
  next.textContent = "Next";
  controls.append(previous, next);
  listening.append(example, controls);

  let position = 0;
  const update = (): void => {
    const current = manifest.examples[position];
    if (!current) return;
    renderExample(example, runId, current, position, manifest.examples.length);
    previous.disabled = position === 0;
    next.disabled = position === manifest.examples.length - 1;
  };
  previous.addEventListener("click", () => {
    position = Math.max(0, position - 1);
    update();
  });
  next.addEventListener("click", () => {
    position = Math.min(manifest.examples.length - 1, position + 1);
    update();
  });

  results.append(heading, durationSummary, tableWrap);
  if (manifest.examples.length > 0) {
    results.append(listening);
    update();
  }
};

const load = async (runId: string): Promise<void> => {
  if (!/^[a-zA-Z0-9_-]+$/.test(runId)) {
    throw new Error("Run ID may contain only letters, numbers, hyphens, and underscores.");
  }
  const base = `/reports/${encodeURIComponent(runId)}`;
  const [manifestResponse, summaryResponse] = await Promise.all([
    fetch(`${base}/manifest.json`, { cache: "no-store" }),
    fetch(`${base}/summary.json`, { cache: "no-store" }),
  ]);
  if (!manifestResponse.ok || !summaryResponse.ok) {
    throw new Error(`No complete comparison report was found for run ${runId}.`);
  }
  const manifest = (await manifestResponse.json()) as ComparisonManifest;
  const summary = (await summaryResponse.json()) as ComparisonSummary;
  render(runId, manifest, summary);
  workflowReady.evaluation = true;
  workflowReady.comparison = true;
  updateWorkflowAccess();
  history.replaceState(null, "", `?run=${encodeURIComponent(runId)}`);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const runId = input.value.trim();
  status.className = "status";
  status.textContent = "Loading comparison…";
  results.hidden = true;
  try {
    await load(runId);
    status.textContent = `Loaded run ${runId}.`;
  } catch (error) {
    status.className = "status error";
    status.textContent = error instanceof Error ? error.message : "Unable to load comparison.";
  }
});

multiForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const path = multiInput.value.trim().replace(/^\/+/, "");
  multiStatus.textContent = "Loading conditions…";
  try {
    if (!path || path.split("/").some((part) => part === ".." || part === ".")) {
      throw new Error("Report path must stay within the reports directory.");
    }
    const response = await fetch(`/reports/${path}`, { cache: "no-store" });
    if (!response.ok) throw new Error("No multi-condition report was found.");
    renderMulti((await response.json()) as MultiConditionReport);
    multiStatus.textContent = "Loaded multi-condition report.";
  } catch (error) {
    multiStatus.className = "status error";
    multiStatus.textContent = error instanceof Error ? error.message : "Unable to load report.";
  }
});

trainingForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const experiment = trainingExperiments[trainingExperiment.value as keyof typeof trainingExperiments];
  const values = new FormData(trainingForm);
  const overrides = [...values.entries()]
    .filter(([name]) => name !== "experiment")
    .map(([name, value]) => `--${name.replaceAll("_", "-")} ${String(value)}`)
    .join(" ");
  const commandName = trainingExperiment.value === "study1" ? "conditioned-train" : "study";
  const command = `docker compose run --rm research \\\n+  singalign-${commandName} \\\n+  --config ${experiment.config} \\\n+  --index data/interim/pjs/index.jsonl \\\n+  --splits data/interim/pjs/splits.json${overrides ? ` \\\n+  ${overrides}` : ""}`;
  trainingCommand.hidden = false;
  trainingCommand.textContent = command;
  if (trainingExperiment.value === "study2") return;
  trainingCommand.textContent += "\n\nLaunching through http://localhost:8000 …";
  fetch("http://localhost:8000/training", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      experiment: "conditioned",
      parameters: Object.fromEntries(
        [...values.entries()].filter(([name]) => name !== "experiment").map(([name, value]) => [name, Number(value)]),
      ),
    }),
  }).then(async (response) => {
    if (!response.ok) throw new Error((await response.json()).detail ?? "Training launch failed.");
    const result = (await response.json()) as { job_id: string };
    trainingCommand.textContent = `${command}\n\nJob started: ${result.job_id}\nProgress: queued`;
    const poll = async (): Promise<void> => {
      const statusResponse = await fetch(`http://localhost:8000/training/${result.job_id}`);
      const status = (await statusResponse.json()) as { status: string };
      trainingCommand.textContent = `${command}\n\nJob ${result.job_id}\nProgress: ${status.status}`;
      if (!["completed-or-unknown", "exited", "dead"].includes(status.status)) {
        window.setTimeout(poll, 2000);
      }
    };
    window.setTimeout(poll, 1000);
    workflowReady.evaluation = true;
    updateWorkflowAccess();
  }).catch((error: unknown) => {
    trainingCommand.textContent = `${command}\n\nAPI unavailable: ${error instanceof Error ? error.message : "launch failed"}`;
  });
});

const initialRun = new URLSearchParams(location.search).get("run");
if (initialRun) {
  input.value = initialRun;
  form.requestSubmit();
} else {
  status.textContent = "Loading the latest local comparison…";
  fetch("/reports/latest.json", { cache: "no-store" })
    .then(async (response) => {
      if (!response.ok) throw new Error("No latest comparison is available yet.");
      return (await response.json()) as LatestComparison;
    })
    .then((latest) => {
      input.value = latest.run_id;
      form.requestSubmit();
    })
    .catch(() => {
      status.textContent = "Enter a run ID after generating a comparison report.";
    });
}
