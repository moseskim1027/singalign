import "./styles.css";
import type {
  ComparisonExample,
  ComparisonManifest,
  ComparisonSummary,
  LatestComparison,
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
  <footer>
    <strong>Interpretation boundary:</strong> this view is not a blinded listening study. Generated examples use approximate Griffin-Lim reconstruction.
  </footer>
`;

const form = document.querySelector<HTMLFormElement>("#run-form");
const input = document.querySelector<HTMLInputElement>("#run-id");
const status = document.querySelector<HTMLElement>("#status");
const results = document.querySelector<HTMLElement>("#results");

if (!form || !input || !status || !results) {
  throw new Error("comparison controls are missing");
}

const number = (value: number): string =>
  new Intl.NumberFormat("en", {
    maximumSignificantDigits: 5,
    signDisplay: "exceptZero",
  }).format(value);

const duration = (value: number): string =>
  new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(value);

const label = (name: string): string => name.replaceAll("_", " ");

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

  results.append(heading, tableWrap);
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
