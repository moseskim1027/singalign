import "./styles.css";

type Study = "study1" | "study2";
type Step = "setup" | "training" | "results";
const app = document.querySelector<HTMLElement>("#app");
if (!app) throw new Error("application root is missing");
const studies = {
  study1: { name: "Same-singer synthesis", description: "Reconstruct the PJS vocalist from score, phonemes, timing, pitch, and observed F0.", setup: "A same-singer diagnostic using the fixed PJS song-disjoint split." },
  study2: { name: "Content-and-melody transfer", description: "Preserve source vocal content, timing, and melody over a different instrumental.", setup: "A controlled source/target remix with explicit pair and alignment settings." },
} as const;
app.innerHTML = `<div class="shell"><header class="topbar"><div class="brand-mark">S</div><div><div class="brand">SingAlign</div><div class="subtitle">Research workspace</div></div><div class="dataset">PJS · reproducible studies</div></header><main id="content"></main></div>`;
const content = document.querySelector<HTMLElement>("#content")!;

function landing(): void {
  content.innerHTML = `<section class="landing-card"><div class="eyebrow">Research workflow</div><h1>Choose a study</h1><p class="lead">Run a focused, reproducible singing experiment.</p><div class="study-grid">${(["study1", "study2"] as Study[]).map((id) => `<label class="study-card"><input type="radio" name="study" value="${id}" ${id === "study1" ? "checked" : ""}><span class="study-number">${id === "study1" ? "01" : "02"}</span><span><strong>${studies[id].name}</strong><small>${studies[id].description}</small></span></label>`).join("")}</div><p id="landing-note" class="note"></p><button id="continue" class="primary">Continue <span>→</span></button></section>`;
  const note = document.querySelector<HTMLElement>("#landing-note")!;
  const selected = (): Study => document.querySelector<HTMLInputElement>('input[name="study"]:checked')!.value as Study;
  const update = (): void => { note.textContent = studies[selected()].setup; };
  document.querySelectorAll<HTMLInputElement>('input[name="study"]').forEach((input) => input.addEventListener("change", update));
  update();
  document.querySelector<HTMLButtonElement>("#continue")!.onclick = () => workflow(selected());
}

function workflow(study: Study): void {
  let step: Step = "setup"; let runId = "";
  const steps: Step[] = ["setup", "training", "results"];
  const draw = (): void => {
    content.innerHTML = `<div class="workflow-head"><div><div class="eyebrow">Study ${study === "study1" ? "01" : "02"}</div><h1>${studies[study].name}</h1><p>${studies[study].description}</p></div><button id="change" class="quiet">Change study</button></div><nav class="stepper">${steps.map((item, i) => `<button class="step ${item === step ? "active" : ""}" data-step="${item}"><span>${i + 1}</span>${item.charAt(0).toUpperCase() + item.slice(1)}</button>`).join("")}</nav><section id="stage" class="stage"></section>`;
    document.querySelector<HTMLButtonElement>("#change")!.onclick = landing;
    document.querySelectorAll<HTMLButtonElement>(".step").forEach((button) => { button.onclick = () => { step = button.dataset.step as Step; draw(); }; });
    const stage = document.querySelector<HTMLElement>("#stage")!;
    if (step === "setup") stage.innerHTML = `<div class="card"><h2>Study setup</h2><p>${studies[study].setup} Results will be tracked with MLflow, Git revision, seed, and split fingerprint.</p><div class="facts"><div><b>Dataset</b><span>PJS v1.1</span></div><div><b>Split</b><span>80 / 10 / 10</span></div><div><b>Tracking</b><span>MLflow</span></div></div><button id="next" class="primary">Continue to training <span>→</span></button></div>`;
    if (step === "training") stage.innerHTML = study === "study1" ? `<div class="card"><h2>Train Study 1</h2><p>Start the score-conditioned baseline. The MLflow run ID will carry forward to evaluation.</p><button id="run" class="primary">Start training</button><pre id="status" class="output" hidden></pre></div>` : `<div class="card"><h2>Run Study 2 transfer</h2><p>Start the Docker transfer control with a declared source and target.</p><label>Source vocal<input id="source" value="data/interim/source.wav"></label><label>Target instrumental<input id="target" value="data/interim/target.wav"></label><button id="run" class="primary">Start transfer</button><pre id="status" class="output" hidden></pre></div>`;
    if (step === "results") stage.innerHTML = `<div class="card"><h2>Results</h2><p>Run evaluation and review objective diagnostics and engineering audio artifacts. These results do not establish human preference or unseen-singer generalization.</p><div class="result-row"><span>Training MLflow run</span><code>${runId || "Not available yet"}</code></div><button id="evaluate" class="primary">Run evaluation</button><pre id="evaluation-status" class="output" hidden></pre></div>`;
    if (step === "training") { const next = document.createElement("button"); next.className = "primary continue-training"; next.textContent = "Continue to results →"; next.disabled = true; stage.append(next); const enable = (): void => { const status = document.querySelector<HTMLElement>("#status"); if (status && /Progress: (exited|completed|dead)/.test(status.textContent || "")) next.disabled = false; else window.setTimeout(enable, 1000); }; next.onclick = () => { step = "results"; draw(); }; window.setTimeout(enable, 1000); }
    document.querySelector<HTMLButtonElement>("#next")?.addEventListener("click", () => { step = "training"; draw(); });
    document.querySelector<HTMLButtonElement>("#evaluate")?.addEventListener("click", () => { const output = document.querySelector<HTMLElement>("#evaluation-status")!; output.hidden = false; output.textContent = `Evaluation queued for training run ${runId || "(pending)"}.\nUse the generated MLflow report in the comparison tools.`; });
    document.querySelector<HTMLButtonElement>("#run")?.addEventListener("click", () => launch(study));
  };
  const launch = (id: Study): void => { const out = document.querySelector<HTMLElement>("#status")!; out.hidden = false; out.textContent = "Launching Docker job…"; const body = id === "study1" ? { experiment: "conditioned", parameters: {} } : { experiment: "study2", source: (document.querySelector<HTMLInputElement>("#source")!).value, target: (document.querySelector<HTMLInputElement>("#target")!).value, output: "reports/study-2/transfer.wav" }; fetch("http://localhost:8000/training", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(async (response) => { if (!response.ok) throw new Error((await response.json()).detail); const result = await response.json() as { job_id: string }; out.textContent = `Job ${result.job_id}\nProgress: queued`; const poll = async (): Promise<void> => { const status = await (await fetch(`http://localhost:8000/training/${result.job_id}`)).json() as { status: string; mlflow_run_id?: string }; runId = status.mlflow_run_id ?? runId; out.textContent = `Job ${result.job_id}\nProgress: ${status.status}\nMLflow run: ${runId || "pending"}`; if (!["completed-or-unknown", "exited", "dead"].includes(status.status)) window.setTimeout(poll, 2000); }; window.setTimeout(poll, 1000); }).catch((error: unknown) => { out.textContent = `Unable to launch: ${error instanceof Error ? error.message : "API unavailable"}`; }); };
  draw();
}
landing();
