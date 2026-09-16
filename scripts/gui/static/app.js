/* DISARM Agent Console — live view of what the classification agent is doing. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// DISARM descriptions embed <br>/<b> markup; escape everything, then re-allow only those
const rich = (s) =>
  esc(s)
    .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
    .replace(/&lt;(\/?)(b|i)&gt;/gi, "<$1$2>");
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
const secs = (n) => (n == null ? "" : `${Number(n).toFixed(n < 10 ? 1 : 0)}s`);

const MODE_HINTS = {
  INVESTIGATE:
    "Only tags a technique when the author or publisher themselves used it. Reporting or quoting someone else's disinformation does not count. Prefers false negatives, and is the only mode that gathers web evidence.",
  RECOGNISE:
    "Tags any technique referenced in the article, whoever performed it. No evidence gathering.",
};

const state = {
  framework: null,
  techniques: {},
  tacticsByShortname: {},
  articles: [],
  current: null,
  evidence: [],
  searches: [],
  llmCalls: 0,
  activeCall: null,
  failures: 0,
  runState: "idle",
  runStartedAt: null,
  runKind: null,
  config: null,
  steps: null, // {label, index, total}
  uploads: [],
  feedCount: 0,
  evidenceContext: [], // techniques the current evidence round is researching
  feedGroup: { article: null, tactic: null }, // open activity groups that events nest into
  expanded: new Set(), // article keys expanded on the Articles tab
};

/* ------------------------------------------------------------------ setup */

async function init() {
  const [fw, defaults] = await Promise.all([
    fetch("/api/framework").then((r) => r.json()),
    fetch("/api/defaults").then((r) => r.json()),
  ]);

  state.framework = fw;
  state.techniques = fw.techniques;
  fw.tactics.forEach((t) => (state.tacticsByShortname[t.shortname] = t));

  $("model_name").value = defaults.config.model_name;
  $("mode").value = defaults.config.mode;
  $("architecture").value = defaults.config.architecture;
  $("local_model").checked = defaults.config.local_model;
  $("check_sub_techniques").checked = defaults.config.check_sub_techniques;
  $("web_search").checked = defaults.config.web_search;
  $("max_search_rounds").value = defaults.config.max_search_rounds;
  $("results_per_query").value = defaults.config.results_per_query;

  renderFramework();
  refreshModeHint();
  refreshDatasetSummary();
  connect();
  setInterval(tick, 500);
}

function techInfo(id) {
  return (
    state.techniques[id] || {
      external_id: id,
      name: "Unknown technique",
      description: "This id is not present in DISARM.json.",
      tactics: [],
      url: null,
      is_sub_technique: String(id).includes("."),
    }
  );
}

const tacticName = (shortname) => state.tacticsByShortname[shortname]?.name || shortname;

/* ------------------------------------------------------------ SSE stream */

function connect() {
  const es = new EventSource("/api/events");
  es.addEventListener("status", (e) => applyStatus(JSON.parse(e.data)));
  es.onmessage = (e) => handle(JSON.parse(e.data));
  es.onerror = () => {
    /* EventSource reconnects on its own; the server replays the run history on reconnect */
  };
}

function applyStatus(status) {
  const env = status.env || {};
  const missing = [];
  if (!env.diffbot_token) missing.push("DIFFBOT_TOKEN");
  if (!env.openai_key) missing.push("OPENAI_API_KEY");
  $("envWarn").textContent = missing.length ? `${missing.join(" & ")} not set` : "";
  setRunState(status.running ? "running" : status.state || "idle");
  if (status.running && status.started_at) state.runStartedAt = status.started_at * 1000;
}

/* --------------------------------------------------------- event handling */

function handle(ev) {
  switch (ev.type) {
    case "run_started":
      resetRun(ev);
      feed("▶", "accent", "Run started", runSummary(ev), ev.ts);
      break;

    case "run_stopping":
      setRunState("stopping");
      feed("■", "amber", "Stop requested", "Finishing the current step, then stopping.", ev.ts);
      break;

    case "run_finished":
      setRunState(ev.state);
      state.activeCall = null;
      setNow(
        { finished: "Run finished", stopped: "Run stopped", failed: "Run failed" }[ev.state] || "Idle",
        ""
      );
      feed("✔", ev.state === "finished" ? "green" : "amber", `Run ${ev.state}`, summariseRun(), ev.ts);
      break;

    case "run_failed":
      state.failures++;
      feed("✖", "red", "Run failed", esc(ev.error), ev.ts);
      break;

    case "dataset_preparing":
      feed("⚙", "amber", "Building the balanced subset", `${esc(ev.subset_path)} does not exist yet — creating it from the full dataset.`, ev.ts);
      break;

    case "dataset_started":
      state.current = null;
      feed(
        "▶",
        "accent",
        "Dataset labelling started",
        `${ev.queued} article(s) queued · ${ev.already_labelled} already in cache · ${ev.total} in subset` +
          `<div class="code" style="margin-top:4px;color:var(--fg-faint)">${esc(ev.cache_path)}</div>`,
        ev.ts
      );
      break;

    case "dataset_stopped":
      feed("■", "amber", "Dataset run stopped", `${ev.completed} labelled this run, ${ev.failed} failed. Progress is kept in the cache.`, ev.ts);
      break;

    case "dataset_finished":
      feed(
        "✔",
        "green",
        "Dataset labelling finished",
        `${ev.labelled} newly labelled · ${ev.skipped} skipped · ${ev.failed} failed` +
          (ev.exports?.length ? `<div class="code" style="margin-top:4px">Wrote: ${ev.exports.map(esc).join(", ")}</div>` : ""),
        ev.ts
      );
      break;

    case "article_started": {
      const article = {
        key: `${ev.index}::${ev.article_id}`,
        index: ev.index,
        total: ev.total,
        id: ev.article_id,
        preview: ev.preview,
        chars: ev.chars,
        url: ev.url,
        publisher: ev.publisher,
        language: ev.language,
        label: ev.label,
        status: "running",
        techniques: [],
        live: [],
        startedAt: ev.ts,
      };
      state.articles.push(article);
      state.current = article;
      state.steps = null;
      // everything this article produces nests under its own group in the activity feed
      openArticleGroup(article, ev.ts);
      renderArticles();
      break;
    }

    case "article_finished": {
      const article = state.current;
      if (article) {
        article.status = "done";
        article.techniques = ev.techniques || [];
        article.duration = ev.duration;
      }
      closeTacticGroup(null);
      feed("✔", "green", `Article ${ev.index} done — ${(ev.techniques || []).length} technique(s)`, tagList(ev.techniques || []), ev.ts);
      closeArticleGroup(article, ev.techniques || []);
      state.current = null;
      renderAll();
      break;
    }

    case "article_failed": {
      const article = state.current;
      state.failures++;
      if (article) article.status = "failed";
      closeTacticGroup(null);
      feed("✖", "red", `Article ${ev.index} failed`, esc(ev.error), ev.ts);
      closeArticleGroup(article, null);
      state.current = null;
      renderArticles();
      break;
    }

    case "plan":
      if (ev.architecture === "taxonomy" && ev.stage === "sub_techniques") {
        state.steps = { label: "Sub-technique checks", index: 0, total: (ev.parents || []).length };
        openSubTechniqueGroup(ev);
      } else if (ev.architecture === "single_clf") {
        state.steps = { label: "Techniques", index: 0, total: ev.technique_count };
        feed("🗺", "accent", "Plan", `Testing all ${ev.technique_count} techniques one prompt at a time.`, ev.ts);
      } else {
        state.steps = { label: "Tactics", index: 0, total: (ev.tactics || []).length };
        feed(
          "🗺",
          "accent",
          "Plan",
          `Working through ${(ev.tactics || []).length} tactic(s)${ev.fast ? " (pre-selected by the model)" : ""}: ` +
            (ev.tactics || []).map((t) => `<span class="tag plain">${esc(tacticName(t))}</span>`).join(""),
          ev.ts
        );
      }
      break;

    case "task":
      if (ev.index && ev.total) state.steps = { label: state.steps?.label || "Steps", index: ev.index, total: ev.total };
      setNow(ev.label, taskDetail(ev));
      break;

    case "tactics_identified":
      feed("🎯", "accent", "Candidate tactics selected", (ev.tactics || []).map((t) => `<span class="tag plain">${esc(tacticName(t))}</span>`).join(""), ev.ts);
      break;

    case "tactic_started":
      // batch_clf / taxonomy work tactic by tactic, so each tactic gets its own subgroup
      openTacticGroup(ev);
      break;

    case "tactic_finished":
      closeTacticGroup(ev);
      break;

    case "techniques_identified":
      if (state.current) state.current.live = dedupe([...state.current.live, ...(ev.techniques || [])]);
      renderTechniques();
      break;

    case "sub_techniques_identified":
      if (state.current) {
        state.current.live = dedupe([...state.current.live.filter((t) => t !== ev.parent), ...(ev.techniques || [])]);
      }
      if (state.steps) state.steps.index = Math.min(state.steps.index + 1, state.steps.total);
      feed(
        "🔬",
        "purple",
        `${esc(ev.parent)} → ${ev.fallback ? "kept as-is" : (ev.techniques || []).join(", ")}`,
        ev.fallback ? "The model picked no sub-technique, so the parent technique stands." : tagList(ev.techniques || []),
        ev.ts
      );
      renderTechniques();
      break;

    case "evidence_started":
      state.evidenceContext = ev.external_ids || [];
      feed(
        "🔍",
        "purple",
        "Evidence gathering started",
        `${(ev.external_ids || []).length} technique(s) need outside knowledge, via ${esc(ev.backend)} (up to ${ev.max_rounds} rounds):<br>` + tagList(ev.external_ids || []),
        ev.ts
      );
      break;

    case "search_round": {
      const round = { articleId: state.current?.id, round: ev.round, maxRounds: ev.max_rounds, queries: ev.queries || [], results: [], ts: ev.ts, forTechniques: state.evidenceContext };
      state.searches.push(round);
      feed("🔎", "cyan", `Web search round ${ev.round}/${ev.max_rounds}`, (ev.queries || []).map((q) => `<div class="query">${esc(q)}</div>`).join(""), ev.ts);
      renderSearches();
      break;
    }

    case "search_results": {
      const round = [...state.searches].reverse().find((r) => r.round === ev.round && !r.results.length);
      if (round) round.results = ev.results || [];
      feed("📥", "cyan", `${ev.count} search result(s)`, (ev.results || []).slice(0, 4).map((r) => `<div class="code" style="color:var(--fg-faint)">${esc(r.title || r.url)}</div>`).join(""), ev.ts);
      renderSearches();
      break;
    }

    case "search_failed":
      feed("⚠", "amber", `Search round ${ev.round} failed`, esc(ev.error) + "<br>The agent carries on with what it already found.", ev.ts);
      break;

    case "evidence_reported":
      (ev.evidence || []).forEach((e) =>
        state.evidence.push({ ...e, articleId: state.current?.id, articleKey: state.current?.key, ts: ev.ts })
      );
      feed("📑", "purple", `Evidence reported for ${(ev.evidence || []).length} technique(s)`, `Gathered in ${secs(ev.duration)}. See the Evidence tab.`, ev.ts);
      renderEvidence();
      renderTechniques();
      break;

    case "llm_call_started":
      state.llmCalls++;
      state.activeCall = { task: ev.task, startedAt: Date.now(), candidates: ev.candidates, model: ev.model };
      break;

    case "llm_call_finished":
      state.activeCall = null;
      break;

    case "classification_complete":
      closeTacticGroup(null);
      feed("🏁", "green", `${esc(ev.architecture)} complete`, `${(ev.techniques || []).length} technique(s) in ${secs(ev.duration)}`, ev.ts);
      break;

    case "log":
      appendLog(ev.level, ev.message);
      break;
  }

  updateStats();
}

const dedupe = (arr) => [...new Set(arr)];

function taskDetail(ev) {
  const bits = [];
  if (ev.tactic) bits.push(`tactic: ${tacticName(ev.tactic)}`);
  if (ev.technique_count) bits.push(`${ev.technique_count} candidates`);
  if (ev.candidate_count) bits.push(`${ev.candidate_count} sub-techniques`);
  if (ev.external_ids) bits.push(ev.external_ids.join(", "));
  return bits.join(" · ");
}

function runSummary(ev) {
  const c = ev.config || {};
  return `<span class="code">${esc(c.model_name)}</span> · ${esc(c.mode)} · ${esc(c.architecture)} · web search ${c.web_search ? "on" : "off"} · sub-techniques ${c.check_sub_techniques ? "on" : "off"}`;
}

function summariseRun() {
  const done = state.articles.filter((a) => a.status === "done").length;
  return `${done} article(s) classified · ${techniqueMap().size} distinct technique(s) · ${state.evidence.length} evidence item(s) · ${state.llmCalls} LLM call(s)`;
}

function articleMeta(a) {
  const bits = [];
  if (a.publisher) bits.push(esc(a.publisher));
  if (a.language) bits.push(esc(a.language));
  if (a.label != null) bits.push(a.label === 1 ? "labelled disinformation" : "labelled trustworthy");
  if (a.chars) bits.push(`${a.chars.toLocaleString()} chars`);
  return bits.length ? `<div class="code" style="color:var(--fg-faint)">${bits.join(" · ")}</div>` : "";
}

function tagList(ids) {
  return ids.map((id) => `<span class="tag" data-tech="${esc(id)}" title="${esc(techInfo(id).name)}">${esc(id)}</span>`).join("");
}

/* ---------------------------------------------------------------- run UI */

function resetRun(ev) {
  Object.assign(state, {
    articles: [], current: null, evidence: [], searches: [],
    llmCalls: 0, activeCall: null, failures: 0, steps: null, feedCount: 0,
    feedGroup: { article: null, tactic: null }, expanded: new Set(),
  });
  state.runKind = ev.kind;
  state.config = ev.config;
  state.runStartedAt = Date.now();
  $("feed").innerHTML = "";
  $("logbox").innerHTML = "";
  setRunState("running");
  renderAll();
}

function setRunState(s) {
  state.runState = s;
  const pill = $("statusPill");
  pill.className = "pill " + s;
  $("statusText").textContent = s;
  $("stopBtn").disabled = !(s === "running" || s === "stopping");
  $("runBtn").disabled = s === "running" || s === "stopping";
  if (s !== "running" && s !== "stopping") state.runStartedAt = null;
}

function setNow(task, sub) {
  $("nowTask").textContent = task;
  $("nowSub").innerHTML = sub || "";
}

function tick() {
  $("elapsed").textContent = state.runStartedAt ? `${Math.round((Date.now() - state.runStartedAt) / 1000)}s` : "";

  if (state.activeCall) {
    const waited = ((Date.now() - state.activeCall.startedAt) / 1000).toFixed(1);
    const cands = state.activeCall.candidates ? ` · ${state.activeCall.candidates} candidate labels` : "";
    $("nowSub").innerHTML = `<span style="color:var(--accent)">⟳ waiting on ${esc(state.activeCall.model)} — ${waited}s</span>${cands}`;
  }

  const bars = state.runState === "running" || state.runState === "stopping";
  $("bars").hidden = !bars;
  if (!bars) return;

  const a = state.current || state.articles[state.articles.length - 1];
  const showArticles = a && a.total > 1;
  $("barArticles").hidden = !showArticles;
  if (showArticles) {
    $("barArticlesVal").textContent = `${a.index} / ${a.total}`;
    $("barArticlesFill").style.width = `${((a.index - 1) / a.total) * 100}%`;
  }

  $("barSteps").hidden = !state.steps;
  if (state.steps) {
    $("barStepsLabel").textContent = state.steps.label;
    $("barStepsVal").textContent = `${state.steps.index} / ${state.steps.total}`;
    $("barStepsFill").style.width = `${state.steps.total ? (state.steps.index / state.steps.total) * 100 : 0}%`;
  }
}

/* -------------------------------------------------------------- activity */

/* The feed nests: run-level events sit at the root, everything an article produces goes
   under that article's group, and under batch_clf/taxonomy each tactic (and the taxonomy
   sub-technique stage) gets its own subgroup inside it. Finished groups collapse to a
   one-line summary so a 16-tactic run stays traversable. */

function makeGroup(cls, summaryHTML, open = true) {
  const group = document.createElement("details");
  group.className = "grp " + cls;
  group.open = open;
  const summary = document.createElement("summary");
  summary.innerHTML = summaryHTML;
  const body = document.createElement("div");
  body.className = "grp-body";
  group.append(summary, body);
  return { el: group, body, set: (html) => (summary.innerHTML = html) };
}

// events land in the innermost open group
function feedTarget() {
  return state.feedGroup.tactic?.body || state.feedGroup.article?.body || $("feed");
}

function groupHead(icon, title, meta) {
  return `<span class="g-ico">${icon}</span><span class="g-title">${title}</span>` + (meta ? `<span class="g-meta">${meta}</span>` : "");
}

function openArticleGroup(article, ts) {
  closeTacticGroup(null);
  const group = makeGroup(
    "art",
    groupHead("📄", `Article ${article.index}/${article.total} — ${esc(article.id)}`, `${clock(ts)} · running`)
  );
  $("feed").appendChild(group.el);
  while ($("feed").children.length > 300) $("feed").removeChild($("feed").firstChild);
  $("feedEmpty").hidden = true;
  state.feedGroup.article = group;
  state.feedGroup.tactic = null;
}

function closeArticleGroup(article, techniques) {
  closeTacticGroup(null);
  const group = state.feedGroup.article;
  if (group && article) {
    const outcome =
      techniques === null
        ? `<span style="color:var(--red)">failed</span>`
        : `${techniques.length} technique(s)${article.duration ? ` · ${secs(article.duration)}` : ""}`;
    group.set(groupHead("📄", `Article ${article.index}/${article.total} — ${esc(article.id)}`, outcome));
    // one article on its own stays open; a queue or dataset run collapses as it goes
    if (article.total > 1) group.el.open = false;
  }
  state.feedGroup.article = null;
  state.feedGroup.tactic = null;
}

function openTacticGroup(ev) {
  closeTacticGroup(null);
  const group = makeGroup(
    "tac",
    groupHead("🎯", `Tactic ${ev.index}/${ev.total}: ${esc(tacticName(ev.tactic))}`, `testing ${ev.technique_count} technique(s)…`)
  );
  (state.feedGroup.article?.body || $("feed")).appendChild(group.el);
  state.feedGroup.tactic = group;
  scrollFeed(group.el);
}

function closeTacticGroup(ev) {
  const group = state.feedGroup.tactic;
  if (group && ev) {
    const found = ev.techniques || [];
    group.set(
      groupHead(
        found.length ? "➕" : "·",
        `Tactic ${ev.index}/${ev.total}: ${esc(tacticName(ev.tactic))}`,
        (found.length ? `<span style="color:var(--green)">${found.length} technique(s)</span> ${tagList(found)}` : "nothing") +
          ` · ${secs(ev.duration)}`
      )
    );
    // once done, a tactic stays open only if it found something AND has detail worth
    // reading underneath (evidence gathering, searches) - otherwise the summary says it all
    group.el.open = found.length > 0 && group.body.children.length > 0;
  }
  state.feedGroup.tactic = null;
}

function openSubTechniqueGroup(ev) {
  closeTacticGroup(null);
  const parents = ev.parents || [];
  const group = makeGroup("tac sub", groupHead("🔬", "Sub-technique breakdown", `${parents.length} parent technique(s): ${tagList(parents)}`));
  (state.feedGroup.article?.body || $("feed")).appendChild(group.el);
  state.feedGroup.tactic = group;
  scrollFeed(group.el);
}

function feed(icon, cls, title, detail, ts) {
  const box = feedTarget();
  const node = document.createElement("div");
  node.className = "ev " + (cls || "");
  node.innerHTML =
    `<div class="ico">${icon}</div><div class="time">${clock(ts)}</div>` +
    `<div class="body"><div class="title"><b>${title}</b></div>${detail ? `<div class="detail">${detail}</div>` : ""}</div>`;
  box.appendChild(node);
  state.feedCount++;
  while (box.children.length > 400) box.removeChild(box.firstChild);
  $("feedEmpty").hidden = true;
  scrollFeed(node);
}

function scrollFeed(node) {
  const panel = document.querySelector(".panels");
  if (panel.scrollHeight - panel.scrollTop - panel.clientHeight < 250) node.scrollIntoView({ block: "end" });
}

/* ------------------------------------------------------------ techniques */

function techniqueMap() {
  const map = new Map();
  for (const a of state.articles) {
    const ids = a.status === "done" ? a.techniques : a.live;
    for (const id of ids || []) {
      if (!map.has(id)) map.set(id, { articles: [], provisional: a.status !== "done" });
      map.get(id).articles.push(a.id);
    }
  }
  return map;
}

// one technique, with its DISARM name and description and whatever the agent researched
// about it. `evidence` is pre-filtered by the caller: every article on the Techniques tab,
// one article's own evidence when an article row is expanded.
function techniqueCard(id, { count = "", provisional = false, evidence = [] } = {}) {
  const t = techInfo(id);
  const tactics = (t.tactics || []).map((name) => `<span class="tag plain">${esc(tacticName(name))}</span>`).join("");
  const countBadge = count ? `<span class="tcard-count">${count}</span>` : "";
  const provBadge = provisional ? `<span class="tcard-count" style="color:var(--amber)">in progress</span>` : "";
  const findings = evidence
    .map(
      (e) =>
        `<div class="findings evidence-note">` +
        `<span class="evidence-kicker">Evidence gathered</span><br>${esc(e.findings)}` +
        (e.sources?.length
          ? `<div class="sources">${e.sources.map((u) => `<a href="${esc(u)}" target="_blank" rel="noreferrer">${esc(u)}</a>`).join("")}</div>`
          : "") +
        `</div>`
    )
    .join("");

  return (
    `<div class="tcard ${t.is_sub_technique ? "sub" : ""}">` +
    `<div class="tcard-head"><span class="tcard-id">${esc(id)}</span><span class="tcard-name">${esc(t.name)}</span>${countBadge}${provBadge}</div>` +
    `<div class="tcard-desc">${rich(t.description)}</div>` +
    findings +
    `<div class="tcard-meta">${tactics}${t.url ? `<a href="${esc(t.url)}" target="_blank" rel="noreferrer">DISARM reference ↗</a>` : ""}</div>` +
    `</div>`
  );
}

function renderTechniques() {
  const map = techniqueMap();
  const box = $("techniques");
  $("techEmpty").hidden = map.size > 0;

  const entries = [...map.entries()].sort((a, b) => b[1].articles.length - a[1].articles.length || a[0].localeCompare(b[0]));
  box.innerHTML = entries
    .map(([id, meta]) =>
      techniqueCard(id, {
        count: state.articles.length > 1 ? `${meta.articles.length} article(s)` : "",
        provisional: meta.provisional,
        evidence: state.evidence.filter((e) => e.external_id === id),
      })
    )
    .join("");
}

/* -------------------------------------------------------------- evidence */

function renderEvidence() {
  const box = $("evidence");
  $("evEmpty").hidden = state.evidence.length > 0;
  box.innerHTML = state.evidence
    .map((e) => {
      const t = techInfo(e.external_id);
      return (
        `<div class="ecard">` +
        `<h4><span class="tcard-id">${esc(e.external_id)}</span> <span>${esc(t.name)}</span>` +
        (e.articleId ? `<span class="tcard-count" style="margin-left:auto">${esc(e.articleId)}</span>` : "") +
        `</h4>` +
        `<div class="tcard-desc" style="margin-top:2px">${rich(t.description)}</div>` +
        `<div class="findings">${esc(e.findings)}</div>` +
        (e.sources?.length
          ? `<div class="sources">${e.sources.map((s) => `<a href="${esc(s)}" target="_blank" rel="noreferrer">${esc(s)}</a>`).join("")}</div>`
          : `<div class="hint">No sources cited.</div>`) +
        `</div>`
      );
    })
    .join("");
}

/* -------------------------------------------------------------- searches */

function renderSearches() {
  const box = $("searches");
  $("searchEmpty").hidden = state.searches.length > 0;
  box.innerHTML = state.searches
    .map(
      (r) =>
        `<div class="round"><h4>Round ${r.round} / ${r.maxRounds}${r.articleId ? ` · ${esc(r.articleId)}` : ""} · ${clock(r.ts)}</h4>` +
        (r.forTechniques?.length ? `<div class="tcard-meta" style="margin:0 0 8px">Researching ${tagList(r.forTechniques)}</div>` : "") +
        r.queries.map((q) => `<div class="query">${esc(q)}</div>`).join("") +
        r.results
          .map(
            (res) =>
              `<div class="result"><a href="${esc(res.url)}" target="_blank" rel="noreferrer">${esc(res.title || res.url)}</a>` +
              `<div class="url">${esc(res.url)}${res.date ? ` · ${esc(res.date)}` : ""}</div>` +
              (res.snippet ? `<div class="snip">${esc(res.snippet)}</div>` : "") +
              `</div>`
          )
          .join("") +
        (r.results.length === 0 ? `<div class="hint">Waiting for results…</div>` : "") +
        `</div>`
    )
    .reverse()
    .join("");
}

/* -------------------------------------------------------------- articles */

function renderArticles() {
  const box = $("articles");
  $("artEmpty").hidden = state.articles.length > 0;

  const done = state.articles.filter((a) => a.status === "done");
  const avg = done.length ? done.reduce((s, a) => s + (a.duration || 0), 0) / done.length : 0;
  $("articleStats").innerHTML = state.articles.length
    ? [
        [state.articles.length, "Seen this run"],
        [done.length, "Classified"],
        [state.failures, "Failed"],
        [avg ? secs(avg) : "—", "Avg per article"],
      ]
        .map(([v, l]) => `<div class="sbox"><b>${v}</b><span>${l}</span></div>`)
        .join("")
    : "";

  // each row expands into the same technique view as the Techniques tab, scoped to
  // this one article - its techniques, and only the evidence gathered for it
  box.innerHTML = state.articles
    .slice()
    .reverse()
    .map((a) => {
      const ids = a.status === "done" ? a.techniques : a.live;
      const evidence = state.evidence.filter((e) => e.articleKey === a.key);
      const cards = (ids || [])
        .map((id) =>
          techniqueCard(id, { provisional: a.status !== "done", evidence: evidence.filter((e) => e.external_id === id) })
        )
        .join("");

      return (
        `<details class="artrow" ${state.expanded.has(a.key) ? "open" : ""}>` +
        `<summary data-artkey="${esc(a.key)}">` +
        `<div class="artrow-head"><b>${esc(a.id)}</b>` +
        `<span class="code" style="color:var(--fg-faint)">${a.index}/${a.total}</span>` +
        `<span class="st ${a.status}">${a.status}${a.duration ? ` · ${secs(a.duration)}` : ""}</span></div>` +
        articleMeta(a) +
        (a.url ? `<div class="sources"><a href="${esc(a.url)}" target="_blank" rel="noreferrer">${esc(a.url)}</a></div>` : "") +
        `<div class="prev">${esc(a.preview)}</div>` +
        (ids?.length
          ? `<div class="tcard-meta">${tagList(ids)}</div>`
          : `<div class="hint">No techniques${a.status === "done" ? "" : " yet"}.</div>`) +
        `</summary>` +
        `<div class="artrow-body">` +
        (cards ||
          `<div class="hint">Nothing to show yet${a.status === "running" ? " - the agent is still working on this article." : "."}</div>`) +
        (evidence.length ? `<p class="hint" style="margin-top:10px">${evidence.length} evidence item(s) gathered for this article.</p>` : "") +
        `</div></details>`
      );
    })
    .join("");
}

// remember which rows are open, so a redraw mid-run does not collapse what is being read
$("articles").addEventListener("click", (e) => {
  const summary = e.target.closest("summary[data-artkey]");
  if (!summary) return;
  if (e.target.closest(".tag[data-tech]")) {
    // a technique tag inside the header is a link to the framework, not a toggle
    e.preventDefault();
    return;
  }
  const key = summary.dataset.artkey;
  if (state.expanded.has(key)) state.expanded.delete(key);
  else state.expanded.add(key);
});

/* ------------------------------------------------------------- framework */

function renderFramework() {
  const sel = $("fwTactic");
  state.framework.tactics.forEach((t) => {
    const o = document.createElement("option");
    o.value = t.shortname;
    o.textContent = `${t.external_id} · ${t.name}`;
    sel.appendChild(o);
  });

  const draw = () => {
    const q = $("fwSearch").value.trim().toLowerCase();
    const tacticFilter = sel.value;
    const out = [];

    for (const tactic of state.framework.tactics) {
      if (tacticFilter && tactic.shortname !== tacticFilter) continue;
      const matches = tactic.technique_ids
        .map((id) => state.techniques[id])
        .filter((t) => !q || t.external_id.toLowerCase().includes(q) || t.name.toLowerCase().includes(q) || t.description.toLowerCase().includes(q));
      if (!matches.length) continue;

      out.push(
        `<div class="tactic-head"><span class="tid">${esc(tactic.external_id)}</span><h3>${esc(tactic.name)}</h3>` +
          `<p>${matches.length} technique(s)</p></div><p class="hint" style="margin:0 0 8px">${rich(tactic.description)}</p>`
      );
      out.push(
        matches
          .map(
            (t) =>
              `<div class="tcard ${t.is_sub_technique ? "sub" : ""}"><div class="tcard-head">` +
              `<span class="tcard-id">${esc(t.external_id)}</span><span class="tcard-name">${esc(t.name)}</span></div>` +
              `<div class="tcard-desc">${rich(t.description)}</div>` +
              (t.url ? `<div class="tcard-meta"><a href="${esc(t.url)}" target="_blank" rel="noreferrer">DISARM reference ↗</a></div>` : "") +
              `</div>`
          )
          .join("")
      );
    }

    const meta = state.framework.meta;
    $("framework").innerHTML =
      `<p class="hint" style="margin-bottom:12px">${meta.technique_count} techniques and ${meta.sub_technique_count} sub-techniques across ${meta.tactic_count} tactics, from ${esc(meta.source)}.</p>` +
      (out.length ? out.join("") : `<div class="empty">Nothing matches that search.</div>`);
  };

  $("fwSearch").addEventListener("input", draw);
  sel.addEventListener("change", draw);
  draw();
}

/* ------------------------------------------------------------------ logs */

const LOG_LABELS = { system_prompt: "SYSTEM", prompt: "PROMPT", response: "RESPONSE", search: "SEARCH", info: "INFO", warn: "WARN" };

function appendLog(level, message) {
  const box = $("logbox");
  const line = document.createElement("span");
  line.className = `logline log-${level}`;
  line.innerHTML = `<span class="lvl">[${LOG_LABELS[level] || level}]</span>${esc(message)}`;
  box.appendChild(line);
  while (box.children.length > 3000) box.removeChild(box.firstChild);
  if ($("logFollow").checked) box.scrollTop = box.scrollHeight;
}

/* ----------------------------------------------------------------- stats */

function updateStats() {
  const map = techniqueMap();
  const queries = state.searches.reduce((n, r) => n + r.queries.length, 0);
  const done = state.articles.filter((a) => a.status === "done").length;
  $("statTech").textContent = map.size;
  $("statEv").textContent = state.evidence.length;
  $("statSearch").textContent = queries;
  $("statCalls").textContent = state.llmCalls;
  $("statArticles").textContent = done;
  $("statFail").textContent = state.failures;
  $("badgeTech").textContent = map.size;
  $("badgeEv").textContent = state.evidence.length;
  $("badgeSearch").textContent = queries;
  $("badgeArt").textContent = state.articles.length;
}

function renderAll() {
  renderTechniques();
  renderEvidence();
  renderSearches();
  renderArticles();
  updateStats();
}

/* -------------------------------------------------------------- controls */

document.querySelectorAll("#mainTabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#mainTabs button").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === btn.dataset.tab));
  });
});

document.querySelectorAll("#sourceTabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#sourceTabs button").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll("[data-srcpanel]").forEach((p) => (p.hidden = p.dataset.srcpanel !== btn.dataset.src));
    $("runBtn").textContent = btn.dataset.src === "dataset" ? "Run dataset labelling" : "Run classification";
    if (btn.dataset.src === "dataset") refreshDatasetSummary();
  });
});

// clicking any technique id jumps to the framework browser filtered to that id
document.addEventListener("click", (e) => {
  const tag = e.target.closest(".tag[data-tech]");
  if (!tag) return;
  document.querySelector('#mainTabs button[data-tab="framework"]').click();
  $("fwTactic").value = "";
  $("fwSearch").value = tag.dataset.tech;
  $("fwSearch").dispatchEvent(new Event("input"));
});

$("mode").addEventListener("change", refreshModeHint);
$("web_search").addEventListener("change", refreshModeHint);

function refreshModeHint() {
  const mode = $("mode").value;
  let hint = MODE_HINTS[mode];
  if (mode === "RECOGNISE" && $("web_search").checked) hint += " Web search is ignored in this mode.";
  $("modeHint").textContent = hint;
  $("searchOpts").hidden = !$("web_search").checked || mode !== "INVESTIGATE";
}

async function refreshDatasetSummary() {
  try {
    const s = await fetch("/api/dataset/summary").then((r) => r.json());
    $("datasetSummary").innerHTML = s.subset_exists
      ? `Cache holds <b>${s.labelled}</b> labelled article(s)${s.total ? ` of <b>${s.total}</b> in the subset` : ""}.<br><span class="code">${esc(s.cache_path)}</span>`
      : `<span class="warn">Subset not found — it will be built from the full dataset on the first run.</span>`;
  } catch {
    $("datasetSummary").textContent = "";
  }
}

/* ------------------------------------------------------------ file input */

const dz = $("dropzone");
dz.addEventListener("click", () => $("fileInput").click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); addFiles(e.dataTransfer.files); });
$("fileInput").addEventListener("change", (e) => addFiles(e.target.files));

async function addFiles(fileList) {
  for (const file of fileList) {
    const text = await file.text();
    // a .json file may hold a list of articles rather than one article's text
    if (file.name.endsWith(".json")) {
      try {
        const parsed = JSON.parse(text);
        const items = Array.isArray(parsed) ? parsed : [parsed];
        items.forEach((item, i) => {
          const body = typeof item === "string" ? item : item.text || item.article_text || item.content || "";
          if (body.trim()) state.uploads.push({ name: item.article_id || item.name || `${file.name}#${i + 1}`, text: body });
        });
        continue;
      } catch { /* not JSON after all - fall through and treat it as plain text */ }
    }
    if (text.trim()) state.uploads.push({ name: file.name, text });
  }
  renderFiles();
}

function renderFiles() {
  $("fileList").innerHTML = state.uploads
    .map((f, i) => `<div class="file"><b>${esc(f.name)}</b><i>${f.text.length.toLocaleString()} ch</i><button data-rm="${i}" title="Remove">×</button></div>`)
    .join("");
}

$("fileList").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-rm]");
  if (!btn) return;
  state.uploads.splice(Number(btn.dataset.rm), 1);
  renderFiles();
});

/* ------------------------------------------------------------ run / stop */

function readConfig() {
  return {
    model_name: $("model_name").value.trim(),
    local_model: $("local_model").checked,
    mode: $("mode").value,
    architecture: $("architecture").value,
    check_sub_techniques: $("check_sub_techniques").checked,
    web_search: $("web_search").checked,
    max_search_rounds: Number($("max_search_rounds").value),
    results_per_query: Number($("results_per_query").value),
  };
}

function showError(msg) {
  const box = $("formError");
  box.hidden = !msg;
  box.textContent = msg || "";
}

$("runBtn").addEventListener("click", async () => {
  showError("");
  const source = document.querySelector("#sourceTabs button.active").dataset.src;
  const config = readConfig();

  let url, body;
  if (source === "dataset") {
    url = "/api/run/dataset";
    body = { config, limit: $("limit").value || null, export: $("export").checked };
  } else {
    const articles =
      source === "paste"
        ? [{ name: "pasted-article", text: $("articleText").value }]
        : state.uploads;
    if (!articles.length || !articles.some((a) => a.text.trim())) {
      showError(source === "paste" ? "Paste some article text first." : "Add at least one file first.");
      return;
    }
    url = "/api/run/articles";
    body = { config, articles };
  }

  setRunState("running");
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    showError(err.error || "Failed to start the run");
    setRunState("idle");
  }
});

$("stopBtn").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));
$("logClear").addEventListener("click", () => ($("logbox").innerHTML = ""));

init();
