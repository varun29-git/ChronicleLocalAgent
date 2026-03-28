const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

const state = {
  serverRuntime: null,
  browserConfig: null,
  browserCapabilities: null,
  browserSession: null,
  activeJobId: null,
  pollTimer: null,
  currentTurn: null,
  hasRenderedIntro: false,
};

const TURN_STAGES = [
  { key: "research", index: "01", title: "Extracting web info" },
  { key: "brain", index: "02", title: "Sending to Chronicle brain" },
  { key: "generate", index: "03", title: "Generating newsletter" },
];

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindComposer();
  hydrateApp().catch((error) => {
    console.error(error);
    appendSystemMessage(error.message || "Chronicle failed to initialize.");
  });
});

function cacheElements() {
  elements.statusPill = document.getElementById("status-pill");
  elements.statusCopy = document.getElementById("status-copy");
  elements.chatThread = document.getElementById("chat-thread");
  elements.composerForm = document.getElementById("composer-form");
  elements.brief = document.getElementById("brief");
  elements.explanationStyle = document.getElementById("explanation-style");
  elements.customStyleField = document.getElementById("custom-style-field");
  elements.generateButton = document.getElementById("generate-button");
}

function bindComposer() {
  elements.explanationStyle.addEventListener("change", () => {
    toggleCustomStyleField(elements.explanationStyle.value === "custom");
  });

  elements.composerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = buildPayload();
    if (!payload.brief) {
      return;
    }

    stopPolling();
    startNewTurn(payload.brief);
    setGenerateBusy(true);

    try {
      await runBrowserGeneration(payload);
    } catch (error) {
      console.error(error);
      appendLogLines([
        `Browser runtime failed: ${error.message || "Unknown error"}`,
      ]);

      if (canUseServerFallback()) {
        updateTurnStatus("Browser runtime failed. Falling back to the host runtime.");
        await startServerGeneration(payload);
      } else {
        finishTurnWithError(error.message || "Chronicle could not generate the issue on this device.");
      }
    }
  });
}

function buildPayload() {
  const formData = new FormData(elements.composerForm);
  return {
    brief: String(formData.get("brief") || "").trim(),
    depth: String(formData.get("depth") || "medium").trim(),
    explanation_style: String(formData.get("explanation_style") || "concise").trim(),
    style_instructions: String(formData.get("style_instructions") || "").trim(),
    days: Number(formData.get("days") || 7),
  };
}

async function hydrateApp() {
  toggleCustomStyleField(false);
  const statusResponse = await fetchJSON("/api/status");
  applyStatus(statusResponse);

  if (!state.hasRenderedIntro) {
    appendAssistantMessage(buildIntroMessage());
    state.hasRenderedIntro = true;
  }

  if (statusResponse.active_job) {
    startRecoveredTurn();
    setGenerateBusy(true);
    renderJobState(statusResponse.active_job);
    startPolling(statusResponse.active_job.id);
  }
}

function applyStatus(statusResponse) {
  state.serverRuntime = statusResponse.runtime;
  state.browserConfig = statusResponse.browser_ai;
  state.browserCapabilities = detectBrowserCapabilities(statusResponse.browser_ai);
  renderHeaderStatus();
}

function renderHeaderStatus() {
  const runtime = state.serverRuntime;
  const browserConfig = state.browserConfig;
  const browserCapabilities = state.browserCapabilities;
  const browserReady = Boolean(browserConfig?.local_model_ready);
  const supportsSlicing = Boolean(browserConfig?.supports_slicing);
  const primaryProfile = browserCapabilities?.candidates?.[0];
  const modelName = browserConfig?.display_name || "Gemma 3n adaptive";

  if (browserReady) {
    setStatusPill(browserCapabilities?.hasWebGPU ? "Browser ready" : "WASM ready", "");
  } else if (runtime?.dependencies_ready && runtime?.model_ready) {
    setStatusPill("Server fallback only", "is-warm");
  } else {
    setStatusPill("Runtime incomplete", "is-danger");
  }

  if (browserReady && primaryProfile) {
    const runtimeMode = browserCapabilities.hasWebGPU ? "WebGPU" : "WASM";
    const sliceText = supportsSlicing ? primaryProfile.sliceLabel : "single local bundle";
    elements.statusCopy.textContent = `${modelName}. ${runtimeMode}. ${sliceText}.`;
  } else if (runtime?.dependencies_ready && runtime?.model_ready) {
    elements.statusCopy.textContent = "Browser bundle unavailable. Host runtime is ready.";
  } else {
    elements.statusCopy.textContent = "Chronicle still needs a complete local runtime path.";
  }
}

function buildIntroMessage() {
  const browserConfig = state.browserConfig;
  const browserCapabilities = state.browserCapabilities;
  const modelName = browserConfig?.display_name || "Gemma 3n adaptive";

  if (browserConfig?.local_model_ready) {
    const firstCandidate = browserCapabilities?.candidates?.[0];
    return [
      "Chronicle is ready.",
      `Local model: ${modelName}`,
      firstCandidate ? `Device target: ${firstCandidate.label}` : "",
      "Pick a mode, choose search depth, and open the finished HTML issue when Chronicle is done.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (state.serverRuntime?.dependencies_ready && state.serverRuntime?.model_ready) {
    return "Chronicle is ready through the host runtime. Pick a mode and Chronicle will write the issue.";
  }

  return "Chronicle is online, but the local runtime still needs a complete model path before generation can succeed.";
}

function startNewTurn(userPrompt) {
  appendUserMessage(userPrompt);
  elements.brief.value = "";
  state.currentTurn = {
    statusNode: appendAssistantMessage("Extracting web info", "Stage"),
    stageNode: appendStageCard(),
    resultNode: null,
    completed: false,
  };
  setStageState("research", "active");
  setStageState("brain", "pending");
  setStageState("generate", "pending");
}

function startRecoveredTurn() {
  state.currentTurn = {
    statusNode: appendAssistantMessage("Resuming Chronicle", "Stage"),
    stageNode: appendStageCard(),
    resultNode: null,
    completed: false,
  };
  setStageState("research", "complete");
  setStageState("brain", "complete");
  setStageState("generate", "active");
}

function updateTurnStatus(text) {
  const turn = ensureCurrentTurn();
  const body = turn.statusNode.querySelector(".message-body");
  body.textContent = text;
  scrollThreadToBottom();
}

function appendLogLines(lines) {
  void lines;
}

function appendResultCard(run, markdown) {
  const turn = ensureCurrentTurn();
  if (turn.completed) {
    return;
  }

  void markdown;
  const card = ensureResultNode();
  const resultTitle = run?.title || "Newsletter ready";
  card.querySelector(".result-title").textContent = resultTitle;
  card.querySelector(".message-body").textContent = "The newsletter is ready. Open the HTML issue to review and edit it.";
  card.querySelector(".result-actions").innerHTML = `
    ${run?.html_url ? `<a class="result-link result-link--primary" href="${escapeHtml(run.html_url)}" target="_blank" rel="noreferrer">Open HTML issue</a>` : ""}
    ${run?.markdown_url ? `<a class="result-link" href="${escapeHtml(run.markdown_url)}" target="_blank" rel="noreferrer">Open markdown</a>` : ""}
  `;
  turn.completed = true;
  scrollThreadToBottom();
}

function ensureResultNode() {
  const turn = ensureCurrentTurn();
  if (turn.resultNode) {
    return turn.resultNode;
  }

  const card = document.createElement("article");
  card.className = "message message--assistant";
  card.innerHTML = `
    <div class="message-card result-card">
      <p class="message-label">Chronicle</p>
      <p class="result-title"></p>
      <div class="result-actions"></div>
      <div class="message-body"></div>
    </div>
  `;
  elements.chatThread.appendChild(card);
  turn.resultNode = card;
  scrollThreadToBottom();
  return card;
}

function updateLiveDraft(text) {
  void text;
}

function upsertReasoningSummary(text) {
  void text;
}

function finishTurnWithError(message) {
  setGenerateBusy(false);
  stopPolling();
  updateTurnStatus("Run failed.");
  setStageState("generate", "error");
  appendSystemMessage(message);
}

async function runBrowserGeneration(payload) {
  setStageState("research", "active");
  updateTurnStatus("Extracting web info");
  const researchResponse = await fetchJSON("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const research = researchResponse.research;
  const packet = buildEditorialPacket(research);
  setStageState("research", "complete");
  setStageState("brain", "active");
  updateTurnStatus("Sending to Chronicle brain");

  let markdown = "";
  let usedServerFallback = false;

  try {
    const aiSession = await withTimeout(
      ensureBrowserSession(),
      20000,
      "Browser model loading took too long.",
    );
    setStageState("brain", "complete");
    setStageState("generate", "active");
    updateTurnStatus("Generating newsletter locally");

    const generatedMarkdown = await generateNewsletterMarkdown(research, packet, aiSession, (partialText) => {
      void partialText;
    });
    const cleaned = stripMarkdownFences(generatedMarkdown).trim();
    if (cleaned.length < 200) {
      throw new Error("Browser model produced too little output.");
    }
    markdown = finalizeNewsletterMarkdown(cleaned, packet);
  } catch (browserError) {
    console.error("[Chronicle] Browser generation FAILED:", browserError?.message || browserError);
    reportClientError("browser_generation", browserError);
    setStageState("brain", "complete");
    setStageState("generate", "active");
    updateTurnStatus("Retrying with simpler prompt");

    try {
      const aiSession = state.browserSession;
      if (aiSession?.model) {
        const retryMarkdown = await generateNewsletterMarkdown(research, packet, aiSession, () => {}, true);
        const retryCleaned = stripMarkdownFences(retryMarkdown).trim();
        console.log("[Chronicle] Retry output length:", retryCleaned.length);
        if (retryCleaned.length >= 150) {
          markdown = finalizeNewsletterMarkdown(retryCleaned, packet);
        } else {
          throw new Error("Retry produced too little: " + retryCleaned.length + " chars");
        }
      } else {
        throw new Error("No browser session available for retry");
      }
    } catch (retryError) {
      console.error("[Chronicle] Retry ALSO FAILED:", retryError?.message || retryError);
      reportClientError("browser_retry", retryError);
      usedServerFallback = true;
      markdown = finalizeNewsletterMarkdown(renderFallbackNewsletter(packet), packet);
    }
  }

  const normalizedMarkdown = finalizeNewsletterMarkdown(markdown, packet);
  const title = extractTitleFromMarkdown(normalizedMarkdown, packet.title);

  updateTurnStatus("Saving newsletter");

  const saveResponse = await fetchJSON("/api/runs/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      brief: payload.brief,
      title,
      markdown: normalizedMarkdown,
      depth: payload.depth,
      explanation_style: payload.explanation_style,
      style_instructions: payload.style_instructions,
      audience: packet.audience,
      tone: packet.tone,
      queries: packet.queries,
      sections: packet.sections,
      sources: packet.selectedSources.map((source) => ({
        ...source,
        source_summary: source.sourceText,
        relevance_score: source.relevanceScore,
      })),
    }),
  });

  setGenerateBusy(false);
  setStageState("generate", "complete", usedServerFallback ? "Backup issue ready" : "Issue ready");
  updateTurnStatus(`Issue ready: ${saveResponse.run.title}`);
  appendResultCard(saveResponse.run, normalizedMarkdown);
  await refreshStatus();
}

async function waitForServerJob(jobId) {
  return new Promise((resolve, reject) => {
    state.activeJobId = jobId;
    stopPolling();
    state.pollTimer = window.setInterval(async () => {
      try {
        const response = await fetchJSON(`/api/jobs/${jobId}`);
        renderJobState(response.job);
        if (response.job.status === "completed") {
          stopPolling();
          resolve(response.job);
        } else if (response.job.status === "failed") {
          stopPolling();
          reject(new Error(response.job.error?.message || "Server generation failed."));
        }
      } catch (error) {
        stopPolling();
        reject(error);
      }
    }, 1500);
  });
}

async function startServerGeneration(payload) {
  const response = await fetchJSON("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.activeJobId = response.job.id;
  renderJobState(response.job);
  startPolling(response.job.id);
}

function renderJobState(job) {
  if (!job) {
    return;
  }

  if (job.status === "queued" || job.status === "running") {
    setGenerateBusy(true);
    setStageState("research", "complete");
    setStageState("brain", "complete");
    setStageState("generate", "active");
  }
  updateTurnStatus(job.message || "Chronicle is working…");

  if (job.status === "failed") {
    setStageState("generate", "error");
    finishTurnWithError(job.error?.message || job.message || "Chronicle failed to finish the run.");
    return;
  }

  if (job.status === "completed") {
    setGenerateBusy(false);
    setStageState("generate", "complete", "Issue ready");
    stopPolling();
    if (job.result) {
      appendResultCard(job.result, "");
    }
    refreshStatus().catch(console.error);
  }
}

function appendJobLogs(job) {
  void job;
}

function startPolling(jobId) {
  state.activeJobId = jobId;
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    try {
      const response = await fetchJSON(`/api/jobs/${jobId}`);
      renderJobState(response.job);
    } catch (error) {
      finishTurnWithError(error.message || "Chronicle lost contact with the local server.");
    }
  }, 1500);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function refreshStatus() {
  const statusResponse = await fetchJSON("/api/status");
  applyStatus(statusResponse);
}

function canUseServerFallback() {
  return Boolean(state.serverRuntime?.dependencies_ready && state.serverRuntime?.model_ready);
}

function ensureCurrentTurn() {
  if (!state.currentTurn) {
    startRecoveredTurn();
  }
  return state.currentTurn;
}

function appendUserMessage(text) {
  return appendMessage("user", text);
}

function appendAssistantMessage(text, label = "") {
  return appendMessage("assistant", text, label);
}

function appendSystemMessage(text, label = "") {
  return appendMessage("system", text, label);
}

function appendMessage(role, text, label = "") {
  const article = document.createElement("article");
  article.className = `message message--${role}`;
  article.innerHTML = `
    <div class="message-card">
      ${label ? `<p class="message-label">${escapeHtml(label)}</p>` : ""}
      <div class="message-body"></div>
    </div>
  `;
  article.querySelector(".message-body").textContent = text;
  elements.chatThread.appendChild(article);
  scrollThreadToBottom();
  return article;
}

function appendStageCard() {
  const article = document.createElement("article");
  article.className = "message message--assistant";
  article.innerHTML = `
    <div class="stage-card">
      <div class="stage-header">
        <div>
          <p class="message-label">Chronicle pipeline</p>
          <p class="stage-title">Three-stage generation flow</p>
        </div>
        <p class="stage-meta">Live</p>
      </div>
      <div class="stage-list">
        ${TURN_STAGES.map((stage) => `
          <div class="stage-item is-pending" data-stage="${stage.key}">
            <div class="stage-index">${stage.index}</div>
            <div class="stage-copy">
              <p class="stage-name">${stage.title}</p>
              <p class="stage-detail">Waiting</p>
            </div>
            <div class="stage-badge">Pending</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
  elements.chatThread.appendChild(article);
  scrollThreadToBottom();
  return article;
}

function setStageState(stageKey, status, detail = "") {
  const turn = ensureCurrentTurn();
  if (!turn.stageNode) {
    return;
  }

  const row = turn.stageNode.querySelector(`[data-stage="${stageKey}"]`);
  if (!row) {
    return;
  }

  row.classList.remove("is-pending", "is-active", "is-complete", "is-error");
  row.classList.add(`is-${status}`);

  const detailNode = row.querySelector(".stage-detail");
  const badgeNode = row.querySelector(".stage-badge");
  const presets = {
    pending: { detail: "Waiting", badge: "Pending" },
    active: { detail: "Working", badge: "Live" },
    complete: { detail: "Complete", badge: "Done" },
    error: { detail: "Blocked", badge: "Error" },
  };
  const preset = presets[status] || presets.pending;
  detailNode.textContent = detail || preset.detail;
  badgeNode.textContent = preset.badge;
  scrollThreadToBottom();
}

function setGenerateBusy(isBusy) {
  elements.generateButton.disabled = isBusy;
  elements.generateButton.textContent = isBusy ? "Working…" : "Send";
}

function toggleCustomStyleField(visible) {
  elements.customStyleField.classList.toggle("is-hidden", !visible);
}

function setStatusPill(text, modifierClass) {
  elements.statusPill.textContent = text;
  elements.statusPill.classList.remove("is-warm", "is-danger");
  if (modifierClass) {
    elements.statusPill.classList.add(modifierClass);
  }
}

function scrollThreadToBottom() {
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function buildEditorialPacket(research) {
  const selectedSources = selectTopSources(research.sources || [], research.depth === "high" ? 10 : 8);

  return {
    brief: research.brief,
    title: research.plan?.title || "Newsletter",
    audience: research.plan?.audience || "General readers",
    tone: research.plan?.tone || "Sharp and analytical",
    days: research.days,
    depth: research.depth,
    explanationStyle: research.explanation_style,
    styleInstructions: research.style_instructions || "",
    queries: research.plan?.queries || [],
    sections: research.plan?.sections || ["What happened", "Why it matters", "What to watch next"],
    marketSnapshot: research.market_snapshot || [],
    selectedSources,
  };
}

function selectTopSources(sources, limit) {
  const deduped = [];
  const seen = new Set();

  sources
    .map((source, index) => ({
      ...source,
      sourceText: trimText(extractEvidenceText(source), 320),
      relevanceScore: rankSourceForDraft(source, index),
    }))
    .sort((left, right) => right.relevanceScore - left.relevanceScore)
    .forEach((source) => {
      const key = normalizeSourceIdentity(source);
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      deduped.push(source);
    });

  return deduped.slice(0, limit);
}

function rankSourceForDraft(source, index) {
  let score = 10 - index * 0.15;
  if (source.snippet) {
    score += 2;
  }
  if (source.source_text) {
    score += Math.min(2, source.source_text.length / 220);
  }
  return score;
}

function normalizeSourceIdentity(source) {
  const titleKey = cleanupSourceTitle(source.title)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

  try {
    const parsed = new URL(source.url || "", window.location.origin);
    if (parsed.hostname.includes("news.google.com")) {
      return `title:${titleKey}`;
    }
    return `url:${parsed.hostname}${parsed.pathname}`;
  } catch (error) {
    return `title:${titleKey}`;
  }
}

function isIndirectSource(source) {
  return /news\.google\.com/i.test(source?.url || "");
}

function cleanupSourceTitle(title) {
  return String(title || "")
    .replace(/\s+-\s+(Reuters|AP News|Associated Press|MSN|AOL\.com|Yahoo!?\s*News|DW\.com)$/i, "")
    .trim();
}

function extractEvidenceText(source) {
  const title = cleanupSourceTitle(source?.title || "");
  const snippet = cleanEvidenceForPrompt(String(source?.snippet || "").trim(), title);
  const article = cleanEvidenceForPrompt(String(source?.article_text || source?.source_text || "").trim(), title);
  if (article) {
    return trimText(article, 240);
  }
  if (snippet) {
    return trimText(snippet, 180);
  }
  return title;
}

function cleanEvidenceForPrompt(text, title = "") {
  let value = String(text || "")
    .replace(/\b(title|url|evidence)\s*:\s*/gi, " ")
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!value) {
    return "";
  }

  if (title) {
    const escapedTitle = escapeRegExp(cleanupSourceTitle(title));
    value = value.replace(new RegExp(`^${escapedTitle}[\\s:;,.\\-–—]+`, "i"), "").trim();
  }

  const sentences = value.match(/[^.!?]+[.!?]?/g) || [value];
  const unique = [];
  const seen = new Set();

  sentences.forEach((sentence) => {
    const cleanedSentence = String(sentence || "").replace(/\s+/g, " ").trim();
    if (!cleanedSentence) {
      return;
    }
    const key = cleanedSentence.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    unique.push(cleanedSentence);
  });

  return unique.join(" ").trim();
}

function finalizeNewsletterMarkdown(markdown, packet) {
  let text = String(markdown || "").trim();
  if (!text) {
    text = renderFallbackNewsletter(packet);
  }
  if (!/^#\s+/m.test(text)) {
    text = `# ${packet.title}\n\n${text}`;
  }
  if (!/\n##\s+Sources\b/i.test(text)) {
    text = `${text.trim()}\n\n${buildSourcesSection(packet)}`;
  }
  return text.trim();
}

function renderFallbackNewsletter(packet) {
  const sources = packet.selectedSources;
  const lines = [
    `# ${packet.title}`,
    "",
    `Chronicle's local model could not complete the writing pass for this issue. Below is a structured summary built from the ${sources.length} source(s) collected for: ${packet.brief}.`,
    "",
  ];

  if (sources.length) {
    lines.push("## Key developments");
    lines.push("");
    sources.slice(0, 5).forEach((source, index) => {
      const title = cleanupSourceTitle(source.title);
      const evidence = extractEvidenceText(source);
      lines.push(`**${title}** [${index + 1}]`);
      lines.push("");
      lines.push(evidence);
      lines.push("");
    });
  } else {
    lines.push("## Note");
    lines.push("");
    lines.push("No web sources were collected for this brief. Try again with a different topic or check your internet connection.");
    lines.push("");
  }

  lines.push(buildSourcesSection(packet));
  return lines.join("\n");
}

function buildSourcesSection(packet) {
  const sources = packet.selectedSources.length
    ? packet.selectedSources
      .map((source, index) => `[${index + 1}]: ${cleanupSourceTitle(source.title)} - ${source.url}`)
      .join("\n")
    : "No external sources were successfully collected.";

  const marketLine = packet.marketSnapshot?.length
    ? "\n[M1]: CoinGecko Markets API - https://www.coingecko.com/"
    : "";

  return `## Sources\n${sources}${marketLine}`;
}


function detectBrowserCapabilities(browserConfig) {
  const hasWebGPU = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const deviceMemory = Number(navigator.deviceMemory || 0);
  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 0);
  const isMobile = typeof navigator !== "undefined"
    && /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  const recommendedProfile = calculateBrowserProfile({
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
    supportsSlicing: Boolean(browserConfig?.supports_slicing),
    maxSlices: Number(browserConfig?.max_slices || 1),
  });
  const candidates = buildBrowserCandidates(browserConfig, recommendedProfile, {
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
  });

  return {
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
    recommendedProfile,
    candidates,
  };
}

async function ensureBrowserSession() {
  if (state.browserSession?.model) {
    return state.browserSession;
  }

  if (!state.browserConfig?.local_model_ready || !state.browserConfig?.local_model_id) {
    throw new Error("No local browser model bundle is available on this server.");
  }

  const { AutoModelForImageTextToText, AutoProcessor, TextStreamer, env } = await import(TRANSFORMERS_CDN);
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = "/models";
  env.useBrowserCache = true;

  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  }

  let lastError = null;
  for (const candidate of state.browserCapabilities.candidates) {
    try {
      const processor = await AutoProcessor.from_pretrained(candidate.model);
      const model = await AutoModelForImageTextToText.from_pretrained(
        candidate.model,
        candidate.modelOptions,
      );
      state.browserSession = {
        processor,
        model,
        TextStreamer,
        profile: candidate,
      };
      return state.browserSession;
    } catch (error) {
      lastError = error;
      appendLogLines([`Model candidate failed: ${candidate.label}`]);
      console.warn(`Chronicle browser candidate failed: ${candidate.label}`, error);
    }
  }

  throw new Error(lastError?.message || "No browser inference backend could be initialized.");
}

async function generateNewsletterMarkdown(research, packet, aiSession, onProgress, isRetry = false) {
  const promptVariants = buildPromptVariants(research, packet, isRetry);
  let selectedVariant = null;
  let selectedInputs = null;
  let inputLength = 0;

  for (const variant of promptVariants) {
    const prepared = await preparePromptInputs(aiSession, variant.promptText);
    if (!prepared.inputLength) {
      continue;
    }
    if (prepared.inputLength > aiSession.profile.maxInputTokens) {
      console.warn(
        `[Chronicle] Prompt variant "${variant.label}" too large: ${prepared.inputLength} tokens (limit ${aiSession.profile.maxInputTokens}).`,
      );
      continue;
    }
    selectedVariant = variant;
    selectedInputs = prepared.inputs;
    inputLength = prepared.inputLength;
    break;
  }

  if (!selectedVariant || !selectedInputs || !inputLength) {
    throw new Error("Chronicle could not fit the newsletter prompt into the local model context window.");
  }

  console.log(
    `[Chronicle] Using prompt variant "${selectedVariant.label}" with ${inputLength} input tokens.`,
  );

  let streamedText = "";
  const streamer = aiSession.TextStreamer
    ? new aiSession.TextStreamer(aiSession.processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        streamedText += text;
        onProgress?.(streamedText);
      },
    })
    : null;

  const output = await withTimeout(
    aiSession.model.generate({
      ...selectedInputs,
      max_new_tokens: aiSession.profile.maxNewTokens,
      do_sample: false,
      repetition_penalty: 1.15,
      streamer,
    }),
    aiSession.profile.generationTimeoutMs,
    "Browser generation timed out before the newsletter finished.",
  );
  const generatedTokens = output.slice(null, [inputLength, null]);
  const decoded = aiSession.processor.batch_decode(generatedTokens, {
    skip_special_tokens: true,
  });
  const generatedText = Array.isArray(decoded) ? decoded[0] : decoded;

  console.log("[Chronicle] Generated text length:", (generatedText || "").length, "streamed:", streamedText.length);
  if (!generatedText && !streamedText.trim()) {
    throw new Error("Browser model returned an empty response.");
  }
  return generatedText || streamedText;
}

function buildModeSystemPrompt(packet) {
  if (packet.explanationStyle === "feynman") {
    return "You are Chronicle. Think silently, find the central argument in the research notes, and explain it simply in markdown.";
  }
  if (packet.explanationStyle === "soc") {
    return "You are Chronicle. Think silently, form one argument, and write the newsletter as a sequence of sharp questions and answers in markdown.";
  }
  if (packet.explanationStyle === "custom" && packet.styleInstructions) {
    return `You are Chronicle. Think silently, form one argument, and follow this style exactly: ${trimText(packet.styleInstructions, 160)}. Write in markdown.`;
  }
  return "You are Chronicle. Think silently, form one clear argument from the research notes, and write a concise analytical newsletter in markdown.";
}

function buildMinimalPrompt(research, packet) {
  return buildNewsletterPrompt(research, packet, {
    sourceLimit: 2,
    evidenceChars: 70,
    briefChars: 120,
    sectionLimit: 3,
    wordRange: "420-560",
  });
}

function buildNewsletterPrompt(research, packet, options = {}) {
  const maxSources = Math.max(1, options.sourceLimit || 3);
  const evidenceChars = Math.max(50, options.evidenceChars || 100);
  const briefChars = Math.max(80, options.briefChars || 180);
  const sectionLimit = Math.max(2, options.sectionLimit || 3);
  const wordRange = options.wordRange || "520-700";
  const sourceLines = packet.selectedSources
    .slice(0, maxSources)
    .map((source, index) => {
      const title = cleanupSourceTitle(source.title);
      const evidence = trimText(extractEvidenceText(source), evidenceChars);
      return `[${index + 1}] ${title}: ${evidence}`;
    })
    .join("\n");

  return `${buildModeSystemPrompt(packet)}

Topic: ${trimText(research.brief, briefChars)}

Research notes:
${sourceLines || "No sources collected."}

Structure:
${packet.sections.slice(0, sectionLimit).join(" | ")}

Task:
- First identify the single strongest through-line across the research notes.
- Then write one newsletter in ${wordRange}.
- Use original prose, not source wording.
- Keep the body free of raw URLs.
- Avoid repeated sentence patterns.
- Cite sources inline as [1], [2].
- Use markdown only: # title, opening paragraph, ## sections, and ## Sources at the end.
- Write only the newsletter markdown.`;
}

function buildPromptVariants(research, packet, isRetry) {
  const variants = [];
  if (!isRetry) {
    variants.push({
      label: "full",
      promptText: buildNewsletterPrompt(research, packet, {
        sourceLimit: 4,
        evidenceChars: 135,
        briefChars: 180,
        sectionLimit: 3,
        wordRange: "520-700",
      }),
    });
  }

  variants.push({
    label: "compact",
    promptText: buildNewsletterPrompt(research, packet, {
      sourceLimit: 3,
      evidenceChars: 95,
      briefChars: 150,
      sectionLimit: 3,
      wordRange: "480-640",
    }),
  });
  variants.push({
    label: "minimal",
    promptText: buildMinimalPrompt(research, packet),
  });

  return variants;
}

async function preparePromptInputs(aiSession, promptText) {
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: promptText,
        },
      ],
    },
  ];
  const prompt = aiSession.processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  });
  const inputs = await aiSession.processor(prompt, null, null, {
    add_special_tokens: false,
  });
  return {
    prompt,
    inputs,
    inputLength: inputs.input_ids?.dims?.at(-1) || 0,
  };
}


function calculateBrowserProfile(config) {
  const maxSlices = Math.max(1, Number(config.maxSlices || 1));
  const supportsSlicing = Boolean(config.supportsSlicing && maxSlices > 1);

  if (!supportsSlicing) {
    return {
      sliceCount: 1,
      percentage: 100,
      label: "Single bundle",
      maxNewTokens: config.hasWebGPU ? 1200 : 900,
      maxInputTokens: config.hasWebGPU ? 1400 : 900,
      reasoningMaxNewTokens: 220,
      reasoningTimeoutMs: config.hasWebGPU ? 90000 : 75000,
      generationTimeoutMs: config.hasWebGPU ? 300000 : 240000,
      temperature: 0.55,
    };
  }

  let sliceCount = 1;
  if (!config.hasWebGPU) {
    sliceCount = 1;
  } else if (config.isMobile) {
    sliceCount = config.deviceMemory >= 12 ? 2 : 1;
  } else if (config.deviceMemory >= 24 && config.hardwareConcurrency >= 16) {
    sliceCount = Math.min(maxSlices, 4);
  } else if (config.deviceMemory >= 16 || config.hardwareConcurrency >= 12) {
    sliceCount = Math.min(maxSlices, 3);
  } else if (config.deviceMemory >= 8 || config.hardwareConcurrency >= 8) {
    sliceCount = Math.min(maxSlices, 2);
  }

  const percentage = Math.round((sliceCount / maxSlices) * 100);
  let maxNewTokens = 900;
  if (percentage >= 50) {
    maxNewTokens = 1200;
  } else if (percentage >= 25) {
    maxNewTokens = 1000;
  }

  return {
    sliceCount,
    percentage,
    label: `${percentage}% slice (${sliceCount}/${maxSlices})`,
    maxNewTokens,
    maxInputTokens: percentage >= 50 ? 1400 : percentage >= 25 ? 1150 : 900,
    reasoningMaxNewTokens: 220,
    reasoningTimeoutMs: percentage >= 50 ? 100000 : 80000,
    generationTimeoutMs: percentage >= 50 ? 240000 : 210000,
    temperature: 0.65,
  };
}

function buildBrowserCandidates(browserConfig, recommendedProfile, deviceConfig) {
  if (!browserConfig?.local_model_ready || !browserConfig?.local_model_id) {
    return [];
  }

  const supportsSlicing = Boolean(browserConfig.supports_slicing && browserConfig.max_slices > 1);
  const candidates = [];
  const preferredSlices = supportsSlicing
    ? buildSliceFallbackChain(recommendedProfile.sliceCount, Number(browserConfig.max_slices || 1))
    : [1];

  if (deviceConfig.hasWebGPU) {
    preferredSlices.forEach((sliceCount) => {
      candidates.push(buildBrowserCandidate(browserConfig, "webgpu", sliceCount, recommendedProfile));
    });
  }

  const wasmSlices = supportsSlicing
    ? buildSliceFallbackChain(Math.min(recommendedProfile.sliceCount, 2), Number(browserConfig.max_slices || 1))
    : [1];
  wasmSlices.forEach((sliceCount) => {
    candidates.push(buildBrowserCandidate(browserConfig, "wasm", sliceCount, recommendedProfile));
  });

  return dedupeCandidates(candidates);
}

function buildBrowserCandidate(browserConfig, device, sliceCount, recommendedProfile) {
  const supportsSlicing = Boolean(browserConfig.supports_slicing && browserConfig.max_slices > 1);
  const normalizedSliceCount = Math.max(1, sliceCount || 1);
  const maxSlices = Math.max(1, Number(browserConfig.max_slices || 1));
  const percentage = Math.round((normalizedSliceCount / maxSlices) * 100);
  const sliceLabel = supportsSlicing
    ? `${percentage}% slice (${normalizedSliceCount}/${maxSlices})`
    : "Single bundle";
  const maxNewTokens = device === "webgpu"
    ? Math.max(800, recommendedProfile.maxNewTokens - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 40)
    : Math.max(800, recommendedProfile.maxNewTokens);
  const maxInputTokens = device === "webgpu"
    ? Math.max(850, recommendedProfile.maxInputTokens - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 120)
    : Math.min(900, recommendedProfile.maxInputTokens);
  const modelOptions = { device };

  if (browserConfig.dtype_map && Object.keys(browserConfig.dtype_map).length) {
    modelOptions.dtype = browserConfig.dtype_map;
  }

  if (supportsSlicing) {
    modelOptions.model_kwargs = { num_slices: normalizedSliceCount };
  }

  return {
    device,
    model: browserConfig.local_model_id,
    label: `${device === "webgpu" ? "WebGPU" : "WASM"} ${sliceLabel}`,
    sliceCount: normalizedSliceCount,
    sliceLabel,
    percentage,
    maxNewTokens,
    maxInputTokens,
    generationTimeoutMs: device === "webgpu"
      ? Math.max(70000, recommendedProfile.generationTimeoutMs - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 10000)
      : Math.min(70000, recommendedProfile.generationTimeoutMs),
    temperature: recommendedProfile.temperature,
    modelOptions,
  };
}

function buildSliceFallbackChain(targetSlices, maxSlices) {
  const chain = [];
  const normalizedTarget = Math.max(1, Math.min(targetSlices || 1, maxSlices || 1));
  [normalizedTarget, Math.ceil(normalizedTarget * 0.75), Math.ceil(normalizedTarget * 0.5), 2, 1]
    .forEach((value) => {
      const sliceCount = Math.max(1, Math.min(value, maxSlices));
      if (!chain.includes(sliceCount)) {
        chain.push(sliceCount);
      }
    });
  return chain;
}

function dedupeCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.device}:${candidate.model}:${candidate.sliceCount}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function reportClientError(context, error) {
  try {
    fetch("/api/client-error", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context,
        message: error?.message || String(error),
        stack: error?.stack || "",
      }),
    }).catch(() => {});
  } catch (_) {}
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function withTimeout(promise, timeoutMs, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]);
}

function stripMarkdownFences(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  return trimmed.replace(/^```[a-zA-Z0-9_-]*\s*/, "").replace(/\s*```$/, "").trim();
}

function extractTitleFromMarkdown(markdown, fallbackTitle) {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallbackTitle;
}

function trimText(text, maximumLength) {
  if (text.length <= maximumLength) {
    return text;
  }
  return `${text.slice(0, maximumLength - 1).trim()}…`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
