const $ = (id) => document.getElementById(id);
const promptInput = $("prompt");
const forgeButton = $("go");
const log = $("log");
const view = $("view");
const steps = [...$("steps").children];
const HISTORY_KEY = "forge_projects_v1";
const THEME_KEY = "forge_theme_v1";
let projects = loadProjects();
let current = null;
let activeTab = "app";
let requestId = 0;

const MODEL_ID = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
let localGeneratorPromise = null;
let progressLine = null;

function updateModelProgress(info) {
  if (info.status !== "progress" || !/model.*onnx/i.test(info.file || "")) return;
  const raw = Number(info.progress);
  if (!Number.isFinite(raw)) return;
  const percent = Math.max(0, Math.min(100, Math.round(raw <= 1 ? raw * 100 : raw)));
  if (!progressLine) {
    progressLine = document.createElement("div");
    log.append(progressLine);
  }
  progressLine.textContent = "Загрузка весов модели: " + percent + "%";
  log.scrollTop = log.scrollHeight;
}

let forceWasm = false;
let TextStreamerClass = null;
let activeDevice = "";

async function detectWebGPU() {
  if (forceWasm || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    return { f16: adapter.features.has("shader-f16") };
  } catch {
    return null;
  }
}

async function getLocalGenerator() {
  if (!localGeneratorPromise) {
    localGeneratorPromise = (async () => {
      appendLog("Загружаю локальную модель Qwen Coder (около 550 МБ, только при первом запуске).");
      const tjs = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0");
      const { pipeline, env } = tjs;
      TextStreamerClass = tjs.TextStreamer;
      env.useBrowserCache = true;
      try {
        if (self.crossOriginIsolated) {
          env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8));
          appendLog("Многопоточный режим CPU: " + env.backends.onnx.wasm.numThreads + " потоков.");
        }
      } catch {}
      const gpu = await detectWebGPU();
      const configs = [];
      if (gpu) configs.push({ device: "webgpu", dtype: gpu.f16 ? "q4f16" : "q4" });
      configs.push({ device: "wasm", dtype: "q4" });
      let lastError;
      for (const cfg of configs) {
        try {
          if (cfg.device === "wasm" && configs.length > 1) appendLog("Запускаю совместимый режим CPU.");
          const gen = await pipeline("text-generation", MODEL_ID, {
            dtype: cfg.dtype,
            device: cfg.device,
            progress_callback: updateModelProgress
          });
          activeDevice = cfg.device;
          appendLog("Режим: " + (cfg.device === "webgpu" ? "видеокарта (WebGPU)" : "процессор (CPU, медленнее)") + ".");
          return gen;
        } catch (error) {
          lastError = error;
          if (cfg.device === "webgpu") appendLog("Не удалось запустить WebGPU; пробую CPU.");
        }
      }
      throw lastError || new Error("Не удалось загрузить локальную модель.");
    })();
  }
  try {
    return await localGeneratorPromise;
  } catch (error) {
    localGeneratorPromise = null;
    throw error;
  }
}

function generatedReply(result) {
  const output = result?.[0]?.generated_text;
  if (Array.isArray(output)) {
    const assistant = output.filter((message) => message.role === "assistant").at(-1);
    return assistant?.content || "";
  }
  return typeof output === "string" ? output : "";
}
function loadProjects() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.filter((item) => item && item.html && item.prompt) : [];
  } catch {
    return [];
  }
}

function saveProjects() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(projects.slice(0, 12)));
  } catch {
    appendLog("Не удалось сохранить историю в этом браузере.");
  }
}

function appendLog(message, state = "") {
  const line = document.createElement("div");
  line.textContent = message;
  if (state) line.className = state;
  log.append(line);
  log.scrollTop = log.scrollHeight;
}

function setStage(active) {
  steps.forEach((step, index) => {
    step.className = index < active ? "done" : index === active ? "run" : "";
  });
}

function safeTitle(value, fallback) {
  const title = typeof value === "string" ? value.trim().replace(/[<>]/g, "") : "";
  return (title || fallback || "Новое приложение").slice(0, 60);
}

function validateHtml(value) {
  if (typeof value !== "string") throw new Error("Модель не вернула HTML-код.");
  let html = value.trim().replace(/^\x60\x60\x60(?:html)?\s*/i, "").replace(/\s*\x60\x60\x60$/, "");
  const start = html.search(/<!doctype\s+html|<html[\s>]/i);
  if (start > 0) html = html.slice(start);
  if (!/<html[\s>]/i.test(html) || !/<\/html\s*>/i.test(html)) {
    throw new Error("Ответ модели не похож на готовую HTML-страницу. Попробуйте уточнить запрос.");
  }
  return html;
}

function renderPreview() {
  view.replaceChildren();
  if (activeTab === "code") {
    const pre = document.createElement("pre");
    pre.className = "code";
    pre.textContent = current.html;
    view.append(pre);
    return;
  }
  const iframe = document.createElement("iframe");
  iframe.className = "preview";
  iframe.title = current.title;
  iframe.setAttribute("sandbox", "allow-scripts allow-forms");
  iframe.srcdoc = current.html;
  view.append(iframe);
}

function updateCurrent() {
  if (!current) return;
  $("url").textContent = "forge://" + current.title.toLowerCase().replace(/[^a-zа-яё0-9]+/gi, "-").replace(/^-|-$/g, "").slice(0, 32);
  $("meta").textContent = current.summary || ("Готово · " + current.html.split("\n").length + " строк · работает в браузере");
  $("copy").disabled = false;
  $("dl").disabled = false;
  $("t1").setAttribute("aria-selected", activeTab === "app");
  $("t2").setAttribute("aria-selected", activeTab === "code");
  renderPreview();
}

function drawHistory() {
  const list = $("hist");
  list.replaceChildren();
  projects.forEach((project) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const name = document.createElement("b");
    const date = document.createElement("span");
    name.textContent = project.title;
    name.style.fontWeight = "500";
    date.textContent = project.createdAt ? new Date(project.createdAt).toLocaleDateString("ru-RU") : "Проект";
    button.append(name, date);
    if (current && current.id === project.id) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => {
      current = project;
      promptInput.value = project.prompt;
      updateCurrent();
      drawHistory();
    });
    item.append(button);
    list.append(item);
  });
}

function setTab(tab) {
  activeTab = tab;
  $("t1").setAttribute("aria-selected", tab === "app");
  $("t2").setAttribute("aria-selected", tab === "code");
  if (current) renderPreview();
}

function resetView() {
  requestId++;
  current = null;
  activeTab = "app";
  promptInput.value = "";
  forgeButton.disabled = false;
  forgeButton.textContent = "Выковать приложение";
  steps.forEach((step) => step.className = "");
  log.textContent = "Горн остыл. Ждём запрос.";
  $("url").textContent = "forge://новый-проект";
  $("meta").textContent = "Приложение ещё не создано";
  $("copy").disabled = true;
  $("dl").disabled = true;
  $("t1").setAttribute("aria-selected", "true");
  $("t2").setAttribute("aria-selected", "false");
  view.innerHTML = '<div class="empty"><div class="anvil" aria-hidden="true">⚒</div><p>Здесь появится готовое приложение. Опишите его слева.</p></div>';
  drawHistory();
  promptInput.focus();
}

async function forge() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    promptInput.focus();
    log.textContent = "Опишите приложение, чтобы начать.";
    return;
  }

  const thisRequest = ++requestId;
  forgeButton.disabled = true;
  forgeButton.textContent = "Куём локально…";
  steps.forEach((step) => step.className = "");
  log.textContent = "";
  progressLine = null;
  setStage(0);
  appendLog("Подготавливаю запрос. Генерация будет выполняться на этом устройстве.");
  setStage(1);

  try {
    const messages = [
      {
        role: "system",
        content: "You are a skilled frontend developer. Build the user's requested app as a complete, functional, self-contained HTML document in Russian. Return only the HTML document, with inline CSS and JavaScript. No Markdown fences, explanations, external libraries, CDNs, network calls, or placeholder features. Implement the requested interactions, responsive layout, form validation, and browser-local persistence when appropriate. Keep the code concise enough to fit in the response."
      },
      {
        role: "user",
        content: "Create this application: " + prompt + "\nReturn only a complete HTML document."
      }
    ];
    const options = { max_new_tokens: 2048, do_sample: false, repetition_penalty: 1.05 };
    let generator = await getLocalGenerator();
    const withProgress = (gen) => {
      const opts = { ...options };
      try {
        if (TextStreamerClass && gen.tokenizer) {
          let count = 0;
          let line = null;
          opts.streamer = new TextStreamerClass(gen.tokenizer, {
            skip_prompt: true,
            callback_function: () => {
              count++;
              if (!line) { line = document.createElement("div"); log.append(line); }
              if (count % 5 === 0) {
                line.textContent = "Генерация: ~" + count + " фрагментов (лимит 2048 токенов)";
                log.scrollTop = log.scrollHeight;
              }
            }
          });
        }
      } catch {}
      return opts;
    };
    if (thisRequest !== requestId) return;
    setStage(2);
    appendLog("Модель загружена. Генерирую приложение на устройстве.");
    let result;
    try {
      result = await generator(messages, withProgress(generator));
    } catch (error) {
      if (activeDevice !== "webgpu") throw error;
      appendLog("Ошибка на видеокарте (" + (error?.message || error).toString().slice(0, 120) + "). Переключаюсь на CPU.");
      forceWasm = true;
      localGeneratorPromise = null;
      progressLine = null;
      generator = await getLocalGenerator();
      if (thisRequest !== requestId) return;
      result = await generator(messages, withProgress(generator));
    }
    if (thisRequest !== requestId) return;

    setStage(3);
    appendLog("Проверяю результат и запускаю изолированный предпросмотр.");
    const html = validateHtml(generatedReply(result));
    const generatedDocument = new DOMParser().parseFromString(html, "text/html");
    const project = {
      id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
      prompt,
      title: safeTitle(generatedDocument.title, prompt),
      summary: "Создано локальной моделью Qwen2.5-Coder-0.5B-Instruct",
      html,
      createdAt: new Date().toISOString()
    };
    current = project;
    projects = [project, ...projects.filter((item) => item.id !== project.id)].slice(0, 12);
    saveProjects();
    activeTab = "app";
    setStage(4);
    updateCurrent();
    drawHistory();
    forgeButton.textContent = "Выковать ещё раз";
    appendLog("Приложение создано локально и готово к работе.", "ok");
  } catch (error) {
    if (thisRequest !== requestId) return;
    steps.forEach((step) => step.className = "");
    appendLog(error?.message || "Не удалось запустить локальную модель.", "error");
    appendLog("Для первой загрузки нужна сеть и около 550 МБ свободного места; после загрузки веса кэшируются браузером.");
    forgeButton.textContent = "Попробовать снова";
  } finally {
    if (thisRequest === requestId) forgeButton.disabled = false;
  }
}

$("t1").addEventListener("click", () => setTab("app"));
$("t2").addEventListener("click", () => setTab("code"));
forgeButton.addEventListener("click", forge);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) forge();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    promptInput.value = chip.dataset.p;
    promptInput.focus();
  });
});
$("new").addEventListener("click", resetView);
$("copy").addEventListener("click", async () => {
  if (!current) return;
  try {
    await navigator.clipboard.writeText(current.html);
    $("copy").textContent = "Скопировано";
  } catch {
    setTab("code");
    $("copy").textContent = "Выделите код вручную";
  }
  setTimeout(() => $("copy").textContent = "Скопировать код", 1600);
});
$("dl").addEventListener("click", () => {
  if (!current) return;
  const blob = new Blob([current.html], { type: "text/html;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = current.title.toLowerCase().replace(/[^a-zа-яё0-9]+/gi, "-").replace(/^-|-$/g, "") + ".html";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "dark" || (!root.dataset.theme && !matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch {}
});
try {
  const theme = localStorage.getItem(THEME_KEY);
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch {}
drawHistory();
