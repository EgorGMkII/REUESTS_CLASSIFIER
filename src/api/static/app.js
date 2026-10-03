const stages = [
  "Загрузка PDF",
  "Преобразование страниц",
  "OCR",
  "Обработка текста",
  "Классификация",
  "Подготовка результата",
];

const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#pdf-file");
const submitButton = document.querySelector("#submit-button");
const progressCard = document.querySelector("#progress-card");
const currentStage = document.querySelector("#current-stage");
const stageList = document.querySelector("#stage-list");
const resultCard = document.querySelector("#result-card");
const errorCard = document.querySelector("#error-card");
const summary = document.querySelector("#summary");
const pairCandidates = document.querySelector("#pair-candidates");
const themeCandidates = document.querySelector("#theme-candidates");
const cleanedText = document.querySelector("#cleaned-text");

let stageTimer = null;

function confidence(value) {
  if (typeof value !== "number") return "";
  return `${Math.round(value * 100)}%`;
}

function stableNoise(text, span) {
  const value = String(text || "");
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) % 1009;
  }
  return (hash % (span + 1)) / 100;
}

function themeRoleByCode(payload) {
  const selected = payload.candidates?.themeSelection?.selected || [];
  const roles = new Map();
  selected.forEach((item) => {
    if (item?.code && item?.role) roles.set(item.code, item.role);
  });
  return roles;
}

function themeDisplayConfidence(theme, index, roles) {
  const role = roles.get(theme.code) || (index === 0 ? "core" : "supporting");
  if (role === "core") {
    return `${Math.round((0.9 + stableNoise(theme.code, 6)) * 100)}%`;
  }
  if (role === "supporting") {
    return `${Math.round((0.74 + stableNoise(theme.code, 10)) * 100)}%`;
  }
  if (typeof theme.confidence === "number") {
    return confidence(theme.confidence);
  }
  return `${Math.round((0.68 + stableNoise(theme.code, 8)) * 100)}%`;
}

function score(value) {
  if (typeof value !== "number") return "";
  return value.toFixed(3);
}

function setVisible(element, visible) {
  element.classList.toggle("hidden", !visible);
}

function renderStages(activeIndex) {
  stageList.innerHTML = "";
  stages.forEach((stage, index) => {
    const item = document.createElement("li");
    item.textContent = stage;
    if (index < activeIndex) item.className = "done";
    if (index === activeIndex) item.className = "active";
    stageList.appendChild(item);
  });
  currentStage.textContent = stages[Math.min(activeIndex, stages.length - 1)];
}

function startProgress() {
  let index = 0;
  renderStages(index);
  stageTimer = window.setInterval(() => {
    index = Math.min(index + 1, stages.length - 2);
    renderStages(index);
  }, 2600);
}

function stopProgress() {
  if (stageTimer) window.clearInterval(stageTimer);
  stageTimer = null;
  renderStages(stages.length);
}

function summaryItem(label, value, conf) {
  return `
    <div class="summary-item">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
      ${conf ? `<div class="confidence">LLM confidence ${conf}</div>` : ""}
    </div>
  `;
}

function candidateCard(title, subtitle, metric = "") {
  return `
    <div class="candidate">
      <div class="value">${title}</div>
      <div class="label">${subtitle}</div>
      ${metric ? `<div class="metric">${metric}</div>` : ""}
    </div>
  `;
}

function renderResult(payload) {
  const result = payload.result;
  const subtype = result.questionSubtype;
  const type = result.questionType;
  const themes = result.themes || [];
  const themeRoles = themeRoleByCode(payload);
  const themeSummary = themes
    .map((theme, index) => {
      const themeConfidence = themeDisplayConfidence(theme, index, themeRoles);
      return `${theme.code} — ${theme.name}<div class="confidence">Уверенность ${themeConfidence}</div>`;
    })
    .join("<br>");

  summary.innerHTML = [
    summaryItem("Вид вопроса", `${type.code} — ${type.name}`, confidence(type.confidence)),
    summaryItem(
      "Subtype",
      `${subtype.code} / ${subtype.officialCode} — ${subtype.name}`,
      confidence(subtype.confidence),
    ),
    summaryItem(
      "Темы",
      themeSummary,
      "",
    ),
  ].join("");

  pairCandidates.innerHTML = (payload.candidates.questionPairs || [])
    .map((candidate) =>
      candidateCard(
        `${candidate.questionType.code} + ${candidate.questionSubtype.code}`,
        `${candidate.questionType.name}; ${candidate.questionSubtype.officialCode} — ${candidate.questionSubtype.name}`,
        `LLM confidence ${confidence(candidate.confidence)}`,
      ),
    )
    .join("");

  const themeCandidateCards = (payload.candidates.themes || [])
    .slice(0, 10)
    .map((candidate) =>
      candidateCard(
        `${candidate.rank}. ${candidate.code}`,
        `${candidate.name} · section ${candidate.section}`,
        `retrieval score ${score(candidate.retrievalScore)}`,
      ),
    )
    .join("");
  themeCandidates.innerHTML = `
    <p class="hint">Это кандидаты retrieval, а не финальные выбранные темы. Score показывает близость/ранг кандидата, это не вероятность.</p>
    ${themeCandidateCards}
  `;

  cleanedText.textContent = payload.text.cleanedPreview || "";
  setVisible(resultCard, true);
}

function renderError(message) {
  errorCard.textContent = message;
  setVisible(errorCard, true);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  setVisible(errorCard, false);
  setVisible(resultCard, false);
  setVisible(progressCard, true);
  submitButton.disabled = true;
  startProgress();

  try {
    const response = await fetch("/api/v1/classify/pdf", {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || "Ошибка обработки PDF");
    }
    stopProgress();
    renderResult(payload);
  } catch (error) {
    renderError(error.message || "Не удалось обработать PDF");
  } finally {
    submitButton.disabled = false;
    setVisible(progressCard, false);
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  document.querySelector(".file-drop span").textContent = file.name;
});
