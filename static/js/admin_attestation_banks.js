const PAGE_SIZE = 25;

function parseList(value) {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }
  return [];
}

function bankBase(ctx) {
  const bankId = Number(ctx.state.selectedAttestationAdminBankId || 0);
  return bankId ? `/api/admin/attestation-banks/${bankId}` : "";
}

export function renderAdminAttestationBank(ctx) {
  ctx.setChrome({ showBack: true });
  const bank = ctx.state.selectedAttestationAdminBank;
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(bank?.title || "Керування розділом")}</h1>
      <p class="page-subtitle">Назва, пошук, додавання та редагування питань.</p>
      <div id="attestation-bank-manager">
        <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
      </div>
    </section>
  `;
}

export async function loadAdminAttestationBank(ctx, offset = ctx.state.attestationAdminOffset || 0) {
  if (ctx.state.currentScreen !== "admin-attestation-bank") return;
  const base = bankBase(ctx);
  const root = document.querySelector("#attestation-bank-manager");
  if (!base || !root) {
    ctx.goBack();
    return;
  }
  const topic = ctx.state.attestationAdminTopic || "";
  const query = ctx.state.attestationAdminQuery || "";
  try {
    const payload = await ctx.api(
      `${base}/questions?topic=${encodeURIComponent(topic)}&q=${encodeURIComponent(query)}&offset=${Math.max(0, offset)}&limit=${PAGE_SIZE}`,
    );
    if (ctx.state.currentScreen !== "admin-attestation-bank") return;
    ctx.state.selectedAttestationAdminBank = payload.bank;
    ctx.state.attestationAdminOffset = payload.offset;
    const topics = payload.topics || [];
    root.closest(".screen-content")?.querySelector(".page-title")?.replaceChildren(
      document.createTextNode(payload.bank.title),
    );
    root.innerHTML = `
      <form class="attestation-bank-title-form" id="attestation-bank-title-form">
        <label class="field">
          <span class="field__label">Назва розділу</span>
          <input class="input" id="attestation-bank-title" value="${ctx.escapeHtml(payload.bank.title)}" maxlength="160">
        </label>
        <button class="btn btn--primary" type="submit">Зберегти назву</button>
      </form>

      <div class="attestation-bank-toolbar">
        <button class="btn btn--primary btn--block" id="attestation-question-add" type="button">Додати питання</button>
        <form class="attestation-bank-filters" id="attestation-bank-filters">
          <label class="field">
            <span class="field__label">Підрозділ</span>
            <select class="input" id="attestation-bank-topic">
              <option value="">Усі підрозділи</option>
              ${topics.map((item) => `<option value="${ctx.escapeHtml(item.topic)}" ${item.topic === topic ? "selected" : ""}>${ctx.escapeHtml(item.topic)} (${item.questions_count})</option>`).join("")}
            </select>
          </label>
          <label class="field">
            <span class="field__label">Пошук</span>
            <input class="input" id="attestation-bank-query" value="${ctx.escapeHtml(query)}" placeholder="Номер або текст питання">
          </label>
          <button class="btn" type="submit">Знайти</button>
        </form>
      </div>

      <div class="group">
        <div class="group__label">Питання · ${payload.total}</div>
        <div class="group__list" id="attestation-managed-question-list"></div>
      </div>
      <div class="apk-pagination" id="attestation-managed-pagination">
        <button class="btn btn--sm" type="button" id="attestation-managed-prev" ${payload.has_prev ? "" : "disabled"}>Назад</button>
        <span>${payload.total ? payload.offset + 1 : 0}–${Math.min(payload.offset + payload.limit, payload.total)} із ${payload.total}</span>
        <button class="btn btn--sm" type="button" id="attestation-managed-next" ${payload.has_next ? "" : "disabled"}>Далі</button>
      </div>
    `;

    const list = root.querySelector("#attestation-managed-question-list");
    if (!payload.items.length) {
      list.innerHTML = '<div class="empty empty--inline"><h2>Питань не знайдено</h2><p>Змініть фільтр або додайте нове питання.</p></div>';
    } else {
      payload.items.forEach((item) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "cell";
        row.innerHTML = `
          <span class="cell__icon cell__icon--purple">${ctx.escapeHtml(item.qnum ?? "—")}</span>
          <span class="cell__body">
            <span class="cell__title">${ctx.escapeHtml(item.question)}</span>
            <span class="cell__subtitle">${ctx.escapeHtml(item.topic)}${item.managed_manually ? " · змінено вручну" : ""}</span>
          </span>
          <span class="cell__chevron" aria-hidden="true"></span>
        `;
        row.addEventListener("click", () => {
          ctx.state.selectedAttestationAdminQuestionId = Number(item.id);
          ctx.navigate("admin-attestation-question");
        });
        list.append(row);
      });
    }

    root.querySelector("#attestation-bank-title-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const title = root.querySelector("#attestation-bank-title").value.trim();
      if (!title) return ctx.setMessage("error", "Вкажіть назву розділу.");
      try {
        const saved = await ctx.api(base, { method: "PATCH", body: { title } });
        ctx.state.selectedAttestationAdminBank = saved;
        ctx.setMessage("success", "Назву розділу збережено.");
        await loadAdminAttestationBank(ctx, payload.offset);
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
    root.querySelector("#attestation-question-add").addEventListener("click", () => {
      ctx.state.selectedAttestationAdminQuestionId = null;
      ctx.navigate("admin-attestation-question");
    });
    root.querySelector("#attestation-bank-filters").addEventListener("submit", (event) => {
      event.preventDefault();
      ctx.state.attestationAdminTopic = root.querySelector("#attestation-bank-topic").value;
      ctx.state.attestationAdminQuery = root.querySelector("#attestation-bank-query").value.trim();
      ctx.state.attestationAdminOffset = 0;
      void loadAdminAttestationBank(ctx, 0);
    });
    root.querySelector("#attestation-bank-topic").addEventListener("change", (event) => {
      ctx.state.attestationAdminTopic = event.currentTarget.value;
      ctx.state.attestationAdminOffset = 0;
      void loadAdminAttestationBank(ctx, 0);
    });
    root.querySelector("#attestation-managed-prev").addEventListener("click", () => {
      void loadAdminAttestationBank(ctx, Math.max(0, payload.offset - payload.limit));
    });
    root.querySelector("#attestation-managed-next").addEventListener("click", () => {
      void loadAdminAttestationBank(ctx, payload.offset + payload.limit);
    });
  } catch (error) {
    root.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

export function renderAdminAttestationQuestion(ctx) {
  ctx.setChrome({ showBack: true });
  const isNew = !ctx.state.selectedAttestationAdminQuestionId;
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${isNew ? "Нове питання" : "Редагування питання"}</h1>
      <p class="page-subtitle">Усі зміни одразу зберігаються в базі.</p>
      <div id="attestation-managed-question-editor">
        <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
      </div>
    </section>
  `;
}

function appendChoiceRow(ctx, container, choice = "", checked = false) {
  const row = document.createElement("div");
  row.className = "attestation-managed-choice";
  row.innerHTML = `
    <div class="attestation-managed-choice__header">
      <strong class="attestation-managed-choice__number"></strong>
      <label class="attestation-correct-toggle">
        <span>Правильна</span>
        <span class="switch"><input class="attestation-managed-correct" type="checkbox" ${checked ? "checked" : ""}><span class="switch__track"></span></span>
      </label>
    </div>
    <textarea class="textarea attestation-managed-choice-text" placeholder="Варіант відповіді">${ctx.escapeHtml(choice)}</textarea>
    <button class="btn btn--sm btn--destructive attestation-managed-choice-remove" type="button">Видалити варіант</button>
  `;
  row.querySelector(".attestation-managed-choice-remove").addEventListener("click", () => {
    if (container.children.length <= 2) {
      ctx.setMessage("error", "Потрібно залишити щонайменше два варіанти.");
      return;
    }
    row.remove();
    renumberChoices(container);
  });
  container.append(row);
  renumberChoices(container);
}

function renumberChoices(container) {
  [...container.children].forEach((row, index) => {
    row.querySelector(".attestation-managed-choice__number").textContent = `Варіант ${index + 1}`;
  });
}

function renderQuestionForm(ctx, question) {
  const root = document.querySelector("#attestation-managed-question-editor");
  if (!root) return;
  const isNew = !question?.id;
  const bank = ctx.state.selectedAttestationAdminBank;
  const defaultTopic = ctx.state.attestationAdminTopic || "Основний розділ";
  const choices = parseList(question?.choices);
  const correct = new Set(parseList(question?.correct).map(Number));
  root.innerHTML = `
    <form class="attestation-managed-question-form" id="attestation-managed-question-form">
      <div class="attestation-managed-meta">
        <label class="field">
          <span class="field__label">Підрозділ</span>
          <input class="input" id="attestation-managed-topic" value="${ctx.escapeHtml(question?.topic || defaultTopic)}" maxlength="240">
        </label>
        <label class="field">
          <span class="field__label">Номер питання</span>
          <input class="input" id="attestation-managed-qnum" type="number" min="1" value="${ctx.escapeHtml(question?.qnum ?? "")}" placeholder="Автоматично">
        </label>
      </div>
      <label class="field">
        <span class="field__label">Текст питання</span>
        <textarea class="textarea attestation-managed-question-text" id="attestation-managed-question-text">${ctx.escapeHtml(question?.question || "")}</textarea>
      </label>
      <label class="attestation-managed-shuffle">
        <span><strong>Перемішувати варіанти</strong><small>Вимкніть, якщо порядок відповідей важливий.</small></span>
        <span class="switch"><input id="attestation-managed-shuffle" type="checkbox" ${question?.shuffle_choices === false ? "" : "checked"}><span class="switch__track"></span></span>
      </label>
      <div class="attestation-managed-choices" id="attestation-managed-choices"></div>
      <button class="btn" id="attestation-managed-choice-add" type="button">Додати варіант</button>
      <button class="btn btn--primary btn--block" type="submit">${isNew ? "Створити питання" : "Зберегти зміни"}</button>
      ${isNew ? "" : '<button class="btn btn--destructive btn--block" id="attestation-managed-question-delete" type="button">Видалити питання</button>'}
    </form>
  `;
  const container = root.querySelector("#attestation-managed-choices");
  const initialChoices = choices.length >= 2 ? choices : ["", "", "", ""];
  initialChoices.forEach((choice, index) => appendChoiceRow(ctx, container, choice, correct.has(index + 1)));
  root.querySelector("#attestation-managed-choice-add").addEventListener("click", () => {
    if (container.children.length >= 12) return ctx.setMessage("error", "Максимум 12 варіантів.");
    appendChoiceRow(ctx, container);
  });
  root.querySelector("#attestation-managed-question-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const rows = [...container.querySelectorAll(".attestation-managed-choice")];
    const answerChoices = rows.map((row) => row.querySelector(".attestation-managed-choice-text").value.trim());
    const answerCorrect = rows.flatMap((row, index) => row.querySelector(".attestation-managed-correct").checked ? [index + 1] : []);
    const topic = root.querySelector("#attestation-managed-topic").value.trim();
    const text = root.querySelector("#attestation-managed-question-text").value.trim();
    const qnumRaw = root.querySelector("#attestation-managed-qnum").value.trim();
    if (!topic || !text || answerChoices.some((item) => !item) || !answerCorrect.length) {
      ctx.setMessage("error", "Заповніть питання, усі варіанти та позначте правильну відповідь.");
      return;
    }
    const body = {
      topic,
      qnum: qnumRaw ? Number(qnumRaw) : null,
      question: text,
      choices: answerChoices,
      correct: answerCorrect,
      shuffle_choices: root.querySelector("#attestation-managed-shuffle").checked,
    };
    const base = bankBase(ctx);
    const url = isNew ? `${base}/questions` : `${base}/questions/${question.id}`;
    try {
      await ctx.api(url, { method: isNew ? "POST" : "PATCH", body });
      await ctx.goBack();
      ctx.setMessage("success", isNew ? "Питання створено." : "Питання збережено.");
    } catch (error) {
      ctx.setMessage("error", error.message);
    }
  });
  root.querySelector("#attestation-managed-question-delete")?.addEventListener("click", async () => {
    if (!window.confirm("Видалити це питання?")) return;
    try {
      await ctx.api(`${bankBase(ctx)}/questions/${question.id}`, { method: "DELETE" });
      await ctx.goBack();
      ctx.setMessage("success", "Питання видалено.");
    } catch (error) {
      ctx.setMessage("error", error.message);
    }
  });
  if (bank?.title) root.closest(".screen-content")?.querySelector(".page-subtitle")?.append(` · ${bank.title}`);
}

export async function loadAdminAttestationQuestion(ctx) {
  if (ctx.state.currentScreen !== "admin-attestation-question") return;
  const questionId = Number(ctx.state.selectedAttestationAdminQuestionId || 0);
  if (!bankBase(ctx)) {
    ctx.goBack();
    return;
  }
  if (!questionId) {
    renderQuestionForm(ctx, null);
    return;
  }
  try {
    const payload = await ctx.api(`${bankBase(ctx)}/questions/${questionId}`);
    if (ctx.state.currentScreen === "admin-attestation-question") {
      renderQuestionForm(ctx, payload.question);
    }
  } catch (error) {
    const root = document.querySelector("#attestation-managed-question-editor");
    if (root) root.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}
