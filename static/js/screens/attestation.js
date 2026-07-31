export function partRows(count, size = 50) {
  const total = Math.max(0, Number(count) || 0);
  const chunk = Math.max(1, Number(size) || 50);
  const rows = [];
  for (let start = 1, part = 1; start <= total; start += chunk, part += 1) {
    rows.push({
      part,
      start,
      end: Math.min(total, start + chunk - 1),
    });
  }
  return rows;
}


export function startOptions(section) {
  const access = section?.access || "demo";
  if (access !== "full") {
    return [
      {
        mode: "demo",
        count: Math.max(0, Number(section?.demo_count) || 0),
        locked: false,
      },
      { mode: "random", count: 50, locked: true },
    ];
  }
  return [
    {
      mode: "random",
      count: Math.min(50, Math.max(0, Number(section?.count) || 0)),
      locked: false,
    },
  ];
}


function catalog(ctx) {
  return ctx.state.bootstrap?.catalog?.attestation || {
    access: "demo",
    sections: [],
  };
}


function sectionCell(ctx, section, index) {
  const icons = ["К", "ДС", "МК", "ЗК", "∑"];
  const tints = ["blue", "teal", "indigo", "orange", "purple"];
  const demo = section.demo_count || 0;
  const accessLabel = section.locked
    ? `Демо: ${demo} · повний банк за підпискою`
    : `${section.count} питань · ${section.parts} частин`;
  return `
    <button class="cell" type="button" data-attestation-section="${ctx.escapeHtml(section.key)}">
      <span class="cell__icon cell__icon--${tints[index] || "purple"}">${icons[index] || "✓"}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(section.title)}</span>
        <span class="cell__subtitle">${ctx.escapeHtml(accessLabel)}</span>
      </span>
      <span class="cell__detail">${section.count}</span>
      <span class="cell__chevron" aria-hidden="true"></span>
    </button>
  `;
}


export function renderAttestation(ctx) {
  const data = catalog(ctx);
  const verified = data.sections.reduce(
    (sum, section) => sum + (section.key === "all" ? 0 : Number(section.count) || 0),
    0,
  );
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="muted-sm" style="font-weight:700; letter-spacing:.08em; text-transform:uppercase;">Офіційний перелік</div>
      <h1 class="page-title">Перший етап атестації</h1>
      <p class="page-subtitle">Чотири нормативні розділи. Питання з PDF проходять окрему перевірку тексту, варіантів і правильної відповіді.</p>

      <div class="stat-strip">
        ${ctx.statPill("Перевірено", String(verified))}
        ${ctx.statPill("Розділів", "4")}
        ${ctx.statPill("Доступ", data.access === "full" ? "Повний" : "Демо")}
      </div>

      ${ctx.group({
        header: "Оберіть модуль",
        children: data.sections.map((section, index) => sectionCell(ctx, section, index)).join(""),
        footer: data.access === "full"
          ? "Доступні послідовні частини по 50 і випадковий набір із 50 питань."
          : "У демо доступні перші 10 питань кожного модуля; у «Всі питання» — 40.",
      })}
    </section>
  `;

  ctx.refs.mainPanel.querySelectorAll("[data-attestation-section]").forEach((button) => {
    button.addEventListener("click", () => {
      ctx.state.selectedAttestationSection = button.dataset.attestationSection;
      ctx.navigate("attestation-parts");
    });
  });
}


export function renderAttestationParts(ctx) {
  const data = catalog(ctx);
  const selectedKey = ctx.state.selectedAttestationSection || "constitution";
  const section = data.sections.find((item) => item.key === selectedKey);
  if (!section) {
    ctx.navigate("attestation", { replace: true });
    return;
  }

  const options = startOptions({ ...section, access: data.access });
  const parts = data.access === "full" ? partRows(section.count) : [];
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="muted-sm" style="font-weight:700; letter-spacing:.08em; text-transform:uppercase;">Перший етап атестації</div>
      <h1 class="page-title">${ctx.escapeHtml(section.title)}</h1>
      <p class="page-subtitle">${section.count} перевірених питань у банку.</p>

      <div class="stat-strip">
        ${ctx.statPill("Питань", String(section.count))}
        ${ctx.statPill("Демо", String(section.demo_count))}
        ${ctx.statPill("Частин", String(section.parts))}
      </div>

      <div class="sticky-cta" id="attestation-primary-actions" style="position:static;"></div>

      ${parts.length
        ? ctx.group({
            header: "Послідовні частини",
            children: parts.map((row) => `
              <button class="cell" type="button" data-attestation-part="${row.part}">
                <span class="cell__icon cell__icon--indigo">${row.part}</span>
                <span class="cell__body">
                  <span class="cell__title">Частина ${row.part}</span>
                  <span class="cell__subtitle">Питання ${row.start}–${row.end}</span>
                </span>
                <span class="cell__detail">${row.end - row.start + 1}</span>
                <span class="cell__chevron" aria-hidden="true"></span>
              </button>
            `).join(""),
            footer: "Кожне питання показується повністю; після відповіді одразу видно правильний варіант.",
          })
        : ctx.group({
            header: "Повний банк",
            children: `
              <button class="cell" type="button" id="attestation-unlock">
                <span class="cell__icon cell__icon--purple">🔒</span>
                <span class="cell__body">
                  <span class="cell__title">Відкрити всі ${section.count} питань</span>
                  <span class="cell__subtitle">Частини по 50 та випадковий тест</span>
                </span>
                <span class="cell__chevron" aria-hidden="true"></span>
              </button>
            `,
            footer: "Тріальний доступ не відкриває повний банк атестації.",
          })}
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#attestation-primary-actions");
  const primary = options[0];
  if (primary?.mode === "demo") {
    actions.append(
      ctx.actionButton(
        `Почати демо · ${primary.count} питань`,
        () => startAttestation(ctx, section.key, "demo", 1),
        "block",
      ),
    );
  } else {
    actions.append(
      ctx.actionButton(
        `Випадкові ${primary.count} питань`,
        () => startAttestation(ctx, section.key, "random", 1),
        "block",
      ),
    );
  }

  ctx.refs.mainPanel.querySelectorAll("[data-attestation-part]").forEach((button) => {
    button.addEventListener("click", () => {
      void startAttestation(
        ctx,
        section.key,
        "part",
        Number(button.dataset.attestationPart),
      );
    });
  });
  ctx.refs.mainPanel.querySelector("#attestation-unlock")?.addEventListener(
    "click",
    () => void ctx.openPayment("full"),
  );
}


export async function startAttestation(ctx, section, mode, part = 1) {
  try {
    ctx.state.currentView = await ctx.api("/api/attestation/start", {
      method: "POST",
      body: { section, mode, part },
    });
    ctx.impact("medium");
    ctx.render();
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}
