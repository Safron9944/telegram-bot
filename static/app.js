import "/static/js/app.js?v=20260810-admin-access-status-01";

const confirmedFinishButtons = new WeakSet();
const pendingFinishButtons = new WeakSet();

function showFinishConfirmation() {
  const message = "Завершити тестування? Поточне тестування буде завершено.";
  const webApp = window.Telegram?.WebApp;

  if (webApp?.showConfirm) {
    return new Promise((resolve) => {
      try {
        webApp.showConfirm(message, (confirmed) => resolve(Boolean(confirmed)));
      } catch (_) {
        resolve(window.confirm(message));
      }
    });
  }

  return Promise.resolve(window.confirm(message));
}

document.addEventListener(
  "click",
  async (event) => {
    const button = event.target.closest?.("#question-actions button, #feedback-actions button");
    if (!button || button.textContent.trim() !== "Завершити") return;

    if (confirmedFinishButtons.has(button)) {
      confirmedFinishButtons.delete(button);
      return;
    }

    event.preventDefault();
    event.stopImmediatePropagation();

    if (pendingFinishButtons.has(button)) return;
    pendingFinishButtons.add(button);

    try {
      const confirmed = await showFinishConfirmation();
      if (!confirmed || !button.isConnected) return;

      confirmedFinishButtons.add(button);
      button.click();
    } finally {
      pendingFinishButtons.delete(button);
    }
  },
  true,
);
