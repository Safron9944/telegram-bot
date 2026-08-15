export const tg = window.Telegram?.WebApp || null;

let telegramBackAttached = false;
let closingConfirmationEnabled = false;
let fullscreenEventsAttached = false;

function syncFullscreenState() {
  document.documentElement.classList.toggle("telegram-fullscreen", Boolean(tg?.isFullscreen));
}

export function initializeTelegram(onBack) {
  if (!tg) {
    return;
  }

  tg.ready();
  tg.expand();

  if (!fullscreenEventsAttached) {
    tg.onEvent?.("fullscreenChanged", syncFullscreenState);
    fullscreenEventsAttached = true;
  }
  syncFullscreenState();

  if (tg.isVersionAtLeast?.("8.0") && typeof tg.requestFullscreen === "function") {
    tg.requestFullscreen();
  }

  if (!telegramBackAttached) {
    tg.BackButton?.onClick?.(() => {
      onBack();
    });
    telegramBackAttached = true;
  }
}

export function impact() {}

export function setTelegramBackButton(showBack) {
  if (showBack) {
    tg?.BackButton?.show?.();
    return;
  }
  tg?.BackButton?.hide?.();
}

export function syncClosingConfirmation(view) {
  const shouldProtect = Boolean(
    view && (view.mode === "pretest" || view.screen === "question" || view.screen === "feedback" || view.screen === "open-practice"),
  );

  if (shouldProtect && !closingConfirmationEnabled) {
    tg?.enableClosingConfirmation?.();
    closingConfirmationEnabled = true;
    return;
  }

  if (!shouldProtect && closingConfirmationEnabled) {
    tg?.disableClosingConfirmation?.();
    closingConfirmationEnabled = false;
  }
}
