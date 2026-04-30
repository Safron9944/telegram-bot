export const tg = window.Telegram?.WebApp || null;

let telegramBackAttached = false;
let closingConfirmationEnabled = false;

export function initializeTelegram(onBack) {
  if (!tg) {
    return;
  }

  tg.ready();
  tg.expand();

  if (!telegramBackAttached) {
    tg.BackButton?.onClick?.(() => {
      onBack();
    });
    telegramBackAttached = true;
  }
}

export function impact(style = "light") {
  tg?.HapticFeedback?.impactOccurred?.(style);
}

export function setTelegramBackButton(showBack) {
  if (showBack) {
    tg?.BackButton?.show?.();
    return;
  }
  tg?.BackButton?.hide?.();
}

export function syncClosingConfirmation(view) {
  const shouldProtect = Boolean(
    view && (view.mode === "pretest" || view.screen === "question" || view.screen === "feedback"),
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
