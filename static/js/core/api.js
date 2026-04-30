import { tg } from "./telegram.js";

function getAuthHeaders() {
  const params = new URLSearchParams(window.location.search);
  const headers = {};
  const initData = tg?.initData || params.get("initData") || "";
  if (initData) {
    headers["X-Telegram-Init-Data"] = initData;
  }

  const devUserId = params.get("dev_user_id");
  if (devUserId) {
    headers["X-Debug-User-Id"] = devUserId;
    headers["X-Debug-First-Name"] = params.get("dev_first_name") || "Dev";
    headers["X-Debug-Last-Name"] = params.get("dev_last_name") || "User";
    headers["X-Debug-Username"] = params.get("dev_username") || "dev_user";
  }
  return headers;
}

export async function api(path, options = {}) {
  const config = {
    method: options.method || "GET",
    headers: {
      ...getAuthHeaders(),
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  };

  if (options.body) {
    config.body = JSON.stringify(options.body);
  }

  const response = await fetch(path, config);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      detail?.message ||
      (typeof detail === "string" ? detail : null) ||
      payload?.message ||
      "Сталася помилка.";
    const error = new Error(message);
    error.code = detail?.code || payload?.code || "request_failed";
    throw error;
  }
  return payload;
}
