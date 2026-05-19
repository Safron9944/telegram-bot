import { tg } from "./telegram.js?v=20260519-minimal-16";

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
  const isFormData = options.body instanceof FormData;
  const controller = options.timeoutMs ? new AbortController() : null;
  const timeoutId = controller
    ? window.setTimeout(() => controller.abort(), options.timeoutMs)
    : null;
  const config = {
    method: options.method || "GET",
    headers: {
      ...getAuthHeaders(),
      ...(options.body && !isFormData ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
    ...(controller ? { signal: controller.signal } : {}),
  };

  if (options.body) {
    config.body = isFormData ? options.body : JSON.stringify(options.body);
  }

  let response;
  try {
    response = await fetch(path, config);
  } catch (error) {
    if (error?.name === "AbortError") {
      const timeoutError = new Error("Сервер не відповів вчасно. Спробуйте відкрити Mini App ще раз.");
      timeoutError.code = "request_timeout";
      throw timeoutError;
    }
    throw error;
  } finally {
    if (timeoutId) window.clearTimeout(timeoutId);
  }

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
