import test from "node:test";
import assert from "node:assert/strict";

import {
  partRows,
  startOptions,
} from "../../static/js/screens/attestation.js";
import {
  changedFields,
  reviewReasonLabels,
} from "../../static/js/screens/admin-attestation.js";


test("partRows makes 50-question ranges", () => {
  assert.deepEqual(partRows(121), [
    { part: 1, start: 1, end: 50 },
    { part: 2, start: 51, end: 100 },
    { part: 3, start: 101, end: 121 },
  ]);
});


test("demo exposes only the fixed demo action", () => {
  assert.deepEqual(
    startOptions({
      access: "demo",
      key: "constitution",
      demo_count: 10,
    }),
    [
      { mode: "demo", count: 10, locked: false },
      { mode: "random", count: 50, locked: true },
    ],
  );
});


test("review model exposes OCR and answer differences", () => {
  assert.deepEqual(
    reviewReasonLabels(["low_ocr_confidence", "page_break_not_closed"]),
    [
      "Низька впевненість OCR",
      "Незавершене перенесення між сторінками",
    ],
  );
  assert.deepEqual(
    changedFields(
      { question: "Питання", choices: ["А", "Б"] },
      { question: "Питання?", choices: ["А", "В"] },
    ),
    ["question", "choices"],
  );
});
