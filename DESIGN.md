---
name: Test_Customs administration
description: Visual and interaction rules for the redesigned administrative surface.
colors:
  primary: "#10233f"
  link: "#193554"
  canvas: "#f4f6f8"
  surface: "#ffffff"
  text: "#101828"
  secondary-text: "#536273"
  tertiary-text: "#677383"
  separator: "#e3e8ef"
  positive: "#17603b"
  danger: "#a12a2a"
  focus: "#3971ad"
  access-toggle: "#2f6fed"
rounded: {button: "14px", panel: "18px", group: "24px", pill: "999px"}
spacing: {gap: "8px", gutter: "16px", section: "24px"}
---

# Design System: Test_Customs administration

## Overview

These rules cover the redesigned administrative screens of the Ukrainian Telegram Mini App. Home and learning screens retain their existing design.
Use the implemented light surfaces, compact hierarchy, and literal action labels. Source authority: `static/styles/admin.css`, `tokens.css`, `prototype.css`, and the admin navigation, settings, and screen modules; the approved scope is in `docs/superpowers/specs/2026-10-14-admin-panel-clarity-design.md`.

## Colors

Use canvas behind white panels, text for primary content, and the admin secondary/tertiary text overrides for supporting copy. Thin separators distinguish rows; navy identifies primary buttons and selected filters, with link navy for navigation.
Positive and danger colors accompany explicit text. Partial access and administrator badges use pale blue (#eaf2ff) with blue text (#24529c); checked access switches use access-toggle. Color alone never conveys access, selection, or an error.

## Typography

Keep the system stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`. Page headings are (25px, 1.25); compact profile names (21px, 1.35); section headings (17px, 1.4); row names (15px, weight 650, 1.45).
Introductions use (14px, 1.5), supporting copy (13px, 1.5–1.6), and metadata (12px). Inputs use (16px). Wrap names, section titles, statuses, and navigation subtitles; allow long unbroken names to wrap anywhere.

## Layout

Keep administration in one vertical task column with an app maximum width of (840px), (16px) content gutters, and (24px) separation between task groups. At widths of at least (700px), add (28px) screen side padding; below (360px), use (12px) gutters and stack profile summary terms and values.
The profile starts with a small initials avatar, name, Telegram ID, and access summary with “Керувати доступом”. Learning results and reference dates follow; deletion occupies a separate lower disclosure. Do not give static dates navigation chevrons.

## Elevation & Depth

Separate content with white surfaces and subtle borders. Primary buttons retain the existing soft navy shadow (`0 8px 22px rgba(16, 35, 63, 0.15)`); avoid adding decorative elevation. Admin content does not use staggered entrance animations; retain the shared screen transition and reduced-motion behavior.

## Shapes

Use button rounding for standard controls, panel rounding for profile/access blocks and inputs, and group rounding for the hub list. Full-width buttons use (16px) corners; compact statistics use (10px). Filters and status badges are pills; initials avatars are circles.

## Components

The hub has five destinations: “Користувачі”, “Доступ і тарифи”, “Матеріали”, “Повідомлення”, and “Налаштування”, each with an explanatory subtitle. Shared administrative navigation uses Telegram BackButton when available, one browser Back fallback otherwise, and “До меню адміністратора” on child screens.
User search accepts a name or Telegram ID and retains query, full-package filter, and page on return. Filters are exactly “Усі”, “З повним пакетом”, and “Без повного пакета”; they describe the full package rather than all section access.
Present the full package, access to learning sections, and visibility of additional materials as separate groups. Label switches “Відкрити доступ” or “Показувати в меню”; visibility never grants paid access. Count available paid sections only when both access and visibility are true, and explain hidden granted or shown locked sections separately.
Access changes save immediately. Keep an inline polite status region, disable access controls during the request, and retain a per-user pending lock across rerenders and navigation. Confirm from the server response; on failure restore the saved state and show the error. Restore focus and scroll after the update; “До картки користувача” is navigation.
The tariffs form edits only the full-package price in Telegram Stars; individual prices link to their section settings. The settings form edits only the support contact. Both show loading/saving state, disable controls while pending, and expose retry on load failure. Messages keep text, recipients, and sending result explicit; materials own import and content search.
Use native buttons, labelled inputs and checkboxes, `aria-pressed` filters, and polite live status/list updates. Buttons, navigation actions, filters, and icon actions have at least (44px) targets; input and labelled toggle rows have at least (48px) height. Keyboard focus uses a (3px) focus outline with (3px) offset, including the visible switch track.

## Do's and Don'ts

- Keep package state, paid access, and menu visibility distinct in copy and hierarchy; name each action by its actual result.
- Preserve wrapped text, list context, shared navigation, server-confirmed feedback, and protected administrator accounts.
- Keep one clear primary task per form; do not duplicate package-price editing or disguise navigation as saving.
- Reuse existing icons and initials; do not invent branding, image assets, financial metrics, access claims, or rules for the user-facing home and learning surfaces.
