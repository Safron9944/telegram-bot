---
name: Test_Customs administration
description: A compact client directory for purchases, permissions, content, and system tools.
colors:
  bg: "#ffffff"
  surface: "#f7f9fb"
  ink: "#101828"
  muted: "#667085"
  line: "#e3e8ef"
  accent: "#10233f"
  selected: "#e8eef5"
  button-text: "#ffffff"
  warning: "#795113"
  warning-bg: "#fff5e4"
  danger: "#a33430"
  exterior: "#f4f6f8"
typography:
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    fontSize: "24px"
    fontWeight: 650
    lineHeight: 1.3
    letterSpacing: "-0.025em"
  mobile-headline:
    fontSize: "21px"
    fontWeight: 650
    lineHeight: 1.4
  title:
    fontSize: "22px"
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: "-0.02em"
  section:
    fontSize: "16px"
    fontWeight: 650
    lineHeight: 1.4
  client-name:
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.4
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    fontSize: "14px"
    lineHeight: 1.6
  client-fact:
    fontSize: "13px"
    lineHeight: 1.4
  metadata:
    fontSize: "12px"
    lineHeight: 1.5
  input:
    fontSize: "16px"
rounded:
  subtle: "4px"
  control: "5px"
  mobile-navigation: "6px"
  inspector: "8px"
spacing:
  tight: "4px"
  action: "8px"
  field: "12px"
  inspector: "16px"
  section: "18px"
  shell: "20px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.button-text}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  button-secondary:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  input:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.ink}"
    typography: "{typography.input}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  navigation-selected:
    backgroundColor: "{colors.selected}"
    textColor: "{colors.accent}"
    rounded: "{rounded.control}"
    padding: "12px"
  filter-selected:
    backgroundColor: "{colors.selected}"
    textColor: "{colors.accent}"
    rounded: "{rounded.subtle}"
    padding: "9px 10px"
  inspector:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.inspector}"
    padding: "16px"
---

# Design System: Test_Customs administration

## Overview

**Creative North Star: "Картотека клієнтів"**

Compact white/gray records, navy actions, and small corners follow the app. Green/mint colors and spacious mobile rows were rejected.

Authority: effective workspace CSS/modules and `prototype.css` navy; `core/theme.js` fixes light mode. See `PRODUCT.md` and `docs/superpowers/specs/2026-10-14-admin-mobile-clarity-design.md`.

**Key Characteristics:**

- Three spaces: «Клієнти», «Матеріали», «Система».
- Mobile bottom navigation; desktop directory and inspector.
- Purchases, manual permissions, and legacy access stay distinct.
- Telegram device/content safe areas protect controls and records.

## Colors

Frontmatter maps to `--aw-*`; accent uses `--navy` for actions, disclosures, and focus. White holds work, cool gray inspectors, blue-gray selection. Ink carries facts; muted carries metadata; lines divide rows. Warning/danger require labels. Exterior frames the app.

## Typography

Use recorded system type: headings (24/21px desktop/mobile), names (15px), facts (13px), metadata (12px). Titles/sections retain their roles. Inputs use (16px), mobile filter (14px). Amounts are tabular; long content wraps.

## Layout

Desktop: maximum (1240px), padding (20px), sticky sidebar (152px), gap (20px). Directory/inspector columns: `minmax(0,1.2fr) minmax(320px,1fr)`, gap (20px); inspector padding (16px). System form bodies reach (680px).

At (960px) and below, hide brand intro. Compact heading/exit sit above work; three destinations stay fixed below. Main padding: (12px) plus top safe inset, at least (12px) at either side, and (82px) plus bottom inset. Navigation applies side/bottom insets. Safe values combine browser/device and Telegram content insets, viewport updates, and fullscreen controls fallback.

Selected client/content details hide parent heading/view tabs, replace the directory, and remove inspector fill/border/padding. Local return starts detail. Headers use normal flow without a sticky backdrop. Landscape (540–960px), height at most (500px), combines heading/view switch and search/filter columns; safe top/side insets remain. At (360px), paired payment fields stack.

**The Safe Edges Rule.** Telegram chrome and bottom navigation must leave the first control and final record reachable.

## Elevation & Depth

Flat surfaces use fills/separators without shadows. Admin entrance/shell motion is disabled; shared route transition remains. Hover uses `brightness(.96)`; text actions underline.

## Shapes

Frames/rows are square; rounded elements use recorded roles. Icons: navigation (20px), exit/search (18px), chevron (16px). Actions/fields have minimum height (44px), destinations (48px).

## Components

Navigation: «Клієнти», «Матеріали», «Система», with `aria-current="page"`. Bottom «Клієнти» returns to the directory, preserving state/focus. Views: «Картотека»/«Оплати». Desktop filters use pressed state; mobile select restores focus through `data-aw-focus="client-filter"`.

Rows pair one information column (name/product/source/ID) with recorded amount and chevron. Mixed records name manual products; missing payment uses one «Оплата не зафіксована»; unknown amounts stay explicit. Minimum (72px), vertical padding (10px), tonal selection/surface hover.

**The Evidence Rule.** Visible access or a manual permission is never proof of payment; recorded sums never become current catalog prices.

Details separate purchase/manual/legacy evidence. Customer changes require review; content saves directly with «До каталогу» return. Drafts, locks, list context, and focus survive repaint/navigation in app state. Server success clears drafts; failure retains them. Purchase/deletion and publication protections remain intact.

Native controls retain keyboard behavior. Focus: navy (3px) outline/(3px) offset. Disabled opacity: (.65). Lists/saves use polite status; customer errors use alerts.

## Do's and Don'ts

- **Do** preserve compact records, fixed light colors, bottom navigation, safe insets, wrapping, and focus.
- **Do** preserve honest evidence, manual product names, drafts, and pending locks.
- **Don't** restore green/mint colors, spacious rows, upper mobile destinations, or dark admin mode.
- **Don't** infer payment from access/prices or make a three-column text table or large card grid.
