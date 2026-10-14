---
name: Test_Customs administration
description: A client file workspace for purchases, permissions, content, and system tools.
colors:
  bg: "#ffffff"
  surface: "#f3f6f5"
  ink: "#172b2a"
  muted: "#52676b"
  line: "#d8e2df"
  accent: "#215e53"
  selected: "#e7f0ec"
  button-text: "#ffffff"
  warning: "#795113"
  warning-bg: "#fff5e4"
  danger: "#a33430"
  exterior: "#edf2f0"
  dark-bg: "#152320"
  dark-surface: "#20302b"
  dark-ink: "#e3eee7"
  dark-muted: "#b5c9c0"
  dark-line: "#42594e"
  dark-accent: "#acd8c7"
  dark-selected: "#284a3d"
  dark-button-text: "#152a21"
  dark-warning: "#efcd91"
  dark-warning-bg: "#3c3220"
  dark-danger: "#f3aba5"
  dark-exterior: "#101c17"
typography:
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    fontSize: "28px"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  title:
    fontSize: "22px"
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: "-0.02em"
  section:
    fontSize: "16px"
    fontWeight: 650
    lineHeight: 1.4
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    fontSize: "14px"
    lineHeight: 1.6
  supporting:
    fontSize: "13px"
    lineHeight: 1.6
  metadata:
    fontSize: "12px"
    lineHeight: 1.6
  input:
    fontSize: "16px"
rounded:
  subtle: "4px"
  control: "5px"
  inspector: "6px"
spacing:
  tight: "4px"
  action: "8px"
  field: "12px"
  gutter: "16px"
  inspector: "20px"
  section: "24px"
  shell: "28px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.button-text}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  button-primary-dark:
    backgroundColor: "{colors.dark-accent}"
    textColor: "{colors.dark-button-text}"
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
    padding: "20px"
---

# Design System: Test_Customs administration

## Overview

**Creative North Star: "Картотека клієнтів"**

The administration is a compact working directory: find a client, inspect the evidence, review a change, and confirm it. White and pale green surfaces, teal actions, system typography, and flat rows support that work. The final workspace replaces the earlier navy, rounded admin hub.

These rules apply only to administration. The user-facing home and learning screens retain their existing design. The built authority is `static/styles/admin_workspace.css` and `static/js/admin_navigation.js`, `admin_workspace.js`, `admin_customer_view.js`, `admin_content_workspace.js`, and `admin_settings.js`; the inherited font comes from `static/styles/tokens.css` and `base.css`. Product facts live in `PRODUCT.md`.

**Key Characteristics:**

- Three task spaces: «Клієнти», «Контент», and «Система».
- Flat directories with a separate inspector on desktop and a separate detail view on mobile.
- Payment evidence, manual permissions, and unverified legacy access have distinct labels and sections.
- Both themes use the same hierarchy, spacing, and restrained corner language.

## Colors

Frontmatter values are normative. Light keys map directly to the `--aw-*` properties on `body.admin-mode`; `dark-*` keys are their replacements under `html[data-theme="dark"]`. Exterior is the body background around the app.

### Primary

Deep teal accent identifies primary actions, selected navigation, disclosure text, positive status text, and keyboard focus. In dark mode it becomes pale mint, with dark text on filled buttons. Selected surfaces provide a quieter green tint behind active rows and filters.

### Neutral

Background is the main work area; surface is the pale inspector and list-header tone. Ink carries names and facts, muted carries explanations and metadata, and line separates rows. The dark equivalents preserve these roles. Both exterior colors are intentional parts of the workspace frame; detector advisories on these two literal backgrounds do not authorize changing them.

Warning and warning-bg identify access that needs attention or has unverified provenance. Danger marks errors and destructive text actions. Every status also has a literal Ukrainian label; color is supporting evidence.

## Typography

Use the inherited system stack for operational headings, body, and controls. There is no separate decorative display face. The frontmatter records the workspace hierarchy: page headline, inspector title, section title, body, supporting text, metadata, and input.

Page headings become (25px) at the mobile breakpoint. Directory rows use (14px, 1.5), with names at weight (600); the navigation brand is (18px) on desktop and (17px) on mobile. Inputs remain (16px). Amounts in payment and material rows use tabular numerals. Long names, references, product titles, and inspector text wrap rather than truncate.

## Layout

The admin app expands to (1360px) maximum width with (24px) outer padding. Its desktop shell has a (176px) sticky navigation column, a (28px) gap, and a flexible work area. Client and content directories use `minmax(0, 1.2fr) minmax(320px, 1fr)` columns separated by (24px). The inspector has (20px) padding; list rows use (16px 10px). System forms retain a narrower body of (680px).

At widths of (960px) and below, navigation becomes three horizontal destinations above the work area. Main padding is (20px 16px 32px). Selecting a client or material hides its directory and opens the detail view; «До списку клієнтів» or «До каталогу» returns to the stored list context. The inspector loses its desktop border, fill, and padding. Client rows become two columns with status on a full-width third line.

At (360px) and below, main padding becomes (18px 12px 28px), destination icons disappear, and paired payment fields stack. Use the recorded spacing for action gaps, fields, and sections; keep flat row separators instead of converting directories into a card grid.

## Elevation & Depth

The workspace app, inspector, rows, and restyled primary buttons are flat, without shadows. Tonal surfaces and thin borders establish structure. Direct admin content children have entrance animation disabled; the shared route transition remains inherited from `prototype.css`. Hover on standard buttons and destinations uses `brightness(.96)`; text actions underline without a motion effect.

**The Flat Rows Rule.** Directory records stay flat with separators; selection uses a tonal fill.

## Shapes

Controls and desktop navigation use restrained control rounding. Filters and warning surfaces use subtle rounding; inspectors and retained form containers use inspector rounding. The app frame and directory rows are square. Mobile destinations round only the top two corners (4px). Inline SVG navigation icons are (20px) on desktop and (18px) on mobile.

## Components

Navigation labels are exactly «Клієнти», «Контент», and «Система», with «До застосунку» as the exit. The active destination uses `aria-current="page"`. Client views «Картотека» and «Оплати» use an underline and `aria-pressed`; filters are «Усі», «З покупками», «Ручний доступ», and «Потребують уваги». Search accepts a name or Telegram ID, and client lists load (20) records per page.

The customer inspector starts with name and Telegram ID. «Покупки» shows product, known amount, currency, source, date, reference, and status. Unknown amounts and dates are explicit. «Ручні дозволи» contains reason and revoke actions; legacy access has its own warning. Missing paid material and blocked purchased access have distinct explanations. Available sections, history, and learning/profile details are native disclosures.

**The Evidence Rule.** A permission or visible section never serves as proof of payment; purchase evidence, manual permission, and unverified legacy access remain distinct.

Grant, revoke, restore, and payment-entry actions open an inline draft, then «Перевірте зміну» before «Підтвердити». Review shows the client, product or effect, reason, and payment amount when applicable. Cancel discards the draft. Payment entry records already verified evidence; it does not collect or refund money. Pending state is held per client across rerenders and navigation, disables that client's change actions, and shows «Зберігаємо…». Success uses returned detail; failure retains the draft and displays the server error. Revoking a manual permission does not cancel purchases. Deletion uses a native confirmation and protects administrators, the current admin, profiles with purchase history, and unverified legacy access.

Client drafts, query, filter, offset, and reconciliation progress are retained in current application state. They are not persisted across a browser reload. Selecting another client preserves that client's pending request and draft independently. Reconciliation has an explicit start/continue action and inline progress; it reads available Telegram Stars history.

Content uses the same directory/inspector pattern for the full package and materials. It owns prices, publication, questions, subsections, order, imports, and search. Title, price, and publication drafts are retained per material in current application state; saving is a direct form submit with controls disabled while pending, a status region, and draft retention on failure. These forms do not use the customer's review step. Publication copy explains that catalog visibility preserves purchased rights. Removing material uses a native confirmation; published or purchased material is preserved for existing owners, while new unpublished drafts do not open to package owners.

System contains support, client messages, and the action journal. Support loads before enabling its field and save button; pending submission disables both, and load failure exposes retry. Prices belong to Content. Journal entries show action, product, reason, date, administrator, and client when present.

Native buttons, labelled inputs, selects, textareas, and disclosure summaries retain keyboard behavior. Standard actions and fields have minimum (44px) height; main destinations have minimum (48px). Focus uses the accent outline (3px) with (3px) offset. Client selection focuses its heading, draft opening focuses reason, review focuses confirm, and return focuses the selected row. Client repaint restores named input or tagged-control focus and scroll when available; Content restores its tagged heading or row. These are inline views, without modal focus trapping. Lists and save feedback use polite status regions; customer errors use alerts. Disabled controls use opacity (.65).

## Do's and Don'ts

- **Do** preserve both themes, compact system typography, flat rows, and restrained corners.
- **Do** keep purchases, manual permissions, and legacy access visibly distinct, with literal status and action labels.
- **Do** preserve list context, per-record drafts, pending locks, review boundaries, and server-confirmed results.
- **Don't** reintroduce the discarded five-destination navy hub or large rounded cards into this admin workspace.
- **Don't** turn unknown amounts into current prices, manual access into payment proof, or publication into a promise of paid access.
- **Don't** apply these admin rules to the user-facing home and learning screens.
