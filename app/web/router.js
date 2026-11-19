"use strict";

/* ── Theme ───────────────────────────────────────────────────────────────── */

const THEME_KEY = "kernelforge.theme.v1";
const hlDark  = document.getElementById("hl-dark");
const hlLight = document.getElementById("hl-light");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  if (hlDark)  hlDark.disabled  = (theme === "light");
  if (hlLight) hlLight.disabled = (theme === "dark");
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  }
}

function savedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function persistTheme(t) {
  try { localStorage.setItem(THEME_KEY, t); } catch { /* storage disabled */ }
}

function initTheme() {
  const stored = savedTheme();
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(stored || preferred);
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || "dark";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  persistTheme(next);
}

/* Listen for OS-level preference changes (only when user hasn't set their own) */
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  if (!savedTheme()) applyTheme(e.matches ? "dark" : "light");
});

/* ── Route definitions ───────────────────────────────────────────────────── */

const ROUTES = {
  "studio":       { pageId: "page-studio",       navHref: "#/studio" },
  "results":      { pageId: "page-results",       navHref: "#/results" },
  "library":      { pageId: "page-library",       navHref: "#/library" },
  "how-it-works": { pageId: "page-how-it-works",  navHref: "#/how-it-works" },
  "dataset":      { pageId: "page-dataset",       navHref: "#/dataset" },
  "about":        { pageId: "page-about",         navHref: "#/about" },
};

const DEFAULT_ROUTE = "studio";

function currentRoute() {
  const hash = location.hash.replace(/^#\/?/, "").split("?")[0].trim();
  return ROUTES[hash] ? hash : DEFAULT_ROUTE;
}

let _activeRoute = null;

function navigate(routeKey) {
  const route = ROUTES[routeKey] || ROUTES[DEFAULT_ROUTE];

  /* Show/hide pages */
  Object.entries(ROUTES).forEach(([key, r]) => {
    const el = document.getElementById(r.pageId);
    if (el) el.hidden = key !== routeKey;
  });

  /* Update nav active state */
  document.querySelectorAll(".nav-link").forEach((a) => {
    const href = a.getAttribute("href").replace(/^#\/?/, "");
    const active = href === routeKey;
    a.classList.toggle("active", active);
    a.setAttribute("aria-current", active ? "page" : "false");
  });

  _activeRoute = routeKey;

  /* On first arrival at Studio, trigger studio init if not yet done */
  if (routeKey === "studio" && typeof window.studioReady === "function") {
    window.studioReady();
  }
}

function handleHashChange() {
  const route = currentRoute();
  navigate(route);
  /* Close mobile nav if open */
  const mobileNav = document.getElementById("mobile-nav");
  if (mobileNav) mobileNav.hidden = true;
  const burger = document.getElementById("burger-btn");
  if (burger) burger.setAttribute("aria-expanded", "false");
}

/* ── Mobile nav ──────────────────────────────────────────────────────────── */

function initMobileNav() {
  const burger = document.getElementById("burger-btn");
  const mobileNav = document.getElementById("mobile-nav");
  if (!burger || !mobileNav) return;

  burger.addEventListener("click", () => {
    const open = mobileNav.hidden;
    mobileNav.hidden = !open;
    burger.setAttribute("aria-expanded", String(open));
  });

  /* Close on outside click */
  document.addEventListener("click", (e) => {
    if (!mobileNav.hidden && !mobileNav.contains(e.target) && !burger.contains(e.target)) {
      mobileNav.hidden = true;
      burger.setAttribute("aria-expanded", "false");
    }
  });

  /* Close on nav link click */
  mobileNav.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      mobileNav.hidden = true;
      burger.setAttribute("aria-expanded", "false");
    });
  });
}

/* ── Bootstrap ───────────────────────────────────────────────────────────── */

function initRouter() {
  initTheme();

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.addEventListener("click", toggleTheme);

  initMobileNav();

  window.addEventListener("hashchange", handleHashChange);
  handleHashChange(); /* run on page load */
}

document.addEventListener("DOMContentLoaded", initRouter);
