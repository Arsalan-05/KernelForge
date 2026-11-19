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
  "home":         { pageId: "page-home",          navHref: "#/" },
  "studio":       { pageId: "page-studio",        navHref: "#/studio" },
  "results":      { pageId: "page-results",       navHref: "#/results" },
  "library":      { pageId: "page-library",       navHref: "#/library" },
  "how-it-works": { pageId: "page-how-it-works",  navHref: "#/how-it-works" },
  "dataset":      { pageId: "page-dataset",       navHref: "#/dataset" },
  "about":        { pageId: "page-about",         navHref: "#/about" },
};

const DEFAULT_ROUTE = "home";

function currentRoute() {
  const raw = location.hash.replace(/^#\/?/, "").split("?")[0].trim();
  /* Empty hash or bare "#/" → home */
  if (!raw || raw === "") return "home";
  return ROUTES[raw] ? raw : DEFAULT_ROUTE;
}

let _activeRoute = null;

function navigate(routeKey) {
  const route = ROUTES[routeKey] || ROUTES[DEFAULT_ROUTE];

  /* Show/hide pages */
  Object.entries(ROUTES).forEach(([key, r]) => {
    const el = document.getElementById(r.pageId);
    if (el) el.hidden = key !== routeKey;
  });

  /* Show/hide the studio banner only on the studio page */
  const banner = document.getElementById("banner");
  if (banner) banner.hidden = routeKey !== "studio";

  /* Show/hide model/gpu pills only on studio page */
  const modelPill = document.getElementById("model-pill");
  const gpuPill   = document.getElementById("gpu-pill");
  if (modelPill) modelPill.hidden = routeKey !== "studio";
  if (gpuPill)   gpuPill.hidden   = routeKey !== "studio";

  /* Update nav active state */
  document.querySelectorAll(".nav-link").forEach((a) => {
    /* Normalize both the href and the routeKey for comparison:
       "#/" → "home", "#/studio" → "studio", etc. */
    const rawHref = a.getAttribute("href").replace(/^#\/?/, "");
    /* For the home link (#/), rawHref will be empty string */
    const hrefKey = rawHref === "" ? "home" : rawHref;
    const active = hrefKey === routeKey;
    a.classList.toggle("active", active);
    a.setAttribute("aria-current", active ? "page" : "false");
  });

  _activeRoute = routeKey;

  /* On first arrival at Studio, trigger studio init if not yet done */
  if (routeKey === "studio" && typeof window.studioReady === "function") {
    window.studioReady();
  }

  /* When navigating to studio on desktop, ensure sidebar is open */
  if (routeKey === "studio" && isDesktop()) {
    setSidebarOpen(true);
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

/* ── Sidebar drawer ───────────────────────────────────────────────────────── */

function isDesktop() {
  return window.matchMedia("(min-width: 981px)").matches;
}

function setSidebarOpen(open) {
  const studio  = document.querySelector(".studio");
  const sidebar = document.querySelector(".sidebar");
  const drawerBtn = document.getElementById("sidebar-drawer-btn");
  const legacyBtn = document.getElementById("sidebar-toggle");

  if (!studio || !sidebar) return;

  /* Apply classes */
  studio.classList.toggle("sidebar-open", open);
  sidebar.classList.toggle("open", open);        /* compat: app.js checks .open on sidebar */

  /* Update aria-expanded on both toggle buttons */
  if (drawerBtn) drawerBtn.setAttribute("aria-expanded", String(open));
  if (legacyBtn) legacyBtn.setAttribute("aria-expanded", String(open));
}

function initSidebarDrawer() {
  const drawerBtn = document.getElementById("sidebar-drawer-btn");
  if (!drawerBtn) return;

  /* Initial state: open on desktop, closed on mobile */
  setSidebarOpen(isDesktop());

  drawerBtn.addEventListener("click", () => {
    const studio = document.querySelector(".studio");
    const isOpen = studio ? studio.classList.contains("sidebar-open") : false;
    setSidebarOpen(!isOpen);
  });

  /* Re-evaluate on resize: auto-open if going from mobile→desktop while closed */
  window.matchMedia("(min-width: 981px)").addEventListener("change", (e) => {
    if (e.matches) {
      /* Switched to desktop: open sidebar if currently closed */
      const studio = document.querySelector(".studio");
      if (studio && !studio.classList.contains("sidebar-open")) {
        setSidebarOpen(true);
      }
    } else {
      /* Switched to mobile: close sidebar */
      setSidebarOpen(false);
    }
  });
}

/* ── Bootstrap ───────────────────────────────────────────────────────────── */

function initRouter() {
  initTheme();

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.addEventListener("click", toggleTheme);

  initMobileNav();
  initSidebarDrawer();

  window.addEventListener("hashchange", handleHashChange);
  handleHashChange(); /* run on page load */
}

document.addEventListener("DOMContentLoaded", initRouter);
