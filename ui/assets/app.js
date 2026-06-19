const API = "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

function showToast(msg, type = "success") {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// CSV Type → CSS class
const TYPE_CLASS = {
  "{G}": "grass", "{R}": "fire", "{W}": "water", "{L}": "lightning",
  "{P}": "psychic", "{F}": "fighting", "{D}": "darkness", "{M}": "metal",
  "{C}": "colorless", "竜": "dragon", "{A}": "any",
};

function typeBadge(typeStr) {
  if (!typeStr) return "";
  const cls = TYPE_CLASS[typeStr] || "colorless";
  const label = typeStr.replace(/[{}]/g, "");
  return `<span class="type-badge type-${cls}">${label}</span>`;
}

function hpBar(hp, maxHp) {
  const ratio = maxHp ? hp / maxHp : 1;
  const pct = Math.max(0, Math.min(100, ratio * 100));
  const cls = pct > 50 ? "" : pct > 20 ? "med" : "low";
  return `
    <div class="hp-bar-bg">
      <div class="hp-bar-fill ${cls}" style="width:${pct}%"></div>
    </div>`;
}

function debounce(fn, ms = 300) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function qs(sel, root = document) { return root.querySelector(sel); }
function qsa(sel, root = document) { return [...root.querySelectorAll(sel)]; }

function setActive(linkEl) {
  qsa(".nav-link").forEach(l => l.classList.remove("active"));
  if (linkEl) linkEl.classList.add("active");
}
