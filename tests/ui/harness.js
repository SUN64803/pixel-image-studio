// Shared helpers for the Electron UI smoke tests (tests/ui/run.js).
//
// The tests serve the real host + studio files from disk and stub the engine
// API, so they exercise the actual renderer code without a GPU or a model.
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");

const ROOT = process.env.PIXELSMITH_ROOT
  ? path.resolve(process.env.PIXELSMITH_ROOT)
  : path.resolve(__dirname, "..", "..");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let failures = 0;
let passes = 0;
function check(name, ok, extra) {
  const line = (ok ? "PASS" : "FAIL") + "  " + name + (extra ? "  " + extra : "");
  console.log(line);
  if (ok) passes += 1;
  else failures += 1;
}
function summary() {
  console.log("");
  console.log(`${passes} passed, ${failures} failed`);
  return failures;
}

// Static file server with a tiny engine stub. `modelLoadMs` controls how long
// /api/model/load reports "loading" for; power POSTs take `powerMs`.
function startServer(options = {}) {
  const modelLoadMs = options.modelLoadMs == null ? 1500 : options.modelLoadMs;
  const powerMs = options.powerMs == null ? 1200 : options.powerMs;
  let modelState = "not_loaded";
  let loadingUntil = 0;

  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const json = (obj, code = 200) => {
      res.writeHead(code, { "content-type": "application/json" });
      res.end(JSON.stringify(obj));
    };
    if (url === "/api/health") {
      if (modelState === "loading" && Date.now() > loadingUntil) modelState = "ready";
      return json({
        model: { state: modelState, loaded: modelState === "ready" },
        gpu: { name: "stub GPU", vram_total: 10737418240, vram_used: 0 },
      });
    }
    if (url === "/api/model/load") {
      modelState = "loading";
      loadingUntil = Date.now() + modelLoadMs;
      return json({ ok: true, started: true });
    }
    if (url === "/api/model/unload") {
      modelState = "not_loaded";
      return json({ ok: true });
    }
    if (url === "/api/power") {
      const body = {
        available: true,
        supported: true,
        limit: 250,
        default_limit: 250,
        saved_mode: "balanced",
        temp: 40,
        draw: 50,
        modes: [
          { id: "low", watts: 150 },
          { id: "balanced", watts: 200 },
          { id: "performance", watts: 250 },
        ],
      };
      if (req.method === "POST") {
        setTimeout(() => json({ ...body, limit: 200 }), powerMs);
      } else {
        json(body);
      }
      return;
    }
    if (url.startsWith("/api/") || url.startsWith("/files/")) {
      return json({ detail: "ui-test stub" }, 404);
    }
    let file = path.join(ROOT, url);
    if (url.endsWith("/")) file = path.join(file, "index.html");
    fs.readFile(file, (err, buf) => {
      if (err) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, {
        "content-type": MIME[path.extname(file)] || "application/octet-stream",
      });
      res.end(buf);
    });
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      resolve({ server, port: server.address().port });
    });
  });
}

function makeWindow(port, options = {}) {
  const { BrowserWindow } = require("electron");
  const win = new BrowserWindow({
    show: false,
    width: 1400,
    height: 900,
    webPreferences: {
      preload: options.preload ? path.join(ROOT, "electron", "preload.js") : undefined,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // the studio caches its old bundle in the default session; keep each
      // run isolated so a stale app.js cannot mask a regression
      partition: "ui-test-" + Date.now() + "-" + Math.random().toString(36).slice(2),
    },
  });
  return win;
}

async function waitFor(fn, timeoutMs = 5000, intervalMs = 50) {
  const started = Date.now();
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() - started >= timeoutMs) return null;
    await sleep(intervalMs);
  }
}

module.exports = {
  ROOT,
  sleep,
  check,
  summary,
  startServer,
  makeWindow,
  waitFor,
};
