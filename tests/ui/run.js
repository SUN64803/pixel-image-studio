// Electron UI smoke tests for PixelSmith Image Studio.
//
// Run from the repository root with:
//   bash tests/ui/run.sh
//
// These cover the bugs that unit tests cannot see: the in-app confirm dialog,
// caret/focus after closing it, translated dialog text, the blocking overlay
// (host and studio), the app-dialog IPC round-trip, and the preload file-path
// bridge inside the studio iframe.
"use strict";

const { app, ipcMain } = require("electron");
const path = require("path");
const {
  ROOT,
  sleep,
  check,
  summary,
  startServer,
  makeWindow,
  waitFor,
} = require("./harness");

// Each suite destroys its window; without this Electron would quit as soon as
// the first window closes.
app.on("window-all-closed", () => {});

// Expected noise: the stub engine answers 404 for feature endpoints, and
// Electron warns about CSP in dev builds (the packaged app sets one).
function isExpectedConsoleError(message) {
  return (
    /Electron Security Warning/.test(message) ||
    /Content Security Policy/.test(message) ||
    /ui-test stub/.test(message)
  );
}

async function studioSuite(port) {
  console.log("\n── studio: confirm dialog + caret ──────────────────────────");
  const win = makeWindow(port);
  const errors = [];
  win.webContents.on("console-message", (_e, level, message) => {
    if (level >= 2 && !isExpectedConsoleError(message)) errors.push(message);
  });
  await win.loadURL(`http://127.0.0.1:${port}/studio/`);
  await sleep(1200);

  const opened = await win.webContents.executeJavaScript(`(() => {
    const modal = document.querySelector("#confirmModal");
    const btn = document.querySelector("#clearHistory");
    if (!modal || !btn) return { ok: false, reason: "missing element" };
    btn.focus();
    btn.click();
    return { ok: true };
  })()`);
  check("confirm modal exists", opened.ok, JSON.stringify(opened));
  await sleep(200);
  const visible = await win.webContents.executeJavaScript(
    `!document.querySelector("#confirmModal").classList.contains("hidden")`
  );
  check("confirm modal opens on demand", visible);

  await win.webContents.executeJavaScript(
    `document.querySelector("#confirmCancel").click()`
  );
  await sleep(200);
  const closed = await win.webContents.executeJavaScript(`(() => {
    const modal = document.querySelector("#confirmModal");
    return {
      hidden: modal.classList.contains("hidden"),
      active: document.activeElement && document.activeElement.id,
    };
  })()`);
  check("cancel closes the dialog", closed.hidden, JSON.stringify(closed));
  check("focus returns to the opener", closed.active === "clearHistory", closed.active);

  const typed = await win.webContents.executeJavaScript(`(() => {
    const p = document.querySelector("#prompt");
    p.focus();
    p.value = "";
    return document.activeElement === p;
  })()`);
  await win.webContents.insertText("ui caret test");
  await sleep(150);
  const value = await win.webContents.executeJavaScript(
    `document.querySelector("#prompt").value`
  );
  check("prompt focused after the dialog", typed);
  check("typing reaches the prompt", value === "ui caret test", JSON.stringify(value));

  await win.webContents.executeJavaScript(`document.querySelector("#clearHistory").click()`);
  await sleep(150);
  await win.webContents.sendInputEvent({ type: "keyDown", keyCode: "Escape" });
  await win.webContents.sendInputEvent({ type: "keyUp", keyCode: "Escape" });
  await sleep(200);
  const escClosed = await win.webContents.executeJavaScript(
    `document.querySelector("#confirmModal").classList.contains("hidden")`
  );
  check("Escape cancels the dialog", escClosed);
  check("no console errors", errors.length === 0, errors.join(" | "));
  win.destroy();
}

async function i18nDialogSuite(port) {
  console.log("\n── studio: dialog text in TH and EN ────────────────────────");
  const win = makeWindow(port);
  await win.loadURL(`http://127.0.0.1:${port}/studio/`);
  await sleep(1200);

  const read = () => win.webContents.executeJavaScript(`(() => ({
    title: document.querySelector("#confirmTitle").textContent,
    body: document.querySelector("#confirmText").textContent,
    cancel: document.querySelector("#confirmCancel").textContent,
    ok: document.querySelector("#confirmOk").textContent,
    busy: document.querySelector("#busyText").textContent,
  }))()`);

  await win.webContents.executeJavaScript(`document.querySelector("#clearHistory").click()`);
  await sleep(250);
  const th = await read();
  check("TH title not blank", !!th.title.trim(), JSON.stringify(th.title));
  check("TH message not blank", !!th.body.trim(), JSON.stringify(th.body.slice(0, 40)));
  check("TH cancel not blank", !!th.cancel.trim(), JSON.stringify(th.cancel));
  check("TH confirm not blank", !!th.ok.trim(), JSON.stringify(th.ok));
  check("TH busy text not blank", !!th.busy.trim(), JSON.stringify(th.busy));
  await win.webContents.executeJavaScript(`document.querySelector("#confirmCancel").click()`);
  await sleep(150);

  await win.webContents.executeJavaScript(`window.I18N.setLang("en")`);
  await sleep(350);
  await win.webContents.executeJavaScript(`document.querySelector("#clearHistory").click()`);
  await sleep(250);
  const en = await read();
  check("EN title translated", en.title === "Confirm action", JSON.stringify(en.title));
  check("EN cancel translated", en.cancel === "Cancel", JSON.stringify(en.cancel));
  check("EN confirm translated", en.ok === "Confirm", JSON.stringify(en.ok));
  check("EN busy text translated", /Applying changes/.test(en.busy), JSON.stringify(en.busy.slice(0, 50)));
  check("EN message has text", !!en.body.trim(), JSON.stringify(en.body.slice(0, 40)));
  await win.webContents.executeJavaScript(`document.querySelector("#confirmCancel").click()`);
  await sleep(150);

  await win.webContents.executeJavaScript(`window.I18N.setLang("th")`);
  await sleep(350);
  await win.webContents.executeJavaScript(`document.querySelector("#clearHistory").click()`);
  await sleep(250);
  const th2 = await read();
  check(
    "TH restored after switching back",
    th2.title === "ยืนยันการทำรายการ" && th2.cancel === "ยกเลิก",
    JSON.stringify(th2.title)
  );
  await win.webContents.executeJavaScript(`document.querySelector("#confirmCancel").click()`);
  win.destroy();
}

async function hostBlockingSuite(port) {
  console.log("\n── host: blocking overlay around model load ───────────────");
  const win = makeWindow(port);
  await win.loadURL(`http://127.0.0.1:${port}/host/`);
  await sleep(1200);
  const hasHostOverlay = await win.webContents.executeJavaScript(
    `!!document.querySelector("#blocking")`
  );
  check("host blocking overlay exists", hasHostOverlay);

  await win.webContents.executeJavaScript(`document.querySelector("#loadModel").click()`);
  await sleep(300);
  const shown = await win.webContents.executeJavaScript(`(() => {
    const b = document.querySelector("#blocking");
    return b && !b.classList.contains("hidden")
      ? document.querySelector("#blockingText").textContent
      : "";
  })()`);
  check("host overlay shows during model load", !!shown, JSON.stringify(shown));
  const hidden = await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `document.querySelector("#blocking").classList.contains("hidden")`
      ),
    8000
  );
  check("host overlay hides when the model is ready", !!hidden);
  win.destroy();
}

async function studioBlockingSuite(port) {
  console.log("\n── host: studio blocking covers the whole window ──────────");
  const win = makeWindow(port);
  await win.loadURL(`http://127.0.0.1:${port}/host/`);
  await sleep(1500);
  const frame = win.webContents.mainFrame.frames.find((f) => f.url.includes("/studio/"));
  check("studio iframe loaded", !!frame);
  if (!frame) {
    win.destroy();
    return;
  }
  const clicked = await frame.executeJavaScript(`(() => {
    const b = document.querySelector('#powerModes .power-mode[data-mode="balanced"]');
    if (!b) return false;
    b.click();
    return true;
  })()`);
  check("power mode button exists", clicked);
  await sleep(300);
  const overlay = await win.webContents.executeJavaScript(`(() => {
    const b = document.querySelector("#blocking");
    const btn = document.querySelector("#loadModel");
    const r = btn.getBoundingClientRect();
    const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return {
      shown: b && !b.classList.contains("hidden"),
      text: document.querySelector("#blockingText").textContent,
      coversBar: !!top && (top.id === "blocking" || b.contains(top)),
    };
  })()`);
  check("host overlay shows for a studio operation", !!overlay.shown, JSON.stringify(overlay.text));
  check("overlay covers the host top bar", !!overlay.coversBar);
  check(
    "overlay text is not blank",
    /กำลังเปลี่ยนค่า|พลังงาน/.test(overlay.text),
    JSON.stringify(overlay.text)
  );
  const localHidden = await frame.executeJavaScript(
    `document.querySelector("#busyOverlay").classList.contains("hidden")`
  );
  check("studio local overlay stays hidden when hosted", localHidden);
  const hidden = await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `document.querySelector("#blocking").classList.contains("hidden")`
      ),
    8000
  );
  check("host overlay hides when the studio operation ends", !!hidden);
  win.destroy();
}

async function appDialogSuite(port) {
  console.log("\n── host: in-app dialog IPC round-trip ─────────────────────");
  const win = makeWindow(port, { preload: true });
  let result = null;
  const onResult = (_e, msg) => { result = msg; };
  ipcMain.on("app-dialog-result", onResult);
  await win.loadURL(`http://127.0.0.1:${port}/host/`);
  await sleep(1500);

  win.webContents.send("app-dialog", {
    id: 41,
    title: "ui-test title",
    message: "ui-test message",
    detail: "ui-test detail",
    buttons: ["เปิดโฟลเดอร์", "ปิด"],
    defaultId: 0,
    cancelId: 1,
  });
  const rendered = await waitFor(
    () =>
      win.webContents.executeJavaScript(`(() => {
        const d = document.querySelector("#appDialog");
        return d && !d.hidden
          ? {
              title: document.querySelector("#appDialogTitle").textContent,
              message: document.querySelector("#appDialogMessage").textContent,
              detail: document.querySelector("#appDialogDetail").textContent,
              buttons: [...document.querySelectorAll("#appDialogActions button")].map((b) => b.textContent),
            }
          : null;
      })()`),
    4000
  );
  check("app dialog renders", !!rendered, JSON.stringify(rendered));
  if (rendered) {
    check("app dialog title", rendered.title === "ui-test title", JSON.stringify(rendered.title));
    check("app dialog message", rendered.message === "ui-test message");
    check("app dialog detail", rendered.detail === "ui-test detail");
    check(
      "app dialog buttons",
      JSON.stringify(rendered.buttons) === JSON.stringify(["เปิดโฟลเดอร์", "ปิด"]),
      JSON.stringify(rendered.buttons)
    );
  }
  await win.webContents.executeJavaScript(
    `document.querySelectorAll("#appDialogActions button")[1].click()`
  );
  const replied = await waitFor(() => Promise.resolve(result), 3000);
  check(
    "dialog reply reaches the main process",
    !!replied && replied.id === 41 && replied.response === 1,
    JSON.stringify(replied)
  );
  const closed = await win.webContents.executeJavaScript(
    `document.querySelector("#appDialog").hidden`
  );
  check("app dialog closes after the reply", closed);

  // Escape must answer with cancelId rather than leaving the dialog stuck.
  result = null;
  win.webContents.send("app-dialog", {
    id: 42,
    title: "escape test",
    message: "escape",
    buttons: ["OK", "Cancel"],
    defaultId: 0,
    cancelId: 1,
  });
  await sleep(400);
  await win.webContents.sendInputEvent({ type: "keyDown", keyCode: "Escape" });
  await win.webContents.sendInputEvent({ type: "keyUp", keyCode: "Escape" });
  const escReply = await waitFor(() => Promise.resolve(result), 3000);
  check(
    "Escape answers with cancelId",
    !!escReply && escReply.id === 42 && escReply.response === 1,
    JSON.stringify(escReply)
  );

  ipcMain.removeListener("app-dialog-result", onResult);
  win.destroy();
}

async function iframeBridgeSuite(port) {
  console.log("\n── studio iframe: preload file-path bridge ────────────────");
  const win = makeWindow(port, { preload: true });
  await win.loadURL(`http://127.0.0.1:${port}/host/`);
  await sleep(1800);
  const frame = win.webContents.mainFrame.frames.find((f) => f.url.includes("/studio/"));
  check("studio iframe loaded", !!frame);
  if (!frame) {
    win.destroy();
    return;
  }
  const out = await frame.executeJavaScript(`(() => {
    const f = new File(["hello"], "hello.txt", { type: "text/plain" });
    let value = null, error = null, hasBridge = false;
    try {
      hasBridge = !!(window.parent && window.parent.pixelApp &&
                     typeof window.parent.pixelApp.getPathForFile === "function");
      value = window.parent.pixelApp.getPathForFile(f);
    } catch (e) {
      error = String(e);
    }
    return { hasBridge, value, error };
  })()`);
  check("parent bridge exposes getPathForFile", out.hasBridge, JSON.stringify(out));
  check("getPathForFile does not throw", out.error === null, JSON.stringify(out.error));
  check(
    "getPathForFile returns a string",
    typeof out.value === "string",
    JSON.stringify(out.value)
  );
  const hostBridge = await win.webContents.executeJavaScript(
    `typeof window.pixelApp.getPathForFile === "function"`
  );
  check("host window exposes getPathForFile", hostBridge);
  win.destroy();
}

async function main() {
  if (process.argv.includes("--help")) {
    console.log("usage: electron tests/ui/run.js  (or bash tests/ui/run.sh)");
    app.exit(0);
    return;
  }
  await app.whenReady();
  const { server, port } = await startServer();
  try {
    await studioSuite(port);
    await i18nDialogSuite(port);
    await hostBlockingSuite(port);
    await studioBlockingSuite(port);
    await appDialogSuite(port);
    await iframeBridgeSuite(port);
  } catch (err) {
    check("suite crashed", false, String((err && err.stack) || err));
  } finally {
    server.close();
  }
  const failed = summary();
  app.exit(failed === 0 ? 0 : 1);
}

main();
