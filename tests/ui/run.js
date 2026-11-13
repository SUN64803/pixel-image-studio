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

async function backupSuite(port) {
  console.log("\n── studio: backup / restore ───────────────────────────────");
  const win = makeWindow(port);
  await win.loadURL(`http://127.0.0.1:${port}/studio/`);
  await sleep(1200);

  const present = await win.webContents.executeJavaScript(`(() => ({
    exportBtn: !!document.querySelector("#backupExport"),
    importBtn: !!document.querySelector("#backupImport"),
    input: !!document.querySelector("#backupFile"),
    accept: (document.querySelector("#backupFile") || {}).accept || "",
    state: (document.querySelector("#backupState") || {}).textContent || "",
  }))()`);
  check("backup export button exists", present.exportBtn);
  check("backup import button exists", present.importBtn);
  check("backup file input accepts zip", /zip/i.test(present.accept), present.accept);

  await win.webContents.executeJavaScript(
    `document.querySelector("#backupExport").click()`
  );
  await sleep(120);
  const busy = await win.webContents.executeJavaScript(`(() => {
    const b = document.querySelector("#busyOverlay");
    return b && !b.classList.contains("hidden")
      ? (document.querySelector("#busyText") || {}).textContent : "";
  })()`);
  check("backup export shows blocking status", /สำรองข้อมูล/.test(busy || ""), JSON.stringify(busy));

  const done = await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `document.querySelector("#backupState").textContent.includes("ไฟล์")`
      ),
    6000
  );
  check("backup state reports the result", !!done);
  const after = await win.webContents.executeJavaScript(`(() => ({
    state: document.querySelector("#backupState").textContent,
    toast: (document.querySelector("#toasts") || {}).textContent || "",
    overlayHidden: document.querySelector("#busyOverlay").classList.contains("hidden"),
  }))()`);
  check("backup state shows file count and size", /7 ไฟล์/.test(after.state), after.state);
  check("backup toast appears", /สำรองข้อมูลแล้ว/.test(after.toast), after.toast.slice(0, 80));
  check("blocking overlay hides when export ends", after.overlayHidden);

  // Import: feed a File through the hidden input, confirm, then dismiss reload.
  const fed = await win.webContents.executeJavaScript(`(() => {
    const input = document.querySelector("#backupFile");
    if (!input) return false;
    const dt = new DataTransfer();
    dt.items.add(new File([new Uint8Array([1, 2, 3])], "backup.zip", { type: "application/zip" }));
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  check("backup file input accepts a file", fed);
  await sleep(250);
  const confirmOpen = await win.webContents.executeJavaScript(
    `!document.querySelector("#confirmModal").classList.contains("hidden")`
  );
  check("restore asks for confirmation", confirmOpen);
  await win.webContents.executeJavaScript(`document.querySelector("#confirmOk").click()`);
  await sleep(900);
  const summary = await win.webContents.executeJavaScript(`(() => ({
    open: !document.querySelector("#confirmModal").classList.contains("hidden"),
    title: document.querySelector("#confirmTitle").textContent,
    body: document.querySelector("#confirmText").textContent,
  }))()`);
  check("restore reports a summary", summary.open && /กู้คืนแล้ว 7 รายการ/.test(summary.body), summary.body.slice(0, 90));
  check("restore summary dialog is titled", /กู้คืนสำเร็จ/.test(summary.title), summary.title);
  await win.webContents.executeJavaScript(`document.querySelector("#confirmCancel").click()`);
  await sleep(200);

  // EN mode: the export toast and restore dialog must be translated.
  await win.webContents.executeJavaScript(`window.I18N.setLang("en")`);
  await sleep(300);
  await win.webContents.executeJavaScript(
    `document.querySelector("#backupExport").click()`
  );
  await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `document.querySelector("#backupState").textContent.includes("files")`
      ),
    6000
  );
  const enState = await win.webContents.executeJavaScript(
    `document.querySelector("#backupState").textContent`
  );
  check("EN backup state translated", /7 files/.test(enState), enState);
  await win.webContents.executeJavaScript(`window.I18N.setLang("th")`);
  win.destroy();
}

async function trainerSuite(port) {
  console.log("\n── studio: trainer datasets ───────────────────────────────");
  const win = makeWindow(port);
  await win.loadURL(`http://127.0.0.1:${port}/studio/`);
  await sleep(1200);

  // The train pane only renders its dataset after the API stub responds.
  await win.webContents.executeJavaScript(
    `document.querySelector('#panelTabs .panel-tab[data-pane="train"]').click()`
  );
  await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `!!document.querySelector('#trainerDs option[value="ds-test"]')`
      ),
    6000
  );
  check(
    "manual button available before selecting a dataset",
    await win.webContents.executeJavaScript(`!!document.querySelector("#guideOpenTop")`)
  );
  await win.webContents.executeJavaScript(`(() => {
    const sel = document.querySelector("#trainerDs");
    sel.value = "ds-test";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
  })()`);
  await sleep(300);

  const view = await win.webContents.executeJavaScript(`(() => {
    const row = document.querySelector("#captionList .caption-row");
    return {
      active: !document.querySelector("#trainerActive").classList.contains("hidden"),
      meta: document.querySelector("#dsMeta").textContent,
      kind: document.querySelector("#dsKindEdit").value,
      extra: document.querySelector("#dsExtraTriggers").value,
      trigger: row.querySelector(".caption-trigger").value,
      tags: row.querySelector(".caption-tags").value,
      repeats: row.querySelector(".caption-repeats").value,
      caption: row.querySelector(".caption-input").value,
      multiHidden: getComputedStyle(row.querySelector(".caption-role")).display === "none",
      audit: document.querySelector("#dsAudit").textContent,
      hint: document.querySelector("#trainHint").textContent,
      calc: document.querySelector("#calcSteps").textContent,
    };
  })()`);
  check("dataset panel opens", view.active);
  check("dataset kind rendered", view.kind === "style", view.kind);
  check("extra triggers rendered", view.extra === "armor form", view.extra);
  check("per-image trigger rendered", view.trigger === "armor form", view.trigger);
  check("per-image tags rendered", view.tags === "front view", view.tags);
  check("per-image repeats rendered", view.repeats === "3", view.repeats);
  check("caption text rendered", view.caption === "posing", view.caption);
  check("multi selector hidden for style datasets", view.multiHidden);
  check(
    "kind-aware audit shows the style target",
    /เป้าหมาย/.test(view.audit) && /20–30/.test(view.audit) && !/เดี่ยว/.test(view.audit),
    view.audit.slice(0, 90)
  );
  check(
    "style training hint replaces the character one",
    /💡 style:/.test(view.hint) && !/character:/.test(view.hint),
    view.hint.slice(0, 80)
  );
  check(
    "steps calculator counts repeats × epochs",
    /30 steps/.test(view.calc) && /3 samples\/epoch/.test(view.calc),
    view.calc
  );

  // Calculator → steps field
  await win.webContents.executeJavaScript(`(() => {
    document.querySelector("#calcEpochs").value = "5";
    document.querySelector("#calcEpochs").dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector("#calcApply").click();
  })()`);
  await sleep(200);
  const applied = await win.webContents.executeJavaScript(
    `document.querySelector("#trainSteps").value`
  );
  check("apply calculator writes the steps field", applied === "15", applied);

  // Preset → rank/steps/scheduler
  await win.webContents.executeJavaScript(`(() => {
    document.querySelector("#trainPreset").value = "character";
    document.querySelector("#trainPresetApply").click();
  })()`);
  await sleep(200);
  const preset = await win.webContents.executeJavaScript(`(() => ({
    rank: document.querySelector("#trainRank").value,
    steps: document.querySelector("#trainSteps").value,
    lr: document.querySelector("#trainLr").value,
    scheduler: document.querySelector("#trainScheduler").value,
  }))()`);
  check("preset sets rank", preset.rank === "32", preset.rank);
  check("preset sets steps", preset.steps === "1200", preset.steps);
  check("preset sets learning rate", preset.lr === "0.0001", preset.lr);
  check("preset sets scheduler", preset.scheduler === "cosine", preset.scheduler);

  // Phase-2 recipe controls exist and presets set them.
  const phase2Fields = await win.webContents.executeJavaScript(`(() => ({
    bucketing: !!document.querySelector("#trainBucketing"),
    captionDropout: !!document.querySelector("#trainCaptionDropout"),
    captionShuffle: !!document.querySelector("#trainCaptionShuffle"),
    keepTokens: !!document.querySelector("#trainKeepTokens"),
    noiseOffset: !!document.querySelector("#trainNoiseOffset"),
    minSnr: !!document.querySelector("#trainMinSnr"),
    rsLora: !!document.querySelector("#trainRsLora"),
    targetModules: !!document.querySelector("#trainTargetModules"),
    resume: !!document.querySelector("#trainResumeState"),
  }))()`);
  check(
    "phase-2 recipe controls rendered",
    Object.values(phase2Fields).every(Boolean),
    JSON.stringify(phase2Fields)
  );

  await win.webContents.executeJavaScript(`(() => {
    document.querySelector("#trainPreset").value = "style";
    document.querySelector("#trainPresetApply").click();
  })()`);
  await sleep(200);
  const stylePreset = await win.webContents.executeJavaScript(`(() => ({
    bucketing: document.querySelector("#trainBucketing").checked,
    shuffle: document.querySelector("#trainCaptionShuffle").checked,
    minSnr: document.querySelector("#trainMinSnr").value,
    noise: document.querySelector("#trainNoiseOffset").value,
    keep: document.querySelector("#trainKeepTokens").value,
  }))()`);
  check("style preset enables bucketing", stylePreset.bucketing);
  check("style preset enables caption shuffle", stylePreset.shuffle);
  check("style preset sets min-SNR", stylePreset.minSnr === "5", stylePreset.minSnr);
  check("style preset sets noise offset", stylePreset.noise === "0.02", stylePreset.noise);
  check("style preset sets keep tokens", stylePreset.keep === "1", stylePreset.keep);

  // Dataset QC report renders.
  await win.webContents.executeJavaScript(
    `document.querySelector("#dsQcRun").click()`
  );
  await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `document.querySelector("#dsQcResult").textContent.includes("ข้อมูลพร้อมเทรน")`
      ),
    5000
  );
  const qc = await win.webContents.executeJavaScript(`(() => ({
    visible: !document.querySelector("#dsQcResult").classList.contains("hidden"),
    text: document.querySelector("#dsQcResult").textContent,
  }))()`);
  check("QC report visible", qc.visible);
  check("QC report shows recommendation", /ข้อมูลพร้อมเทรน/.test(qc.text));

  // Sample generation button queues a job with the selected LoRA.
  const sampleLoraReady = await win.webContents.executeJavaScript(
    `!!document.querySelector('#sampleLora option[value="lora-test"]')`
  );
  check("sample LoRA picker populated", sampleLoraReady);
  await win.webContents.executeJavaScript(`(() => {
    document.querySelector("#sampleLora").value = "lora-test";
    document.querySelector("#sampleRun").click();
  })()`);
  await sleep(400);

  // In-app training manual: opens, searches, deep-links and re-renders in EN.
  await win.webContents.executeJavaScript(
    `document.querySelector("#guideOpen").click()`
  );
  await waitFor(
    () =>
      win.webContents.executeJavaScript(
        `!document.querySelector("#guideModal").classList.contains("hidden") &&
         document.querySelectorAll("#guideContent .guide-section").length > 20`
      ),
    5000
  );
  const guideTh = await win.webContents.executeJavaScript(`(() => ({
    title: document.querySelector("#guideTitle").textContent,
    sections: document.querySelectorAll("#guideContent .guide-section").length,
    toc: document.querySelectorAll("#guideToc .guide-toc-item").length,
    text: document.querySelector("#guideContent").textContent,
  }))()`);
  check("guide opens with all sections", guideTh.sections >= 30, String(guideTh.sections));
  check("guide TOC populated", guideTh.toc === guideTh.sections, `${guideTh.toc}/${guideTh.sections}`);
  check("guide TH title", /คู่มือการเทรน/.test(guideTh.title), guideTh.title);
  check(
    "guide TH covers bucketing + recommendations",
    /bucketing/i.test(guideTh.text) && /แนะนำ/.test(guideTh.text)
  );

  await win.webContents.executeJavaScript(`(() => {
    const s = document.querySelector("#guideSearch");
    s.value = "rsLoRA";
    s.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);
  await sleep(200);
  const guideSearch = await win.webContents.executeJavaScript(`(() => ({
    sections: document.querySelectorAll("#guideContent .guide-section").length,
    text: document.querySelector("#guideContent").textContent,
  }))()`);
  check(
    "guide search filters to rsLoRA",
    guideSearch.sections >= 1 && guideSearch.sections < guideTh.sections,
    String(guideSearch.sections)
  );
  check("guide search keeps matching content", /rsLoRA/.test(guideSearch.text));

  await win.webContents.executeJavaScript(`(() => {
    const s = document.querySelector("#guideSearch");
    s.value = "";
    s.dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector('[data-guide-target="rank"]').click();
  })()`);
  await sleep(200);
  const guideToc = await win.webContents.executeJavaScript(`(() => ({
    active: document.querySelector("#guideToc .guide-toc-item.active")?.textContent || "",
    rank: !!document.querySelector("#guide-sec-rank"),
  }))()`);
  check("guide TOC deep-link selects rank", guideToc.rank && /Rank/.test(guideToc.active), guideToc.active);

  await win.webContents.executeJavaScript(`window.I18N.setLang("en")`);
  await sleep(400);
  const guideEn = await win.webContents.executeJavaScript(`(() => ({
    title: document.querySelector("#guideTitle").textContent,
    text: document.querySelector("#guideContent").textContent,
    search: document.querySelector("#guideSearch").placeholder,
    footer: document.querySelector("#guideFoot").textContent,
  }))()`);
  check("guide EN title", guideEn.title === "LoRA training manual", guideEn.title);
  check(
    "guide EN content translated",
    /Aspect-ratio bucketing/.test(guideEn.text) && /recommended/i.test(guideEn.text)
  );
  check("guide EN search placeholder", /Search a topic/.test(guideEn.search), guideEn.search);
  check("guide footer shows language", /EN/.test(guideEn.footer), guideEn.footer);

  await win.webContents.executeJavaScript(
    `document.querySelector("#guideClose").click()`
  );
  await sleep(200);
  const guideClosed = await win.webContents.executeJavaScript(
    `document.querySelector("#guideModal").classList.contains("hidden")`
  );
  check("guide closes", guideClosed);
  await win.webContents.executeJavaScript(`window.I18N.setLang("th")`);
  await sleep(300);

  // EN mode: dynamically rendered trainer labels must be translated.
  await win.webContents.executeJavaScript(`window.I18N.setLang("en")`);
  await sleep(400);
  const en = await win.webContents.executeJavaScript(`(() => ({
    meta: document.querySelector("#dsMeta").textContent,
    hint: document.querySelector("#trainHint").textContent,
    kindOption: document.querySelector('#dsKindEdit option[value="style"]').textContent,
  }))()`);
  check("EN dataset kind translated", en.meta === "Style", en.meta);
  check("EN style hint translated", /Style:/.test(en.hint), en.hint.slice(0, 60));
  check("EN kind option translated", en.kindOption === "Style", en.kindOption);
  await win.webContents.executeJavaScript(`window.I18N.setLang("th")`);
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
    await backupSuite(port);
    await trainerSuite(port);
  } catch (err) {
    check("suite crashed", false, String((err && err.stack) || err));
  } finally {
    server.close();
  }
  const failed = summary();
  app.exit(failed === 0 ? 0 : 1);
}

main();
