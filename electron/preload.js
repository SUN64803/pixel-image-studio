// PixelSmith Image Studio — preload (PART A).
// The renderer stays sandboxed; it reaches Node only through window.pixelApp.
const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("pixelApp", {
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
  // File.path was removed in Electron 32 — this is the supported way to get
  // the absolute path of a file chosen through <input type="file">.
  getPathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file);
    } catch (e) {
      return "";
    }
  },
  openFolder: (which) => ipcRenderer.invoke("open-folder", String(which || "root")),
  openProgramFolder: () => ipcRenderer.invoke("open-folder", "root"),
  info: () => ipcRenderer.invoke("app-info"),
  // first-run installer progress (main process → installer page)
  onInstallUpdate: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => {
      try { cb(payload); } catch (e) { /* page handler bug must not crash IPC */ }
    };
    ipcRenderer.on("install:update", handler);
    return () => ipcRenderer.removeListener("install:update", handler);
  },
  // in-app message boxes (About, install guide, download finished) — the main
  // process sends the payload, the host page renders it and replies with the
  // index of the clicked button.
  onAppDialog: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => {
      try { cb(payload); } catch (e) { /* page handler bug must not crash IPC */ }
    };
    ipcRenderer.on("app-dialog", handler);
    return () => ipcRenderer.removeListener("app-dialog", handler);
  },
  dialogResult: (id, response) => ipcRenderer.send("app-dialog-result", { id, response }),
  // engine crash recovery: main process → host page
  restartEngine: () => ipcRenderer.invoke("restart-engine"),
  onEngineStatus: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => {
      try { cb(payload); } catch (e) { /* page handler bug must not crash IPC */ }
    };
    ipcRenderer.on("engine-status", handler);
    return () => ipcRenderer.removeListener("engine-status", handler);
  },
});
