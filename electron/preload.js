// PixelSmith Image Studio — preload (PART A).
// The renderer stays sandboxed; it reaches Node only through window.pixelApp.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("pixelApp", {
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
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
});
