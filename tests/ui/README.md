# UI smoke tests

Electron tests for behaviour that unit tests cannot see: focus/caret handling,
dialog rendering, and the host ↔ studio overlay bridge. They serve the real
`host/` and `studio/` files and stub the engine API, so no GPU, model, or
network is needed.

```bash
bash tests/ui/run.sh          # from the repo root
# or
cd electron && npm run test:ui
```

Requires the Electron dev dependency (`cd electron && npm install`). A display
is needed; on a headless machine install `xvfb` and the script uses
`xvfb-run` automatically. Set `ELECTRON=/path/to/electron` to use a different
binary.

## What is covered

| Suite | Checks |
|---|---|
| confirm dialog + caret | dialog opens from Clear history, Cancel restores focus to the opener, typing reaches the prompt afterwards, Escape cancels, no console errors |
| dialog text TH/EN | every dialog/toast/blocking string is non-empty in Thai, translated in English, and restored when switching back |
| host blocking | the full-window overlay appears while the model loads and hides when it is ready |
| studio → host blocking | a studio operation asks the host to show the overlay, the overlay covers the top bar, the studio's own overlay stays hidden, and the host overlay hides afterwards |
| app dialog IPC | `app-dialog` payload renders title/message/detail/buttons, the clicked index travels back to the main process, and Escape answers with `cancelId` |
| iframe bridge | the studio iframe reaches `parent.pixelApp.getPathForFile` and it returns a string without throwing |

`tests/ui/run.js` prints `PASS`/`FAIL` per check and exits non-zero when
anything fails. `tests/ui/harness.js` holds the static server and helpers.
