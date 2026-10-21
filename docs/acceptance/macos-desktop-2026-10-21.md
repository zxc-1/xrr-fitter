# macOS desktop input acceptance — 2026-10-21

Source baseline: `6be4d94fd65e254deffc4941901f7b3fe7a8b80f`, tree
`c9ad6b4a5228225b7139155c96863c25b98dcfbe`. The checkout was clean during input
acceptance. Python 3.12.13, PySide6/Qt 6.11.2, native `cocoa`, Apple Silicon.

The observer prepared a disposable synthetic three-layer project and isolated
settings outside the repository. CUA supplied desktop keyboard, mouse and scroll
input. During acceptance input, the observer recorded events and file hashes;
it did not synthesize quit, mouse, key or drop events, or answer confirmation
dialogs. The separate observer teardown closed the structure-tree test window
after its recorded checks; that closure is not quit-acceptance evidence. No owner
data was used.

| Scenario | Result | Evidence checked |
|---|---|---|
| Command-Q, then Cancel | PASS | Spontaneous application Quit reached the real save prompt; window, dirty project identity, saved bytes and draft bytes were unchanged |
| Command-Q, then Save | PASS | Saved project includes the prepared expert-mode edit, draft removed, event loop exited and window destroyed |
| Native application menu Quit, then Cancel | PASS | Real menu and prompt used; window, dirty project and both file hashes remained unchanged |
| Command-Q, then Discard | PASS | Original saved bytes preserved, draft removed and window destroyed |
| Scroll over an unfocused thickness editor | PASS | Native wheel event reached the editor; inspector scrollbar moved from 0 to 81, project identity and parameters stayed unchanged |
| Start and cancel structure drag | PASS | Real mouse input entered the native drag loop; Escape preserved all five rows and the three-component model |
| Accepted structure drop/reorder | UNVERIFIED | One-stage and staged CUA drags entered `QDrag.exec` but produced no `dragEnterEvent`, `dragMoveEvent` or `dropEvent`; a later click ended the drag without a model change |
| Dock Quit and OS logout | NOT_RUN | Dock did not expose a usable target through the available desktop tool; application-menu Quit is not Dock or logout evidence |
| Physical trackpad inertia | NOT_RUN | Available scroll input produced `NoScrollPhase`; no physical momentum sequence was observed |

All test windows exited. No production change was needed for the verified paths.
The structured observer reports, screenshots, saved synthetic projects and JSONL
events are retained under the external evidence directory
`macos-followups-20261021-170318`. `quit-command/acceptance.json`,
`quit-dock/acceptance.json`, `quit-dock/wheel-unfocused.json` and
`tree-desktop/acceptance.json` identify each result; the `quit-dock` directory name
records the attempted task, while its report explicitly excludes Dock acceptance.

The unresolved drag observation does not establish a production-specific defect.
The controlled Qt lifecycle regressions and keyboard reordering have their own
coverage, and do not substitute for a completed native desktop drop. Repeat the
remaining paths with a physical device or a tool exposing the required native
input sequence before claiming them as accepted.
