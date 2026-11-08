# Terminal workbench

Install the optional interface with `python -m pip install '.[tui]'` from the
checkout, then run `rephix ui`. The underlying commands still print JSON and keep
their existing command-line syntax. `rephix ui --plain` prints a linear command
catalog without requiring a terminal or the optional interface dependency.

The workbench is a quiet, borderless layout: a searchable command list on the
left, and the selected command's description, its argument field and output on the
right. A status marker in the header always pairs a symbol with its color:
`○ Ready`, `● Running · 4s`, `■ Stopping…`, `✓ Done`, `✗ Exit 2`.

| Key | Where | Action |
| --- | --- | --- |
| type | command list | filter by name or description; the best match is selected |
| ↑ ↓ / Ctrl+P Ctrl+N, PgUp PgDn | command list | choose a command |
| Enter | command list | open the command's argument field |
| Enter (or F5) | argument field | run the command |
| F1 | anywhere | show or hide the command's options (its `--help`) |
| Tab / Shift+Tab | anywhere | move between list, arguments and output |
| Esc | anywhere | close options, step back, or clear the filter |
| Ctrl+C | anywhere | stop a running command; quit when idle |

Below 80 columns the list and the command become two pages: Enter opens a command
and Esc returns to the list. Below 48×12 the workbench asks for a larger window.
Draft arguments are kept per command while you browse. Argument errors, such as an
unclosed quote, appear under the field instead of starting a run.

Managed installation changes require exiting the workbench first so its runtime
lease can protect the active environment. Run those commands through the stable
launcher described in [installation](INSTALLATION.md).

Enter or F5 captures the displayed arguments at the keypress, so subsequent edits affect
only a future run. It executes that captured command through the same Python installation as the
workbench. Arguments use shell-style quoting, but no shell is invoked: pipelines,
variable expansion, and shell substitutions are not supported. Quote paths with
spaces; use forward slashes for Windows paths. Read the arguments before running
commands that write files. Output includes errors and exit status. Scroll up in
the output to pause following; move to its end to follow again. The viewport retains
the last 200,000 characters; use a command's `--output` option for full reports.

Ctrl+C requests interruption while a command runs. Wait for its exit before
quitting; committed evidence stays available for the command's normal resume path.
On Windows, stopping terminates the child process, so unfinished work depends on
its durable checkpoint boundaries. On POSIX, Ctrl+Z suspends the interface and
restores terminal modes until resumed; a running child may continue. Ctrl+C exits
when idle. The list's filter and the argument field own printable keys. The
workbench does not capture the mouse, so terminal text selection keeps working.

The terminal keeps its own background. `--theme light` selects foreground tones for light backgrounds; the default suits dark ones.

`NO_COLOR=1` or `--color never` disables color and preserves a reverse-video
selection marker. `--color always` overrides `NO_COLOR`, with a portable ANSI palette if the
terminal library otherwise selects monochrome. No Nerd Font, image protocol,
animation, or network connection is required. For screen readers, redirected output,
or very small terminals, use the ordinary CLI or `ui --plain`.

The interface uses prompt_toolkit's buffered rendering, input parser, resize
handling and terminal restoration. An outer POSIX guard also restores the complete
mode snapshot around toolkit setup and execution. Verified locally in a real Linux PTY at 40, 80,
and 120 columns: launch, editing, execution, help, resize, exit, and exact restoration
of terminal modes. Automated input-routing tests also exercise help, command execution,
and monochrome mode. macOS CI also verified PTY launch, exit, cooked line input and
restoration. Windows is covered by the native console verification below.

The lifecycle PTY check exercises a cooked input line after exit before comparing
the complete terminal-mode snapshot. On macOS, the kernel sets the transient PENDIN
state when canonical input is restored and clears it on the next input; this is
[defined by XNU](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/tty.c).
The test checks actual line input and every mode field without masking that state.

## Windows console verification

`examples/windows_console_check.py` (workflow `windows console`) installs the wheel
into a managed prefix whose path contains spaces and non-ASCII characters. It then
drives the installed `rephix.exe` interactively, with nothing redirected, through two
independent console paths:

* **ConPTY:** a real pseudoconsole (pywinpty), the layer Windows Terminal uses. The
  VT stream is rendered into a screen model (pyte) that the checks read.
* **Classic console host:** a real conhost window (`CREATE_NEW_CONSOLE`). Keys are
  sent with `WriteConsoleInputW` and the screen is read with
  `ReadConsoleOutputCharacterW`. Ctrl+C in cooked mode uses
  `GenerateConsoleCtrlEvent`, and resizing moves the window as a user drag would.

Matrix: Windows Server 2022 (10.0.20348) and 2025 (10.0.26100), each with Python 3.10
and 3.13; shells cmd.exe, Windows PowerShell 5.1 and PowerShell 7.

| Area | Verified in both hosts unless noted |
| --- | --- |
| Launch | Fresh shell; `--version`; quoted output paths with spaces, `é`, `ü` and CJK; non-ASCII arguments round-trip; usage errors exit 2 without a traceback |
| Rendering | Unicode welcome with no raw escape codes; full-screen workbench; filter, F1 options, editing, Enter to run, `✓ Done` |
| Resize | Two-page layout at 60×24, "Terminal too small" at 40×10, full layout again at 110×30 (ConPTY on both releases; conhost window on Server 2022) |
| Interruption | Ctrl+C stops a running workbench command and its child process; Ctrl+C when idle exits to a working shell with echo and a visible cursor; Ctrl+C interrupts plain commands in cmd and PowerShell 7 with no "Terminate batch job" prompt and no orphan processes |
| Long runs and resume | `text-recover` with its oracle subprocess is interrupted, leaves no orphans, and resumes from its durable transcript |
| Concurrency | Two commands through the launcher at once; installation changes are refused while either runs |
| Redirection | `>` redirection, pipes to an ANSI code page, `ui --plain` in a PowerShell pipeline, and `ui` without a console explains `--plain` |
| Lifecycle | Update and uninstall are refused while the workbench runs, then update, prune, uninstall and reinstall succeed |

Not exercised (host limitations, not passes):

* Ctrl+C on a plain command in a **conhost window under PowerShell**: the synthetic
  `GenerateConsoleCtrlEvent` does not interrupt even a plain Python control program
  there. The same combination through ConPTY receives a real Ctrl+C keystroke and
  passes.
* Resizing a **conhost window on Server 2025**: the host ignores the window-size
  change. Resize handling on Server 2025 is verified through ConPTY.
* The Windows Terminal application itself was not driven; it hosts ConPTY.

Defects found and fixed by this verification:

* Ctrl+C through `rephix.cmd` produced cmd.exe's "Terminate batch job (Y/N)?" prompt.
  Use the native `rephix.exe` launcher; `rephix.cmd` remains for existing scripts and
  still behaves that way.
* A bare `rephix` piped to an ANSI code page crashed on the box-drawing wordmark.
* The colored welcome printed raw escape codes in classic console windows.
* The launcher's "shared" runtime lease was exclusive on Windows, so a second
  concurrent command failed.
