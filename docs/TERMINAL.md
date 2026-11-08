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
and monochrome mode. macOS CI also verified PTY launch, exit, cooked line input and restoration. Full
manual macOS and Windows console exploration remains unverified.

The lifecycle PTY check exercises a cooked input line after exit before comparing
the complete terminal-mode snapshot. On macOS, the kernel sets the transient PENDIN
state when canonical input is restored and clears it on the next input; this is
[defined by XNU](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/tty.c).
The test checks actual line input and every mode field without masking that state.
