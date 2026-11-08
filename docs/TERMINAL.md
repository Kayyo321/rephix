# Terminal workbench

Install the optional interface with `python -m pip install '.[tui]'` from the
checkout, then run `rephix ui`. The underlying commands still print JSON and keep
their existing command-line syntax. `rephix ui --plain` prints a linear command
catalog without requiring a terminal or the optional interface dependency.

The workbench uses a restrained dark palette, a command list beside output on wide
screens, and a single output pane on smaller screens. Command arguments remain
editable at every size. At 90 columns or wider, use Tab to focus the command list,
arrows or j/k to select, and Enter to edit. Draft arguments are preserved when moving
between commands. At narrower sizes, type a command or use the plain catalog to
find one. F1 opens help for the command in the editor; Escape closes help.

Managed installation changes require exiting the workbench first so its runtime
lease can protect the active environment. Run those commands through the stable
launcher described in [installation](INSTALLATION.md).

F5 captures the displayed arguments at the keypress, so subsequent edits affect
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
when idle; q exits from the command list. Text input owns printable keys, including
q and j. The workbench does not capture the mouse.

`--theme light` selects the light palette; the default is dark.

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
