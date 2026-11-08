# Install, update, recover, and remove Rephix

Rephix 0.3 supports ordinary Python virtual-environment installations and a managed
installation with transactional updates. Use a dedicated user-owned directory for
the managed installation. Administrative privileges are unnecessary. The bootstrap
requires Python with `venv` and `ensurepip`; on Linux distributions that split these
out, install the distribution's Python venv package first.

## Build and install

From a checkout of the desired release tag or commit:

```bash
python -m venv .build-env
# POSIX; on Windows use .build-env\Scripts\python.exe below
.build-env/bin/python -m pip install build
.build-env/bin/python -m build
python src/rephix/installation.py install \
  --prefix "$HOME/.local/share/rephix-app" \
  --wheel dist/rephix-0.3.0-py3-none-any.whl --extras tui,hf
"$HOME/.local/share/rephix-app/rephix" --version
"$HOME/.local/share/rephix-app/rephix" doctor --require-hf
"$HOME/.local/share/rephix-app/rephix" ui
```

For core-only affine recovery, omit `--extras`. Available extras are `tui`, `hf`,
`diffusion`, `gguf`, and `analysis`. CPU is sufficient for local fixtures; model
operations need enough RAM for source, destination, and probe temporaries.
For GPU use, prepare a compatible Torch wheelhouse or constraints file according
to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The installer doesn't choose a GPU driver or install system software.

On Windows PowerShell the equivalent bootstrap is:

```powershell
python -m venv .build-env
.build-env\Scripts\python.exe -m pip install build
.build-env\Scripts\python.exe -m build
python src\rephix\installation.py install `
  --prefix "$env:LOCALAPPDATA\Rephix" `
  --wheel dist\rephix-0.3.0-py3-none-any.whl --extras tui,hf
& "$env:LOCALAPPDATA\Rephix\rephix.cmd" --version
```

The stable launcher is `PREFIX/rephix` on POSIX or `PREFIX\rephix.cmd` on Windows.
You can add PREFIX to PATH, or invoke it by absolute path. It uses the base Python
that created the install; keep that Python installed. The cross-platform equivalent
is `python PREFIX/launch.py ...`. Do not move the managed directory: virtual
environments contain absolute paths. Reinstall at a new prefix instead.

The bootstrap itself uses only the standard library. It creates a fresh environment
at its final location, installs the wheel and selected dependencies, checks dependency
consistency, checks the installed version, and runs the affine recovery demo. Only
then does it atomically publish the active-environment pointer. It records the wheel
SHA-256, version and extras. Each environment includes `requirements.lock.txt`, a
`pip freeze --all` inventory; local wheel paths in it refer to the original install
location. Keep the original wheel and an offline wheelhouse for portable reproduction.

To pin dependencies, supply `--constraints /absolute/constraints.txt`. To prohibit
index access, also supply `--wheelhouse /absolute/wheels`. Prepare the wheelhouse
using `python -m pip download 'dist/rephix-0.3.0-py3-none-any.whl[tui,hf]' --dest wheels`.
Record its files and hashes if exact environment reproduction is required. Without
constraints or a wheelhouse, dependencies resolve within the release's declared
ranges, so repeated installations can select different compatible dependency builds.

## Update and rollback

Build the next release wheel outside PREFIX, stop active model writers, and run:

```bash
rephix installation inspect --verify
rephix installation update --wheel /absolute/releases/rephix-NEXT-py3-none-any.whl
rephix --version
rephix doctor
```

Use the managed launcher's full path if it isn't on PATH. Update inherits the active
installation's extras unless `--extras` is supplied; `--extras ''` selects core only.
It builds and validates a new environment without modifying the old environment or
user directories. Dependency, installation, or smoke-test failure leaves the active
version unchanged. Ctrl+C cleans up a partial environment where possible. A forced
process kill can leave a recorded `preparing` slot; inspect and prune it after the
writer exits. The installation lock is released by the OS on process exit.

The previous environment remains available:

```bash
rephix installation rollback
rephix installation prune
rephix installation prune --apply
```

Rollback validates the previous environment before switching. Prune previews or
removes older and interrupted environments, retaining current and previous versions.
Updates don't delete historical environments automatically. Prune after verification
to reclaim disk space. Stop work using an old version before pruning it. The stable launcher holds a runtime lease while commands run. Installation changes
are rejected while those commands (including the workbench) are active; exit them
and retry. Processes started by bypassing the launcher are not tracked, so stop
those explicitly before updating or removing software. Updates do not migrate
running processes.

If an installation must be managed from another Python environment, run
`python src/rephix/installation.py inspect --prefix PREFIX --verify` and the analogous
update/rollback/uninstall commands. Removing the environment that runs the manager
is rejected before deactivation; use the stable launcher or external bootstrap.

## Data formats and migration

Installation manifest schema 1 and session journal schema 1 are supported. Existing
layer evidence schema 1 is read and resumed in place; new layer journals use schema
2 with lossless compression. Bundle manifests remain schema 1. No migration of user
sessions, evidence or configuration is needed for 0.2-to-0.3 updates, and the installer
does not open or rewrite those files. Unsupported future schemas fail explicitly.

Application rollback switches software and dependencies; it doesn't reverse user
work. A pre-schema-2 application cannot read new schema-2 layer evidence. Before
using a newer release to write important state, stop its writer and copy the complete
session/evidence directory to a separate backup if you need backward rollback. Keep
that backup until the update is accepted. Never copy just a live SQLite database
while its writer or write-ahead log is active. There is no hidden configuration file
or global cache migration; experiment paths remain explicit command arguments.

## Uninstall with explicit retention choices

```bash
rephix installation uninstall
```

This removes the managed Python environments and their installed dependencies.
It retains PREFIX/data, PREFIX/config, PREFIX/evidence and PREFIX/checkpoints by
default. Reinstalling through the retained launcher preserves them. These directories
are optional organizational locations; existing commands still use their explicit
input/output paths and do not relocate your artifacts into PREFIX.

Choose any combination when you intentionally want those managed directories removed:

```bash
rephix installation uninstall --purge-evidence --purge-checkpoints
rephix installation uninstall --purge-data --purge-config --purge-evidence --purge-checkpoints --remove-bootstrap
```

Each flag removes only its named directory inside PREFIX. Symlink purge targets and
symlink environment roots are rejected. Files elsewhere, shared Hugging Face caches,
other Python installations, and unrelated files inside PREFIX are retained. Software
removal disables new launches before deleting environments; rerun uninstall after
an interruption or filesystem error. User directories remain separate from software.
By default the small bootstrap, manifest and coordination lock remain for inspection,
retry and reinstall. `--remove-bootstrap` also removes the launcher, manager and
manifest. A persistent coordination lock remains to prevent concurrent operations
from opening different lock files. After reviewing retained files and stopping all
operations, manually remove the dedicated prefix to remove that final record. If
you removed the bootstrap while retaining data, install into a new prefix and use
the retained artifacts by their explicit paths. No PATH or shell configuration is changed.

## Ordinary pip installations

For an existing environment, use `python -m pip install --upgrade /path/rephix.whl`
and `python -m pip uninstall rephix`. Pip uninstalls the package; dependencies and user
artifacts remain. To remove its dependencies too, delete only the dedicated venv
after deactivating it. Pip updates are in-place and do not provide the managed
update's activation/rollback guarantees. A pip installation can create a separate
managed installation using `rephix installation install --prefix ... --wheel ...`.

## Help and diagnosis

`rephix --help`, `rephix COMMAND --help`, `rephix --version`, `rephix doctor`, and
`rephix installation inspect --verify` provide the supported command and environment
checks. `doctor --workspace PATH` checks free space on the relevant filesystem;
`doctor --model LOCAL_PATH` reads a local model configuration without downloading.
A missing optional dependency reports the extra to install. For damaged managed
software, run the external standard-library bootstrap to inspect or reinstall.
See [supported environments and limitations](SUPPORT.md).
