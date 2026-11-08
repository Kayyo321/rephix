# Install, update, recover, and remove Rephix

Rephix 0.4 supports ordinary Python virtual-environment installations and a managed
installation with transactional updates. Use a dedicated user-owned directory for
the managed installation. Administrative privileges are unnecessary. The installer
requires Python 3.10+ with `venv` and `ensurepip`; on Linux distributions that split
these out, install the distribution's Python venv package first.

## Install a published release

Every GitHub release publishes the wheel, the source distribution, the standalone
standard-library installer `rephix-installer.py`, `BUILD-INFO.json` (version and exact
commit) and `SHA256SUMS`. The wheel is pure Python (`py3-none-any`), so the same
artifact serves Linux, macOS and Windows on x86-64 and ARM64.

```bash
curl -LO https://github.com/Kayyo321/rephix/releases/download/v0.4.0/rephix-installer.py
curl -LO https://github.com/Kayyo321/rephix/releases/download/v0.4.0/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS   # macOS: shasum -a 256 -c --ignore-missing
python3 rephix-installer.py install --prefix "$HOME/.local/share/rephix-app" \
  --release latest --extras tui
"$HOME/.local/share/rephix-app/rephix" --version
```

On Windows PowerShell:

```powershell
$base = "https://github.com/Kayyo321/rephix/releases/download/v0.4.0"
Invoke-WebRequest "$base/rephix-installer.py" -OutFile rephix-installer.py
Invoke-WebRequest "$base/SHA256SUMS" -OutFile SHA256SUMS
(Get-FileHash rephix-installer.py -Algorithm SHA256).Hash.ToLower()   # compare with SHA256SUMS
py -3 rephix-installer.py install --prefix "$env:LOCALAPPDATA\Rephix" --release latest --extras tui
& "$env:LOCALAPPDATA\Rephix\rephix.exe" --version
```

`--release latest` asks the GitHub Releases API for the latest **stable** release (the
`/releases/latest` endpoint never returns drafts or prereleases). `--release vX.Y.Z`
selects that release by tag. A prerelease is installable only this way; `latest` never selects one, and drafts are never installable. The installer downloads the wheel, checks it
against the release's `SHA256SUMS` and, when the channel provides one, the asset
digest GitHub records server-side, and installs only on agreement. Any mismatch, unreachable channel, or
missing asset stops with an error before anything changes. `REPHIX_RELEASES_API`
overrides the channel URL for mirrors and tests. Provenance attestations can be
checked with `gh attestation verify rephix-0.4.0-py3-none-any.whl --repo Kayyo321/rephix`.

The managed launcher is `PREFIX/rephix` on POSIX and `PREFIX\rephix.exe` on Windows.
The `.exe` is pip's standard console launcher, so Ctrl+C reaches Rephix directly.
`PREFIX\rephix.cmd` remains for existing scripts, but `cmd.exe` asks "Terminate
batch job (Y/N)?" after Ctrl+C through a batch file, so prefer `rephix.exe`.
`python PREFIX/launch.py ...` is the cross-platform equivalent. Add PREFIX to PATH
yourself if you want: the installer never changes PATH, shell startup files or the
Windows user environment. The launcher uses the base Python that created the install;
keep it installed. Do not move the managed directory (virtual environments contain
absolute paths). Reinstall at a new prefix instead.

For core-only use, omit `--extras`. Available extras are `tui`, `hf`, `diffusion`,
`gguf`, and `analysis`. CPU is sufficient for local fixtures. For GPU use, prepare a
compatible Torch wheelhouse or constraints file according to the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The installer does not choose a GPU driver or install system software.

### From a local wheel or checkout

`--wheel PATH` installs a wheel you built or downloaded yourself, for example from a
checkout of a release tag:

```bash
python -m venv .build-env
.build-env/bin/python -m pip install build
.build-env/bin/python -m build
python src/rephix/installation.py install --prefix "$HOME/.local/share/rephix-app" \
  --wheel dist/rephix-0.4.0-py3-none-any.whl --extras tui,hf
```

The bootstrap itself uses only the standard library. It creates a fresh environment
at its final location, installs the wheel and selected dependencies, checks dependency
consistency, checks the installed version, and runs the affine recovery demo. Only
then does it atomically publish the active-environment pointer. It records the wheel
SHA-256, version and extras. Each environment includes `requirements.lock.txt`, a
`pip freeze --all` inventory; local wheel paths in it refer to the original install
location. Keep the original wheel and an offline wheelhouse for portable reproduction.

To pin dependencies, supply `--constraints /absolute/constraints.txt`. To prohibit
index access, also supply `--wheelhouse /absolute/wheels`. Prepare the wheelhouse
using `python -m pip download 'dist/rephix-0.4.0-py3-none-any.whl[tui,hf]' --dest wheels`.
Record its files and hashes if exact environment reproduction is required. Without
constraints or a wheelhouse, dependencies resolve within the release's declared
ranges, so repeated installations can select different compatible dependency builds.

## Update and rollback

Stop active model writers, then:

```bash
rephix installation check-update          # installed vs latest stable release
rephix installation update --release latest
rephix --version
rephix doctor
```

`update --wheel /absolute/rephix-X-py3-none-any.whl` installs a wheel you supply.
Updating to the release that is already active, with the same extras, is reported as
`already_current` and leaves the environment unchanged. It still refreshes the
launcher and manager, which repairs an interrupted refresh or a bootstrap left by an
older manager. Add `--force` to rebuild the environment anyway.
After a successful update, the launcher and manager files in PREFIX are refreshed from
the new version. An unchanged `rephix.cmd` is never rewritten, because `cmd.exe`
rereads a running batch file. A running `rephix.exe` is renamed aside and replaced,
never overwritten in place.

### Upgrading from 0.2 or 0.3

* **pip installation of 0.2.0 (the previous public release):**
  `python -m pip install --upgrade rephix-0.4.0-py3-none-any.whl` in the same
  environment, after verifying the wheel against `SHA256SUMS`. Existing evidence files
  and outputs need no migration.
* **Managed installation of 0.3.0:** the 0.3 manager has no release discovery. Either
  run the 0.4 installer against the existing prefix, which updates and adopts the 0.4
  manager:
  `python rephix-installer.py update --prefix PREFIX --release latest`,
  or download and verify the 0.4.0 wheel and run
  `PREFIX/rephix installation update --wheel /absolute/rephix-0.4.0-py3-none-any.whl`.
  The second route keeps the 0.3 manager until a later update through the 0.4
  installer. Both routes keep the manifest at schema 1, so 0.3 rollback still works.

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

Installation manifest schema 1 and session journal schema 1 are supported; 0.4 keeps
the 0.3 manifest format unchanged. Existing
layer evidence schema 1 is read and resumed in place; new layer journals use schema
2 with lossless compression. Bundle manifests remain schema 1. No migration of user
sessions, evidence or configuration is needed for 0.2/0.3-to-0.4 updates, and the installer
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
retry and reinstall. `--remove-bootstrap` also removes the launchers, manager and
manifest. On Windows a launcher that is still running (for example `rephix.exe`
running this very command) is renamed aside and deleted by a short helper once it
exits. When the command runs through `rephix.cmd`, that batch file is deleted after
the hosting `cmd.exe` exits, because cmd.exe rereads a running batch file. The report
lists both under `deferred_removal`. A persistent coordination lock remains to prevent concurrent operations
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
