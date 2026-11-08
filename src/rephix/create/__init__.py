"""Create Mode: build a base-compatible checkpoint from unknown-model evidence or a corpus.

See docs/CREATE_MODE.md (usage) and docs/CREATE_MODE_METHOD.md (method and claims).
"""

from .cli import add_subcommands

__all__ = ["add_subcommands"]
