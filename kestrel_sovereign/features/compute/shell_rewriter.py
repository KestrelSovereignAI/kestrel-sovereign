"""Fail-closed shell parsing and safe-delete command generation.

The compute feature must never decide that a pathname is safe at rewrite time
and then emit a real ``rm`` for execution later.  Filesystem topology can
change between those moments.  This module therefore uses Tree-sitter's Bash
grammar to identify command nodes and replaces every statically-provable
``rm`` invocation with POSIX-shell code. Authorized temporary/agent-owned
paths are revalidated before a real delete; every other operand moves into a
unique, exclusively-created trash directory at runtime.

Commands whose identity is dynamic, nested deletion in substitutions, and
utilities that execute another command are deliberately rejected.  A rejected
script is safer than a partially-understood script that can permanently delete
host data.
"""

import re
import shlex
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import tree_sitter_bash
from tree_sitter import Language, Node, Parser


class ShellRewriteError(ValueError):
    """Raised when a shell deletion cannot be classified without ambiguity."""


# These commands interpret one or more arguments as another command or mutate
# command lookup.  Supporting them safely requires recursively governing their
# own language/configuration, so the compute shell subset rejects them.
_COMMAND_DISPATCHERS = frozenset(
    {
        ".",
        "alias",
        "bash",
        "builtin",
        "busybox",
        "chroot",
        "command",
        "dash",
        "doas",
        "enable",
        "env",
        "eval",
        "exec",
        "find",
        "ksh",
        "nice",
        "nohup",
        "parallel",
        "setsid",
        "sh",
        "source",
        "stdbuf",
        "sudo",
        "time",
        "timeout",
        "trap",
        "xargs",
        "zsh",
    }
)

_DYNAMIC_WORD_NODES = frozenset(
    {
        "ansi_c_string",
        "arithmetic_expansion",
        "command_substitution",
        "expansion",
        "process_substitution",
        "simple_expansion",
    }
)

_SUPPORTED_SHORT_OPTIONS = frozenset("rfivIR")
_SUPPORTED_LONG_OPTIONS = frozenset(
    {
        "--dir",
        "--force",
        "--no-preserve-root",
        "--one-file-system",
        "--preserve-root",
        "--recursive",
        "--verbose",
    }
)
_TRASH_SAFE_SHORT_OPTIONS = frozenset("fRr")
_TRASH_SAFE_LONG_OPTIONS = frozenset({"--force", "--recursive"})
_RM_WORD_PATTERN = re.compile(r"(?<![A-Za-z0-9_])rm(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class _RemoveArguments:
    targets: tuple[str, ...]
    options: tuple[str, ...]
    force: bool
    recursive: bool
    trash_incompatible_options: tuple[str, ...]


def _iter_nodes(
    node: Node,
    ancestors: tuple[str, ...] = (),
) -> Iterator[tuple[Node, tuple[str, ...]]]:
    yield node, ancestors
    child_ancestors = (*ancestors, node.type)
    for child in node.named_children:
        yield from _iter_nodes(child, child_ancestors)


def _contains_node_type(node: Node, node_types: frozenset[str]) -> bool:
    return any(descendant.type in node_types for descendant, _ in _iter_nodes(node))


def _is_rm_command(command: str) -> bool:
    """Return whether a static shell command word names ``rm`` directly."""
    return PurePosixPath(command).name == "rm"


class ShellScriptRewriter:
    """Rewrite the supported Bash/POSIX-shell subset without raw deletion."""

    def __init__(
        self,
        *,
        trash_dir: str | Path,
        workdir: str | None = None,
        assert_delete_allowed: Callable[[str], None] | None = None,
        assert_command_allowed: Callable[[str], None] | None = None,
        direct_delete_root: Callable[[str], str | None] | None = None,
        current_agent_data_root: str | Path | None = None,
    ) -> None:
        self._trash_dir = str(trash_dir)
        self._workdir = workdir
        self._assert_delete_allowed = assert_delete_allowed
        self._assert_command_allowed = assert_command_allowed
        self._direct_delete_root = direct_delete_root
        self._current_agent_data_root = (
            str(current_agent_data_root) if current_agent_data_root else None
        )

        language = Language(tree_sitter_bash.language())
        self._parser = Parser(language)

    def rewrite(self, content: str) -> str:
        """Return ``content`` with every supported ``rm`` command softened.

        Tree-sitter intentionally treats a backslash-newline as trivia between
        tokens, while the shell removes it before tokenization.  Rather than
        maintain a second offset map, this valid-but-ambiguous spelling is an
        explicit unsupported construct and fails closed.
        """
        if "\\\n" in content or "\\\r\n" in content:
            raise ShellRewriteError(
                "Shell line continuations are unsupported by safe-delete rewriting"
            )

        source = content.encode("utf-8")
        tree = self._parser.parse(source)
        if tree.root_node.has_error:
            if not _RM_WORD_PATTERN.search(content):
                return content
            raise ShellRewriteError(
                "Refusing to rewrite syntactically invalid shell code"
            )

        for node, _ in _iter_nodes(tree.root_node):
            if node.type != "heredoc_body":
                continue
            body = source[node.start_byte : node.end_byte].decode("utf-8")
            if _RM_WORD_PATTERN.search(body):
                raise ShellRewriteError(
                    "Refusing to classify rm text inside a here-document"
                )

        replacements: list[tuple[int, int, bytes]] = []
        for node, ancestors in _iter_nodes(tree.root_node):
            if node.type == "function_definition":
                self._reject_command_override(node, source)
            if node.type == "file_redirect":
                if self._assert_command_allowed is not None:
                    self._assert_command_allowed(
                        source[node.start_byte : node.end_byte].decode("utf-8")
                    )
                continue
            if node.type != "command":
                continue

            command_name = node.child_by_field_name("name")
            if command_name is None:
                continue
            command = self._static_word(command_name, source, role="command")
            command_basename = PurePosixPath(command).name

            if (
                command_basename in _COMMAND_DISPATCHERS
                and self._dispatcher_may_hide_rm(
                    node,
                    source,
                )
            ):
                raise ShellRewriteError(
                    f"Unsupported command-dispatching shell utility: {command}"
                )

            if _is_rm_command(command):
                if (
                    "command_substitution" in ancestors
                    or "process_substitution" in ancestors
                ):
                    raise ShellRewriteError(
                        "rm inside command/process substitution is unsupported"
                    )
                if "redirected_statement" in ancestors:
                    raise ShellRewriteError(
                        "Redirection on rm is unsupported by safe-delete rewriting"
                    )
                if any(
                    child.type == "variable_assignment" for child in node.named_children
                ):
                    raise ShellRewriteError(
                        "Environment assignments preceding rm are unsupported"
                    )
                replacement = self._rewrite_rm_node(node, source)
                replacements.append(
                    (node.start_byte, node.end_byte, replacement.encode())
                )
                continue

            if self._assert_command_allowed is not None:
                self._assert_command_allowed(
                    source[node.start_byte : node.end_byte].decode("utf-8")
                )

        rewritten = source
        for start, end, replacement in sorted(replacements, reverse=True):
            rewritten = rewritten[:start] + replacement + rewritten[end:]
        return rewritten.decode("utf-8")

    def _reject_command_override(self, node: Node, source: bytes) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = self._static_word(name_node, source, role="function name")
        if name == "command":
            raise ShellRewriteError(
                "Refusing a function that overrides the POSIX command builtin"
            )

    def _dispatcher_may_hide_rm(self, node: Node, source: bytes) -> bool:
        """Return whether a wrapper argument is dynamic or names rm."""
        for argument in node.children_by_field_name("argument"):
            if _contains_node_type(argument, _DYNAMIC_WORD_NODES):
                return True
            value = self._static_word(argument, source, role="argument")
            if _RM_WORD_PATTERN.search(value):
                return True
        return False

    def _rewrite_rm_node(self, node: Node, source: bytes) -> str:
        arguments = tuple(
            self._static_word(argument, source, role="rm argument")
            for argument in node.children_by_field_name("argument")
        )
        parsed = self._parse_rm_arguments(arguments)

        if self._assert_delete_allowed is not None:
            for target in parsed.targets:
                self._assert_delete_allowed(target)

        direct_roots = tuple(
            self._direct_delete_root(target)
            if self._direct_delete_root is not None
            else None
            for target in parsed.targets
        )
        if all(root is not None for root in direct_roots):
            return self._direct_delete_operation(
                parsed,
                tuple(root for root in direct_roots if root is not None),
            )

        if parsed.trash_incompatible_options:
            options = ", ".join(parsed.trash_incompatible_options)
            raise ShellRewriteError(
                f"rm option(s) cannot be preserved by trash-backed deletion: {options}"
            )

        return self._trash_delete_operation(parsed)

    def _direct_delete_operation(
        self,
        parsed: _RemoveArguments,
        direct_roots: tuple[str, ...],
    ) -> str:
        """Emit a guarded real rm for paths authorized for true deletion."""
        status = f"_kestrel_rm_status_{uuid4().hex}"
        lines = ["(", f"  {status}=0;", "  set --;"]

        for target, expected_root in zip(
            parsed.targets,
            direct_roots,
            strict=True,
        ):
            item_status = f"_kestrel_rm_item_{uuid4().hex}"
            resolved_var = f"_kestrel_rm_resolved_{uuid4().hex}"
            root_var = f"_kestrel_rm_root_{uuid4().hex}"
            lines.extend(
                self._direct_target_guard(
                    target=target,
                    expected_root=expected_root,
                    force=parsed.force,
                    item_status=item_status,
                    overall_status=status,
                    resolved_var=resolved_var,
                    root_var=root_var,
                )
            )

        quoted_options = " ".join(shlex.quote(option) for option in parsed.options)
        rm_command = "command -p rm"
        if quoted_options:
            rm_command += f" {quoted_options}"
        rm_command += ' -- "$@"'
        lines.extend(
            [
                '  if command -p test "$#" -gt 0; then',
                f"    {rm_command} || {status}=$?;",
                "  fi;",
                f'  exit "${{{status}}}";',
                ")",
            ]
        )
        return "\n".join(lines)

    def _direct_target_guard(
        self,
        *,
        target: str,
        expected_root: str,
        force: bool,
        item_status: str,
        overall_status: str,
        resolved_var: str,
        root_var: str,
    ) -> list[str]:
        quoted_target = shlex.quote(target)
        quoted_root = shlex.quote(expected_root)
        refusal = shlex.quote(
            f"Refusing rm path whose runtime ownership changed: {target}"
        )
        lines = [
            f"  {item_status}=0;",
            (
                f"  if command -p test -e {quoted_target} || "
                f"command -p test -L {quoted_target}; then"
            ),
            f"    if command -p test -L {quoted_target}; then",
            f"      command -p printf '%s\\n' {refusal} >&2;",
            f"      {item_status}=1;",
            "    else",
            (
                f"      {resolved_var}=$(command -p realpath -- "
                f"{quoted_target}) || {item_status}=1;"
            ),
            (
                f"      {root_var}=$(command -p realpath -- "
                f"{quoted_root}) || {item_status}=1;"
            ),
            f'      if command -p test "${{{item_status}}}" -eq 0; then',
            f'        if command -p test "${{{root_var}}}" != {quoted_root}; then',
            f"          command -p printf '%s\\n' {refusal} >&2;",
            f"          {item_status}=1;",
            "        else",
            f'          case "${{{resolved_var}}}" in',
            f'            "${{{root_var}}}"|"${{{root_var}}}"/*) ;;',
            "            *)",
            f"              command -p printf '%s\\n' {refusal} >&2;",
            f"              {item_status}=1;",
            "              ;;",
            "          esac;",
            "        fi;",
            "      fi;",
            f'      if command -p test "${{{item_status}}}" -eq 0; then',
            f'        set -- "$@" "${{{resolved_var}}}";',
            "      fi;",
            "    fi;",
            "  else",
        ]
        if force:
            lines.append("    :;")
        else:
            lines.extend(
                [
                    (
                        "    command -p printf '%s\\n' "
                        + shlex.quote(f"rm target does not exist: {target}")
                        + " >&2;"
                    ),
                    f"    {item_status}=1;",
                ]
            )
        lines.extend(
            [
                "  fi;",
                f'  command -p test "${{{item_status}}}" -eq 0 || {overall_status}=1;',
            ]
        )
        return lines

    def _trash_delete_operation(self, parsed: _RemoveArguments) -> str:
        """Emit independent runtime-unique trash moves for every operand."""
        operation_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        overall_status = f"_kestrel_rm_status_{uuid4().hex}"
        lines = ["(", f"  {overall_status}=0;"]

        for target in parsed.targets:
            item_status = f"_kestrel_rm_item_{uuid4().hex}"
            resolved_var = f"_kestrel_rm_resolved_{uuid4().hex}"
            source_var = f"_kestrel_rm_source_{uuid4().hex}"
            subdir_var = f"_kestrel_rm_trash_{uuid4().hex}"
            lines.extend(
                self._trash_target_operation(
                    target=target,
                    operation_prefix=operation_prefix,
                    force=parsed.force,
                    recursive=parsed.recursive,
                    item_status=item_status,
                    overall_status=overall_status,
                    resolved_var=resolved_var,
                    source_var=source_var,
                    subdir_var=subdir_var,
                )
            )

        lines.extend(
            [
                f'  exit "${{{overall_status}}}";',
                ")",
            ]
        )
        return "\n".join(lines)

    def _trash_target_operation(
        self,
        *,
        target: str,
        operation_prefix: str,
        force: bool,
        recursive: bool,
        item_status: str,
        overall_status: str,
        resolved_var: str,
        source_var: str,
        subdir_var: str,
    ) -> list[str]:
        quoted_target = shlex.quote(target)
        quoted_root = shlex.quote(self._trash_dir)
        template = str(Path(self._trash_dir) / f"{operation_prefix}_XXXXXXXX")
        quoted_template = shlex.quote(template)
        lines = [
            f"  {item_status}=0;",
            (
                f"  if command -p test -e {quoted_target} || "
                f"command -p test -L {quoted_target}; then"
            ),
            (
                f"    {resolved_var}=$(command -p realpath -- "
                f"{quoted_target}) || {item_status}=1;"
            ),
            f'    {source_var}="${{{resolved_var}}}";',
            f"    if command -p test -L {quoted_target}; then",
            f"      {source_var}={quoted_target};",
            "    fi;",
        ]

        lines.extend(self._runtime_agent_data_guard(target, resolved_var, item_status))

        if not recursive:
            lines.extend(
                [
                    f'    if command -p test "${{{item_status}}}" -eq 0 && '
                    f"command -p test -d {quoted_target} && "
                    f"! command -p test -L {quoted_target}; then",
                    (
                        "      command -p printf '%s\\n' "
                        + shlex.quote(
                            f"Refusing to remove directory without -r: {target}"
                        )
                        + " >&2;"
                    ),
                    f"      {item_status}=1;",
                    "    fi;",
                ]
            )

        lines.extend(
            [
                f'    if command -p test "${{{item_status}}}" -eq 0; then',
                (
                    f"      command -p mkdir -p -- {quoted_root} && "
                    f"{subdir_var}=$(command -p mktemp -d {quoted_template}) "
                    f"|| {item_status}=1;"
                ),
                f'      if command -p test "${{{item_status}}}" -eq 0; then',
                (
                    f'        command -p mv -- "${{{source_var}}}" '
                    f'"${{{subdir_var}}}/" || {item_status}=1;'
                ),
                f'        if command -p test "${{{item_status}}}" -ne 0; then',
                f'          command -p rmdir -- "${{{subdir_var}}}" 2>/dev/null || :;',
                "        fi;",
                "      fi;",
                "    fi;",
                "  else",
            ]
        )
        if force:
            lines.append("    :;")
        else:
            lines.extend(
                [
                    (
                        "    command -p printf '%s\\n' "
                        + shlex.quote(f"rm target does not exist: {target}")
                        + " >&2;"
                    ),
                    f"    {item_status}=1;",
                ]
            )
        lines.extend(
            [
                "  fi;",
                (
                    f'  command -p test "${{{item_status}}}" -eq 0 '
                    f"|| {overall_status}=1;"
                ),
            ]
        )
        return lines

    def _runtime_agent_data_guard(
        self,
        target: str,
        resolved_var: str,
        item_status: str,
    ) -> list[str]:
        refusal = shlex.quote(
            f"Refusing to delete another agent's runtime data: {target}"
        )
        lines = [f'    if command -p test "${{{item_status}}}" -eq 0; then']
        if self._current_agent_data_root:
            quoted_own = shlex.quote(self._current_agent_data_root)
            lines.extend(
                [
                    f'      case "${{{resolved_var}}}" in',
                    f"        {quoted_own}|{quoted_own}/*) ;;",
                    "        */agent_data|*/agent_data/*)",
                    f"          command -p printf '%s\\n' {refusal} >&2;",
                    f"          {item_status}=1;",
                    "          ;;",
                    "      esac;",
                ]
            )
        else:
            lines.extend(
                [
                    f'      case "${{{resolved_var}}}" in',
                    "        */agent_data|*/agent_data/*)",
                    f"          command -p printf '%s\\n' {refusal} >&2;",
                    f"          {item_status}=1;",
                    "          ;;",
                    "      esac;",
                ]
            )
        lines.append("    fi;")
        return lines

    def _static_word(self, node: Node, source: bytes, *, role: str) -> str:
        if _contains_node_type(node, _DYNAMIC_WORD_NODES):
            raise ShellRewriteError(f"Dynamic {role} is unsupported")

        fragment = source[node.start_byte : node.end_byte].decode("utf-8")
        try:
            words = shlex.split(fragment, comments=False, posix=True)
        except ValueError as exc:
            raise ShellRewriteError(f"Malformed {role}: {fragment!r}") from exc
        if len(words) != 1 or not words[0]:
            raise ShellRewriteError(f"Non-static {role}: {fragment!r}")

        value = words[0]
        if role in {"command", "function name"} and any(
            character in value for character in "*?[]{}~"
        ):
            raise ShellRewriteError(f"Expanded {role} is unsupported: {fragment!r}")
        return value

    def _parse_rm_arguments(self, arguments: tuple[str, ...]) -> _RemoveArguments:
        targets: list[str] = []
        options: list[str] = []
        trash_incompatible: list[str] = []
        force = False
        recursive = False
        options_ended = False

        for argument in arguments:
            if not options_ended and argument == "--":
                options_ended = True
                continue

            if not options_ended and argument.startswith("--"):
                option, separator, value = argument.partition("=")
                if option == "--interactive" and (
                    not separator or value in {"always", "never", "once"}
                ):
                    options.append(argument)
                    trash_incompatible.append(argument)
                    continue
                if argument not in _SUPPORTED_LONG_OPTIONS:
                    raise ShellRewriteError(
                        f"Unsupported rm option (fail-closed): {argument}"
                    )
                options.append(argument)
                force = force or argument == "--force"
                recursive = recursive or argument == "--recursive"
                if argument not in _TRASH_SAFE_LONG_OPTIONS:
                    trash_incompatible.append(argument)
                continue

            if not options_ended and argument.startswith("-") and argument != "-":
                option_characters = set(argument[1:])
                unsupported = option_characters - _SUPPORTED_SHORT_OPTIONS
                if unsupported:
                    unsupported_options = "".join(sorted(unsupported))
                    raise ShellRewriteError(
                        "Unsupported rm option(s) (fail-closed): "
                        f"-{unsupported_options}"
                    )
                options.append(argument)
                force = force or "f" in option_characters
                recursive = recursive or bool(option_characters & {"r", "R"})
                incompatible_characters = option_characters - _TRASH_SAFE_SHORT_OPTIONS
                if incompatible_characters:
                    trash_incompatible.append(
                        "-" + "".join(sorted(incompatible_characters))
                    )
                continue

            if any(character in argument for character in "*?[]{}~"):
                raise ShellRewriteError(
                    f"Expanded or globbed rm target is unsupported: {argument!r}"
                )
            targets.append(argument)

        if not targets:
            raise ShellRewriteError("Refusing to rewrite rm without a target")
        return _RemoveArguments(
            tuple(targets),
            tuple(options),
            force,
            recursive,
            tuple(trash_incompatible),
        )
