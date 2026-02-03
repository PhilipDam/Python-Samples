r"""
Definition/Production-driven tokenizer (Python)
-----------------------------------------------

SPEC FORMAT (UTF-8 text)

Lines beginning with '#' or ';' are comments. Blank lines are ignored.
Sections are [definitions], [tokens], [skip]. Section names are case-insensitive.

[definitions]
NAME = <Python re pattern>      # Python 're' syntax (no surrounding / /)
IDENT = [A-Za-z_][A-Za-z0-9_]*
INT   = [0-9]+
WS    = \s+
COMMENT = //[^\r\n]*
# Multiline comments (non-nested) using inline DOTALL so '.' spans newlines:
MLCOMMENT = (?s:/\*.*?\*/)

[tokens]
Identifier = IDENT
Integer    = INT
Equals     = '='
Plus       = '+'
Minus      = '-'
Star       = '*'
Slash      = '/'
LParen     = '('
RParen     = ')'
Trivia     = WS | COMMENT | MLCOMMENT

[skip]
Trivia

RUN
----
python tokenizer.py spec.txt input.txt
# If input file is omitted, reads from stdin.

NOTES
-----
- Longest-match wins at each position. Ties: earlier declaration in [tokens].
- Literals use single quotes; double single-quotes escape a quote inside: 'can''t'
- Your spec file is plain text; write \s, \d, etc. normally there.
- For nested block comments, you'd need a small manual scanner (depth counter).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional, Set


# ------------------------------ Data types ------------------------------

@dataclass(frozen=True)
class SourceLocation:
    offset: int
    line: int
    column: int

    def __str__(self) -> str:
        return f"line {self.line}, col {self.column}"


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    start: SourceLocation
    end: SourceLocation

    def __str__(self) -> str:
        return f"{self.kind} \"{self.lexeme}\" @ {self.start}..{self.end}"


class SpecParseException(Exception):
    pass


class TokenizationException(Exception):
    def __init__(self, message: str, location: SourceLocation) -> None:
        super().__init__(f"{message} at {location}")
        self.location = location


@dataclass
class AltElement:
    is_literal: bool
    value: str

    def __str__(self) -> str:
        return f"'{self.value}'" if self.is_literal else self.value


@dataclass
class TokenProduction:
    token_name: str
    elements: List[AltElement]

    def __str__(self) -> str:
        return f"{self.token_name} = " + " | ".join(str(e) for e in self.elements)


@dataclass
class TokenSpec:
    definitions: Dict[str, str]
    productions: List[TokenProduction]
    skip: Set[str]


# ------------------------------ Spec parser ------------------------------

_SECTION_RE = re.compile(r"^\s*\[(?P<sec>[A-Za-z]+)\]\s*$")
_DEF_RE     = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rx>.+?)\s*$")
_TOKEN_RE   = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rhs>.+?)\s*$")
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_spec(spec_text: str) -> TokenSpec:
    class Section:
        NONE, DEF, TOK, SKIP = range(4)

    section = Section.NONE
    definitions: Dict[str, str] = {}
    productions: List[TokenProduction] = []
    skip: Set[str] = set()

    seen_defs: Set[str] = set()
    seen_toks: Set[str] = set()

    for idx, raw in enumerate(_iter_lines(spec_text), start=1):
        line = raw.strip()

        if not line or line.startswith("#") or line.startswith(";"):
            continue

        msec = _SECTION_RE.match(line)
        if msec:
            sec = msec.group("sec").lower()
            if   sec == "definitions": section = Section.DEF
            elif sec == "tokens":      section = Section.TOK
            elif sec == "skip":        section = Section.SKIP
            else:
                raise SpecParseException(f"Unknown section [{sec}] (line {idx})")
            continue

        if section == Section.DEF:
            m = _DEF_RE.match(raw)
            if not m:
                raise SpecParseException(f"Invalid definition syntax on line {idx}")
            name = m.group("name")
            rx   = m.group("rx").strip()
            if name.lower() in (n.lower() for n in seen_defs):
                raise SpecParseException(f"Duplicate definition '{name}' on line {idx}")
            try:
                re.compile(rx)
            except re.error as ex:
                raise SpecParseException(f"Invalid regex for '{name}' on line {idx}: {ex}") from ex
            definitions[name] = rx
            seen_defs.add(name)

        elif section == Section.TOK:
            m = _TOKEN_RE.match(raw)
            if not m:
                raise SpecParseException(f"Invalid token production syntax on line {idx}")
            name = m.group("name")
            if name.lower() in (n.lower() for n in seen_toks):
                raise SpecParseException(f"Duplicate token '{name}' on line {idx}")
            rhs = m.group("rhs")
            elements: List[AltElement] = []
            for alt in _split_alternatives(rhs):
                s = alt.strip()
                if not s:
                    continue
                lit = _as_quoted_literal(s)
                if lit is not None:
                    elements.append(AltElement(is_literal=True, value=lit))
                else:
                    if not _VALID_NAME.match(s):
                        raise SpecParseException(
                            f"Invalid reference '{s}' in token {name} (line {idx})")
                    elements.append(AltElement(is_literal=False, value=s))
            if not elements:
                raise SpecParseException(f"Token '{name}' has no alternatives (line {idx})")
            productions.append(TokenProduction(token_name=name, elements=elements))
            seen_toks.add(name)

        elif section == Section.SKIP:
            tok = line
            if not _VALID_NAME.match(tok):
                raise SpecParseException(f"Invalid token name '{tok}' in [skip] (line {idx})")
            skip.add(tok)

        else:
            raise SpecParseException(f"Content outside a section near line {idx}")

    for prod in productions:
        for e in prod.elements:
            if not e.is_literal and e.value not in definitions:
                raise SpecParseException(
                    f"Token '{prod.token_name}' references undefined definition '{e.value}'")

    return TokenSpec(definitions=definitions, productions=productions, skip=skip)


def _iter_lines(text: str) -> Iterable[str]:
    buf = io.StringIO(text)
    while True:
        line = buf.readline()
        if line == "":
            break
        yield line


def _as_quoted_literal(s: str) -> Optional[str]:
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        inner = s[1:-1].replace("''", "'")
        return inner
    return None


def _split_alternatives(rhs: str) -> Iterable[str]:
    out: List[str] = []
    sb: List[str] = []
    in_quote = False
    i = 0
    while i < len(rhs):
        c = rhs[i]
        if c == "'":
            if in_quote and i + 1 < len(rhs) and rhs[i + 1] == "'":
                sb.append("''")
                i += 2
                continue
            in_quote = not in_quote
            sb.append(c)
            i += 1
        elif c == "|" and not in_quote:
            out.append("".join(sb))
            sb.clear()
            i += 1
        else:
            sb.append(c)
            i += 1
    if sb:
        out.append("".join(sb))
    return out


# ------------------------------ Tokenizer ------------------------------

class Tokenizer:
    r"""
    Compiles a per-token regex (OR of alts) and enforces longest-match at each position.
    Tie-breaker: token declaration order.
    """

    def __init__(self, spec: TokenSpec, flags: int = re.ASCII) -> None:
        r"""
        flags default to re.ASCII (stable ASCII semantics for \w, \d, etc.).
        Use 0 (no flags) for full Unicode if your definitions expect it.
        """
        self._skip = {name.lower() for name in spec.skip}
        self._compiled: List[Tuple[str, re.Pattern[str]]] = []
        self._flags = flags

        for prod in spec.productions:
            alts: List[str] = []
            for e in prod.elements:
                if e.is_literal:
                    alts.append(re.escape(e.value))
                else:
                    alts.append(f"(?:{spec.definitions[e.value]})")
            # IMPORTANT FIX: no '^' anchor — we rely on .match(text, pos) to start at offset
            pattern = r"(?:" + "|".join(alts) + r")"
            try:
                rx = re.compile(pattern, flags=self._flags)
            except re.error as ex:
                raise SpecParseException(
                    f"Failed to compile token '{prod.token_name}': {ex}") from ex
            self._compiled.append((prod.token_name, rx))

    def tokenize(self, text: str) -> Iterable[Token]:
        offset = 0
        line = 1
        col = 1
        n = len(text)

        while offset < n:
            best_name: Optional[str] = None
            best_match: Optional[re.Match[str]] = None

            # Try all tokens at this position; choose the longest match
            for name, rx in self._compiled:
                m = rx.match(text, offset)
                if m:
                    if best_match is None or len(m.group(0)) > len(best_match.group(0)):
                        best_match = m
                        best_name = name

            if best_match is None or best_name is None:
                loc = SourceLocation(offset, line, col)
                preview = text[offset: offset + 20].replace("\r", "\\r").replace("\n", "\\n")
                raise TokenizationException(f"No token matches starting at '{preview}…'", loc)

            lexeme = best_match.group(0)
            start = SourceLocation(offset, line, col)

            # advance
            line, col = _advance_position(lexeme, line, col)
            offset += len(lexeme)
            end = SourceLocation(offset, line, col)

            if best_name.lower() not in self._skip:
                yield Token(kind=best_name, lexeme=lexeme, start=start, end=end)


def _advance_position(fragment: str, line: int, col: int) -> Tuple[int, int]:
    i = 0
    L = len(fragment)
    while i < L:
        c = fragment[i]
        if c == "\r":
            if i + 1 < L and fragment[i + 1] == "\n":
                i += 2
            else:
                i += 1
            line += 1
            col = 1
        elif c == "\n":
            line += 1
            col = 1
            i += 1
        else:
            col += 1
            i += 1
    return line, col


# ------------------------------ CLI / Demo ------------------------------

def _escape_newlines(s: str) -> str:
    return s.replace("\r", "\\r").replace("\n", "\\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Definition/production-driven tokenizer")
    p.add_argument("spec", help="Path to spec file")
    p.add_argument("input", nargs="?", help="Path to input file (default: stdin)")
    p.add_argument("--unicode", action="store_true",
                   help="Use full Unicode regex behavior (disable re.ASCII)")
    args = p.parse_args(argv)

    try:
        with open(args.spec, "r", encoding="utf-8") as f:
            spec_text = f.read()
        spec = parse_spec(spec_text)

        flags = 0 if args.unicode else re.ASCII
        tokenizer = Tokenizer(spec, flags=flags)

        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                input_text = f.read()
        else:
            input_text = sys.stdin.read()

        for tok in tokenizer.tokenize(input_text):
            print(f"{tok.kind}\t{_escape_newlines(tok.lexeme)}\t@ {tok.start}")

        return 0

    except SpecParseException as ex:
        print(f"SPEC ERROR: {ex}", file=sys.stderr)
        return 1
    except TokenizationException as ex:
        print(f"LEX ERROR: {ex}", file=sys.stderr)
        return 3
    except Exception as ex:  # unexpected
        print(f"FATAL: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    sys.exit(main())
