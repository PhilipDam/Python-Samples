# regex_automata_engine.py
#
# Extension requested:
#   - Dump NFA states/transitions as a readable table
#   - Dump DFA transition structure as a readable table
#   - Dump Minimal-DFA transition structure as a readable table
#
# Tables are printed in a “compact but still tabular” way:
#   - NFA: one row per transition (and eps list shown per state)
#   - DFA / MinDFA: one row per (state, destination) group with symbols aggregated
#
# Why not print a gigantic matrix (states × all 130 symbols)?
#   - Domain can be 128 + BOF + EOF = 130 columns, which becomes unreadable fast.
#   - Grouping symbols by destination gives a real “table” that humans can digest.
#
# Constraints preserved:
#   - No capturing; no lookaround
#   - No multiline comments/docstrings: only single-line '#' comments

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Set, Deque, Iterable
from collections import deque


# =============================================================================
# AST definitions
# =============================================================================

class Node:
    pass


@dataclass(frozen=True)
class Alternation(Node):
    options: List[Node]


@dataclass(frozen=True)
class Concatenation(Node):
    parts: List[Node]  # empty => ε


@dataclass(frozen=True)
class Quantifier(Node):
    expr: Node
    min_count: int
    max_count: Optional[int]  # None => unbounded


@dataclass(frozen=True)
class Group(Node):
    expr: Node


@dataclass(frozen=True)
class Literal(Node):
    ch: str


@dataclass(frozen=True)
class Dot(Node):
    pass


@dataclass(frozen=True)
class AnchorBegin(Node):
    pass


@dataclass(frozen=True)
class AnchorEnd(Node):
    pass


@dataclass(frozen=True)
class EscapeClass(Node):
    code: str  # d D w W s S


CharClassItem = Union["CharItem", "RangeItem", "EscapeItem"]


@dataclass(frozen=True)
class CharItem:
    ch: str


@dataclass(frozen=True)
class RangeItem:
    start: str
    end: str


@dataclass(frozen=True)
class EscapeItem:
    code: str


@dataclass(frozen=True)
class CharClass(Node):
    negated: bool
    items: List[CharClassItem]


# =============================================================================
# Parser
# =============================================================================

class RegexParseError(ValueError):
    pass


class Parser:
    # Grammar:
    #   regex         := alternation
    #   alternation   := concatenation ('|' concatenation)*
    #   concatenation := repetition*            # ε allowed
    #   repetition    := atom quantifier?
    #   quantifier    := '*' | '+' | '?' | '{m}' | '{m,}' | '{m,n}'
    #   atom          := literal | '.' | '^' | '$' | escape | group | charclass
    #
    # Disallowed:
    #   - "(?" constructs (lookaround / inline flags / etc.)
    #   - capturing groups (none exist here)

    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def parse(self) -> Node:
        node = self._parse_alternation()
        if not self._eof():
            raise RegexParseError(f"Unexpected trailing input at pos {self.i}: {self._peek_remaining()!r}")
        return self._simplify(node)

    def _parse_alternation(self) -> Node:
        options = [self._parse_concatenation()]
        while self._peek() == '|':
            self._eat('|')
            options.append(self._parse_concatenation())
        if len(options) == 1:
            return options[0]
        return Alternation(options)

    def _parse_concatenation(self) -> Node:
        parts: List[Node] = []
        while True:
            c = self._peek()
            if c is None or c in ')|':
                break
            parts.append(self._parse_repetition())
        if not parts:
            return Concatenation([])
        if len(parts) == 1:
            return parts[0]
        return Concatenation(parts)

    def _parse_repetition(self) -> Node:
        atom = self._parse_atom()
        c = self._peek()
        if c in ('*', '+', '?'):
            q = self._eat_any('*+?')
            if q == '*':
                return Quantifier(atom, 0, None)
            if q == '+':
                return Quantifier(atom, 1, None)
            return Quantifier(atom, 0, 1)
        if c == '{':
            m, n = self._parse_brace_quantifier()
            return Quantifier(atom, m, n)
        return atom

    def _parse_brace_quantifier(self) -> Tuple[int, Optional[int]]:
        self._eat('{')
        m = self._parse_int(required=True)
        if self._peek() == '}':
            self._eat('}')
            return m, m
        self._eat(',')
        if self._peek() == '}':
            self._eat('}')
            return m, None
        n = self._parse_int(required=True)
        self._eat('}')
        if n < m:
            raise RegexParseError(f"Invalid quantifier range {{{m},{n}}}: n < m")
        return m, n

    def _parse_atom(self) -> Node:
        c = self._peek()
        if c is None:
            raise RegexParseError("Unexpected end of input while parsing atom")

        if c == '(':
            return self._parse_group()
        if c == '[':
            return self._parse_charclass()
        if c == '\\':
            return self._parse_escape()
        if c == '.':
            self._eat('.')
            return Dot()
        if c == '^':
            self._eat('^')
            return AnchorBegin()
        if c == '$':
            self._eat('$')
            return AnchorEnd()

        if c in '*+?}':
            raise RegexParseError(f"Quantifier symbol {c!r} cannot appear here (pos {self.i})")
        if c in '|)':
            raise RegexParseError(f"Unexpected {c!r} while parsing atom (pos {self.i})")

        self.i += 1
        return Literal(c)

    def _parse_group(self) -> Node:
        self._eat('(')
        if self._peek() == '?':
            raise RegexParseError(
                f"Unsupported construct '(?' at pos {self.i-1}. "
                "Lookaround/inline constructs are not supported."
            )
        expr = self._parse_alternation()
        if self._peek() != ')':
            raise RegexParseError(f"Unclosed '(' starting near pos {self.i}")
        self._eat(')')
        return Group(expr)

    def _parse_escape(self) -> Node:
        self._eat('\\')
        c = self._peek()
        if c is None:
            raise RegexParseError("Dangling backslash at end of input")
        self.i += 1

        if c in ('d', 'D', 'w', 'W', 's', 'S'):
            return EscapeClass(c)

        if c == 'n':
            return Literal('\n')
        if c == 'r':
            return Literal('\r')
        if c == 't':
            return Literal('\t')
        if c == 'f':
            return Literal('\f')
        if c == 'v':
            return Literal('\v')

        return Literal(c)

    def _parse_charclass(self) -> Node:
        self._eat('[')
        negated = False
        if self._peek() == '^':
            negated = True
            self._eat('^')

        items: List[CharClassItem] = []
        first = True
        while True:
            c = self._peek()
            if c is None:
                raise RegexParseError("Unclosed '[' character class")
            if c == ']' and not first:
                self._eat(']')
                break
            first = False
            items.append(self._parse_charclass_item())

        return CharClass(negated, items)

    def _parse_charclass_item(self) -> CharClassItem:
        c = self._peek()
        if c is None:
            raise RegexParseError("Unexpected end in character class")

        if c == '\\':
            self._eat('\\')
            esc = self._peek()
            if esc is None:
                raise RegexParseError("Dangling backslash in character class")
            self.i += 1

            if esc in ('d', 'D', 'w', 'W', 's', 'S'):
                left: CharClassItem = EscapeItem(esc)
            elif esc == 'n':
                left = CharItem('\n')
            elif esc == 'r':
                left = CharItem('\r')
            elif esc == 't':
                left = CharItem('\t')
            else:
                left = CharItem(esc)
        else:
            self.i += 1
            left = CharItem(c)

        # IMPORTANT BUGFIX (from earlier discussion):
        # If we see '-' and it is immediately before ']', then '-' is a literal.
        # We must NOT consume it here, or it gets lost and classes like [+-] become [+].
        if self._peek() == '-' and isinstance(left, CharItem):
            if self._peek_ahead(1) == ']' or self._peek_ahead(1) is None:
                # Do NOT consume '-', just return left.
                # The outer loop will parse '-' as the next CharItem.
                return left

            # Otherwise treat as a range left-right.
            self._eat('-')
            r = self._peek()
            if r is None:
                raise RegexParseError("Unclosed range in character class")

            if r == '\\':
                self._eat('\\')
                esc2 = self._peek()
                if esc2 is None:
                    raise RegexParseError("Dangling backslash in range endpoint")
                self.i += 1
                if esc2 in ('d', 'D', 'w', 'W', 's', 'S'):
                    raise RegexParseError("Range endpoint cannot be an escape class like \\d")
                if esc2 == 'n':
                    right_ch = '\n'
                elif esc2 == 'r':
                    right_ch = '\r'
                elif esc2 == 't':
                    right_ch = '\t'
                else:
                    right_ch = esc2
            else:
                self.i += 1
                right_ch = r

            return RangeItem(left.ch, right_ch)

        return left

    def _parse_int(self, required: bool) -> int:
        start = self.i
        while (c := self._peek()) is not None and c.isdigit():
            self.i += 1
        if self.i == start:
            if required:
                raise RegexParseError(f"Expected integer at pos {self.i}")
            return 0
        return int(self.text[start:self.i])

    def _peek(self) -> Optional[str]:
        if self.i >= len(self.text):
            return None
        return self.text[self.i]

    def _peek_ahead(self, k: int) -> Optional[str]:
        j = self.i + k
        if j >= len(self.text):
            return None
        return self.text[j]

    def _eat(self, ch: str) -> None:
        if self._peek() != ch:
            raise RegexParseError(f"Expected {ch!r} at pos {self.i}, found {self._peek()!r}")
        self.i += 1

    def _eat_any(self, chars: str) -> str:
        c = self._peek()
        if c is None or c not in chars:
            raise RegexParseError(f"Expected one of {chars!r} at pos {self.i}, found {c!r}")
        self.i += 1
        return c

    def _eof(self) -> bool:
        return self.i >= len(self.text)

    def _peek_remaining(self) -> str:
        return self.text[self.i:self.i + 30]

    def _simplify(self, node: Node) -> Node:
        if isinstance(node, Alternation):
            flat: List[Node] = []
            for opt in node.options:
                sopt = self._simplify(opt)
                if isinstance(sopt, Alternation):
                    flat.extend(sopt.options)
                else:
                    flat.append(sopt)
            return Alternation(flat)

        if isinstance(node, Concatenation):
            flat: List[Node] = []
            for part in node.parts:
                spart = self._simplify(part)
                if isinstance(spart, Concatenation):
                    flat.extend(spart.parts)
                else:
                    flat.append(spart)
            if len(flat) == 1:
                return flat[0]
            return Concatenation(flat)

        if isinstance(node, Quantifier):
            return Quantifier(self._simplify(node.expr), node.min_count, node.max_count)

        if isinstance(node, Group):
            return Group(self._simplify(node.expr))

        return node


def parse_regex(pattern: str) -> Node:
    return Parser(pattern).parse()


# =============================================================================
# Symbol domain (finite alphabet + BOF/EOF)
# =============================================================================

class SymbolDomain:
    BOF = object()
    EOF = object()

    def __init__(self, max_char_code: int = 127):
        if max_char_code < 0:
            raise ValueError("max_char_code must be >= 0")
        self.max_char_code = max_char_code
        self._size = (max_char_code + 1) + 2
        self._bof_index = max_char_code + 1
        self._eof_index = max_char_code + 2

    @property
    def size(self) -> int:
        return self._size

    @property
    def bof_index(self) -> int:
        return self._bof_index

    @property
    def eof_index(self) -> int:
        return self._eof_index

    def index_of_symbol(self, sym: object) -> int:
        if sym is self.BOF:
            return self._bof_index
        if sym is self.EOF:
            return self._eof_index
        if isinstance(sym, int) and 0 <= sym <= self.max_char_code:
            return sym
        raise ValueError(f"Symbol out of domain: {sym!r}")

    def sym_stream(self, s: str) -> List[object]:
        out: List[object] = [self.BOF]
        for ch in s:
            code = ord(ch)
            if code > self.max_char_code:
                raise ValueError(f"Character {ch!r} (ord={code}) outside domain 0..{self.max_char_code}")
            out.append(code)
        out.append(self.EOF)
        return out


# =============================================================================
# NFA representation
# =============================================================================

def _empty_label(domain: SymbolDomain) -> List[bool]:
    return [False] * domain.size


def _label_single(domain: SymbolDomain, idx: int) -> List[bool]:
    lab = [False] * domain.size
    lab[idx] = True
    return lab


def _label_union(a: List[bool], b: List[bool]) -> List[bool]:
    return [x or y for x, y in zip(a, b)]


def _label_invert_chars_only(domain: SymbolDomain, lab: List[bool]) -> List[bool]:
    out = lab[:]
    for i in range(domain.max_char_code + 1):
        out[i] = not out[i]
    out[domain.bof_index] = False
    out[domain.eof_index] = False
    return out


@dataclass
class NFAState:
    eps: Set[int]
    trans: List[Tuple[List[bool], int]]  # (label, dst)


class NFA:
    def __init__(self, domain: SymbolDomain):
        self.domain = domain
        self.states: List[NFAState] = []
        self.accept: Optional[int] = None

    def new_state(self) -> int:
        sid = len(self.states)
        self.states.append(NFAState(eps=set(), trans=[]))
        return sid

    def add_epsilon(self, src: int, dst: int) -> None:
        self.states[src].eps.add(dst)

    def add_transition(self, src: int, label: List[bool], dst: int) -> None:
        self.states[src].trans.append((label, dst))

    def epsilon_closure(self, start_set: Set[int]) -> Set[int]:
        closure = set(start_set)
        q: Deque[int] = deque(start_set)
        while q:
            s = q.popleft()
            for t in self.states[s].eps:
                if t not in closure:
                    closure.add(t)
                    q.append(t)
        return closure

    def move_on_symbol_index(self, state_set: Set[int], sym_idx: int) -> Set[int]:
        out: Set[int] = set()
        for s in state_set:
            for lab, dst in self.states[s].trans:
                if lab[sym_idx]:
                    out.add(dst)
        return out


@dataclass(frozen=True)
class Fragment:
    start: int
    accept: int


# =============================================================================
# Thompson construction
# =============================================================================

class ThompsonBuilder:
    def __init__(self, domain: SymbolDomain):
        self.domain = domain
        self.nfa = NFA(domain)

    def build(self, node: Node) -> Tuple[NFA, int, int]:
        frag = self._build_node(node)
        self.nfa.accept = frag.accept
        return self.nfa, frag.start, frag.accept

    def _epsilon_fragment(self) -> Fragment:
        s = self.nfa.new_state()
        a = self.nfa.new_state()
        self.nfa.add_epsilon(s, a)
        return Fragment(s, a)

    def _build_label_atom(self, label: List[bool]) -> Fragment:
        s = self.nfa.new_state()
        a = self.nfa.new_state()
        self.nfa.add_transition(s, label, a)
        return Fragment(s, a)

    def _build_node(self, node: Node) -> Fragment:
        if isinstance(node, Literal):
            return self._build_label_atom(self._label_for_literal(node.ch))

        if isinstance(node, Dot):
            return self._build_label_atom(self._label_for_dot())

        if isinstance(node, EscapeClass):
            return self._build_label_atom(self._label_for_escape(node.code))

        if isinstance(node, CharClass):
            return self._build_label_atom(self._label_for_charclass(node))

        if isinstance(node, AnchorBegin):
            return self._build_label_atom(_label_single(self.domain, self.domain.bof_index))

        if isinstance(node, AnchorEnd):
            return self._build_label_atom(_label_single(self.domain, self.domain.eof_index))

        if isinstance(node, Group):
            return self._build_node(node.expr)

        if isinstance(node, Concatenation):
            if not node.parts:
                return self._epsilon_fragment()
            frag = self._build_node(node.parts[0])
            for part in node.parts[1:]:
                right = self._build_node(part)
                self.nfa.add_epsilon(frag.accept, right.start)
                frag = Fragment(frag.start, right.accept)
            return frag

        if isinstance(node, Alternation):
            s = self.nfa.new_state()
            a = self.nfa.new_state()
            for opt in node.options:
                of = self._build_node(opt)
                self.nfa.add_epsilon(s, of.start)
                self.nfa.add_epsilon(of.accept, a)
            return Fragment(s, a)

        if isinstance(node, Quantifier):
            return self._build_repeat(node.expr, node.min_count, node.max_count)

        raise TypeError(f"Unsupported AST node: {type(node).__name__}")

    def _build_kleene_star(self, expr: Node) -> Fragment:
        s = self.nfa.new_state()
        a = self.nfa.new_state()
        body = self._build_node(expr)
        self.nfa.add_epsilon(s, a)
        self.nfa.add_epsilon(s, body.start)
        self.nfa.add_epsilon(body.accept, a)
        self.nfa.add_epsilon(body.accept, body.start)
        return Fragment(s, a)

    def _build_repeat(self, expr: Node, m: int, n: Optional[int]) -> Fragment:
        if m < 0:
            raise ValueError("min repeat must be >= 0")
        if n is not None and n < m:
            raise ValueError("max repeat must be >= min repeat")

        if m == 0:
            frag = self._epsilon_fragment()
        else:
            frag = self._build_node(expr)
            for _ in range(m - 1):
                nxt = self._build_node(expr)
                self.nfa.add_epsilon(frag.accept, nxt.start)
                frag = Fragment(frag.start, nxt.accept)

        if n is None:
            star = self._build_kleene_star(expr)
            self.nfa.add_epsilon(frag.accept, star.start)
            return Fragment(frag.start, star.accept)

        k = n - m
        cur_accept = frag.accept
        for _ in range(k):
            new_accept = self.nfa.new_state()
            self.nfa.add_epsilon(cur_accept, new_accept)
            copy = self._build_node(expr)
            self.nfa.add_epsilon(cur_accept, copy.start)
            self.nfa.add_epsilon(copy.accept, new_accept)
            cur_accept = new_accept

        return Fragment(frag.start, cur_accept)

    def _label_for_literal(self, ch: str) -> List[bool]:
        code = ord(ch)
        if code > self.domain.max_char_code:
            raise ValueError(f"Literal {ch!r} outside domain 0..{self.domain.max_char_code}")
        return _label_single(self.domain, code)

    def _label_for_dot(self) -> List[bool]:
        lab = _empty_label(self.domain)
        for i in range(self.domain.max_char_code + 1):
            lab[i] = True
        nl = ord('\n')
        if nl <= self.domain.max_char_code:
            lab[nl] = False
        lab[self.domain.bof_index] = False
        lab[self.domain.eof_index] = False
        return lab

    def _label_for_escape(self, code: str) -> List[bool]:
        base = _empty_label(self.domain)

        if code in ('d', 'D'):
            for c in range(ord('0'), ord('9') + 1):
                if c <= self.domain.max_char_code:
                    base[c] = True

        elif code in ('w', 'W'):
            for c in range(ord('0'), ord('9') + 1):
                if c <= self.domain.max_char_code:
                    base[c] = True
            for c in range(ord('A'), ord('Z') + 1):
                if c <= self.domain.max_char_code:
                    base[c] = True
            for c in range(ord('a'), ord('z') + 1):
                if c <= self.domain.max_char_code:
                    base[c] = True
            u = ord('_')
            if u <= self.domain.max_char_code:
                base[u] = True

        elif code in ('s', 'S'):
            for c in [ord(' '), ord('\t'), ord('\n'), ord('\r'), ord('\f'), ord('\v')]:
                if c <= self.domain.max_char_code:
                    base[c] = True

        else:
            raise ValueError(f"Unknown escape class: \\{code}")

        base[self.domain.bof_index] = False
        base[self.domain.eof_index] = False
        if code in ('D', 'W', 'S'):
            return _label_invert_chars_only(self.domain, base)
        return base

    def _label_for_charclass(self, cc: CharClass) -> List[bool]:
        lab = _empty_label(self.domain)
        for it in cc.items:
            if isinstance(it, CharItem):
                code = ord(it.ch)
                if code <= self.domain.max_char_code:
                    lab[code] = True
            elif isinstance(it, RangeItem):
                a = ord(it.start)
                b = ord(it.end)
                if a > b:
                    a, b = b, a
                lo = max(0, a)
                hi = min(self.domain.max_char_code, b)
                for code in range(lo, hi + 1):
                    lab[code] = True
            elif isinstance(it, EscapeItem):
                lab = _label_union(lab, self._label_for_escape(it.code))
            else:
                raise TypeError(f"Unknown charclass item: {type(it).__name__}")

        lab[self.domain.bof_index] = False
        lab[self.domain.eof_index] = False
        if cc.negated:
            return _label_invert_chars_only(self.domain, lab)
        return lab


# =============================================================================
# NFA simulation
# =============================================================================

def nfa_fullmatch(nfa: NFA, start: int, accept: int, domain: SymbolDomain, text: str) -> bool:
    stream = domain.sym_stream(text)
    current = nfa.epsilon_closure({start})
    for sym in stream:
        idx = domain.index_of_symbol(sym)
        moved = nfa.move_on_symbol_index(current, idx)
        current = nfa.epsilon_closure(moved)
        if not current:
            return False
    return accept in current


# =============================================================================
# DFA + subset construction
# =============================================================================

@dataclass
class DFA:
    domain: SymbolDomain
    start: int
    accept: Set[int]
    trans: List[List[int]]  # trans[state][sym_idx] -> next_state

    def fullmatch(self, text: str) -> bool:
        stream = self.domain.sym_stream(text)
        s = self.start
        for sym in stream:
            idx = self.domain.index_of_symbol(sym)
            s = self.trans[s][idx]
        return s in self.accept


def nfa_to_dfa(nfa: NFA, start: int, accept: int) -> DFA:
    domain = nfa.domain

    def closure_of(states: Set[int]) -> frozenset[int]:
        return frozenset(nfa.epsilon_closure(states))

    start_set = closure_of({start})
    state_id: Dict[frozenset[int], int] = {start_set: 0}
    dfa_sets: List[frozenset[int]] = [start_set]
    dfa_trans: List[List[int]] = []
    dfa_accept: Set[int] = set()

    q: Deque[int] = deque([0])
    while q:
        sid = q.popleft()
        nfa_set = dfa_sets[sid]

        row = [-1] * domain.size
        if accept in nfa_set:
            dfa_accept.add(sid)

        for sym_idx in range(domain.size):
            moved = nfa.move_on_symbol_index(set(nfa_set), sym_idx)
            if not moved:
                continue
            nxt = closure_of(moved)
            if not nxt:
                continue
            if nxt not in state_id:
                nid = len(dfa_sets)
                state_id[nxt] = nid
                dfa_sets.append(nxt)
                q.append(nid)
            row[sym_idx] = state_id[nxt]

        dfa_trans.append(row)

    dead_needed = any(any(t == -1 for t in row) for row in dfa_trans)
    if dead_needed:
        dead = len(dfa_trans)
        dead_row = [dead] * domain.size
        for row in dfa_trans:
            for i in range(domain.size):
                if row[i] == -1:
                    row[i] = dead
        dfa_trans.append(dead_row)

    return DFA(domain=domain, start=0, accept=dfa_accept, trans=dfa_trans)


# =============================================================================
# Hopcroft minimization
# =============================================================================

@dataclass
class MinDFA:
    domain: SymbolDomain
    start: int
    accept: Set[int]
    trans: List[List[int]]

    def fullmatch(self, text: str) -> bool:
        stream = self.domain.sym_stream(text)
        s = self.start
        for sym in stream:
            idx = self.domain.index_of_symbol(sym)
            s = self.trans[s][idx]
        return s in self.accept


def minimize_dfa(dfa: DFA) -> MinDFA:
    domain = dfa.domain
    n = len(dfa.trans)

    A = set(dfa.accept)
    NA = set(range(n)) - A

    P: List[Set[int]] = []
    if A:
        P.append(A)
    if NA:
        P.append(NA)

    W: Deque[Set[int]] = deque()
    if len(P) == 2:
        W.append(A if len(A) <= len(NA) else NA)
    elif len(P) == 1:
        W.append(P[0])

    rev: List[List[Set[int]]] = []
    for _ in range(domain.size):
        rev.append([set() for _ in range(n)])
    for src in range(n):
        row = dfa.trans[src]
        for sym_idx in range(domain.size):
            dst = row[sym_idx]
            rev[sym_idx][dst].add(src)

    while W:
        splitter = W.popleft()
        for sym_idx in range(domain.size):
            X: Set[int] = set()
            for dst in splitter:
                X |= rev[sym_idx][dst]
            if not X:
                continue

            newP: List[Set[int]] = []
            for Y in P:
                Y1 = Y & X
                Y2 = Y - X
                if Y1 and Y2:
                    newP.append(Y1)
                    newP.append(Y2)

                    in_worklist = any(blk == Y for blk in W)
                    if in_worklist:
                        W.remove(Y)
                        W.append(Y1)
                        W.append(Y2)
                    else:
                        W.append(Y1 if len(Y1) <= len(Y2) else Y2)
                else:
                    newP.append(Y)
            P = newP

    block_of: Dict[int, int] = {}
    for bi, blk in enumerate(P):
        for s in blk:
            block_of[s] = bi

    m = len(P)
    min_trans: List[List[int]] = [[0] * domain.size for _ in range(m)]
    min_accept: Set[int] = set()

    for bi, blk in enumerate(P):
        rep = next(iter(blk))
        for sym_idx in range(domain.size):
            min_trans[bi][sym_idx] = block_of[dfa.trans[rep][sym_idx]]
        if rep in dfa.accept:
            min_accept.add(bi)

    min_start = block_of[dfa.start]
    return MinDFA(domain=domain, start=min_start, accept=min_accept, trans=min_trans)


# =============================================================================
# Compilation bundle
# =============================================================================

@dataclass
class AutomataBundle:
    ast: Node
    nfa: NFA
    nfa_start: int
    nfa_accept: int
    dfa: DFA
    mindfa: MinDFA


def compile_regex(pattern: str, max_char_code: int = 127) -> AutomataBundle:
    domain = SymbolDomain(max_char_code=max_char_code)
    ast = parse_regex(pattern)
    builder = ThompsonBuilder(domain)
    nfa, ns, na = builder.build(ast)
    dfa = nfa_to_dfa(nfa, ns, na)
    mindfa = minimize_dfa(dfa)
    return AutomataBundle(ast=ast, nfa=nfa, nfa_start=ns, nfa_accept=na, dfa=dfa, mindfa=mindfa)


# =============================================================================
# Pretty-print helpers for transition labels and tables
# =============================================================================

def _is_printable_ascii(code: int) -> bool:
    return 32 <= code <= 126


def _escape_char(ch: str) -> str:
    # Render characters safely in table output.
    if ch == '\n':
        return r'\n'
    if ch == '\r':
        return r'\r'
    if ch == '\t':
        return r'\t'
    if ch == '\\':
        return r'\\'
    if ch == "'":
        return r"\'"
    if _is_printable_ascii(ord(ch)):
        return ch
    return f"\\x{ord(ch):02X}"


def label_to_ranges(domain: SymbolDomain, lab: List[bool]) -> List[str]:
    # Convert a boolean label bitset into compact human-readable ranges.
    # Returns a list of strings, e.g.:
    #   ["'0'-'9'", "'A'-'Z'", "'_'", "BOF"]
    out: List[str] = []

    # BOF / EOF (rare: used by anchors).
    if lab[domain.bof_index]:
        out.append("BOF")
    if lab[domain.eof_index]:
        out.append("EOF")

    # Character ranges.
    i = 0
    last = domain.max_char_code
    while i <= last:
        if not lab[i]:
            i += 1
            continue
        j = i
        while j + 1 <= last and lab[j + 1]:
            j += 1

        if i == j:
            ch = chr(i)
            out.append(f"'{_escape_char(ch)}'")
        else:
            a = chr(i)
            b = chr(j)
            out.append(f"'{_escape_char(a)}'-'{_escape_char(b)}'")

        i = j + 1

    return out


def format_label(domain: SymbolDomain, lab: List[bool], max_chunks: int = 10) -> str:
    # Format label ranges into a single string, optionally truncating.
    chunks = label_to_ranges(domain, lab)
    if not chunks:
        return "∅"
    if len(chunks) <= max_chunks:
        return ", ".join(chunks)
    head = ", ".join(chunks[:max_chunks])
    return f"{head}, ... (+{len(chunks)-max_chunks} more)"


def _make_table(rows: List[List[str]], headers: List[str]) -> str:
    # Very small ASCII table formatter.
    # rows are list of columns as strings; headers define column names.
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(r[c]))

    def fmt_row(r: List[str]) -> str:
        parts = []
        for c in range(cols):
            parts.append(r[c].ljust(widths[c]))
        return " | ".join(parts)

    line = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), line]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def dump_nfa_table(nfa: NFA, start: int, accept: int) -> str:
    # NFA table: one row per labeled transition; epsilon list shown per-state.
    #
    # Columns:
    #   State | Markers | Epsilons | Dest | Symbols
    rows: List[List[str]] = []
    for s, st in enumerate(nfa.states):
        markers = []
        if s == start:
            markers.append("START")
        if s == accept:
            markers.append("ACCEPT")
        mark = ",".join(markers) if markers else ""

        eps = "{" + ",".join(str(x) for x in sorted(st.eps)) + "}" if st.eps else "{}"

        if not st.trans:
            # Still show the state even if it has no labeled transitions.
            rows.append([str(s), mark, eps, "", ""])
            continue

        first_row = True
        for lab, dst in st.trans:
            # On subsequent rows for same state, blank out repeated columns for readability.
            s_col = str(s) if first_row else ""
            m_col = mark if first_row else ""
            e_col = eps if first_row else ""
            rows.append([s_col, m_col, e_col, str(dst), format_label(nfa.domain, lab)])
            first_row = False

    return _make_table(rows, ["State", "Markers", "Eps", "Dest", "Symbols"])


def dump_dfa_table(domain: SymbolDomain, trans: List[List[int]], start: int, accept: Set[int], title: str) -> str:
    # DFA/MinDFA table: group symbols by destination for each state.
    #
    # Columns:
    #   State | Markers | Dest | Symbols
    rows: List[List[str]] = []

    for s in range(len(trans)):
        markers = []
        if s == start:
            markers.append("START")
        if s in accept:
            markers.append("ACCEPT")
        mark = ",".join(markers) if markers else ""

        # Group by destination: dst -> label bitset (union of all symbols leading to dst).
        groups: Dict[int, List[bool]] = {}
        for sym_idx, dst in enumerate(trans[s]):
            if dst not in groups:
                groups[dst] = _empty_label(domain)
            groups[dst][sym_idx] = True

        # Sort destinations for stable output.
        dests = sorted(groups.keys())
        first_row = True
        for dst in dests:
            s_col = str(s) if first_row else ""
            m_col = mark if first_row else ""
            rows.append([s_col, m_col, str(dst), format_label(domain, groups[dst])])
            first_row = False

    table = _make_table(rows, ["State", "Markers", "Dest", "Symbols"])
    return f"{title}\n{table}"


# =============================================================================
# AST dumper (still useful)
# =============================================================================

def dump_ast(node: Node, indent: str = "", is_last: bool = True) -> str:
    branch = "└─ " if is_last else "├─ "
    next_indent = indent + ("   " if is_last else "│  ")

    def head(label: str) -> str:
        return f"{indent}{branch}{label}\n"

    if isinstance(node, Alternation):
        s = head("Alternation |")
        for idx, opt in enumerate(node.options):
            s += dump_ast(opt, next_indent, idx == len(node.options) - 1)
        return s

    if isinstance(node, Concatenation):
        if not node.parts:
            return head("Concatenation (ε)")
        s = head("Concatenation")
        for idx, part in enumerate(node.parts):
            s += dump_ast(part, next_indent, idx == len(node.parts) - 1)
        return s

    if isinstance(node, Quantifier):
        max_part = "∞" if node.max_count is None else str(node.max_count)
        s = head(f"Quantifier {{{node.min_count},{max_part}}}")
        s += dump_ast(node.expr, next_indent, True)
        return s

    if isinstance(node, Group):
        s = head("Group (non-capturing)")
        s += dump_ast(node.expr, next_indent, True)
        return s

    if isinstance(node, Literal):
        return head(f"Literal {node.ch!r}")

    if isinstance(node, Dot):
        return head("Dot .")

    if isinstance(node, AnchorBegin):
        return head("AnchorBegin ^")

    if isinstance(node, AnchorEnd):
        return head("AnchorEnd $")

    if isinstance(node, EscapeClass):
        return head(f"EscapeClass \\{node.code}")

    if isinstance(node, CharClass):
        prefix = "^" if node.negated else ""
        s = head(f"CharClass [{prefix}...]")
        for idx, it in enumerate(node.items):
            last = idx == len(node.items) - 1
            if isinstance(it, CharItem):
                label = f"Char {it.ch!r}"
            elif isinstance(it, RangeItem):
                label = f"Range {it.start!r}-{it.end!r}"
            elif isinstance(it, EscapeItem):
                label = f"EscapeItem \\{it.code}"
            else:
                label = f"Item {it!r}"
            s += f"{next_indent}{'└─ ' if last else '├─ '}{label}\n"
        return s

    return head(f"UnknownNode {node!r}")


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    # Practical float regex.
    float_regex = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
    # Practical email regex.
    email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    demos = [
        ("C-like float", float_regex, ["1", "-3.14", "-3.14E-2", "+3.0", "e10", "1ee2", ""]),
        ("Email", email_regex, ["a@b.com", "john.doe+tag@sub.example.co", "bad@", "@bad.com", "a@b.c", ""]),
    ]

    for name, pat, tests in demos:
        print("=" * 100)
        print(f"{name} regex:")
        print(pat)
        print("-" * 100)

        bundle = compile_regex(pat, max_char_code=127)

        print("AST:")
        print(dump_ast(Concatenation([bundle.ast])))

        print(f"NFA states:    {len(bundle.nfa.states)}")
        print(f"DFA states:    {len(bundle.dfa.trans)}")
        print(f"MinDFA states: {len(bundle.mindfa.trans)}")

        """
        print("-" * 100)
        print("NFA transition table:")
        print(dump_nfa_table(bundle.nfa, bundle.nfa_start, bundle.nfa_accept))

        print("-" * 100)
        print(dump_dfa_table(bundle.dfa.domain, bundle.dfa.trans, bundle.dfa.start, bundle.dfa.accept, "DFA transition table:"))

        print("-" * 100)
        print(dump_dfa_table(bundle.mindfa.domain, bundle.mindfa.trans, bundle.mindfa.start, bundle.mindfa.accept, "MinDFA transition table:"))
        """

        print("-" * 100)
        print("Tests (NFA vs DFA vs MinDFA):")
        for t in tests:
            nfa_ok = nfa_fullmatch(bundle.nfa, bundle.nfa_start, bundle.nfa_accept, bundle.nfa.domain, t)
            dfa_ok = bundle.dfa.fullmatch(t)
            min_ok = bundle.mindfa.fullmatch(t)
            print(f"  {t!r:28}  NFA={nfa_ok}  DFA={dfa_ok}  MinDFA={min_ok}")

    print("=" * 100)


if __name__ == "__main__":
    main()
