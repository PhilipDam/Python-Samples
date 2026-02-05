#!/usr/bin/env python3
"""
Sudoku Solver (Deduction-First, then Guessing) — FULL VERBOSE VERSION

This file is deliberately “teaching code”: heavy comments, explicit reasoning,
and (as requested) *very verbose logging*.

WHAT IT PRINTS (ALL MOVES)
--------------------------
For every usable action the solver applies, it prints:

  1) What happened:
       - a placement:   "Place D at rXcY"
       - an elimination:"Eliminate D from rXcY"

  2) Why it is valid (a human-readable explanation of the technique).

Additionally (this is the big upgrade):
---------------------------------------
Every time we PLACE a digit, we also print *every peer-elimination* caused by
that placement:

  - If we place digit D in a cell, then every peer cell (same row/col/box)
    cannot contain D, so we remove candidate D from those peer cells.

This includes placements coming from:
  - the named deduction techniques
  - propagation (forced singles)
  - guessing / trial branches

This is extremely verbose. A typical solve can produce thousands of lines.
If you ever want less output, set PRINT_PEER_ELIMINATIONS=False.

SOLVING STRATEGY
----------------
We repeatedly apply deduction techniques in a fixed order. As soon as a technique
finds a usable action, we apply it, print it, and restart the technique list.

If no deduction technique makes progress, we switch to Guessing (backtracking):
  - choose an empty cell with the fewest candidates (MRV heuristic)
  - try each candidate recursively

TECHNIQUES IMPLEMENTED (in order)
---------------------------------
  1) Naked Singles
  2) Hidden Singles
  3) Naked Pairs/Twins
  4) Hidden Pairs
  5) Naked Triples
  6) Naked Quads
  7) Hidden Triples
  8) Hidden Quads
  9) Pointing Pairs/Triples
 10) Box/Line Reduction (Claiming)
 11) X-Wing
 12) Swordfish
 13) XYZ-Wing
 14) Coloring (two-coloring on strong links)
 15) Chains (AIC – limited digit-focused, bounded depth)
 16) Forcing Chains (contradiction probing on 2–3 candidate cells)
 17) Unique Rectangle (UR Type 1)
 18) Guessing (Trial & Error / Backtracking)

Input format
------------
Provide either:
- an 81-character string (digits 1-9, '.' or '0' for blanks), OR
- 9 lines of 9 characters (spaces ignored).

Example:
  python sudoku_solver.py "53..7....6..195....98....6.8...6...34..8..6...2...3.6....28....419..5....8..79"
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import sys

DIGITS = set(range(1, 10))

# Toggle: print every peer-elimination caused by placements (very verbose).
PRINT_PEER_ELIMINATIONS = True


# =============================================================================
# Indexing helpers and precomputed constraint structure
# =============================================================================

def rc_to_idx(r: int, c: int) -> int:
    """Map (row, col) in 0..8 x 0..8 to cell index 0..80."""
    return r * 9 + c

def idx_to_rc(i: int) -> tuple[int, int]:
    """Map cell index 0..80 to (row, col)."""
    return divmod(i, 9)

def box_index(r: int, c: int) -> int:
    """Map (r,c) to a 3x3 box index 0..8."""
    return (r // 3) * 3 + (c // 3)

def box_cells(b: int) -> list[int]:
    """Return the 9 cell indices belonging to box b."""
    br = (b // 3) * 3
    bc = (b % 3) * 3
    return [rc_to_idx(br + dr, bc + dc) for dr in range(3) for dc in range(3)]

ROWS = [[rc_to_idx(r, c) for c in range(9)] for r in range(9)]
COLS = [[rc_to_idx(r, c) for r in range(9)] for c in range(9)]
BOXES = [box_cells(b) for b in range(9)]
UNITS = ROWS + COLS + BOXES  # 27 units

# PEERS[i] = all cells that share row, col, or box with i (excluding i).
PEERS: list[set[int]] = []
for i in range(81):
    r, c = idx_to_rc(i)
    b = box_index(r, c)
    peer_set = set(ROWS[r] + COLS[c] + BOXES[b])
    peer_set.remove(i)
    PEERS.append(peer_set)


# =============================================================================
# Action (for logging + applying)
# =============================================================================

@dataclass
class Action:
    """
    One atomic solver step.

    kind:
      - "place": set a cell to a digit
      - "elim" : remove a digit from a cell's candidate set
    """
    kind: str          # "place" | "elim"
    idx: int
    digit: int
    technique: str
    explanation: str


def print_action(a: Action, indent: int = 0) -> None:
    """Human-readable logging for one action."""
    r, c = idx_to_rc(a.idx)
    pad = "  " * indent
    if a.kind == "place":
        print(f"{pad}[{a.technique}] Place {a.digit} at r{r+1}c{c+1}")
    else:
        print(f"{pad}[{a.technique}] Eliminate {a.digit} from r{r+1}c{c+1}")
    print(f"{pad}  Reason: {a.explanation}")


# =============================================================================
# Sudoku state + candidate bookkeeping
# =============================================================================

class Sudoku:
    """
    grid[i] : placed digit (0 means empty)
    cand[i] : set of candidate digits for cell i (only meaningful if grid[i]==0)

    Invariant:
      - For filled cells, cand[i] == {grid[i]}
      - For empty cells, cand[i] contains only digits not present in peers.
    """
    def __init__(self, grid: list[int]):
        self.grid = grid[:]
        self.cand = [set() for _ in range(81)]
        self._init_candidates()

    def clone(self) -> "Sudoku":
        """Deep clone for forcing chains / guessing branches."""
        s = Sudoku(self.grid)
        s.cand = [c.copy() for c in self.cand]
        return s

    def is_solved(self) -> bool:
        return all(v != 0 for v in self.grid)

    # -------------------------------------------------------------------------
    # Initialization and consistency checks
    # -------------------------------------------------------------------------

    def _init_candidates(self) -> None:
        """Compute candidate sets from the current grid."""
        for i in range(81):
            v = self.grid[i]
            if v != 0:
                self.cand[i] = {v}
            else:
                used = {self.grid[p] for p in PEERS[i] if self.grid[p] != 0}
                self.cand[i] = DIGITS - used
        self._assert_consistent()

    def _assert_consistent(self) -> None:
        """Raise ValueError if a contradiction exists."""
        # No empty cell may have zero candidates
        for i in range(81):
            if self.grid[i] == 0 and len(self.cand[i]) == 0:
                r, c = idx_to_rc(i)
                raise ValueError(f"Contradiction: r{r+1}c{c+1} has no candidates")

        # No duplicate digit in any unit
        for unit in UNITS:
            seen = set()
            for i in unit:
                v = self.grid[i]
                if v != 0:
                    if v in seen:
                        r, c = idx_to_rc(i)
                        raise ValueError(f"Contradiction: duplicate {v} in a unit (at r{r+1}c{c+1})")
                    seen.add(v)

    # -------------------------------------------------------------------------
    # Core primitives
    # -------------------------------------------------------------------------

    def _place_silent(self, idx: int, digit: int) -> None:
        """
        Place digit (silent, no Action creation) and update candidate sets:
          - set grid[idx]
          - set cand[idx] = {digit}
          - remove digit from all peers' candidate sets
        """
        if self.grid[idx] != 0 and self.grid[idx] != digit:
            raise ValueError("Contradiction: overwriting different digit")

        self.grid[idx] = digit
        self.cand[idx] = {digit}

        for p in PEERS[idx]:
            if self.grid[p] == 0:
                self.cand[p].discard(digit)

    def _place_with_peer_elims(self, idx: int, digit: int, source_technique: str) -> list[Action]:
        """
        Place digit and return a list of peer-elimination Actions (if enabled).
        This is the mechanism that makes the log “complete” per your request.
        """
        if self.grid[idx] != 0 and self.grid[idx] != digit:
            raise ValueError("Contradiction: overwriting different digit")

        self.grid[idx] = digit
        self.cand[idx] = {digit}

        # If peer-elimination printing is disabled, still do the elimination silently.
        if not PRINT_PEER_ELIMINATIONS:
            for p in PEERS[idx]:
                if self.grid[p] == 0:
                    self.cand[p].discard(digit)
            return []

        # Otherwise, generate a separate elimination Action for each peer that loses this candidate.
        elim_actions: list[Action] = []
        r0, c0 = idx_to_rc(idx)

        for p in PEERS[idx]:
            if self.grid[p] != 0:
                continue
            if digit in self.cand[p]:
                self.cand[p].remove(digit)
                rp, cp = idx_to_rc(p)
                elim_actions.append(Action(
                    kind="elim",
                    idx=p,
                    digit=digit,
                    technique="Peer Elimination",
                    explanation=(
                        f"Placed {digit} at r{r0+1}c{c0+1}, so no peer in the same row/column/box may be {digit}. "
                        f"Remove {digit} from r{rp+1}c{cp+1}."
                        + (f" (source: {source_technique})" if source_technique else "")
                    )
                ))

        return elim_actions

    def _eliminate_silent(self, idx: int, digit: int) -> None:
        """Eliminate a candidate from an empty cell (silent)."""
        if self.grid[idx] != 0:
            if self.grid[idx] == digit:
                raise ValueError("Contradiction: eliminating placed digit")
            return
        self.cand[idx].discard(digit)

    # -------------------------------------------------------------------------
    # Propagation (forced singles) with full logging + peer-elims
    # -------------------------------------------------------------------------

    def _propagate_forced_singles_as_actions(self) -> list[Action]:
        """
        Repeatedly:
          - If an empty cell has exactly one candidate, it must be placed.

        Returns a list of Actions that includes:
          - the forced placements
          - (optionally) every peer-elimination caused by those forced placements
        """
        log: list[Action] = []
        changed = True

        while changed:
            changed = False

            for i in range(81):
                if self.grid[i] == 0 and len(self.cand[i]) == 1:
                    d = next(iter(self.cand[i]))
                    r, c = idx_to_rc(i)

                    forced_place = Action(
                        kind="place",
                        idx=i,
                        digit=d,
                        technique="Propagation (Forced Naked Single)",
                        explanation=(
                            f"After the previous step, r{r+1}c{c+1} has only candidate {d} left, so it must be {d}."
                        )
                    )
                    log.append(forced_place)

                    # Place + peer eliminations
                    log.extend(self._place_with_peer_elims(i, d, forced_place.technique))

                    changed = True

            self._assert_consistent()

        return log

    def apply(self, action: Action) -> list[Action]:
        """
        Apply one explicit action, then propagate forced singles.

        Returns:
          A list of implied actions (peer eliminations from placements +
          forced placements + peer eliminations from forced placements).
        """
        implied: list[Action] = []

        if action.kind == "place":
            implied.extend(self._place_with_peer_elims(action.idx, action.digit, action.technique))
        elif action.kind == "elim":
            self._eliminate_silent(action.idx, action.digit)
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        implied.extend(self._propagate_forced_singles_as_actions())
        self._assert_consistent()
        return implied

    # -------------------------------------------------------------------------
    # Pretty print
    # -------------------------------------------------------------------------

    def pretty(self) -> str:
        out = []
        for r in range(9):
            row = []
            for c in range(9):
                v = self.grid[rc_to_idx(r, c)]
                row.append(str(v) if v != 0 else ".")
                if c in (2, 5):
                    row.append("|")
            out.append(" ".join(row))
            if r in (2, 5):
                out.append("-" * 21)
        return "\n".join(out)


# =============================================================================
# Technique implementations
# Each returns a list[Action]; solver uses only the first (one atomic step).
# =============================================================================

def technique_naked_singles(s: Sudoku) -> list[Action]:
    for i in range(81):
        if s.grid[i] == 0 and len(s.cand[i]) == 1:
            d = next(iter(s.cand[i]))
            r, c = idx_to_rc(i)
            return [Action(
                kind="place",
                idx=i,
                digit=d,
                technique="Naked Single",
                explanation=(
                    f"r{r+1}c{c+1} has only one candidate: {d}. "
                    f"All other digits are excluded by row/column/box constraints."
                )
            )]
    return []

def technique_hidden_singles(s: Sudoku) -> list[Action]:
    for unit in UNITS:
        positions = {d: [] for d in DIGITS}
        for i in unit:
            if s.grid[i] == 0:
                for d in s.cand[i]:
                    positions[d].append(i)
        for d, cells in positions.items():
            if len(cells) == 1:
                i = cells[0]
                r, c = idx_to_rc(i)
                return [Action(
                    kind="place",
                    idx=i,
                    digit=d,
                    technique="Hidden Single",
                    explanation=(
                        f"In this unit, digit {d} can appear in only one cell: r{r+1}c{c+1}. "
                        f"Therefore r{r+1}c{c+1} must be {d}."
                    )
                )]
    return []

def technique_naked_pairs(s: Sudoku) -> list[Action]:
    for unit in UNITS:
        pairs: dict[tuple[int, int], list[int]] = {}
        for i in unit:
            if s.grid[i] == 0 and len(s.cand[i]) == 2:
                key = tuple(sorted(s.cand[i]))
                pairs.setdefault(key, []).append(i)

        for pair, cells in pairs.items():
            if len(cells) == 2:
                a, b = pair
                for j in unit:
                    if j in cells or s.grid[j] != 0:
                        continue
                    for d in (a, b):
                        if d in s.cand[j]:
                            rj, cj = idx_to_rc(j)
                            r1, c1 = idx_to_rc(cells[0])
                            r2, c2 = idx_to_rc(cells[1])
                            return [Action(
                                kind="elim",
                                idx=j,
                                digit=d,
                                technique="Naked Pair",
                                explanation=(
                                    f"In this unit, cells r{r1+1}c{c1+1} and r{r2+1}c{c2+1} form a Naked Pair {{ {a},{b} }}. "
                                    f"Thus {a} and {b} must occupy those two cells, so r{rj+1}c{cj+1} cannot be {d}."
                                )
                            )]
    return []

def technique_hidden_pairs(s: Sudoku) -> list[Action]:
    for unit in UNITS:
        pos = {d: [] for d in DIGITS}
        for i in unit:
            if s.grid[i] == 0:
                for d in s.cand[i]:
                    pos[d].append(i)

        for a, b in combinations(range(1, 10), 2):
            if len(pos[a]) == 2 and pos[a] == pos[b]:
                cells = pos[a]
                for i in cells:
                    extras = s.cand[i] - {a, b}
                    if extras:
                        d_elim = next(iter(extras))
                        r, c = idx_to_rc(i)
                        r1, c1 = idx_to_rc(cells[0])
                        r2, c2 = idx_to_rc(cells[1])
                        return [Action(
                            kind="elim",
                            idx=i,
                            digit=d_elim,
                            technique="Hidden Pair",
                            explanation=(
                                f"In this unit, digits {a} and {b} can only appear in r{r1+1}c{c1+1} and r{r2+1}c{c2+1}. "
                                f"So those two cells must be {{ {a},{b} }}, hence remove {d_elim} from r{r+1}c{c+1}."
                            )
                        )]
    return []

def technique_naked_k(s: Sudoku, k: int) -> list[Action]:
    # General naked k-tuple (k=3 => triple, k=4 => quad)
    for unit in UNITS:
        empties = [i for i in unit if s.grid[i] == 0]
        candidate_cells = [i for i in empties if 1 <= len(s.cand[i]) <= k]

        for cells in combinations(candidate_cells, k):
            union = set().union(*(s.cand[i] for i in cells))
            if len(union) != k:
                continue

            for j in empties:
                if j in cells:
                    continue
                for d in union:
                    if d in s.cand[j]:
                        rj, cj = idx_to_rc(j)
                        coords = ", ".join([f"r{idx_to_rc(i)[0]+1}c{idx_to_rc(i)[1]+1}" for i in cells])
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique=f"Naked {k}-tuple",
                            explanation=(
                                f"In this unit, cells {coords} collectively contain exactly digits {sorted(union)}. "
                                f"Those digits must be placed within those {k} cells, so remove {d} from r{rj+1}c{cj+1}."
                            )
                        )]
    return []

def technique_hidden_k(s: Sudoku, k: int) -> list[Action]:
    # General hidden k-tuple (k=3 => triple, k=4 => quad)
    for unit in UNITS:
        pos = {d: [] for d in DIGITS}
        for i in unit:
            if s.grid[i] == 0:
                for d in s.cand[i]:
                    pos[d].append(i)

        for digits in combinations(range(1, 10), k):
            cells_union = set().union(*(pos[d] for d in digits))
            if len(cells_union) != k:
                continue

            allowed = set(digits)
            for i in cells_union:
                extras = s.cand[i] - allowed
                if extras:
                    d_elim = next(iter(extras))
                    r, c = idx_to_rc(i)
                    cell_list = [f"r{idx_to_rc(j)[0]+1}c{idx_to_rc(j)[1]+1}" for j in sorted(cells_union)]
                    return [Action(
                        kind="elim",
                        idx=i,
                        digit=d_elim,
                        technique=f"Hidden {k}-tuple",
                        explanation=(
                            f"In this unit, digits {list(digits)} are restricted to exactly cells {cell_list}. "
                            f"Therefore those cells can only contain {list(digits)}, so remove {d_elim} from r{r+1}c{c+1}."
                        )
                    )]
    return []

def technique_pointing_pairs_triples(s: Sudoku) -> list[Action]:
    for b in range(9):
        box = BOXES[b]
        pos = {d: [] for d in DIGITS}
        for i in box:
            if s.grid[i] == 0:
                for d in s.cand[i]:
                    pos[d].append(i)

        for d, locs in pos.items():
            if len(locs) < 2:
                continue
            rows = {idx_to_rc(i)[0] for i in locs}
            cols = {idx_to_rc(i)[1] for i in locs}

            if len(rows) == 1:
                r = next(iter(rows))
                for j in ROWS[r]:
                    if j not in box and s.grid[j] == 0 and d in s.cand[j]:
                        rj, cj = idx_to_rc(j)
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Pointing Pair/Triple",
                            explanation=(
                                f"In box {b+1}, all candidates for digit {d} are on row {r+1}. "
                                f"So {d} must be placed in that box on row {r+1}, hence remove {d} from r{rj+1}c{cj+1}."
                            )
                        )]

            if len(cols) == 1:
                c = next(iter(cols))
                for j in COLS[c]:
                    if j not in box and s.grid[j] == 0 and d in s.cand[j]:
                        rj, cj = idx_to_rc(j)
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Pointing Pair/Triple",
                            explanation=(
                                f"In box {b+1}, all candidates for digit {d} are on column {c+1}. "
                                f"So {d} must be placed in that box on column {c+1}, hence remove {d} from r{rj+1}c{cj+1}."
                            )
                        )]
    return []

def technique_box_line_reduction(s: Sudoku) -> list[Action]:
    # Row claiming
    for r in range(9):
        for d in range(1, 10):
            locs = [i for i in ROWS[r] if s.grid[i] == 0 and d in s.cand[i]]
            if len(locs) < 2:
                continue
            boxes = {box_index(*idx_to_rc(i)) for i in locs}
            if len(boxes) == 1:
                b = next(iter(boxes))
                for j in BOXES[b]:
                    if j not in ROWS[r] and s.grid[j] == 0 and d in s.cand[j]:
                        rj, cj = idx_to_rc(j)
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Box/Line Reduction",
                            explanation=(
                                f"In row {r+1}, all candidates for {d} lie in box {b+1}. "
                                f"So {d} cannot appear elsewhere in that box outside row {r+1}; "
                                f"remove {d} from r{rj+1}c{cj+1}."
                            )
                        )]

    # Column claiming
    for c in range(9):
        for d in range(1, 10):
            locs = [i for i in COLS[c] if s.grid[i] == 0 and d in s.cand[i]]
            if len(locs) < 2:
                continue
            boxes = {box_index(*idx_to_rc(i)) for i in locs}
            if len(boxes) == 1:
                b = next(iter(boxes))
                for j in BOXES[b]:
                    if j not in COLS[c] and s.grid[j] == 0 and d in s.cand[j]:
                        rj, cj = idx_to_rc(j)
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Box/Line Reduction",
                            explanation=(
                                f"In column {c+1}, all candidates for {d} lie in box {b+1}. "
                                f"So {d} cannot appear elsewhere in that box outside column {c+1}; "
                                f"remove {d} from r{rj+1}c{cj+1}."
                            )
                        )]
    return []

def technique_x_wing(s: Sudoku) -> list[Action]:
    # Row-based X-Wing
    for d in range(1, 10):
        row_positions = []
        for r in range(9):
            cols = sorted({idx_to_rc(i)[1] for i in ROWS[r] if s.grid[i] == 0 and d in s.cand[i]})
            if len(cols) == 2:
                row_positions.append((r, tuple(cols)))

        for (r1, cpair1), (r2, cpair2) in combinations(row_positions, 2):
            if cpair1 == cpair2:
                c1, c2 = cpair1
                for r in range(9):
                    if r in (r1, r2):
                        continue
                    for c in (c1, c2):
                        j = rc_to_idx(r, c)
                        if s.grid[j] == 0 and d in s.cand[j]:
                            return [Action(
                                kind="elim",
                                idx=j,
                                digit=d,
                                technique="X-Wing",
                                explanation=(
                                    f"For digit {d}, rows {r1+1} and {r2+1} each allow {d} only in columns {c1+1} and {c2+1}. "
                                    f"Thus {d} is locked to those columns in those rows, so eliminate {d} from r{r+1}c{c+1}."
                                )
                            )]

    # Column-based X-Wing
    for d in range(1, 10):
        col_positions = []
        for c in range(9):
            rows = sorted({idx_to_rc(i)[0] for i in COLS[c] if s.grid[i] == 0 and d in s.cand[i]})
            if len(rows) == 2:
                col_positions.append((c, tuple(rows)))

        for (c1, rpair1), (c2, rpair2) in combinations(col_positions, 2):
            if rpair1 == rpair2:
                r1, r2 = rpair1
                for c in range(9):
                    if c in (c1, c2):
                        continue
                    for r in (r1, r2):
                        j = rc_to_idx(r, c)
                        if s.grid[j] == 0 and d in s.cand[j]:
                            return [Action(
                                kind="elim",
                                idx=j,
                                digit=d,
                                technique="X-Wing",
                                explanation=(
                                    f"For digit {d}, columns {c1+1} and {c2+1} each allow {d} only in rows {r1+1} and {r2+1}. "
                                    f"Thus {d} is locked to those rows in those columns, so eliminate {d} from r{r+1}c{c+1}."
                                )
                            )]
    return []

def technique_swordfish(s: Sudoku) -> list[Action]:
    # Row-based Swordfish
    for d in range(1, 10):
        row_to_cols: dict[int, list[int]] = {}
        for r in range(9):
            cols = sorted({idx_to_rc(i)[1] for i in ROWS[r] if s.grid[i] == 0 and d in s.cand[i]})
            if 2 <= len(cols) <= 3:
                row_to_cols[r] = cols

        for rset in combinations(row_to_cols.keys(), 3):
            cols_union = sorted(set().union(*(row_to_cols[r] for r in rset)))
            if len(cols_union) != 3:
                continue
            for r in range(9):
                if r in rset:
                    continue
                for c in cols_union:
                    j = rc_to_idx(r, c)
                    if s.grid[j] == 0 and d in s.cand[j]:
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Swordfish",
                            explanation=(
                                f"For digit {d}, rows {[x+1 for x in rset]} restrict {d} to columns {[x+1 for x in cols_union]}. "
                                f"So remove {d} from r{r+1}c{c+1}."
                            )
                        )]

    # Column-based Swordfish
    for d in range(1, 10):
        col_to_rows: dict[int, list[int]] = {}
        for c in range(9):
            rows = sorted({idx_to_rc(i)[0] for i in COLS[c] if s.grid[i] == 0 and d in s.cand[i]})
            if 2 <= len(rows) <= 3:
                col_to_rows[c] = rows

        for cset in combinations(col_to_rows.keys(), 3):
            rows_union = sorted(set().union(*(col_to_rows[c] for c in cset)))
            if len(rows_union) != 3:
                continue
            for c in range(9):
                if c in cset:
                    continue
                for r in rows_union:
                    j = rc_to_idx(r, c)
                    if s.grid[j] == 0 and d in s.cand[j]:
                        return [Action(
                            kind="elim",
                            idx=j,
                            digit=d,
                            technique="Swordfish",
                            explanation=(
                                f"For digit {d}, columns {[x+1 for x in cset]} restrict {d} to rows {[x+1 for x in rows_union]}. "
                                f"So remove {d} from r{r+1}c{c+1}."
                            )
                        )]
    return []

def technique_xyz_wing(s: Sudoku) -> list[Action]:
    # Pivot: 3 candidates; Wings: 2 candidates each; eliminate the shared digit z.
    bivals = [i for i in range(81) if s.grid[i] == 0 and len(s.cand[i]) == 2]
    triv = [i for i in range(81) if s.grid[i] == 0 and len(s.cand[i]) == 3]

    for p in triv:
        P = set(s.cand[p])
        for a in PEERS[p]:
            if a not in bivals:
                continue
            A = set(s.cand[a])
            if not A.issubset(P):
                continue
            for b in PEERS[p]:
                if b == a or b not in bivals:
                    continue
                B = set(s.cand[b])
                if not B.issubset(P):
                    continue

                inter = A & B
                if len(inter) != 1:
                    continue
                z = next(iter(inter))

                # Candidate elimination applies to any cell that sees BOTH wings
                common = PEERS[a] & PEERS[b]
                for t in common:
                    if s.grid[t] == 0 and z in s.cand[t]:
                        rp, cp = idx_to_rc(p)
                        ra, ca = idx_to_rc(a)
                        rb, cb = idx_to_rc(b)
                        rt, ct = idx_to_rc(t)
                        return [Action(
                            kind="elim",
                            idx=t,
                            digit=z,
                            technique="XYZ-Wing",
                            explanation=(
                                f"Pivot r{rp+1}c{cp+1} has candidates {sorted(P)}. "
                                f"Wings r{ra+1}c{ca+1}={sorted(A)} and r{rb+1}c{cb+1}={sorted(B)} share digit {z}. "
                                f"Any cell that sees both wings cannot be {z}, so remove {z} from r{rt+1}c{ct+1}."
                            )
                        )]
    return []

def _strong_links_for_digit(s: Sudoku, d: int) -> list[tuple[int, int]]:
    # Strong link: a unit has exactly two candidates for digit d.
    edges: list[tuple[int, int]] = []
    for unit in UNITS:
        cells = [i for i in unit if s.grid[i] == 0 and d in s.cand[i]]
        if len(cells) == 2:
            edges.append((cells[0], cells[1]))
    return edges

def technique_coloring(s: Sudoku) -> list[Action]:
    # Simple two-coloring on strong-link graph.
    for d in range(1, 10):
        edges = _strong_links_for_digit(s, d)
        if not edges:
            continue

        adj: dict[int, list[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        color: dict[int, int] = {}
        for start in adj.keys():
            if start in color:
                continue
            color[start] = 0
            stack = [start]
            while stack:
                x = stack.pop()
                for y in adj.get(x, []):
                    if y not in color:
                        color[y] = 1 - color[x]
                        stack.append(y)

        colored = list(color.keys())
        if not colored:
            continue

        # Eliminate: if a cell sees both colors, it cannot be d.
        for i in range(81):
            if s.grid[i] != 0 or d not in s.cand[i]:
                continue
            sees0 = any((c in PEERS[i]) and (color.get(c) == 0) for c in colored)
            sees1 = any((c in PEERS[i]) and (color.get(c) == 1) for c in colored)
            if sees0 and sees1:
                r, c = idx_to_rc(i)
                return [Action(
                    kind="elim",
                    idx=i,
                    digit=d,
                    technique="Coloring",
                    explanation=(
                        f"For digit {d}, strong-link coloring yields two color classes. "
                        f"r{r+1}c{c+1} sees a candidate of each color, so r{r+1}c{c+1} cannot be {d}."
                    )
                )]
    return []

def technique_aic_limited(s: Sudoku, max_depth: int = 6) -> list[Action]:
    """
    Limited AIC (Alternating Inference Chains), digit-focused and bounded depth.

    Nodes: "cell i can be digit d".
    Strong edge: exactly 2 candidates for d in a unit (either/or).
    Weak edge: two candidates for d in same unit cannot both be true (peer relation).

    We alternate strong/weak edges up to max_depth and look for a simple contradiction loop
    that implies the start candidate cannot be true.
    """
    for d in range(1, 10):
        nodes = [i for i in range(81) if s.grid[i] == 0 and d in s.cand[i]]
        if len(nodes) < 4:
            continue

        strong = {i: set() for i in nodes}
        for unit in UNITS:
            locs = [i for i in unit if s.grid[i] == 0 and d in s.cand[i]]
            if len(locs) == 2:
                a, b = locs
                if a in strong: strong[a].add(b)
                if b in strong: strong[b].add(a)

        weak = {i: {p for p in PEERS[i] if s.grid[p] == 0 and d in s.cand[p]} for i in nodes}

        for start in nodes:
            queue = [(start, 0, 0)]     # (node, next_type 0=strong 1=weak, depth)
            visited = {(start, 0)}

            while queue:
                node, next_type, depth = queue.pop(0)
                if depth >= max_depth:
                    continue

                nbrs = strong[node] if next_type == 0 else weak[node]
                for nb in nbrs:
                    state = (nb, 1 - next_type)
                    if state in visited:
                        continue
                    visited.add(state)

                    # Simple contradiction closure:
                    if nb in weak[start] and next_type == 0:
                        r, c = idx_to_rc(start)
                        return [Action(
                            kind="elim",
                            idx=start,
                            digit=d,
                            technique="AIC (limited)",
                            explanation=(
                                f"For digit {d}, an alternating strong/weak chain starting at r{r+1}c{c+1} closes into a contradiction loop. "
                                f"So eliminate {d} from r{r+1}c{c+1}."
                            )
                        )]

                    queue.append((nb, 1 - next_type, depth + 1))
    return []

def technique_forcing_chains(s: Sudoku) -> list[Action]:
    """
    Contradiction probing (lightweight forcing chains):

    Pick an empty cell with 2–3 candidates (small branching).
    For each candidate, assume it, propagate forced singles silently.
    If contradiction occurs, that candidate is impossible => eliminate it.
    """
    choices = [(len(s.cand[i]), i) for i in range(81) if s.grid[i] == 0 and len(s.cand[i]) > 1]
    if not choices:
        return []
    choices.sort()
    _, idx = choices[0]
    options = sorted(s.cand[idx])
    if len(options) > 3:
        return []

    r, c = idx_to_rc(idx)

    for d in options:
        trial = s.clone()
        try:
            # silent place + silent propagation
            trial._place_silent(idx, d)

            changed = True
            while changed:
                changed = False
                for i in range(81):
                    if trial.grid[i] == 0 and len(trial.cand[i]) == 1:
                        dd = next(iter(trial.cand[i]))
                        trial._place_silent(i, dd)
                        changed = True
                trial._assert_consistent()

        except ValueError:
            return [Action(
                kind="elim",
                idx=idx,
                digit=d,
                technique="Forcing Chains",
                explanation=(
                    f"Assume r{r+1}c{c+1}={d} and propagate forced singles. This yields a contradiction, "
                    f"so r{r+1}c{c+1} cannot be {d}."
                )
            )]

    return []

def technique_unique_rectangle(s: Sudoku) -> list[Action]:
    """
    Unique Rectangle (UR Type 1):
      - Consider any 2x2 rectangle formed by two rows and two columns.
      - If three cells are exactly {a,b} and the fourth is {a,b,x,...},
        remove an extra candidate from the fourth to avoid a deadly pattern.
    """
    for r1, r2 in combinations(range(9), 2):
        for c1, c2 in combinations(range(9), 2):
            rect = [rc_to_idx(r1, c1), rc_to_idx(r1, c2), rc_to_idx(r2, c1), rc_to_idx(r2, c2)]
            if any(s.grid[i] != 0 for i in rect):
                continue

            csets = [s.cand[i] for i in rect]
            pair_cells = [cs for cs in csets if len(cs) == 2]
            if len(pair_cells) < 3:
                continue

            pair = pair_cells[0]
            if any(cs != pair for cs in pair_cells):
                continue

            for i in rect:
                if s.cand[i] != pair:
                    extras = s.cand[i] - pair
                    if extras:
                        d_elim = next(iter(extras))
                        rr, cc = idx_to_rc(i)
                        return [Action(
                            kind="elim",
                            idx=i,
                            digit=d_elim,
                            technique="Unique Rectangle",
                            explanation=(
                                f"Three cells in this 2x2 rectangle are exactly {sorted(pair)}. "
                                f"To prevent a deadly two-solution swap, remove extra candidate {d_elim} from r{rr+1}c{cc+1}."
                            )
                        )]
    return []


# =============================================================================
# Technique list (order matters)
# =============================================================================

TECHNIQUES = [
    ("Naked Singles", technique_naked_singles),
    ("Hidden Singles", technique_hidden_singles),
    ("Naked Pairs", technique_naked_pairs),
    ("Hidden Pairs", technique_hidden_pairs),
    ("Naked Triples", lambda s: technique_naked_k(s, 3)),
    ("Naked Quads",   lambda s: technique_naked_k(s, 4)),
    ("Hidden Triples", lambda s: technique_hidden_k(s, 3)),
    ("Hidden Quads",   lambda s: technique_hidden_k(s, 4)),
    ("Pointing Pairs/Triples", technique_pointing_pairs_triples),
    ("Box/Line Reduction", technique_box_line_reduction),
    ("X-Wing", technique_x_wing),
    ("Swordfish", technique_swordfish),
    ("XYZ-Wing", technique_xyz_wing),
    ("Coloring", technique_coloring),
    ("Chains (AIC)", technique_aic_limited),
    ("Forcing Chains", technique_forcing_chains),
    ("Unique Rectangle", technique_unique_rectangle),
]


# =============================================================================
# Deduction loop (prints ALL actions including implied ones)
# =============================================================================

def deduction_loop(s: Sudoku, indent: int = 0) -> bool:
    """
    Apply techniques in order until stuck or solved.

    Prints:
      - the explicit technique action
      - then every implied action returned by Sudoku.apply():
          - peer eliminations caused by placements (if enabled)
          - forced placements from propagation
          - peer eliminations from those forced placements
    """
    try:
        while True:
            if s.is_solved():
                return True

            made_progress = False

            for _, fn in TECHNIQUES:
                acts = fn(s)
                if acts:
                    a = acts[0]
                    print_action(a, indent=indent)

                    implied = s.apply(a)
                    for ia in implied:
                        print_action(ia, indent=indent)

                    made_progress = True
                    break  # restart from technique #1

            if not made_progress:
                return True

    except ValueError as e:
        print(("  " * indent) + f"[Contradiction] {e}")
        return False


# =============================================================================
# Guessing / Backtracking (prints guess + implied moves)
# =============================================================================

def solve_with_guessing(s: Sudoku, depth: int = 0) -> Sudoku | None:
    """
    Backtracking with MRV (minimum remaining values):
      - Exhaust deductions at this branch
      - If unsolved, choose the empty cell with smallest candidate count
      - Try each candidate recursively
    """
    if not deduction_loop(s, indent=depth):
        return None
    if s.is_solved():
        return s

    # MRV choice
    choices = [(len(s.cand[i]), i) for i in range(81) if s.grid[i] == 0]
    choices.sort()
    _, idx = choices[0]
    r, c = idx_to_rc(idx)

    for d in sorted(s.cand[idx]):
        trial = s.clone()

        guess = Action(
            kind="place",
            idx=idx,
            digit=d,
            technique="Guess",
            explanation=f"Trial assumption (branch depth {depth}): set r{r+1}c{c+1}={d}."
        )

        print_action(guess, indent=depth)

        try:
            implied = trial.apply(guess)
            for ia in implied:
                print_action(ia, indent=depth)
        except ValueError:
            print(("  " * depth) + f"[Guess] Contradiction immediately; backtrack.")
            continue

        result = solve_with_guessing(trial, depth + 1)
        if result is not None:
            return result

        print(("  " * depth) + f"[Guess] Backtrack from r{r+1}c{c+1}={d}")

    return None


# =============================================================================
# Input parsing and main
# =============================================================================

def parse_grid_from_text(text: str) -> list[int]:
    """
    Extract digits and '.' from text; ignore whitespace and other characters.
    Expect exactly 81 cells after filtering.
    """
    raw = "".join(ch for ch in text if ch in "0123456789.")
    if len(raw) != 81:
        raise ValueError(f"Expected 81 cells, got {len(raw)} after filtering")
    return [0 if ch in ".0" else int(ch) for ch in raw]

def solve_sudoku(text: str) -> Sudoku:
    s = Sudoku(parse_grid_from_text(text))

    print("Initial puzzle:")
    print(s.pretty())
    print()

    ok = deduction_loop(s, indent=0)
    if not ok:
        raise ValueError("Puzzle is inconsistent from the start.")

    if s.is_solved():
        print("\nSolved by deduction!")
        return s

    print("\n[Stuck] No more deductions apply. Switching to Guessing (Trial & Error).")
    solved = solve_with_guessing(s, depth=0)
    if solved is None:
        raise ValueError("No solution found (unexpected for a solvable Sudoku).")
    return solved

if __name__ == "__main__":
    puzzle_text = sys.argv[1] if len(sys.argv) >= 2 else sys.stdin.read()
    solved = solve_sudoku(puzzle_text)

    print("\nFinal solution:")
    print(solved.pretty())

"""
    Puzzle: 1...34..8.7.68..3...821.7.4.54.9.68.91.5.8.2..8.3....53.59.6871..6....4...1.7.2..
    Puzzle: 100805030500090680000000507020010300000948006000600000970006000600070104010000000
"""