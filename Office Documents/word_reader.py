#!/usr/bin/env python3
"""
docx_format_breaks.py

Reads a .docx (Microsoft Word) file and emits the document text while inserting
explicit START/END markers whenever formatting context changes:

- Paragraph boundaries + paragraph style changes
- Run-level bold / italic (including bold+italic)
- Run style changes (character styles)
- Lists (bullet vs numbered when detectable) + list level + list start/end
- Tables + row/cell boundaries (with table start/end)

Output is plain text with lightweight markers like:
  [[P_START style="Normal" list="bullet" lvl=0]]
  [[B_START]]bold text[[B_END]]
  [[I_START]]italic text[[I_END]]
  [[TABLE_START]]
  [[CELL_START r=0 c=1]]...

Notes / limitations (Word is… complicated):
- “Effective” bold/italic can be inherited (run -> character style -> paragraph style -> defaults).
  This script resolves inheritance partially: run formatting first, then run style,
  then paragraph style, then Normal. It won’t perfectly match Word in every edge case.
- List type (bullet vs numbered) requires reading numbering definitions. This script
  parses the numbering part and usually identifies bullet/decimal/etc per level.
  If it can’t, it will label the list type as "list".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

from docx import Document
from docx.document import Document as DocxDocument
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

# python-docx exposes the underlying XML via ._element, which is sometimes required
# for features like numbering/list detection.
from docx.oxml.ns import qn


# -----------------------------
# Output helpers (markers)
# -----------------------------

def emit(out, s: str) -> None:
    out.write(s)


def emit_line(out, s: str) -> None:
    out.write(s + "\n")


def esc_attr(s: str) -> str:
    # Small attribute escape for marker readability.
    return (s or "").replace('"', '\\"')


# -----------------------------
# Block iteration (Paragraph / Table in document order)
# -----------------------------

Block = Union[Paragraph, Table]


def iter_block_items(parent: Union[DocxDocument, _Cell]) -> Iterator[Block]:
    """
    Yield each paragraph and table child within *parent*, in document order.
    Works for the main document and for table cells (which can contain paragraphs/tables).
    """
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


# -----------------------------
# Numbering (list) parsing
# -----------------------------

@dataclass(frozen=True)
class ListInfo:
    num_id: str
    ilvl: int
    fmt: str  # "bullet", "decimal", "lowerLetter", etc. (Word numFmt)
    kind: str  # "bullet" | "numbered" | "list"


def _safe_int(x: Optional[str], default: int = 0) -> int:
    try:
        return int(x) if x is not None else default
    except ValueError:
        return default


def build_numbering_maps(doc: Document) -> Tuple[Dict[str, str], Dict[str, Dict[int, str]]]:
    """
    Returns:
      numId_to_abstractId: { "5": "3", ... }
      abstractId_to_lvlFmt: { "3": {0:"bullet", 1:"decimal", ...}, ... }
    """
    numId_to_abstractId: Dict[str, str] = {}
    abstractId_to_lvlFmt: Dict[str, Dict[int, str]] = {}

    try:
        numbering_part = doc.part.numbering_part
    except Exception:
        return numId_to_abstractId, abstractId_to_lvlFmt

    root = numbering_part._element  # lxml element

    # Map numId -> abstractNumId
    for num in root.findall(qn("w:num")):
        num_id = num.get(qn("w:numId"))
        abs_elm = num.find(qn("w:abstractNumId"))
        abs_id = abs_elm.get(qn("w:val")) if abs_elm is not None else None
        if num_id and abs_id:
            numId_to_abstractId[num_id] = abs_id

    # Map abstractNumId -> per-level numFmt
    for absnum in root.findall(qn("w:abstractNum")):
        abs_id = absnum.get(qn("w:abstractNumId"))
        if not abs_id:
            continue

        lvl_map: Dict[int, str] = {}
        for lvl in absnum.findall(qn("w:lvl")):
            ilvl = _safe_int(lvl.get(qn("w:ilvl")), 0)
            fmt_elm = lvl.find(qn("w:numFmt"))
            fmt = fmt_elm.get(qn("w:val")) if fmt_elm is not None else ""
            if fmt:
                lvl_map[ilvl] = fmt

        if lvl_map:
            abstractId_to_lvlFmt[abs_id] = lvl_map

    return numId_to_abstractId, abstractId_to_lvlFmt


def paragraph_list_info(
    p: Paragraph,
    numId_to_abstractId: Dict[str, str],
    abstractId_to_lvlFmt: Dict[str, Dict[int, str]],
) -> Optional[ListInfo]:
    """
    Detect whether this paragraph is in a list, and if so return list metadata.
    """
    pPr = p._p.pPr
    if pPr is None or pPr.numPr is None:
        return None

    numId_elm = pPr.numPr.numId
    ilvl_elm = pPr.numPr.ilvl

    num_id = numId_elm.val if numId_elm is not None else None
    ilvl = int(ilvl_elm.val) if ilvl_elm is not None and ilvl_elm.val is not None else 0

    if not num_id:
        return None

    abs_id = numId_to_abstractId.get(str(num_id))
    fmt = ""
    if abs_id:
        fmt = abstractId_to_lvlFmt.get(abs_id, {}).get(ilvl, "")

    # Classify into bullet/numbered/list
    if fmt == "bullet":
        kind = "bullet"
    elif fmt in ("decimal", "decimalZero", "upperRoman", "lowerRoman",
                 "upperLetter", "lowerLetter", "ordinal", "cardinalText",
                 "ordinalText", "hex", "chineseCounting", "aiueo", "iroha"):
        kind = "numbered"
    else:
        kind = "list"

    return ListInfo(num_id=str(num_id), ilvl=ilvl, fmt=fmt or "unknown", kind=kind)


# -----------------------------
# “Effective” formatting resolution (best-effort)
# -----------------------------

@dataclass
class RunFormat:
    bold: bool
    italic: bool
    rstyle: str  # character style name (or "")
    # You can extend this with underline, color, etc.


def _style_name(style) -> str:
    try:
        return style.name or ""
    except Exception:
        return ""


def resolve_bool_chain(*vals: Optional[bool], default: bool = False) -> bool:
    """
    Return the first non-None boolean from vals, else default.
    """
    for v in vals:
        if v is not None:
            return bool(v)
    return default


def effective_run_format(run, para: Paragraph, doc: Document) -> RunFormat:
    """
    Best-effort inheritance for bold/italic:
      run.font -> run.style.font -> paragraph.style.font -> Normal.style.font -> default False
    """
    run_style = getattr(run, "style", None)
    para_style = getattr(para, "style", None)
    normal_style = None
    try:
        normal_style = doc.styles["Normal"]
    except Exception:
        normal_style = None

    bold = resolve_bool_chain(
        run.bold,
        getattr(getattr(run_style, "font", None), "bold", None),
        getattr(getattr(para_style, "font", None), "bold", None),
        getattr(getattr(normal_style, "font", None), "bold", None),
        default=False,
    )
    italic = resolve_bool_chain(
        run.italic,
        getattr(getattr(run_style, "font", None), "italic", None),
        getattr(getattr(para_style, "font", None), "italic", None),
        getattr(getattr(normal_style, "font", None), "italic", None),
        default=False,
    )

    rstyle_name = _style_name(run_style)
    return RunFormat(bold=bold, italic=italic, rstyle=rstyle_name)


# -----------------------------
# Streaming emitter with state transitions
# -----------------------------

@dataclass
class StreamState:
    in_table: int = 0
    # Paragraph context
    pstyle: str = ""
    in_list: bool = False
    list_kind: str = ""
    list_num_id: str = ""
    list_lvl: int = 0
    # Run context
    bold: bool = False
    italic: bool = False
    rstyle: str = ""


def close_run_markers(out, st: StreamState) -> None:
    # Close in reverse “nested” order (style, italic, bold) for nicer symmetry.
    if st.rstyle:
        emit(out, f'[[RSTYLE_END name="{esc_attr(st.rstyle)}"]]')
        st.rstyle = ""
    if st.italic:
        emit(out, "[[I_END]]")
        st.italic = False
    if st.bold:
        emit(out, "[[B_END]]")
        st.bold = False


def open_run_markers(out, st: StreamState, new_fmt: RunFormat) -> None:
    # Open bold/italic then rstyle so the style “wraps” the content last.
    if new_fmt.bold and not st.bold:
        emit(out, "[[B_START]]")
        st.bold = True
    if new_fmt.italic and not st.italic:
        emit(out, "[[I_START]]")
        st.italic = True
    if new_fmt.rstyle and new_fmt.rstyle != st.rstyle:
        emit(out, f'[[RSTYLE_START name="{esc_attr(new_fmt.rstyle)}"]]')
        st.rstyle = new_fmt.rstyle


def switch_run_format(out, st: StreamState, new_fmt: RunFormat) -> None:
    """
    Transition from current run formatting to new formatting by emitting END/START markers.
    """
    # If rstyle changed, close old and open new.
    if st.rstyle != new_fmt.rstyle:
        if st.rstyle:
            emit(out, f'[[RSTYLE_END name="{esc_attr(st.rstyle)}"]]')
        st.rstyle = ""
        # (Re-open later after bold/italic transitions.)

    # Close markers that must end
    if st.italic and not new_fmt.italic:
        emit(out, "[[I_END]]")
        st.italic = False
    if st.bold and not new_fmt.bold:
        emit(out, "[[B_END]]")
        st.bold = False

    # Open markers that must begin
    if new_fmt.bold and not st.bold:
        emit(out, "[[B_START]]")
        st.bold = True
    if new_fmt.italic and not st.italic:
        emit(out, "[[I_START]]")
        st.italic = True

    # Now re-open rstyle if needed
    if new_fmt.rstyle and new_fmt.rstyle != st.rstyle:
        emit(out, f'[[RSTYLE_START name="{esc_attr(new_fmt.rstyle)}"]]')
        st.rstyle = new_fmt.rstyle


def start_paragraph(out, st: StreamState, pstyle: str, li: Optional[ListInfo]) -> None:
    # Close any lingering run markers from previous paragraph.
    close_run_markers(out, st)
    emit(out, "\n")  # paragraph break (even before markers) to visually separate blocks

    # List transitions
    new_in_list = li is not None
    if st.in_list and (not new_in_list or li is None):
        emit_line(out, f'[[LIST_END kind="{st.list_kind}" numId="{st.list_num_id}"]]')
        st.in_list = False
        st.list_kind = ""
        st.list_num_id = ""
        st.list_lvl = 0

    if new_in_list:
        # Start list when entering a new list or changing list identity.
        if (not st.in_list) or (st.list_num_id != li.num_id):
            emit_line(out, f'[[LIST_START kind="{li.kind}" numId="{li.num_id}"]]')
            st.in_list = True
            st.list_kind = li.kind
            st.list_num_id = li.num_id
            st.list_lvl = li.ilvl
        else:
            st.list_lvl = li.ilvl

    # Paragraph style transition (we mark each paragraph start explicitly)
    st.pstyle = pstyle or ""
    if li is not None:
        emit_line(
            out,
            f'[[P_START style="{esc_attr(st.pstyle)}" list="{li.kind}" lvl={li.ilvl} fmt="{esc_attr(li.fmt)}"]]',
        )
    else:
        emit_line(out, f'[[P_START style="{esc_attr(st.pstyle)}"]]' )


def end_paragraph(out, st: StreamState) -> None:
    close_run_markers(out, st)
    emit_line(out, "[[P_END]]")


def emit_paragraph_text(out, st: StreamState, doc: Document, p: Paragraph) -> None:
    # Emit paragraph runs with run-level formatting transitions.
    for run in p.runs:
        txt = run.text
        if not txt:
            continue

        new_fmt = effective_run_format(run, p, doc)
        switch_run_format(out, st, new_fmt)

        # Emit the actual text as-is. (You could normalize whitespace if desired.)
        emit(out, txt)


def emit_table(out, st: StreamState, doc: Document, table: Table,
               numId_to_abstractId, abstractId_to_lvlFmt) -> None:
    # Enter table
    close_run_markers(out, st)
    emit_line(out, "[[TABLE_START]]")
    st.in_table += 1

    for r_i, row in enumerate(table.rows):
        emit_line(out, f"[[ROW_START r={r_i}]]")
        for c_i, cell in enumerate(row.cells):
            emit_line(out, f"[[CELL_START r={r_i} c={c_i}]]")

            # Each cell can contain paragraphs and nested tables.
            for block in iter_block_items(cell):
                if isinstance(block, Paragraph):
                    pstyle = _style_name(block.style)
                    li = paragraph_list_info(block, numId_to_abstractId, abstractId_to_lvlFmt)
                    start_paragraph(out, st, pstyle, li)
                    emit_paragraph_text(out, st, doc, block)
                    end_paragraph(out, st)
                else:
                    emit_table(out, st, doc, block, numId_to_abstractId, abstractId_to_lvlFmt)

            emit_line(out, f"[[CELL_END r={r_i} c={c_i}]]")
        emit_line(out, f"[[ROW_END r={r_i}]]")

    st.in_table -= 1
    emit_line(out, "[[TABLE_END]]")


def process_document(docx_path: str, out) -> None:
    doc = Document(docx_path)
    numId_to_abstractId, abstractId_to_lvlFmt = build_numbering_maps(doc)

    st = StreamState()

    emit_line(out, f'[[DOC_START path="{esc_attr(docx_path)}"]]')
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            pstyle = _style_name(block.style)
            li = paragraph_list_info(block, numId_to_abstractId, abstractId_to_lvlFmt)
            start_paragraph(out, st, pstyle, li)
            emit_paragraph_text(out, st, doc, block)
            end_paragraph(out, st)
        else:
            emit_table(out, st, doc, block, numId_to_abstractId, abstractId_to_lvlFmt)

    # Close list if still open at end of document
    if st.in_list:
        emit_line(out, f'[[LIST_END kind="{st.list_kind}" numId="{st.list_num_id}"]]')
        st.in_list = False

    close_run_markers(out, st)
    emit_line(out, "[[DOC_END]]")


# -----------------------------
# CLI
# -----------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Emit Word document text with START/END markers when formatting changes."
    )
    ap.add_argument("docx", help="Input .docx file path")
    ap.add_argument(
        "-o", "--output", default="",
        help="Optional output file path. If omitted, prints to stdout."
    )
    args = ap.parse_args(argv)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            process_document(args.docx, f)
    else:
        # Use stdout with utf-8 where possible
        out = sys.stdout
        process_document(args.docx, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
