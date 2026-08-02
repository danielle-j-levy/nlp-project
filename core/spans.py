"""Span selection methods for position-independent KV reuse of code reads.

A *span* is the unit of cache storage and lookup: a contiguous run of the
observation's tokens, keyed by its content hash. Under a position-independent
cache, a span is reusable anywhere, so what determines reuse is purely whether
some earlier observation emitted a span with byte-identical content.

That makes span *boundary placement* the whole game. Two reads can share 95% of
their text and still share zero spans if the boundaries land at different offsets.
The methods below are the candidate boundary rules, in four families:

  read-relative  boundaries counted from the start of THIS observation
                 (fixed_tok / fixed_line / fixed_char). Misaligns whenever a
                 later read starts at a different offset in the same file.
  file-anchored  boundaries at absolute file line numbers (anchor_line).
                 Aligns across reads of the same region, but every insertion or
                 deletion above the region shifts every later boundary.
  content-defined  boundaries where the content itself says so, via a rolling
                 hash (cdc_line) or a syntactic cue (blank / defline / indent0).
                 Shift-invariant by construction.
  structural     boundaries at AST node starts (ast_win / ast_acc). Also
                 shift-invariant, but needs a parseable window.

Every method returns a partition of the SAME line sequence (fixed_tok partitions
the same token sequence), so total tokens are identical across methods and the
reuse fractions are directly comparable.
"""

import ast
import hashlib
import re
import textwrap

# ---------------------------------------------------------------- primitives

def h64(s):
    """Stable 64-bit content hash. Not `hash()`: that is seed-randomized per
    process, which would make CDC boundaries and cache keys irreproducible."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"),
                                          digest_size=8).digest(), "big")


def spans_from_bounds(n, bounds):
    """Turn a set of 0-based start indices into a list of (start, end) line
    ranges covering [0, n). Always includes 0 so the partition is total."""
    b = sorted({0} | {x for x in bounds if 0 < x < n})
    return [(b[i], b[i + 1] if i + 1 < len(b) else n) for i in range(len(b))]


# ------------------------------------------------------- read-relative family

def m_fixed_line(lines, k):
    return spans_from_bounds(len(lines), range(0, len(lines), k))


def m_fixed_char(lines, budget):
    """Fixed character budget, cut at the nearest line boundary. Cutting
    mid-line would only ever lose reuse relative to this, and would break the
    equal-token-total property that keeps the comparison fair."""
    bounds, acc = [], 0
    for i, (_, t) in enumerate(lines):
        if acc >= budget:
            bounds.append(i)
            acc = 0
        acc += len(t) + 1
    return spans_from_bounds(len(lines), bounds)


# fixed_tok is handled in the driver: it partitions the token sequence directly
# rather than the line sequence, so it needs the tokenized form.

# ------------------------------------------------------- file-anchored family

def m_pack_tokens(cum, target, min_tail=0):
    """Greedy line-aligned packing to a token budget.

    A span absorbs whole lines until its total reaches `target`, including the
    line that crosses it, so a 500-token budget yields spans of 503 or 522 rather
    than exactly 500 -- cutting mid-line would split a token run that recurs
    intact elsewhere. A trailing span shorter than `min_tail` is merged back into
    its predecessor instead of being emitted as a runt.

    Boundaries here are read-relative: they are counted from the start of this
    observation, so two reads that open the same file at different offsets land
    their cuts in different places."""
    n = len(cum) - 1
    spans, start = [], 0
    while start < n:
        i = start
        while i < n and cum[i + 1] - cum[start] < target:
            i += 1
        end = min(i + 1, n)
        spans.append((start, end))
        start = end
    if min_tail and len(spans) > 1:
        a, b = spans[-1]
        if cum[b] - cum[a] < min_tail:
            spans[-2] = (spans[-2][0], b)
            spans.pop()
    return spans


def snap_pack(cum, cand, target=500, max_over=250):
    """Target-size packing whose cuts land on content boundaries.

    Uniform packing hits the target size exactly but its cuts are position-
    derived, so an insertion above shifts every boundary below it. Merging a
    content-defined partition up to the target keeps the cuts stable but
    overshoots to the next content unit, and the overshoot is paid twice over in
    a bucket-delivery scheme: bigger buckets mean more retrofit.

    This takes both: accumulate to `target`, then cut at the first candidate
    boundary at or after that point, giving up and cutting at the crossing line
    if no candidate appears within `max_over` extra tokens. Sizes stay near the
    target; cuts sit on content wherever content offers one.

    `cand` is the set of line indices that open a content unit.
    """
    n = len(cum) - 1
    bounds, start = [0], 0
    while start < n:
        j = start + 1
        while j < n and cum[j] - cum[start] < target:
            j += 1
        if j >= n:
            break
        k = j
        while k < n and cum[k] - cum[start] <= target + max_over:
            k += 1
        pick = next((x for x in range(j, k) if x in cand), j)
        bounds.append(pick)
        start = pick
    spans = [(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else n)
             for i in range(len(bounds))]
    # a hard minimum cacheable unit means no bucket may fall below it
    if len(spans) > 1 and cum[spans[-1][1]] - cum[spans[-1][0]] < target:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def pack_file_bounds(lo, run_tok, target):
    """The same packing, but anchored to the FILE: cuts accumulate from the start
    of the contiguous run the cache has reconstructed, not from the start of this
    read, so overlapping reads of an unedited file agree on where spans begin.

    `run_tok` is the per-line token count of that run, starting at absolute line
    `lo`. Returns absolute line numbers that start a span."""
    bounds, acc = {lo}, 0
    for i, t in enumerate(run_tok):
        acc += t
        if acc >= target:
            bounds.add(lo + i + 1)
            acc = 0
    return bounds


def m_anchor_line(lines, k):
    """Boundaries wherever the ABSOLUTE file line number crosses a multiple of
    k. Two overlapping reads of an unedited file agree on these boundaries even
    though they start at different offsets."""
    bounds = [i for i in range(1, len(lines)) if lines[i][0] // k != lines[i - 1][0] // k]
    return spans_from_bounds(len(lines), bounds)


# ----------------------------------------------------- content-defined family

def m_cdc_line(lines, target, min_lines=None, max_lines=None):
    """Content-defined chunking over lines (FastCDC-style, line-granular).

    A boundary falls after any line whose content hash has `log2(target)` low
    bits clear. Because the cut points are a function of content alone, adding
    or removing lines anywhere only perturbs the chunks that actually changed:
    the classic shift-invariance property that read-relative and file-anchored
    boundaries both lack."""
    min_lines = min_lines or max(1, target // 4)
    max_lines = max_lines or target * 4
    mask = (1 << max(1, target.bit_length() - 1)) - 1
    bounds, run = [], 0
    for i, (_, t) in enumerate(lines):
        run += 1
        if run < min_lines:
            continue
        if run >= max_lines or (h64(t) & mask) == 0:
            bounds.append(i + 1)
            run = 0
    return spans_from_bounds(len(lines), bounds)


def m_blank(lines):
    """Boundary at the start of each blank-line-separated paragraph."""
    bounds = [i for i in range(1, len(lines))
              if lines[i][1].strip() and not lines[i - 1][1].strip()]
    return spans_from_bounds(len(lines), bounds)


DEF_RE = re.compile(r"^\s*(async\s+def\s|def\s|class\s|@)")


def m_defline(lines):
    """Parser-free structural split: boundary before each def/class/decorator,
    with a decorator run kept attached to the definition it decorates."""
    bounds = []
    for i in range(1, len(lines)):
        if DEF_RE.match(lines[i][1]) and not lines[i - 1][1].lstrip().startswith("@"):
            bounds.append(i)
    return spans_from_bounds(len(lines), bounds)


def m_indent0(lines):
    """Boundary before each line that starts at column 0 with real content."""
    bounds = [i for i in range(1, len(lines))
              if lines[i][1][:1].strip() and not lines[i][1].startswith((")", "]", "}"))]
    return spans_from_bounds(len(lines), bounds)


def _indent(t):
    return len(t) - len(t.lstrip())


def m_indent_min(lines):
    """`indent0` generalized to a window that starts mid-block: boundary before
    each line at the window's OWN minimum indentation level. Approximates the
    top-level-statement split of an AST without needing the text to parse, which
    is the point of including it -- it isolates how much of the AST methods'
    advantage is really just "split at the outermost indentation level"."""
    body = [_indent(t) for _, t in lines if t.strip()]
    if not body:
        return spans_from_bounds(len(lines), [])
    m = min(body)
    bounds = [i for i in range(1, len(lines))
              if lines[i][1].strip() and _indent(lines[i][1]) == m
              and not lines[i][1].lstrip().startswith((")", "]", "}", "elif", "else", "except", "finally"))]
    return spans_from_bounds(len(lines), bounds)


def merge_to(spans, cum, min_tokens):
    """Merge consecutive spans until each holds >= min_tokens tokens.

    Two purposes. Practically, a cache entry of six tokens is not worth its
    metadata, so any real deployment merges. Experimentally, it gives the
    structural methods -- which otherwise have no size parameter -- a knob, so
    every method can be compared against the others at equal mean span size
    instead of winning merely by being finer-grained."""
    if min_tokens <= 0:
        return spans
    out, cur = [], None
    for a, b in spans:
        cur = (cur[0], b) if cur else (a, b)
        if cum[cur[1]] - cum[cur[0]] >= min_tokens:
            out.append(cur)
            cur = None
    if cur:
        if out:
            out[-1] = (out[-1][0], cur[1])
        else:
            out.append(cur)
    return out


# ----------------------------------------------------------- structural (AST)

def _ast_bounds(text, granularity):
    """Absolute-in-text (1-based) start lines of AST nodes, or None if the text
    does not parse. `granularity='top'` splits at top-level statements only;
    `'func'` additionally splits class bodies into their methods."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        try:
            tree = ast.parse(textwrap.dedent(text))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None

    def start(node):
        decs = getattr(node, "decorator_list", None) or []
        return min([node.lineno] + [d.lineno for d in decs])

    out = set()
    for node in tree.body:
        out.add(start(node))
        if granularity == "func" and isinstance(node, ast.ClassDef):
            for sub in node.body:
                out.add(start(sub))
    return out


MAX_TAIL_DROP = 20


def robust_parse(texts, granularity):
    """Parse a window that an agent viewer cut out of the middle of a file.

    A raw `ast.parse` of such a window fails on most of them, and not for
    interesting reasons: the window opens inside a class or function body, so
    line one is indented and Python reports "unexpected indent". Two repairs,
    both deterministic:

      head  skip to the first line at the window's own minimum indentation and
            dedent by it, so the fragment reads as a well-formed suite. The
            skipped lines cannot be split structurally anyway -- they are the
            tail of a block whose header is off-screen -- so they become one
            leading span.
      tail  a window truncated mid-string or mid-bracket never parses; drop
            trailing lines until it does. The dropped lines attach to the last
            span.

    Returns (offset, bounds) with bounds 1-based within texts[offset:], or None.
    """
    body = [_indent(t) for t in texts if t.strip()]
    if not body:
        return None
    m = min(body)
    off = next((i for i, t in enumerate(texts) if t.strip() and _indent(t) == m), 0)
    cut = [t[m:] if t.strip() else "" for t in texts[off:]]
    for drop in range(min(MAX_TAIL_DROP, max(0, len(cut) - 1)) + 1):
        sub = cut[:len(cut) - drop] if drop else cut
        if not sub:
            break
        b = _ast_bounds("\n".join(sub), granularity)
        if b is not None:
            return off, b
    return None


def m_ast_window(lines, granularity="func"):
    """AST boundaries computed from the observation window alone. Returns
    (spans, ok) where ok=False means the window did not parse and the caller
    should fall back."""
    r = robust_parse([t for _, t in lines], granularity)
    if r is None:
        return None, False
    off, b = r
    return spans_from_bounds(len(lines), {off + x - 1 for x in b}), True


def m_ast_accum(lines, accum_text, accum_first_line, granularity="func"):
    """AST boundaries computed from the file text the cache has ACCUMULATED
    from earlier reads, then projected onto this window by absolute line number.

    This is the stateful variant: a partial window that does not parse on its
    own can still be split structurally if an earlier read gave the cache enough
    of the file to parse. Returns (spans, ok)."""
    if not accum_text:
        return None, False
    r = robust_parse(accum_text.split("\n"), granularity)
    if r is None:
        return None, False
    off, b = r
    abs_bounds = {x + off + accum_first_line - 1 for x in b}   # -> absolute file lines
    idx = {ln: i for i, (ln, _) in enumerate(lines)}
    return spans_from_bounds(len(lines), {idx[a] for a in abs_bounds if a in idx}), True


# ------------------------------------------- content-defined buckets (cdc500)

def line_boundary_values(lines, window_bytes=64):
    """A uniform [0,1) value for the boundary after each line, derived from the
    preceding `window_bytes` of run text and nothing else.

    Deliberately excluded from the hash input: line number, byte offset, file
    path, run index, and the previous boundary. That exclusion is the whole
    point -- it is what lets the value sequence downstream of an edit become
    identical again once `window_bytes` of unchanged text have passed, so the
    chunker re-cuts at the old boundaries instead of shifting everything below.
    """
    body = "\n".join(lines).replace("\r\n", "\n").encode("utf-8", "replace")
    out, off = [], 0
    for ln in lines:
        off += len(ln.encode("utf-8", "replace")) + 1     # +1 for the newline
        w = body[max(0, off - window_bytes):off]
        out.append(int.from_bytes(hashlib.blake2b(w, digest_size=8).digest(),
                                  "big") / 2.0 ** 64)
    return out


def cdc_pack_lines(lines, line_tokens, p_early, p_late,
                   min_tokens=500, target_tokens=650, max_tokens=900,
                   window_bytes=64, blank_boost=1.0):
    """Line-sampled FastCDC: hard minimum, a strict content mask before the
    target, a looser one after it, and a hard maximum as a backstop.

    `blank_boost` multiplies the cut probability at a blank line, which biases
    boundaries toward readable places without snapping to them -- snapping would
    reintroduce a position-dependent search and lose resynchronisation."""
    u = line_boundary_values(lines, window_bytes)
    spans, start, total = [], 0, 0
    for i, nt in enumerate(line_tokens):
        total += nt
        if total < min_tokens:
            continue
        p = p_late if total >= target_tokens else p_early
        if blank_boost != 1.0 and not lines[i].strip():
            p = min(1.0, p * blank_boost)
        if total >= max_tokens or u[i] < p:
            spans.append((start, i + 1))
            start, total = i + 1, 0
    if start < len(lines):
        _finish_tail(spans, start, len(lines), line_tokens, u,
                     min_tokens, target_tokens)
    return spans


def _finish_tail(spans, tail_start, tail_end, line_tokens, u,
                 min_tokens, target_tokens):
    """Rather than merging a short tail into its predecessor unconditionally,
    try to repartition predecessor+tail into two legal buckets, cutting at the
    strongest content boundary that leaves >= min_tokens on each side. Keeps the
    disturbance to the last one or two buckets."""
    tail_tok = sum(line_tokens[tail_start:tail_end])
    if tail_tok >= min_tokens:
        spans.append((tail_start, tail_end))
        return
    if not spans:
        spans.append((tail_start, tail_end))      # whole run under the floor
        return
    a, _ = spans.pop()
    combined = sum(line_tokens[a:tail_end])
    best, left = None, 0
    for b in range(a + 1, tail_end):
        left += line_tokens[b - 1]
        right = combined - left
        if left >= min_tokens and right >= min_tokens:
            d = abs(left - target_tokens) + abs(right - target_tokens)
            cand = (u[b - 1], d, b)
            if best is None or cand < best:
                best = cand
    if best:
        spans.append((a, best[2]))
        spans.append((best[2], tail_end))
    else:
        spans.append((a, tail_end))               # two legal buckets impossible
