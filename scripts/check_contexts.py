"""Find variables in input/tagger_data.tsv that are really functions.

For every DECLARATION, PARAMETER and ATTRIBUTE row, fetch the source file at its
commit, read the declaration, and classify it:

  FUNCTION_DECL  the name is declared as a function (prototype, method declaration)
  LAMBDA         the variable is initialized with a lambda, method reference or
                 anonymous method
  ANON_CLASS     Java: initialized with an anonymous class (a function object such
                 as a Comparator, or an anonymous subclass)
  CALLABLE       the variable's type is a function pointer, callback or functional
                 interface (the variable stores a function, it isn't one)
  INT_AS_BOOL    informational: a non-boolean integer used only as a flag
  OK / NOT_FOUND / NO_SOURCE

Usage:
  python scripts/check_contexts.py scan  [--out output/reports/context_check.tsv]
  python scripts/check_contexts.py apply REVIEWED.tsv

`scan` writes a review sheet with SUGGESTED_CONTEXT pre-filled. Write the
correct context under REVIEW (blank = take the suggestion, KEEP = leave it).
`apply` updates CONTEXT in input/tagger_data.tsv and appends every change to
output/reports/label_changes_<date>.tsv.

Links come from CODE_URL. headnoun_set rows have no CODE_URL; their permalinks are
read from the headnoun annotation file named by $HEADNOUN_DATA (default
~/headnoun_update/headnoun_data.tsv), and are skipped if it doesn't exist. Files
are cached in ~/.cache/scalar_source.
"""
import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "input" / "tagger_data.tsv"
HEADNOUN = Path(os.environ.get("HEADNOUN_DATA", Path.home() / "headnoun_update" / "headnoun_data.tsv"))
CACHE = Path.home() / ".cache" / "scalar_source"
VARIABLE_CONTEXTS = {"DECLARATION", "PARAMETER", "ATTRIBUTE"}

CALLABLE_TYPE = re.compile(
    r"^PFN|PROC$|_cb$|(^|[^a-z])cb$|callback|closure|^(std::)?function\b|^func\b|"
    r"delegate|runnable|callable|consumer|supplier|predicate|comparator|bifunction|"
    r"^function<|unaryoperator|binaryoperator|^action<|^func<",
    re.IGNORECASE,
)
INT_TYPE = re.compile(r"^(unsigned |signed |const |static |volatile )*"
                      r"(int|short|long|char|u?int\d+(_t)?|u\d+|i\d+|s\d+|gint|guint|size_t|DWORD|unsigned)\b",
                      re.IGNORECASE)
JAVA_LIKE = {"Java", "C#"}
LAMBDA_INIT = [  # (pattern, languages it applies to; None = all)
    (re.compile(r"=\s*\[[^\]]*\]\s*(\(|\{|mutable)"), None),                    # C++ lambda
    (re.compile(r"=\s*(\w+|\([^()]*\))\s*->\s*[{(\w]"), {"Java"}),                 # Java lambda
    (re.compile(r"=\s*(async\s*)?(\w+|\([^()]*\))\s*=>"), {"C#"}),                  # C# lambda
    (re.compile(r"=\s*[\w.<>]+::(new|\w+)\s*[;,)]"), {"Java"}),                      # method reference
    (re.compile(r"=\s*delegate\s*[({]"), {"C#"}),                                    # anonymous method
]
ANON_CLASS = re.compile(r"=\s*new\s+[\w.]+\s*(<[^;]*>)?\s*\([^;]*\)\s*\{")      # Java anonymous class


def raw_url(link):
    m = re.match(r"https://github.com/([^/]+/[^/]+)/blob/([^/]+)/([^#]+)(?:#L(\d+))?", link)
    if not m:
        return None, None
    return f"https://raw.githubusercontent.com/{m[1]}/{m[2]}/{m[3]}", int(m[4]) if m[4] else None


def fetch(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.md5(url.encode()).hexdigest()
    if not path.exists():
        tmp = path.with_suffix(".part")
        result = subprocess.run(["curl", "-sfL", "--max-time", "60", "-o", str(tmp), url])
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            return None
        tmp.rename(path)
    return path.read_text(errors="replace")


def load_rows():
    data = pd.read_csv(DATA, sep="\t", dtype=str, keep_default_na=False)
    rows = data.copy()
    rows["LINK"], rows["NAME"], rows["LINE"] = rows.CODE_URL, rows.ORIGINAL_NAME, rows.SRC_LINE
    if HEADNOUN.exists():
        hn = pd.read_csv(HEADNOUN, sep="\t", dtype=str, keep_default_na=False)
        hn["key"] = (hn.gold_split.str.lower() + "|" + hn.pre_tags + "|"
                     + hn.system.str.split("/").str[-1].str.lower())
        index = {}
        for r in hn.itertuples():
            index.setdefault(r.key, []).append(r)
        for i, r in rows[rows.CODE_URL == ""].iterrows():
            key = f"{r.SPLIT.lower()}|{r.GRAMMAR_PATTERN}|{r.SYSTEM_NAME.lower()}"
            matches = index.get(key, [])
            if len(matches) > 1:  # tie-break on the declared type
                matches = [m for m in matches if m.type == r.TYPE] or matches
            if matches:
                rows.at[i, "LINK"], rows.at[i, "NAME"], rows.at[i, "LINE"] = (
                    matches[0].permalink, matches[0].name, matches[0].line)
    return data, rows


def statement_at(lines, line_no, name):
    """Return (line index, statement text) for the declaration of `name` near `line_no`."""
    word = re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")
    for offset in (0, -1, 1, -2, 2, -3, 3, -5, 5):
        i = line_no - 1 + offset
        if 0 <= i < len(lines) and word.search(lines[i]):
            text = lines[i]
            j = i
            while j + 1 < len(lines) and j - i < 6 and not re.search(r"[;{}]\s*(//.*)?$", lines[j]):
                j += 1
                text += " " + lines[j].strip()
            return i, text
    return None, None


def is_function_declaration(before, after_paren):
    """`before` is the statement text up to the name; `after_paren` follows the name's `(`."""
    prefix = before.strip()
    # A return type sits right before the name, with no assignment, call or control flow.
    if not prefix or not re.search(r"([\w*&>\]]|::)$", prefix):
        return False
    if re.search(r"(=|\.|->|\(|,|\?|<<|(?<!:):)\s*$", prefix):
        return False
    if re.search(r"\b(if|while|for|switch|return|throw|case|else|delete)\b", prefix) or "=" in prefix:
        return False
    depth, args = 1, ""
    for ch in after_paren:
        depth += (ch == "(") - (ch == ")")
        if depth == 0:
            break
        args += ch
    args = re.sub(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'', '""', args).strip()
    rest = after_paren[len(args) + 1:] if depth == 0 else ""
    if not re.match(r"\s*(const\b|noexcept\b|override\b|final\b|throws\b|=\s*0|=\s*default|=\s*delete|;|\{|$|->|//|/\*|[A-Z_]+\b)", rest):
        return False
    # Parameter lists are empty, `void`, or contain a type followed by a name or `*`/`&`.
    # Direct initialization (`QFile f(path);`) passes bare expressions.
    if args.startswith("["):  # a lambda passed to a constructor
        return False
    if args in ("", "void") or "..." in args:
        return True
    return bool(re.search(r"[\w>\]]\s*[*&]|[\w>\]]\s+[*&]*\w", args)) and not re.search(r"^\s*[\"'\d]", args)


def classify(row, source):
    name, context, type_ = row.NAME, row.CONTEXT, row.TYPE
    if source is None:
        return "NO_SOURCE", ""
    lines = source.split("\n")
    try:
        line_no = int(row.LINE)
    except ValueError:
        return "NOT_FOUND", ""
    i, stmt = statement_at(lines, line_no, name)
    if stmt is None:
        return "NOT_FOUND", lines[line_no - 1].strip()[:200] if 0 < line_no <= len(lines) else ""
    n = re.escape(name)
    snippet = stmt.strip()[:200]
    if re.search(rf"\(\s*[*&^]\s*(\w+::)?{n}\s*\)\s*\(", stmt):
        return "CALLABLE", snippet                                     # (*name)(...)
    after = re.search(rf"(?<![\w$.]){n}\s*(\(|=|\{{|;|,|\)|\[|:)", stmt)
    before = stmt[:after.start()] if after else ""
    if after and after.group(1) == "(" and context != "PARAMETER" and is_function_declaration(before, stmt[after.end():]):
        return "FUNCTION_DECL", snippet
    tail = stmt[after.start():] if after else stmt
    if any(p.search(tail) for p, langs in LAMBDA_INIT if langs is None or row.LANGUAGE in langs):
        return "LAMBDA", snippet
    if row.LANGUAGE in JAVA_LIKE and ANON_CLASS.search(tail):
        return "ANON_CLASS", snippet
    if type_ and CALLABLE_TYPE.search(type_.strip()):
        return "CALLABLE", snippet
    if (type_ and "bool" not in type_.lower() and INT_TYPE.match(type_.strip())
            and "*" not in type_ and "char" not in type_.lower()):  # pointers and chars get null checks
        uses = re.findall(rf"(if\s*\(\s*!?\s*{n}\s*\)|!\s*{n}\b|\b{n}\s*(&&|\|\|)|(&&|\|\|)\s*{n}\b|\b{n}\s*\?)", source)
        arith = re.findall(rf"\b{n}\s*(\+\+|--|[-+*/%]=?\s*\w|<<|>>|<=?|>=?)|\[\s*{n}\s*\]|\b{n}\s*\[", source)
        if len(uses) >= 1 and not arith:
            return "INT_AS_BOOL", snippet
    return "OK", snippet


def scan(out):
    _, rows = load_rows()
    rows = rows[rows.CONTEXT.isin(VARIABLE_CONTEXTS)].copy()
    urls = {}
    for i, r in rows.iterrows():
        urls[i], line = raw_url(r.LINK)
        if line and not r.LINE:
            rows.at[i, "LINE"] = str(line)
    unique = sorted({u for u in urls.values() if u})
    print(f"{len(rows)} variable rows, {len(unique)} files to read", file=sys.stderr)
    with ThreadPoolExecutor(8) as pool:
        sources = dict(zip(unique, pool.map(fetch, unique)))
    results = []
    for i, r in rows.iterrows():
        kind, snippet = classify(r, sources.get(urls[i]))
        suggested = {"FUNCTION_DECL": "FUNCTION", "LAMBDA": "FUNCTION"}.get(kind, r.CONTEXT)
        results.append(dict(KIND=kind, TAGGER_DATA_LINE=r.TAGGER_DATA_LINE, NAME=r.NAME, SPLIT=r.SPLIT,
                            GRAMMAR_PATTERN=r.GRAMMAR_PATTERN, CONTEXT=r.CONTEXT, SUGGESTED_CONTEXT=suggested,
                            TYPE=r.TYPE, LANGUAGE=r.LANGUAGE, SYSTEM_NAME=r.SYSTEM_NAME,
                            DATA_SOURCE=r.DATA_SOURCE, SOURCE=snippet, LINK=r.LINK, REVIEW="", NOTES=""))
    report = pd.DataFrame(results)
    order = {"FUNCTION_DECL": 0, "LAMBDA": 1, "ANON_CLASS": 2, "CALLABLE": 3, "INT_AS_BOOL": 4, "NOT_FOUND": 5, "NO_SOURCE": 6, "OK": 7}
    report = report.sort_values(["KIND", "CONTEXT", "NAME"], key=lambda s: s.map(order) if s.name == "KIND" else s)
    report.to_csv(out, sep="\t", index=False)
    print(report.KIND.value_counts().to_string(), file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


def apply(reviewed):
    data = pd.read_csv(DATA, sep="\t", dtype=str, keep_default_na=False)
    review = pd.read_csv(reviewed, sep="\t", dtype=str, keep_default_na=False)
    log_path = REPO / "output" / "reports" / f"label_changes_{datetime.date.today()}.tsv"
    changes = []
    for r in review.itertuples():
        decision = r.REVIEW.strip()
        new_context = r.CONTEXT if decision.upper() == "KEEP" else (decision.upper() or r.SUGGESTED_CONTEXT)
        if new_context == r.CONTEXT:
            continue
        mask = data.TAGGER_DATA_LINE == r.TAGGER_DATA_LINE
        assert mask.sum() == 1, r.TAGGER_DATA_LINE
        data.loc[mask, "CONTEXT"] = new_context
        changes.append(dict(TAGGER_DATA_LINE=r.TAGGER_DATA_LINE, SPLIT=r.SPLIT, FIELD="CONTEXT",
                            OLD=r.CONTEXT, NEW=new_context, REASON=f"context check: {r.KIND}; {r.NOTES}".strip("; "),
                            SOURCE=os.path.basename(reviewed)))
    data.to_csv(DATA, sep="\t", index=False)
    if changes:
        log = pd.DataFrame(changes)
        log.to_csv(log_path, sep="\t", index=False, mode="a", header=not log_path.exists())
    print(f"changed {len(changes)} contexts; logged to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--out", default=str(REPO / "output" / "reports" / f"context_check_{datetime.date.today()}.tsv"))
    a = sub.add_parser("apply")
    a.add_argument("reviewed")
    args = parser.parse_args()
    scan(args.out) if args.cmd == "scan" else apply(args.reviewed)
