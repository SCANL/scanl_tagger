"""
Reconstruct GitHub permalinks (commit + file + line) for identifiers in input/tagger_data.tsv.

The original study kept srcML archives of the "general category" systems but no links.
Each srcML <unit> stores its file path, the SHA-1 of the file's raw contents (`hash`) and
`pos:start` line numbers for every name. This script:

  1. Streams each srcML archive, rebuilding file text to recover git blob hashes, and
     collecting declarations (name, context, type, file, line) for names in tagger_data.
  2. Makes a blob-less clone of the system's repository and finds the first-parent commit
     where the largest share of those blobs is present at once (the snapshot commit).
  3. Matches tagger_data rows by normalized name, preferring the same context and type,
     otherwise the first occurrence.
  4. Merges in the closed-category links (commit + file, no line) for the other systems.

Note: the archives were post-processed by a method-stereotype tool that inserted
<stereotype> elements and "@stereotype" comment lines. Those edits change file hashes for
some files, but `pos:start` still refers to lines in the original file.

Usage:
    python scripts/reconstruct_identifier_links.py \
        --srcml-dir ~/tagger_data_full/general_category_systems \
        --closed ~/tagger_data_full/closed_category_dataset.csv \
        --mirror-dir ~/tagger_data_full/git_mirrors \
        --out-dir output/identifier_links
"""
import argparse
import hashlib
import html
import os
import random
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# tagger_data systems whose srcML archive has a different name
SRCML_ARCHIVE_FOR_SYSTEM = {"corenlp": "java", "gimp.idents": "gimp"}

SNAPSHOT_BEFORE = "2019-07-16 23:59"   # srcML archives were written on this date
CLONE_SINCE = "2016-01-01"             # history depth to fetch; snapshot must be newer
SAMPLE_SIZE = 400                      # exact blobs used to locate the snapshot commit
MAX_TIE_RESCORE = 60                   # tied commits re-scored against all exact blobs

# Empty files are self-closing (<unit ... hash="da39..."/>) and must not swallow the next unit.
UNIT_RE = re.compile(r"<unit ([^>]*\bfilename=\"[^\"]*\"[^>]*?)(?:/>|>(.*?)</unit>)", re.S)
ATTR_RE = re.compile(r'([\w:]+)="([^"]*)"')
XMLNS_RE = re.compile(r'xmlns(?::\w+)?="[^"]*"')
# Some archives only declare src/pos on the root, but bodies still use cpp:, lit: etc.
SRCML_NAMESPACES = {
    "xmlns": "http://www.srcML.org/srcML/src", "xmlns:pos": "http://www.srcML.org/srcML/position",
    "xmlns:cpp": "http://www.srcML.org/srcML/cpp", "xmlns:lit": "http://www.srcML.org/srcML/literal",
    "xmlns:op": "http://www.srcML.org/srcML/operator", "xmlns:type": "http://www.srcML.org/srcML/modifier",
    "xmlns:err": "http://www.srcML.org/srcML/error",
}
STEREOTYPE_RE = re.compile(r"<stereotype>.*?</stereotype>[ \t]*", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SIMPLE_NAME_RE = re.compile(r">([A-Za-z_$][\w$]*)</name>")

FUNCTION_TAGS = {"function", "function_decl"}
# Constructors and destructors are named after their class, so they would shadow the class
# itself; only their parameters are collected.
CONSTRUCTOR_TAGS = {"constructor", "constructor_decl", "destructor", "destructor_decl"}
CLASS_TAGS = {"class", "struct", "union", "enum", "interface", "annotation_defn"}
FORWARD_CLASS_TAGS = {"class_decl", "struct_decl", "union_decl", "enum_decl"}


def normalize(text):
    """Case- and separator-insensitive key: 'm_fooBar', 'm foo Bar' -> 'mfoobar'."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


TYPE_MODIFIERS = {"final", "static", "const", "volatile", "private", "public", "protected",
                  "transient", "mutable", "inline", "virtual", "extern", "constexpr", "struct",
                  "class", "enum", "union", "register", "abstract", "synchronized"}


def type_key(text):
    """Compare types loosely: drop modifiers, generic arguments, pointers and arrays."""
    text = re.sub(r"@[\w.]+(\([^)]*\))?", "", str(text))
    while re.search(r"<[^<>]*>", text):
        text = re.sub(r"<[^<>]*>", "", text)
    words = [w for w in re.split(r"[^\w:.]+", text) if w and w not in TYPE_MODIFIERS]
    return normalize("".join(words))


def local(tag):
    return tag.rsplit("}", 1)[-1]


def iter_units(path, chunk_size=1 << 26):
    """Yield (attrs, body) for each file <unit> without loading the whole archive."""
    buffer = ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        header = handle.read(4096)
        declared = dict(d.split("=", 1) for d in XMLNS_RE.findall(header))
        for prefix, uri in SRCML_NAMESPACES.items():
            declared.setdefault(prefix, f'"{uri}"')
        namespaces = " ".join(f"{k}={v}" for k, v in sorted(declared.items()))
        buffer = header
        while True:
            last = 0
            for match in UNIT_RE.finditer(buffer):
                yield namespaces, dict(ATTR_RE.findall(match.group(1))), match.group(2) or ""
                last = match.end()
            buffer = buffer[last:]
            data = handle.read(chunk_size)
            if not data:
                break
            buffer += data


def exact_blob(body, srcml_hash):
    """Git blob hash of the original file, or None if the text can't be rebuilt exactly."""
    text = html.unescape(TAG_RE.sub("", STEREOTYPE_RE.sub("", body))).encode("utf-8")
    if hashlib.sha1(text).hexdigest() != srcml_hash:
        return None
    return hashlib.sha1(b"blob %d\0" % len(text) + text).hexdigest()


def pick_name(name_el, last):
    """Innermost simple <name>: last part for `ns::Class::method`, first part for `x[10]`."""
    while True:
        parts = [child for child in name_el if local(child.tag) == "name"]
        if not parts:
            text = (name_el.text or "").strip()
            line = name_el.get("{http://www.srcML.org/srcML/position}start", "0:0").split(":")[0]
            return text, int(line)
        name_el = parts[-1] if last else parts[0]


def type_text(el):
    type_el = next((child for child in el if local(child.tag) == "type"), None)
    if type_el is None or type_el.get("ref") == "prev":
        return None
    return " ".join("".join(type_el.itertext()).split())


def collect_declarations(root, targets):
    """
    Return (name, context, type, line) for declarations whose normalized name is in targets,
    plus the first plain use of each target name in the file (context "USAGE") as a fallback.
    """
    found, handled = [], set()

    def record(name_el, context, type_str, last):
        name, line = pick_name(name_el, last)
        if normalize(name) in targets:
            found.append((name, context, type_str or "", line))

    def walk(el, scope):
        tag = local(el.tag)
        child_scope = scope
        if tag in FUNCTION_TAGS:
            name_el = next((c for c in el if local(c.tag) == "name"), None)
            if name_el is not None:
                record(name_el, "FUNCTION", type_text(el), last=True)
            child_scope = "function"
        elif tag in CONSTRUCTOR_TAGS:
            child_scope = "function"
        elif tag in CLASS_TAGS or tag in FORWARD_CLASS_TAGS:
            name_el = next((c for c in el if local(c.tag) == "name"), None)
            if name_el is not None:
                record(name_el, "CLASS" if tag in CLASS_TAGS else "CLASS_FORWARD", "class", last=True)
            child_scope = "class"
        elif tag == "parameter":
            decl = next((c for c in el if local(c.tag) == "decl"), None)
            handled.add(id(decl))
            name_el = None if decl is None else next((c for c in decl if local(c.tag) == "name"), None)
            if name_el is not None:
                record(name_el, "PARAMETER", type_text(decl), last=False)
        elif tag == "decl_stmt":
            context = "ATTRIBUTE" if scope == "class" else "DECLARATION"
            previous_type = None
            for decl in (c for c in el if local(c.tag) == "decl"):
                handled.add(id(decl))
                decl_type = type_text(decl) or previous_type
                previous_type = decl_type
                name_el = next((c for c in decl if local(c.tag) == "name"), None)
                if name_el is not None:
                    record(name_el, context, decl_type, last=False)
        elif tag == "decl" and id(el) not in handled:
            # A <decl> outside a decl_stmt, e.g. `for (auto x : xs)` or K&R parameters. DECLARATION
            # and ATTRIBUTE mean decl_stmt, so these get their own context.
            name_el = next((c for c in el if local(c.tag) == "name"), None)
            if name_el is not None:
                record(name_el, "OTHER_DECL", type_text(el), last=False)
        elif tag == "typedef":
            name_el = next((c for c in el if local(c.tag) == "name"), None)
            if name_el is not None:
                record(name_el, "CLASS", "typedef", last=True)
        for child in el:
            walk(child, child_scope)

    walk(root, "file")
    used = set()
    for name_el in root.iter("{http://www.srcML.org/srcML/src}name"):
        if len(name_el) or not name_el.text:
            continue
        key = normalize(name_el.text)
        if key in targets and key not in used:
            used.add(key)
            record(name_el, "USAGE", "", last=False)
    return found


def scan_archive(archive_path, targets):
    """Return (units, exact_blobs, declarations) for one srcML archive."""
    units, exact, declarations = [], {}, []
    for namespaces, attrs, body in iter_units(archive_path):
        path = attrs.get("filename", "")
        units.append(path)
        srcml_hash = attrs.get("hash")
        if srcml_hash and len(body) < 4_000_000:
            blob = exact_blob(body, srcml_hash)
            if blob:
                exact[path] = blob
        if not targets.intersection(normalize(n) for n in SIMPLE_NAME_RE.findall(body)):
            continue
        try:
            root = ET.fromstring(f"<unit {namespaces}>{body}</unit>")
        except ET.ParseError:
            continue
        for name, context, type_str, line in collect_declarations(root, targets):
            declarations.append((normalize(name), name, context, type_str, path, line,
                                 attrs.get("language", "")))
    return units, exact, declarations


def git(repo, *args, **kwargs):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          check=kwargs.get("check", True)).stdout


def ensure_mirror(url, mirror_dir):
    name = url.rstrip("/").split("github.com/")[-1].replace("/", "__") + ".git"
    repo = os.path.join(mirror_dir, name)
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none",
                        f"--shallow-since={CLONE_SINCE}", url, repo], check=True)
    return repo


def first_parent_commits(repo):
    return git(repo, "rev-list", "--first-parent", "HEAD").split()


def strip_prefix(paths, repo, commits):
    """Number of leading path components to drop so srcML paths match the repository."""
    reference = git(repo, "rev-list", "-1", "--first-parent", f"--before={SNAPSHOT_BEFORE}", "HEAD").strip()
    tree = set(git(repo, "ls-tree", "-r", "--name-only", reference or commits[-1]).splitlines())
    sample = paths[:: max(1, len(paths) // 300)]
    best = max(range(4), key=lambda k: sum(p.split("/", k)[-1] in tree for p in sample if p.count("/") >= k))
    return best


def blob_intervals(repo, path, blob, index):
    """First-parent commit indices (0 = newest) where `path` has content `blob`."""
    log = git(repo, "log", "--first-parent", "--format=C %H", "--raw", "--no-abbrev",
              "--no-renames", "HEAD", "--", path, check=False)
    changes, commit = [], None
    for line in log.splitlines():
        if line.startswith("C "):
            commit = line[2:]
        elif line.startswith(":") and commit:
            changes.append((index.get(commit), line.split()[3]))
    intervals, newer = [], -1
    for position, new_blob in changes:        # newest first
        if position is None:
            continue
        if new_blob == blob:
            intervals.append((newer + 1, position))
        newer = position
    return intervals


def submodules(repo, commit, entries):
    """Map submodule path -> (repository URL, pinned commit) at `commit`."""
    config = git(repo, "show", f"{commit}:.gitmodules", check=False)
    urls, current = {}, None
    for line in config.splitlines():
        line = line.strip()
        if line.startswith("path"):
            current = line.split("=", 1)[1].strip()
        elif line.startswith("url") and current:
            url = line.split("=", 1)[1].strip()
            urls[current] = re.sub(r"\.git$", "", url.replace("git@github.com:", "https://github.com/"))
    return {path: (urls[path], sha) for path, sha in entries.items() if path in urls}


def gitlink_entries(repo, commit):
    entries = {}
    for line in git(repo, "ls-tree", "-r", "--full-tree", commit).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "commit":
            entries[path] = sha
    return entries


def tree_blobs(repo, commit):
    entries = {}
    for line in git(repo, "ls-tree", "-r", "--full-tree", commit).splitlines():
        meta, path = line.split("\t", 1)
        entries[path] = meta.split()[2]
    return entries


def find_snapshot(repo, exact, strip):
    """Pick the first-parent commit where the most exactly-rebuilt files are present."""
    commits = first_parent_commits(repo)
    index = {c: i for i, c in enumerate(commits)}
    pairs = [(p.split("/", strip)[-1], b) for p, b in exact.items() if p.count("/") >= strip]
    random.Random(0).shuffle(pairs)
    sample = pairs[:SAMPLE_SIZE]
    coverage = [0] * (len(commits) + 1)
    for path, blob in sample:
        for start, end in blob_intervals(repo, path, blob, index):
            coverage[start] += 1
            coverage[end + 1] -= 1
    running, scores = 0, []
    for value in coverage[:-1]:
        running += value
        scores.append(running)
    best_score = max(scores) if scores else 0
    tied = [i for i, s in enumerate(scores) if s == best_score]
    if len(tied) > MAX_TIE_RESCORE:
        tied = tied[:: max(1, len(tied) // MAX_TIE_RESCORE)]

    def full_score(i):
        blobs = tree_blobs(repo, commits[i])
        return sum(blobs.get(p) == b for p, b in pairs)

    best = max(tied, key=full_score)
    matched = full_score(best)
    return commits[best], best_score, len(sample), matched, len(pairs), len(tied)


def choose_declaration(candidates, context, type_str):
    """
    Pick one candidate (name, context, type, path, line), in order of preference:
    same context and type, same context, any declaration, first plain use.
    Within a group the first occurrence wins.
    """
    usages = [c for c in candidates if c[1] == "USAGE"]
    declarations = [c for c in candidates if c[1] != "USAGE"]
    if context == "CLASS":
        # A function or variable sharing a class's name (constructor, factory, instance) is
        # not the class. Prefer definitions over forward declarations; never fall back further.
        ranked = (([c for c in declarations if c[1] == "CLASS"], "context"),
                  ([c for c in declarations if c[1] == "CLASS_FORWARD"], "forward-declaration"))
        return next(((group[0], label) for group, label in ranked if group), (None, "none"))
    wanted_type = type_key(type_str)
    ranked = (([c for c in declarations if c[1] == context and type_key(c[2]) == wanted_type], "context+type"),
              ([c for c in declarations if c[1] == context], "context"),
              (declarations, "name-only"),
              (usages, "usage-only"))
    return next(((group[0], label) for group, label in ranked if group), (None, "none"))


def file_lines(repo, commit, path, cache):
    """File at `commit` split on "\n" only (str.splitlines() also breaks on form feeds,
    which would shift line numbers relative to git and srcML)."""
    if path not in cache:
        cache[path] = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"],
                                     capture_output=True, text=True, errors="replace").stdout.split("\n")
    return cache[path]


NOT_A_TYPE = {"return", "else", "new", "delete", "throw", "case", "goto", "if", "while", "for",
              "switch", "sizeof", "typeof", "not", "and", "or", "await", "yield", "assert"}
TYPEISH = r"[\w:<>,*&\[\]\s.@$~]"


def text_declaration(lines, name, context, allow_forward=False):
    """
    Find a line that declares `name` when srcML couldn't parse the declaration (FreeMarker
    templates, macro-heavy C++). Returns (line, kind) or (None, None).
    kind is "function", "variable" or "type". Type forward declarations (`class X;`) and
    `friend class X;` only count when allow_forward is set.
    """
    n = re.escape(name)
    function_re = re.compile(rf"^\s*(?P<prefix>{TYPEISH}*?)(?<![\w$.>]){n}\s*\(")
    variable_re = re.compile(rf"^\s*(?P<prefix>{TYPEISH}*?[\w>\]*&])\s*[*&]*\s*(?<![\w$.>]){n}\s*(=(?!=)|;|\[|,|\)|:(?!:)|\{{)")
    # The name must be followed by something that starts or ends a class declaration, so
    # elaborated types like "class Foo* make()" don't count.
    type_res = [re.compile(rf"\b(class|struct|interface|enum|union|record)\)?\s+(\w+\s+)*{n}"
                           rf"\s*($|[:{{<;(]|final\b|extends\b|implements\b|//|/\*)"),
                re.compile(rf"^\s*}}\s*{n}\s*;"), re.compile(rf"\btypedef\b.*\b{n}\s*;")]

    def typed_prefix(prefix):
        words = re.findall(r"[A-Za-z_$][\w$]*", prefix)
        return bool(words) and words[0] not in NOT_A_TYPE and not prefix.rstrip().endswith(("=", ".", "->"))

    def find_function(i, text):
        match = function_re.search(text)
        if not match:
            return False
        if typed_prefix(match.group("prefix")):
            return True
        # C style: return type on the previous line, name at column 0
        previous = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        return (not match.group("prefix") and text.startswith(name)
                and re.fullmatch(r"[\w\s*:<>,&]+", previous.strip() or "!") is not None
                and previous.strip().split()[0] not in NOT_A_TYPE)

    def find_variable(i, text):
        match = variable_re.search(text)
        return bool(match) and typed_prefix(match.group("prefix"))

    def find_type(i, text):
        if not any(r.search(text) for r in type_res):
            # Keyword on the previous line, e.g. "ATTRIBUTE_ALIGNED16(class)" then "btCollisionShape"
            previous = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
            return (re.match(rf"\s*{n}\s*([:{{]|$)", text) is not None
                    and re.search(r"\b(class|struct|union|enum)\)?\s*$", previous) is not None)
        forward = re.search(rf"\b(class|struct|union|enum)\s+(\w+\s+)*{n}\s*;", text) or "friend" in text.split()
        return allow_forward or not forward

    finders = {"function": find_function, "variable": find_variable, "type": find_type}
    order = {"FUNCTION": ["function", "variable", "type"], "CLASS": ["type"]}
    for kind in order.get(context, ["variable", "function", "type"]):
        for i, text in enumerate(lines):
            if text.lstrip().startswith(("//", "*", "/*", "#")):
                continue
            if finders[kind](i, text):
                return i + 1, kind
    return None, None


def text_fallback(usages, row_context, read_lines):
    """
    Search each file where the name is used for a declaration-looking line. Definitions in
    any file win over forward declarations.
    """
    for allow_forward, quality in ((False, "text-declaration"), (True, "forward-declaration")):
        for name, _, _, path, _ in usages:
            line, kind = text_declaration(read_lines(path), name, row_context, allow_forward)
            if line:
                return (name, f"TEXT_{kind.upper()}", "", path, line), quality
    return None, None


def check_line(lines, name, line):
    """(line, status): `name` on `line`, else the nearest line containing it as a whole word."""
    if not any(lines):
        return line, "file_unreadable"
    if 0 < line <= len(lines) and name in lines[line - 1]:
        return line, "verified"
    word = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
    hits = [i + 1 for i, text in enumerate(lines) if word.search(text)]
    if not hits:
        return line, "not_found"
    nearest = min(hits, key=lambda i: abs(i - line))
    return nearest, f"adjusted({nearest - line:+d})"


def github_file_lines(repo_url, commit, path):
    """Fetch a file from GitHub at `commit` (for submodules we don't mirror); [] if missing."""
    owner_repo = repo_url.rstrip("/").split("github.com/")[-1]
    url = f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{urllib.parse.quote(path)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace").split("\n")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return []


def verify_line(repo, commit, path, name, line, cache):
    """
    Confirm `name` is on `line` of the file at `commit`. srcML lines come from the archived
    file, which can differ slightly from the commit when the file couldn't be rebuilt exactly,
    so fall back to the nearest line containing the name as a whole word.
    """
    return check_line(file_lines(repo, commit, path, cache), name, line)


def process_system(archive, systems, rows, repo_url, mirror_dir, log_dir):
    log = open(os.path.join(log_dir, f"{archive}.log"), "w")
    targets = {normalize(s.replace(" ", "")) for s in rows["SPLIT"]}
    units, exact, declarations = scan_archive(os.path.join(ARGS.srcml_dir, f"{archive}.srcml.xml"), targets)
    print(f"{archive}: {len(units)} files, {len(exact)} rebuilt exactly, "
          f"{len(declarations)} candidate declarations", file=log, flush=True)

    repo = ensure_mirror(repo_url, mirror_dir)
    commits = first_parent_commits(repo)
    strip = strip_prefix(list(exact) or units, repo, commits)
    commit, sample_hits, sample_n, matched, exact_n, tied = find_snapshot(repo, exact, strip)
    commit_date = git(repo, "log", "-1", "--format=%cI", commit).strip()
    print(f"{archive}: commit {commit} ({commit_date}); sample {sample_hits}/{sample_n}; "
          f"all exact files {matched}/{exact_n}; tied commits {tied}", file=log, flush=True)
    blobs_at_commit = tree_blobs(repo, commit)
    gitlinks = submodules(repo, commit, gitlink_entries(repo, commit))

    by_key = defaultdict(list)
    for key, name, context, type_str, path, line, language in declarations:
        by_key[key].append((name, context, type_str, path, line))

    results, file_cache = [], {}
    for _, row in rows.iterrows():
        candidates = by_key.get(normalize(row.SPLIT.replace(" ", "")), [])
        choice, quality = choose_declaration(candidates, row.CONTEXT, row.TYPE)
        if quality in ("usage-only", "none", "forward-declaration"):
            usages = [c for c in candidates if c[1] == "USAGE"]
            text_choice, text_quality = text_fallback(
                usages, row.CONTEXT,
                lambda p: file_lines(repo, commit, p.split("/", strip)[-1], file_cache))
            if text_choice and (quality != "forward-declaration" or text_quality == "text-declaration"):
                choice, quality = text_choice, text_quality
        result = {"TSV_LINE": row.TSV_LINE, "MATCH": quality, "COMMIT": commit, "SNAPSHOT_DATE": commit_date}
        if choice:
            name, context, type_str, path, line = choice
            repo_path = path.split("/", strip)[-1]
            if path in exact:
                file_check = "exact" if blobs_at_commit.get(repo_path) == exact[path] else "differs"
            else:
                file_check = "present" if repo_path in blobs_at_commit else "missing"
            submodule = next((sub for sub in sorted(gitlinks, key=len, reverse=True)
                              if repo_path.startswith(sub + "/")), None)
            if submodule:
                # Vendored code lives in a submodule; link to that repository at the pinned commit.
                sub_url, sub_commit = gitlinks[submodule]
                inner_path = repo_path[len(submodule) + 1:]
                if not sub_url.startswith("https://github.com/"):
                    lines = []
                else:
                    key = (sub_url, sub_commit, inner_path)
                    if key not in file_cache:
                        file_cache[key] = github_file_lines(sub_url, sub_commit, inner_path)
                    lines = file_cache[key]
                line, line_check = check_line(lines, name, line)
                code_url = f"{sub_url}/blob/{sub_commit}/{inner_path}#L{line}"
                file_check = "submodule"
            else:
                line, line_check = verify_line(repo, commit, repo_path, name, line, file_cache)
                code_url = f"{repo_url.rstrip('/')}/blob/{commit}/{repo_path}#L{line}"
            result.update({
                "ORIGINAL_NAME": name, "SRCML_CONTEXT": context, "SRCML_TYPE": type_str,
                "FILE": repo_path, "SRC_LINE": line, "FILE_AT_COMMIT": file_check, "LINE_CHECK": line_check,
                "CODE_URL": code_url,
            })
        results.append(result)
    log.close()
    return archive, results, {"archive": archive, "systems": ",".join(systems), "commit": commit,
                              "date": commit_date, "sample_hits": sample_hits, "sample": sample_n,
                              "exact_matched": matched, "exact_files": exact_n, "tied": tied,
                              "files": len(units)}


def main():
    tagger = pd.read_csv(ARGS.tagger_data, sep="\t", keep_default_na=False)
    tagger["TSV_LINE"] = tagger.index + 2
    tagger["ARCHIVE"] = tagger.SYSTEM_NAME.map(lambda s: SRCML_ARCHIVE_FOR_SYSTEM.get(s, s))
    archives = {f.split(".srcml")[0] for f in os.listdir(ARGS.srcml_dir) if f.endswith(".srcml.xml")}
    if ARGS.only:
        archives &= set(ARGS.only)
    os.makedirs(ARGS.out_dir, exist_ok=True)
    os.makedirs(ARGS.mirror_dir, exist_ok=True)
    log_dir = os.path.join(ARGS.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    link_rows, snapshots = [], []
    with ProcessPoolExecutor(max_workers=ARGS.workers) as pool:
        futures = []
        for archive in sorted(archives):
            rows = tagger[tagger.ARCHIVE == archive]
            if rows.empty:
                continue
            repo_url = rows.GITHUB_URL.iloc[0]
            futures.append(pool.submit(process_system, archive, sorted(rows.SYSTEM_NAME.unique()),
                                       rows, repo_url, ARGS.mirror_dir, log_dir))
        for future in as_completed(futures):
            archive, results, snapshot = future.result()
            link_rows.extend(results)
            snapshots.append(snapshot)
            print(f"done {archive}: commit {snapshot['commit'][:10]} {snapshot['date']} "
                  f"exact {snapshot['exact_matched']}/{snapshot['exact_files']}", flush=True)

    snapshots = pd.DataFrame(snapshots)
    snapshot_path = os.path.join(ARGS.out_dir, "snapshots.csv")
    previous_out = os.path.join(ARGS.out_dir, "tagger_data_with_links.tsv")
    if ARGS.only and os.path.exists(snapshot_path):
        old = pd.read_csv(snapshot_path, keep_default_na=False)
        snapshots = pd.concat([old[~old.archive.isin(snapshots.archive)], snapshots], ignore_index=True)
    snapshots.sort_values("archive").to_csv(snapshot_path, index=False)
    links = pd.DataFrame(link_rows)
    if ARGS.only and os.path.exists(previous_out):
        # Keep links for the archives that weren't rerun.
        old = pd.read_csv(previous_out, sep="\t", keep_default_na=False)
        old = old[(old.LINK_SOURCE == "srcml") & ~old.TSV_LINE.isin(links.TSV_LINE)]
        keep = [c for c in old.columns if c not in tagger.columns or c == "TSV_LINE"]
        links = pd.concat([old[keep], links], ignore_index=True)
    links["LINK_SOURCE"] = links.CODE_URL.notna().map({True: "srcml", False: "none"}) if "CODE_URL" in links else "none"

    # Closed-category systems already have commit + file links (no line numbers).
    closed = pd.read_csv(ARGS.closed, keep_default_na=False)
    closed_by_key = {}
    for _, row in closed.iterrows():
        # tagger_data was re-split after the closed dataset was made ('dn 1x2' vs 'dn 1 x 2'),
        # so also index by the normalized original name.
        closed_by_key.setdefault((row.repository, normalize(row.Name), row.context), row)
        closed_by_key.setdefault((row.repository, normalize(row.Name), None), row)
        closed_by_key.setdefault((row.repository, row.split, row.context), row)
        closed_by_key.setdefault((row.repository, row.split, None), row)
    closed_rows = []
    for _, row in tagger[~tagger.TSV_LINE.isin(links.TSV_LINE)].iterrows():
        hit = closed_by_key.get((row.SYSTEM_NAME, row.SPLIT, row.CONTEXT))
        for key in ((row.SYSTEM_NAME, row.SPLIT, None),
                    (row.SYSTEM_NAME, normalize(row.SPLIT), row.CONTEXT),
                    (row.SYSTEM_NAME, normalize(row.SPLIT), None)):
            if hit is None:
                hit = closed_by_key.get(key)
        if hit is None:
            closed_rows.append({"TSV_LINE": row.TSV_LINE, "LINK_SOURCE": "none", "MATCH": "none"})
            continue
        commit = re.search(r"/blob/([0-9a-f]{40})/", hit.url)
        closed_rows.append({
            "TSV_LINE": row.TSV_LINE, "LINK_SOURCE": "closed_category",
            "MATCH": "context" if hit.context == row.CONTEXT else "name-only",
            "ORIGINAL_NAME": hit.Name, "FILE": hit.file, "COMMIT": commit.group(1) if commit else "",
            "CODE_URL": hit.url,
        })
    links = pd.concat([links, pd.DataFrame(closed_rows)], ignore_index=True)

    merged = tagger.drop(columns=["ARCHIVE"]).merge(links, on="TSV_LINE", how="left")
    merged["LINK_SOURCE"] = merged.LINK_SOURCE.fillna("none")
    if "SRC_LINE" in merged:
        merged["SRC_LINE"] = pd.to_numeric(merged.SRC_LINE, errors="coerce").astype("Int64")
    out = os.path.join(ARGS.out_dir, "tagger_data_with_links.tsv")
    merged.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")
    print(merged.groupby(["LINK_SOURCE", "MATCH"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tagger-data", default="input/tagger_data.tsv")
    parser.add_argument("--srcml-dir", default=os.path.expanduser("~/tagger_data_full/general_category_systems"))
    parser.add_argument("--closed", default=os.path.expanduser("~/tagger_data_full/closed_category_dataset.csv"))
    parser.add_argument("--mirror-dir", default=os.path.expanduser("~/tagger_data_full/git_mirrors"))
    parser.add_argument("--out-dir", default="output/identifier_links")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", help="Limit to these srcML archive names")
    ARGS = parser.parse_args()
    main()
