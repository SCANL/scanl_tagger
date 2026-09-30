"""
Add line numbers to the closed-category links in tagger_data_with_links.tsv.

The closed-category dataset links each identifier to a file at a commit, but not to a line.
For each row this script reads the file at that commit (fetching only that commit, with
file contents on demand), parses it with the `srcml` CLI, and picks the declaration using
the same rules as reconstruct_identifier_links.py. It then confirms the name is on the line.

Run reconstruct_identifier_links.py first. Usage:
    python scripts/add_closed_category_lines.py \
        --links output/identifier_links/tagger_data_with_links.tsv \
        --mirror-dir ~/tagger_data_full/git_mirrors
"""
import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct_identifier_links import (  # noqa: E402
    choose_declaration, collect_declarations, normalize, text_fallback, verify_line,
)

URL_RE = re.compile(r"(https://github\.com/[^/]+/[^/]+)/blob/([0-9a-f]{40})/([^#]*)")
LANGUAGE_BY_EXTENSION = {
    ".c": "C", ".h": "C++", ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++",
    ".hh": "C++", ".hxx": "C++", ".inl": "C++", ".java": "Java", ".cs": "C#",
}


def ensure_commit(repo_url, commit, mirror_dir):
    """Bare repository containing `commit` (history-only; file contents fetched on demand)."""
    repo = os.path.join(mirror_dir, repo_url.split("github.com/")[-1].replace("/", "__") + ".git")
    if not os.path.isdir(repo):
        subprocess.run(["git", "init", "-q", "--bare", repo], check=True)
        subprocess.run(["git", "-C", repo, "remote", "add", "origin", repo_url], check=True)
    has_commit = subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
                                capture_output=True).returncode == 0
    if not has_commit:
        subprocess.run(["git", "-C", repo, "fetch", "-q", "--depth", "1", "--filter=blob:none",
                        "origin", commit], check=True)
    # Mark as a partial clone so `git show` fetches missing file contents on demand.
    subprocess.run(["git", "-C", repo, "config", "remote.origin.promisor", "true"], check=True)
    subprocess.run(["git", "-C", repo, "config", "remote.origin.partialclonefilter", "blob:none"], check=True)
    return repo


def locate(repo, commit, path, rows, language):
    """Return {TSV_LINE: updates} for the rows that point at one file."""
    source = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True).stdout
    if not source:
        return {row.TSV_LINE: {"FILE_AT_COMMIT": "missing", "LINE_CHECK": "file_unreadable"} for row in rows}
    lang = LANGUAGE_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), language)
    xml = subprocess.run(["srcml", "-l", lang, "--position"], input=source, capture_output=True).stdout
    targets = {normalize(row.ORIGINAL_NAME) for row in rows}
    try:
        found = collect_declarations(ET.fromstring(xml), targets)
    except ET.ParseError:
        found = []
    by_key = {}
    for name, context, type_str, line in found:
        by_key.setdefault(normalize(name), []).append((name, context, type_str, path, line))

    cache = {path: source.decode("utf-8", errors="replace").split("\n")}
    updates = {}
    for row in rows:
        candidates = by_key.get(normalize(row.ORIGINAL_NAME), [])
        choice, quality = choose_declaration(candidates, row.CONTEXT, row.TYPE)
        if quality in ("usage-only", "none", "forward-declaration"):
            # srcML misparses some files (FreeMarker templates, macro-heavy C++); look for a
            # declaration-looking line. With no usages from srcML, scan the file directly.
            usages = [c for c in candidates if c[1] == "USAGE"] or [(row.ORIGINAL_NAME, "USAGE", "", path, 0)]
            text_choice, text_quality = text_fallback(usages, row.CONTEXT, lambda _: cache[path])
            if text_choice and (quality != "forward-declaration" or text_quality == "text-declaration"):
                choice, quality = text_choice, text_quality
        if not choice:
            updates[row.TSV_LINE] = {"FILE_AT_COMMIT": "present", "LINE_CHECK": "not_found", "MATCH": "none"}
            continue
        name, context, type_str, _, line = choice
        line, line_check = verify_line(repo, commit, path, name, line, cache)
        updates[row.TSV_LINE] = {
            "MATCH": quality, "SRCML_CONTEXT": context, "SRCML_TYPE": type_str, "SRC_LINE": line,
            "FILE_AT_COMMIT": "present", "LINE_CHECK": line_check,
            "CODE_URL": f"{row.CODE_URL.split('#')[0]}#L{line}",
        }
    return updates


def main():
    links = pd.read_csv(ARGS.links, sep="\t", keep_default_na=False)
    closed = links[links.LINK_SOURCE == "closed_category"].copy()
    parts = closed.CODE_URL.str.extract(URL_RE)
    closed["REPO_URL"], closed["COMMIT"], closed["FILE"] = parts[0], parts[1], parts[2]

    repos = {}
    for (repo_url, commit), _ in closed.groupby(["REPO_URL", "COMMIT"]):
        repos[(repo_url, commit)] = ensure_commit(repo_url, commit, ARGS.mirror_dir)
        print(f"fetched {repo_url} @ {commit[:10]}", flush=True)

    updates = {}
    with ThreadPoolExecutor(max_workers=ARGS.workers) as pool:
        futures = [pool.submit(locate, repos[(repo_url, commit)], commit, path,
                               list(group.itertuples()), group.LANGUAGE.iloc[0])
                   for (repo_url, commit, path), group in closed.groupby(["REPO_URL", "COMMIT", "FILE"])]
        for done, future in enumerate(as_completed(futures), 1):
            updates.update(future.result())
            if done % 100 == 0:
                print(f"{done}/{len(futures)} files", flush=True)

    for column in ("SRCML_CONTEXT", "SRCML_TYPE", "FILE_AT_COMMIT", "LINE_CHECK"):
        if column not in links:
            links[column] = ""
    links["SRC_LINE"] = links.SRC_LINE.astype(object)
    index = links.set_index("TSV_LINE").index
    for tsv_line, values in updates.items():
        row = index.get_loc(tsv_line)
        for column, value in values.items():
            links.at[row, column] = value
    links["SRC_LINE"] = pd.to_numeric(links.SRC_LINE, errors="coerce").astype("Int64")
    links.to_csv(ARGS.links, sep="\t", index=False)

    result = links[links.LINK_SOURCE == "closed_category"]
    print(f"\nUpdated {ARGS.links}")
    print(result.LINE_CHECK.str.replace(r"\(.*", "", regex=True).value_counts().to_string())
    print(result.MATCH.value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links", default="output/identifier_links/tagger_data_with_links.tsv")
    parser.add_argument("--mirror-dir", default=os.path.expanduser("~/tagger_data_full/git_mirrors"))
    parser.add_argument("--workers", type=int, default=8)
    ARGS = parser.parse_args()
    main()
