"""
summarize_repo.py

A utility for summarizing your project directory for LLM context transfer.

How to Use:
-----------
1. Save this as summarize_repo.py in your project root.

2. To get everything (directory tree + all file contents):
    python summarize_repo.py . --mode EVERYTHING

3. To get directory tree + first 10 lines of each file:
    python summarize_repo.py . --mode PARTIAL --max_lines 10

4. To get just the directory tree:
    python summarize_repo.py . --mode TREE

Arguments:
----------
- root_dir: The root directory to summarize (e.g., . for current directory)
- --mode: "EVERYTHING", "PARTIAL", or "TREE" (default: PARTIAL)
- --max_lines: Number of lines per file in PARTIAL mode (default: 10)

Example:
--------
python summarize_repo.py . --mode PARTIAL --max_lines 5

"""

import os
import datetime
import fnmatch

IGNORE_DIRS = {
    ".git", ".venv", ".pdm-build", "__pycache__", ".idea", ".vscode",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".pdm-python"
}
IGNORE_FILE_PATTERNS = (
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".DS_Store"
)
IGNORE_FILE_EXTENSIONS = IGNORE_FILE_PATTERNS
IGNORE_FILES = {
    "pdm.lock", "repo_summary.md"
}

def should_ignore(name):
    # Ignore system/hidden files and folders
    if name in IGNORE_DIRS or name in IGNORE_FILES:
        return True
    if name.startswith(".") and not name in {"README.md", ".gitignore"}:
        return True
    for pat in IGNORE_FILE_PATTERNS:
        if name.endswith(pat):
            return True
    # Ignore any generated summary report
    if fnmatch.fnmatch(name, "repo_summary_*.md"):
        return True
    return False

def summarize_repo(root_dir, out, mode="PARTIAL", max_lines=10):
    """
    Summarizes the directory structure and file contents.
    mode: "EVERYTHING", "PARTIAL", "TREE"
    max_lines: Number of lines per file in PARTIAL mode
    """
    for root, dirs, files in os.walk(root_dir):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        out.write(f"{indent}{os.path.basename(root)}/\n")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if should_ignore(f):
                continue
            file_path = os.path.join(root, f)
            out.write(f"{subindent}{f}\n")
            if mode == "EVERYTHING":
                print_full_content(file_path, out, subindent)
            elif mode == "PARTIAL":
                print_partial_content(file_path, out, max_lines, subindent)
            # If mode is TREE, skip file content

def print_partial_content(file_path, out, max_lines=10, subindent=''):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines[:max_lines]:
                out.write(f"{subindent}    {line.rstrip()}\n")
            if len(lines) > max_lines:
                out.write(f"{subindent}    ...\n")
    except Exception as e:
        out.write(f"{subindent}    [Could not read file: {e}]\n")

def print_full_content(file_path, out, subindent=''):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                out.write(f"{subindent}    {line.rstrip()}\n")
    except Exception as e:
        out.write(f"{subindent}    [Could not read file: {e}]\n")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize project directory.")
    parser.add_argument("root_dir", help="Root directory to summarize")
    parser.add_argument("--mode", choices=["EVERYTHING", "PARTIAL", "TREE"], default="PARTIAL")
    parser.add_argument("--max_lines", type=int, default=10, help="Max lines per file in PARTIAL mode")
    args = parser.parse_args()

    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"repo_summary_{timestamp}.md"

    with open(out_filename, "w", encoding="utf-8") as out:
        out.write(f"# Repository Summary Report\n\n")
        out.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n\n")
        out.write(f"**Parameters:**\n")
        out.write(f"- root_dir: `{args.root_dir}`\n")
        out.write(f"- mode: `{args.mode}`\n")
        out.write(f"- max_lines: `{args.max_lines}`\n\n")
        out.write("---\n\n")
        summarize_repo(args.root_dir, out, mode=args.mode, max_lines=args.max_lines)

    print(f"\nRepository summary written to: {out_filename}\n")