#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _banner.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import importlib.util
import os
from pathlib import Path
from pyfiglet import Figlet, FigletFont

TITLE = "baygaud-PI"
TITLE_FONT  = "ogre"   # Smaller: 'straight', bigger: 'slant'
#TITLE_FONT  = os.environ.get("standar", "smslent", "gothic")
LEFT_MARGIN = 0           # Left indentation (spaces) for the title; 0 = flush left

# --- version loader -----------------------------------------------------------

def load_version_from_version_py(filename: str = "_version.py") -> str:
    """
    Locate _version.py relative to this script and return __version__.
    Return an empty string if the file is missing or cannot be read.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,
        here.parent / filename,            # also try the parent directory
    ]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("pkg_version", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)   # type: ignore[attr-defined]
            ver = getattr(mod, "__version__", "")
            return str(ver) if ver else ""
    return ""

def load_version() -> str:
    """
    Priority order:
      1) Environment variable BAYGAUD_VERSION
      2) __version__ from _version.py in the current or parent directory
      3) Fallback to 'dev'
    For display, automatically prepend 'v.' if not already present to match the existing style.
    """
    ver = os.environ.get("BAYGAUD_VERSION", "").strip()
    if not ver:
        ver = load_version_from_version_py("_version.py")
    if not ver:
        ver = "dev"
    # Keep the existing style: if there's no 'v' prefix, add 'v.' automatically
    vlabel = ver if ver.lower().startswith("v") else f"v.{ver}"
    return vlabel

# --- banner rendering ---------------------------------------------------------

#def figlet_lines(text: str, font: str) -> list[str]:
#    art = Figlet(font=font, width=200).renderText(text).rstrip("\n")
#    return art.splitlines()


def figlet_lines(text: str, font: str) -> list[str]:
    # preferred order: gothic → block → standard → straight
    candidates = (font, "gothic", "block", "standard", "straight")
    available = set(FigletFont.getFonts())
    for f in candidates:
        if f in available:
            art = Figlet(font=f, width=200).renderText(text).rstrip("\n")
            return art.splitlines()
    # fallback: if no figlet fonts are available, return plain text
    return [text]

def visible_width(line: str) -> int:
    return len(line.rstrip(" "))

def print_banner():
    title_lines = figlet_lines(TITLE, TITLE_FONT)
    if LEFT_MARGIN > 0:
        pad = " " * LEFT_MARGIN
        title_lines = [pad + ln for ln in title_lines]

    # 1) Title: print left-aligned
    print()
    for ln in title_lines:
        print(ln)

    # 2) Version: center under the title art width
    title_width = max(visible_width(ln) for ln in title_lines) - LEFT_MARGIN
    if title_width < 0:
        title_width = 0
    version_label = load_version()                    # read from _version.py here
    inner_pad = max((title_width - len(version_label)) // 2, 0)
    print(" " * (LEFT_MARGIN + inner_pad) + version_label)
    print()

if __name__ == "__main__":
    print_banner()

#-- END OF SUB-ROUTINE____________________________________________________________#