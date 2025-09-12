#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import os
from pathlib import Path
from pyfiglet import Figlet, FigletFont

TITLE = "baygaud-PI"
TITLE_FONT  = "ogre"   # 더 작게: 'straight', 더 크게: 'slant'
#TITLE_FONT  = os.environ.get("standar", "smslent", "gothic")
LEFT_MARGIN = 0           # 제목 왼쪽 여백 공백 수 (0이면 완전 좌측)

# --- version loader -----------------------------------------------------------

def load_version_from_version_py(filename: str = "_version.py") -> str:
    """
    현재 스크립트 경로 기준으로 _version.py를 찾아 __version__을 반환.
    없거나 읽기 실패 시 빈 문자열 반환.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,
        here.parent / filename,            # 상위 폴더도 한번 시도
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
    우선순위:
      1) 환경변수 BAYGAUD_VERSION
      2) 로컬/상위 폴더의 _version.py 의 __version__
      3) 폴백 'dev'
    그리고 출력용 라벨은 기존 스타일 유지 위해 'v.' 접두어 자동 부여.
    """
    ver = os.environ.get("BAYGAUD_VERSION", "").strip()
    if not ver:
        ver = load_version_from_version_py("_version.py")
    if not ver:
        ver = "dev"
    # 기존 스타일 유지: 'v.' 프리픽스 없으면 붙여줌
    vlabel = ver if ver.lower().startswith("v") else f"v.{ver}"
    return vlabel

# --- banner rendering ---------------------------------------------------------

#def figlet_lines(text: str, font: str) -> list[str]:
#    art = Figlet(font=font, width=200).renderText(text).rstrip("\n")
#    return art.splitlines()


def figlet_lines(text: str, font: str) -> list[str]:
    # 선호 순서: gothic → block → standard → straight
    candidates = (font, "gothic", "block", "standard", "straight")
    available = set(FigletFont.getFonts())
    for f in candidates:
        if f in available:
            art = Figlet(font=f, width=200).renderText(text).rstrip("\n")
            return art.splitlines()
    # 폴백: 폰트가 전혀 없으면 그냥 텍스트 반환
    return [text]

def visible_width(line: str) -> int:
    return len(line.rstrip(" "))

def print_banner():
    title_lines = figlet_lines(TITLE, TITLE_FONT)
    if LEFT_MARGIN > 0:
        pad = " " * LEFT_MARGIN
        title_lines = [pad + ln for ln in title_lines]

    # 1) 제목: 왼쪽 정렬 출력
    print()
    for ln in title_lines:
        print(ln)

    # 2) 버전: 제목 아트 폭 기준 가운데 정렬
    title_width = max(visible_width(ln) for ln in title_lines) - LEFT_MARGIN
    if title_width < 0:
        title_width = 0
    version_label = load_version()                    # ← 여기서 _version.py 읽음
    inner_pad = max((title_width - len(version_label)) // 2, 0)
    print(" " * (LEFT_MARGIN + inner_pad) + version_label)
    print()

if __name__ == "__main__":
    print_banner()
