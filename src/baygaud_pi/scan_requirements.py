#!/usr/bin/env python3
# scan_requirements_with_latest.py
from __future__ import annotations
import argparse, ast, json, os, sys, sysconfig
from pathlib import Path
from typing import Iterable, Set, Dict, Tuple
import importlib.util
import importlib.metadata as md
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import ssl

DEFAULT_MAPPING = {
    "PIL": "Pillow", "yaml": "PyYAML", "sklearn": "scikit-learn",
    "skimage": "scikit-image", "cv2": "opencv-python",
    "bs4": "beautifulsoup4", "Crypto": "pycryptodome",
    "spectral_cube": "spectral-cube", "radio_beam": "radio-beam",
    "OpenGL": "PyOpenGL", "igraph": "python-igraph",
    "dateutil": "python-dateutil",
}
EXCLUDE_DIRS = {".git",".hg",".svn","__pycache__",".mypy_cache",".pytest_cache",
                ".venv","venv","env","build","dist","site-packages",".tox"}

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts): continue
        yield p

def discover_local_modules(root: Path) -> Set[str]:
    s=set()
    for d in root.iterdir():
        if d.is_dir() and (d/"__init__.py").exists(): s.add(d.name)
    for f in root.iterdir():
        if f.is_file() and f.suffix==".py": s.add(f.stem)
    return s

def is_stdlib_module(name: str) -> bool:
    stdset = getattr(sys, "stdlib_module_names", None)
    if stdset and name in stdset: return True
    spec = importlib.util.find_spec(name)
    if spec is None: return False
    if spec.origin in (None, "built-in"): return True
    std = Path(sysconfig.get_paths().get("stdlib",""))
    try:
        if spec.origin and Path(spec.origin).is_file():
            return std in Path(spec.origin).resolve().parents
        if spec.submodule_search_locations:
            return any(std in Path(p).resolve().parents for p in spec.submodule_search_locations)
    except Exception: pass
    return False

class ImportCollector(ast.NodeVisitor):
    def __init__(self):
        self.required, self.optional, self.dynamic = set(), set(), set()
        self._try_depth=0
    def visit_Try(self, node: ast.Try) -> None:
        self._try_depth+=1
        for n in node.body: self.visit(n)
        self._try_depth-=1
        for n in (node.handlers+node.orelse+node.finalbody): self.visit(n)
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            (self.optional if self._try_depth else self.required).add(a.name.split(".")[0])
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and not node.level:
            (self.optional if self._try_depth else self.required).add(node.module.split(".")[0])
    def visit_Call(self, node: ast.Call) -> None:
        try:
            fn=node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                if fn.value.id=="importlib" and fn.attr=="import_module":
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value,str):
                        self.dynamic.add(node.args[0].value.split(".")[0])
            if isinstance(fn, ast.Name) and fn.id=="__import__":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value,str):
                    self.dynamic.add(node.args[0].value.split(".")[0])
        except Exception: pass
        self.generic_visit(node)

def collect_imports(py_files: Iterable[Path]) -> Tuple[Set[str],Set[str],Set[str]]:
    c=ImportCollector()
    for py in py_files:
        try:
            tree=ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except Exception: continue
        c.visit(tree)
    return c.required, c.optional, c.dynamic

def filter_names(names:Set[str], locals_:Set[str]) -> Set[str]:
    out=set()
    for n in names:
        if n in locals_ or is_stdlib_module(n): continue
        out.add(n)
    return out

def map_to_dist(names:Set[str], mapping:Dict[str,str]) -> Set[str]:
    return {mapping.get(n,n) for n in names}

def pin_installed(pkgs: Iterable[str]) -> Dict[str,str]:
    out={}
    for d in pkgs:
        try: out[d]=f"{d}=={md.version(d)}"
        except md.PackageNotFoundError: out[d]=d
    return out

def fetch_latest(dist: str) -> str:
    """PyPI JSON에서 최신 안정판 버전 조회(접속 실패 시 dist 그대로 반환)."""
    url=f"https://pypi.org/pypi/{dist}/json"
    try:
        # mac에서 종종 SSL 검증 이슈: 기본 컨텍스트 사용
        with urlopen(url, context=ssl.create_default_context(), timeout=8) as r:
            data=json.load(r)
        ver=data["info"]["version"].strip()
        # 가끔 pre/post 릴리스만 있는 경우가 있어 후처리 가능. 여기선 info.version 신뢰.
        return f"{dist}=={ver}"
    except (URLError, HTTPError, KeyError, TimeoutError, ValueError):
        return dist

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--output","-o", default="requirements.generated.txt")
    ap.add_argument("--pin", choices=["none","installed","latest"], default="none")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--mapping", type=str)
    args=ap.parse_args()

    root=Path(args.root).resolve()
    locals_=discover_local_modules(root)
    required, optional, dynamic = collect_imports(iter_py_files(root))
    optional |= dynamic

    required = filter_names(required, locals_)
    optional = filter_names(optional - required, locals_)

    mapping=dict(DEFAULT_MAPPING)
    if args.mapping:
        try:
            mapping.update(json.loads(Path(args.mapping).read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[warn] mapping 읽기 실패: {e}", file=sys.stderr)

    req = sorted(map_to_dist(required, mapping), key=str.lower)
    opt = sorted(map_to_dist(optional, mapping), key=str.lower)

    if args.pin=="installed":
        req_map = pin_installed(req)
        opt_map = pin_installed(opt)
    elif args.pin=="latest":
        req_map = {d: fetch_latest(d) for d in req}
        opt_map = {d: fetch_latest(d) for d in opt}
    else:
        req_map = {d:d for d in req}
        opt_map = {d:d for d in opt}

    lines=[]
    lines.append("# Auto-generated. 검토 후 버전/마커 조정하세요.\n")
    if req_map:
        lines.append("# --- Required ---")
        lines.extend(req_map.values()); lines.append("")
    if opt_map:
        if args.include-optional:
            lines.append("# --- Optional/conditional ---")
            lines.extend(s+"  # optional" for s in opt_map.values())
        else:
            lines.append("# --- Optional/conditional (detected) ---")
            lines.extend("# "+s+"  # optional" for s in opt_map.values())
        lines.append("")
    lines += [
        "# Tips:",
        "# - OS/Python 마커 예: pkg; python_version>='3.12'",
        "# - 모듈명↔배포명 불일치 시 --mapping 으로 보정",
    ]
    Path(args.output).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"[ok] wrote {args.output}")

if __name__=="__main__":
    main()

