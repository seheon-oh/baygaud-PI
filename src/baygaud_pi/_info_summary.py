# ── banner 아래 요약 테이블 출력 서브루틴 ─────────────────────────────────────

from typing import Dict, List, Tuple, Optional
import shutil
import math

def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default

def _box_print(rows,
               title: str | None = None,
               left_margin: int = 0,
               max_width: int | None = None,
               labels: tuple[str, str, str] = ("Quantity", "Value", "Note")) -> None:
    """
    rows: [(col1, col2, col3), ...]
    labels: 헤더 라벨 (기본: Quantity / Value / Note)
    """
    import shutil

    pad = " " * max(left_margin, 0)
    tw = shutil.get_terminal_size((100, 24)).columns
    width = min(max_width or tw, tw)

    def cut(s: str, w: int) -> str:
        return s if len(s) <= w else (s[:max(0, w-1)] + "…")

    # 폭 계산
    name_w = max([len(r[0]) for r in rows] + [len(labels[0]), 4])
    val_w  = max([len(r[1]) for r in rows] + [len(labels[1]), 5])
    note_w = max([len(r[2]) for r in rows] + [len(labels[2]), 4])

    total = name_w + val_w + note_w + 8
    if total > width - left_margin:
        overflow = total - (width - left_margin)
        for target in (2, 1, 0):  # note -> value -> name 순 축소
            if overflow <= 0:
                break
            take = min(overflow, [name_w, val_w, note_w][target] - 4)
            if target == 0:
                name_w -= take
            elif target == 1:
                val_w -= take
            else:
                note_w -= take
            overflow -= take

    hbar = pad + "+" + "-"*(name_w+2) + "+" + "-"*(val_w+2) + "+" + "-"*(note_w+2) + "+"
    if title:
        cap = pad + "+" + "-"*(name_w+val_w+note_w+6) + "+"
        inner_w = name_w + val_w + note_w + 6
        print(cap)
        print(pad + "|" + f" {title} ".center(inner_w) + "|")
        print(cap)

    print(hbar)
    col1, col2, col3 = labels
    # 헤더: Value도 우측 정렬
    header = pad + f"| {col1.ljust(name_w)} | {col2.rjust(val_w)} | {col3.ljust(note_w)} |"
    print(header)
    print(hbar)

    for c1, c2, c3 in rows:
        c1 = cut(c1, name_w)
        c2 = cut(c2, val_w)
        c3 = cut(c3, note_w)

        # Value 컬럼은 항상 우측 정렬
        c1 = c1.ljust(name_w)
        c2 = c2.rjust(val_w)
        c3 = c3.ljust(note_w)

        print(pad + f"| {c1} | {c2} | {c3} |")
    print(hbar)




def get_ray_info():
    """
    Ray 런타임 요약을 dict로 반환.
    Ray 미설치/미초기화면 최소 정보만 반환하거나 None.
    """
    try:
        import ray
    except ImportError:
        return None  # Ray를 안 쓰면 표에 섹션을 안 붙이도록

    info = {"Ray initialized": str(ray.is_initialized())}

    if not ray.is_initialized():
        return info  # 미초기화면 여기까지만

    try:
        # 전체/가용 리소스
        resources = ray.cluster_resources()       # 전체
        avail     = ray.available_resources()     # 가용

        # 노드 수
        try:
            nodes = ray.nodes()
            info["nodes"] = str(len(nodes))
        except Exception:
            pass

        # CPU/GPU
        info["total_CPUs"] = str(int(resources.get("CPU", 0)))
        info["avail_CPUs"] = str(int(avail.get("CPU", 0)))
        if "GPU" in resources or "GPU" in avail:
            info["total_GPUs"] = str(int(resources.get("GPU", 0)))
            info["avail_GPUs"] = str(int(avail.get("GPU", 0)))

        # 메모리 (Ray 2.x는 바이트 단위 키가 있을 수 있음)
        mem_bytes = int(resources.get("memory", 0))
        obj_bytes = int(resources.get("object_store_memory", 0))
        if mem_bytes:
            info["memory_GB"] = f"{mem_bytes/1e9:.1f}"
        if obj_bytes:
            info["object_store_GB"] = f"{obj_bytes/1e9:.1f}"

    except Exception as e:
        info["note"] = f"ray info collection error: {e}"

    return info




# ── 시스템/할당 리소스 요약: total_CPUs, avail_CPUs, memory_GB, object_store_GB ──
def get_runtime_resource_info(_params: dict | None = None) -> dict | None:
    """
    반환 예:
      {
        "Ray initialized": "True" | "False",
        "total_CPUs": "12",                 # 물리 코어 수 (psutil), 실패 시 논리
        "avail_CPUs": "8",                  # baygaud에 할당한 CPU 수(우선순위: _params -> Ray 제한 -> CPU affinity -> 논리)
        "memory_GB": "64.0",                # 시스템 전체 메모리(GB)
        "object_store_GB": "3.2",           # 현재 프로세스 RSS 메모리(GB)
      }
    """
    info = {}
    # 1) Ray 상태
    try:
        import ray
        info["Ray initialized"] = "True" if ray.is_initialized() else "False"
    except Exception:
        info["Ray initialized"] = "False"

    # 2) CPU: 물리/논리/affinity/파라미터/환경변수
    import os
    try:
        import psutil
    except Exception:
        psutil = None

    # total_CPUs: 물리 코어 수 우선
    total_phys = None
    if psutil:
        try:
            total_phys = psutil.cpu_count(logical=False)
        except Exception:
            total_phys = None
    if not total_phys:
        # 물리 실패 시 논리로 대체
        total_phys = os.cpu_count() or 1
    info["Total physical cores"] = str(total_phys)

    # avail_CPUs: baygaud에 "할당"한 수
    # 우선순위: _params['num_cpus'] -> CPU affinity 길이 -> Ray 제한/환경 -> 논리
    avail = None
    if _params and isinstance(_params, dict):
        nc = _params.get("num_cpus_ray")
        if nc:
            try:
                avail = int(nc)
            except Exception:
                try:
                    avail = int(float(nc))
                except Exception:
                    pass

    if avail is None and psutil:
        try:
            # Linux 등에서 affinity가 설정되어 있으면 그 길이가 실사용 가능 코어 수
            p = psutil.Process()
            if hasattr(p, "cpu_affinity"):
                aff = p.cpu_affinity()
                if aff:
                    avail = len(aff)
        except Exception:
            pass

    if avail is None:
        # Ray가 초기화되어 있고 num_cpus 제한이 걸려 있으면 추정
        try:
            if info["Ray initialized"] == "True":
                # 1) ray.init(num_cpus=...)로 제한했다면 환경변수에 노출된 경우가 있음
                env_nc = os.environ.get("RAY_NUM_CPUS")
                if env_nc:
                    avail = int(float(env_nc))
                else:
                    # 2) 노드 리소스에서 이 프로세스가 쓸 수 있는 상한(보수적 추정: 논리 코어 수)
                    avail = os.cpu_count() or total_phys
        except Exception:
            pass

    if avail is None:
        avail = os.cpu_count() or total_phys

    info["Ray allocated cores"] = str(avail)
    info["Sampler allocated cores"] = str(_params['num_cpus_nested_sampling'])
    info["Numba allocated threads"] = str(_params['numba_num_threads'])

    # 3) 메모리: 전체/현재 프로세스 RSS
    total_mem_gb = None
    proc_rss_gb = None
    if psutil:
        try:
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass
        try:
            proc_rss_gb = psutil.Process().memory_info().rss / (1024**3)
        except Exception:
            pass
    if total_mem_gb is None:
        # 매우 보수적인 fallback: 알 수 없음
        total_mem_gb = 0.0
    if proc_rss_gb is None:
        proc_rss_gb = 0.0

    info["System memory (GB)"] = f"{total_mem_gb:.1f}"
    # 요구사항: object_store에는 "지금 코드에 할당된 메모리" → 프로세스 RSS로 표시
    info["Process memory (GB)"] = f"{proc_rss_gb:.1f}"

    info["(y chunk size)"] = '(' + str(_params['y_chunk_size']) + ')'
    info["(gather batch)"] = '(' + str(_params['gather_batch']) + ')'

    return info



from typing import Optional, Dict, List

# ────────────── helpers (no color) ──────────────
def _mk_row(c1: str, c2: str, c3: str, w1: int, w2: int, w3: int, *, left_margin: int = 0) -> str:
    sp = " " * left_margin
    return f"{sp}| {c1:<{w1}} | {c2:>{w2}} | {c3:<{w3}}|"

def _hline(w1: int, w2: int, w3: int, *, left_margin: int = 0) -> str:
    sp = " " * left_margin
    inner = (w1 + w2 + w3) + 9  # 공백/구분 포함
    return f"{sp}+" + "-" * (inner - 2) + "+"

# ────────────── main table ──────────────
def print_cube_summary(
    naxis1: int,
    naxis2: int,
    naxis3: int,
    vel_min_kms: float,
    vel_max_kms: float,
    cdelt3_ms: float,
    *,
    vel_unit_label: str = "km/s",
    cdelt3_unit_label: str = "m/s",
    # 추가(있으면 표에 표시)
    naxis1_s0: int | None = None, naxis1_e0: int | None = None,
    naxis2_s0: int | None = None, naxis2_e0: int | None = None,
    max_ngauss: int | None = None,
    peak_sn_limit: float | None = None,
    y_chunk_size: int | None = None,
    gather_batch: int | None = None,
    # 런타임/Ray
    left_margin: int = 0,
    title: str = "Data cube / key params",
    ray_info: Optional[Dict[str, str]] = None,
) -> None:
    """스크린샷과 동일한 3-컬럼 ASCII 표(모노크롬)."""

    # 고정 폭(스크린샷 기준)
    W1, W2, W3 = 23, 9, 33

    # 헤더
    print(_hline(W1, W2, W3, left_margin=left_margin))
    print(_mk_row(title, "Value", "Note", W1, W2, W3, left_margin=left_margin))
    print(_hline(W1, W2, W3, left_margin=left_margin))

    # 1) Key params
    roi1 = f"[{naxis1_s0} : {naxis1_e0}]" if (naxis1_s0 is not None and naxis1_e0 is not None) else "[:]"
    roi2 = f"[{naxis2_s0} : {naxis2_e0}]" if (naxis2_s0 is not None and naxis2_e0 is not None) else "[:]"
    print(_mk_row("naxis1 (pixels)",      str(naxis1), roi1, W1, W2, W3, left_margin=left_margin))
    print(_mk_row("naxis2 (pixels)",      str(naxis2), roi2, W1, W2, W3, left_margin=left_margin))
    print(_mk_row("naxis3 (channels)",    str(naxis3), "[:]", W1, W2, W3, left_margin=left_margin))
    if max_ngauss is not None:
        print(_mk_row("max_ngauss (number)", str(max_ngauss), "Maximum Gaussian components",
                      W1, W2, W3, left_margin=left_margin))
    if peak_sn_limit is not None:
        print(_mk_row("peak-flux S/N limit", f"{float(peak_sn_limit):.1f}", "Minimum peak-flux S/N",
                      W1, W2, W3, left_margin=left_margin))

    print(_hline(W1, W2, W3, left_margin=left_margin))

    # 2) Velocity / spectral
    sign_note = "(+) spectral axis increasing" if cdelt3_ms >= 0 else "(-) spectral axis decreasing"
    print(_mk_row(f"Velocity min [{vel_unit_label}]", f"{vel_min_kms:.4f}", "", W1, W2, W3, left_margin=left_margin))
    print(_mk_row(f"Velocity max [{vel_unit_label}]", f"{vel_max_kms:.4f}", "", W1, W2, W3, left_margin=left_margin))
    print(_mk_row(f"CDELT3 [{cdelt3_unit_label}]", f"{abs(cdelt3_ms):.2f}", sign_note,
                  W1, W2, W3, left_margin=left_margin))
    print(_mk_row("Spec axis unit check", vel_unit_label, "displayed here should be km/s",
                  W1, W2, W3, left_margin=left_margin))

    print(_hline(W1, W2, W3, left_margin=left_margin))

    # 3) Runtime (Ray)
    print(_mk_row("Runtime (Ray)", "Value", "", W1, W2, W3, left_margin=left_margin))
    print(_hline(W1, W2, W3, left_margin=left_margin))

    if ray_info:
        def _get(k, default=""):
            return str(ray_info.get(k, default))
        for label in [
            "Ray initialized",
            "Total physical cores",
            "Ray allocated cores",
            "Sampler allocated cores",
            "Numba allocated threads",
            "System memory (GB)",
            "Process memory (GB)",
        ]:
            print(_mk_row(label, _get(label), "", W1, W2, W3, left_margin=left_margin))

    if y_chunk_size is not None:
        print(_mk_row("(y chunk size)", f"({y_chunk_size})", "", W1, W2, W3, left_margin=left_margin))
    if gather_batch is not None:
        print(_mk_row("(gather batch)", f"({gather_batch})", "", W1, W2, W3, left_margin=left_margin))

    print(_hline(W1, W2, W3, left_margin=left_margin))






# 간단 YAML 로더 (PyYAML 우선, 없으면 ruamel.yaml)
def _yaml_load(path: str) -> dict:
    try:
        import yaml  # PyYAML
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        from ruamel.yaml import YAML
        y = YAML()
        with open(path, "r", encoding="utf-8") as f:
            return y.load(f) or {}

def print_cube_summary_from_info(
    info: dict | None = None,
    *,
    cube_info: dict | None = None,
    yaml_path: str | None = None,       # ← YAML 파일 경로 추가
    left_margin: int = 0,
    title: str = "Data cube / key params",
    ray_info: dict | None = None,
    prefer: str = "info",               # "info" 우선(기본) 또는 "cube"
) -> None:
    """
    info + (cube_info | yaml_path) 를 병합해 print_cube_summary 호출.
    - yaml_path 가 주어지면 YAML을 읽어 cube_info에 병합합니다(명시 cube_info가 YAML을 덮어씀).
    - prefer="info": 최종 병합에서 info 값이 cube 쪽 값을 덮어씀(기본).
      prefer="cube":  cube 쪽 값이 info를 덮어씀.
    """
    if (info is None) and (cube_info is None) and (yaml_path is None):
        raise ValueError("info, cube_info, yaml_path 중 최소 하나는 제공해야 합니다.")

    # 1) cube 쪽 소스 만들기: YAML → cube_info 순서로 병합(명시 dict 우선)
    base_cube: dict = {}
    if yaml_path:
        base_cube.update(_yaml_load(yaml_path))
    if cube_info:
        base_cube.update(cube_info)

    # 2) 최종 병합 (우선순위 선택)
    a = info or {}
    b = base_cube
    merged = ({**b, **a} if prefer == "info" else {**a, **b})

    # 3) 동의어/키 가져오기 유틸
    def g(keys, default=None):
        if isinstance(keys, str):
            return merged.get(keys, default)
        for k in keys:
            if k in merged:
                return merged[k]
        return default

    # 4) print_cube_summary 호출
    print_cube_summary(
        naxis1=g("naxis1", 0),
        naxis2=g("naxis2", 0),
        naxis3=g("naxis3", 0),
        vel_min_kms=g(["vel_min_kms", "vel_min"], 0.0),
        vel_max_kms=g(["vel_max_kms", "vel_max"], 0.0),
        cdelt3_ms=g(["cdelt3_ms", "cdelt3"], 0.0),
        vel_unit_label=g("vel_unit_label", "km/s"),
        cdelt3_unit_label=g("cdelt3_unit_label", "m/s"),
        naxis1_s0=g("naxis1_s0"),
        naxis1_e0=g("naxis1_e0"),
        naxis2_s0=g("naxis2_s0"),
        naxis2_e0=g("naxis2_e0"),
        max_ngauss=g("max_ngauss"),
        peak_sn_limit=g("peak_sn_limit"),
        y_chunk_size=g("y_chunk_size"),
        gather_batch=g("gather_batch"),
        left_margin=left_margin,
        title=title,
        ray_info=ray_info,
    )
