# 검증 3종
which python
file `which python`                          # → Mach-O 64-bit executable arm64
python -c 'import platform; print(platform.machine())'   # → arm64
python - << 'PY'
import numpy as np; np.show_config()         # → BLAS: openblas / SIMD: NEON, ASIMD
PY

