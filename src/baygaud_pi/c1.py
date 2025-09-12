import sys, platform, sysconfig, struct, os
print("Python:", sys.version.split()[0], platform.platform())
print("CONFIG_ARGS:", sysconfig.get_config_var('CONFIG_ARGS'))  # PGO/LTO 흔적 확인
print("Arch:", platform.machine(), struct.calcsize("P")*8, "bit")
try:
    import numpy as np; print("NumPy:", np.__version__); np.__config__.show()
except Exception as e:
    print("NumPy info error:", e)
try:
    import numba, llvmlite
    print("Numba:", numba.__version__, "llvmlite:", llvmlite.__version__)
except Exception as e:
    print("Numba info error:", e)
print("OPENBLAS_NUM_THREADS=", os.getenv("OPENBLAS_NUM_THREADS"))
print("OMP_NUM_THREADS=", os.getenv("OMP_NUM_THREADS"))
