import numba, numpy as np
print("Numba:", numba.__version__)
print("NumPy:", np.__version__)
import os
print("THREADING_LAYER:", os.getenv("NUMBA_THREADING_LAYER"))
print("CPU_NAME:", os.getenv("NUMBA_CPU_NAME"))
