"""Lanzador de Azarium que funciona desde cualquier directorio.

`py -3 -m azarium.cli` exige estar parado en la carpeta del proyecto. Este
script resuelve su propia ubicacion, asi que se puede invocar con ruta completa
desde cualquier consola.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from azarium.cli import main  # noqa: E402  (necesita el sys.path de arriba)

if __name__ == "__main__":
    raise SystemExit(main())
