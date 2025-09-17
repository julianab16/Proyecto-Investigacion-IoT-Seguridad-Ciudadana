#!C:\Users\Usuario\Documents\Proyecto Investigacion IoT Seguridad Ciudadana\violencia-db\venv\Scripts\python.exe

import sys

from osgeo.gdal import deprecation_warn

# import osgeo_utils.gdalmove as a convenience to use as a script
from osgeo_utils.gdalmove import *  # noqa
from osgeo_utils.gdalmove import main

deprecation_warn("gdalmove")
sys.exit(main(sys.argv))
