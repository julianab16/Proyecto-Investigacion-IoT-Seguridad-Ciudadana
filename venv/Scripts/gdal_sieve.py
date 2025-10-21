#!C:\Users\Usuario\Documents\Proyecto Investigacion IoT Seguridad Ciudadana\violencia-db\venv\Scripts\python.exe

import sys

from osgeo.gdal import deprecation_warn

# import osgeo_utils.gdal_sieve as a convenience to use as a script
from osgeo_utils.gdal_sieve import *  # noqa
from osgeo_utils.gdal_sieve import main

deprecation_warn("gdal_sieve")
sys.exit(main(sys.argv))
