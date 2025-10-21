#!C:\Users\Usuario\Documents\Proyecto Investigacion IoT Seguridad Ciudadana\violencia-db\venv\Scripts\python.exe

import sys

from osgeo.gdal import deprecation_warn

# import osgeo_utils.gdal_merge as a convenience to use as a script
from osgeo_utils.gdal_merge import *  # noqa
from osgeo_utils.gdal_merge import main

deprecation_warn("gdal_merge")
sys.exit(main(sys.argv))
