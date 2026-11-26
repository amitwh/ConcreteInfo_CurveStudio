# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: ConcreteInfo CurveStudio desktop app, one-folder build.

Built by ``apps/desktop/packaging/build.py`` (which generates the icon first and
archives the bundle afterwards); never invoke PyInstaller on this by hand.

What the frozen app needs beyond plain module-graph analysis:

* ``copy_metadata`` — ``importlib.metadata`` reads *installed distributions*, so
  the model-library entry points (group ``curvestudio.models``) only load inside
  the bundle if the dist-info directories (with their ``entry_points.txt``) are
  collected too. Same for the version lookup in ``curvestudio_desktop``.
* ``collect_submodules`` — the model families are imported through entry points,
  never through a static import statement, so PyInstaller cannot see them. The
  lazily generated engineering families are covered by the same call. The
  statistics tool modules are the same shape: the stats registry imports them by
  name from ``TOOL_MODULES`` at first use, so no static import ever reaches them
  (the model-library load warms the catalogue at run time, which is what made
  the missing modules fail the frozen selftest rather than a user's Statistics
  tab).
* ``PySide6.QtSvg`` — QtSvg drives the SVG image format and the icon engine
  plugins; keeping the module explicit keeps those plugins in the bundle.
* matplotlib's ``mpl-data`` (the DejaVu fonts the PDF reports register) and
  reportlab's fonts are data files; their PyInstaller hooks collect them and
  ``build.py`` asserts afterwards that they really landed in the bundle.
* spawn multiprocessing — the Auto-fit pool re-executes the frozen executable
  for every worker; ``curvestudio_desktop.__main__:main`` calls
  ``multiprocessing.freeze_support()`` before anything else, which the frozen
  ``--selftest`` (an Auto-fit through the pool) proves at build time.

The icon is passed via ``CURVESTUDIO_ICON`` (a path build.py generates); on
Linux the ELF format has no icon slot, so it is only used on Windows.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = [
    # keeps the svg imageformat and iconengine Qt plugins in the bundle
    "PySide6.QtSvg",
    # the model families load through entry points, invisible to module analysis
    *collect_submodules("curvestudio_models"),
    *collect_submodules("curvestudio_engineering"),
    # the statistics tools register from modules the stats registry imports by
    # name (TOOL_MODULES) on first use — invisible to module analysis the same way
    *collect_submodules("curvestudio.tools.stats"),
    # the export UI (plotting + reporting) is not imported anywhere in the app's
    # own graph yet; collecting them pulls in matplotlib and reportlab, whose
    # hooks then ship mpl-data and the PDF fonts
    *collect_submodules("curvestudio_plotting"),
    *collect_submodules("curvestudio_reporting"),
]

datas = []
# importlib.metadata (model registry entry points, app version) needs dist-info
for _distribution in (
    "curvestudio-core",
    "curvestudio-models",
    "curvestudio-engineering",
    "curvestudio-plotting",
    "curvestudio-reporting",
    "curvestudio-desktop",
):
    datas += copy_metadata(_distribution)
# every non-Python file in the package, which is exactly what the app needs at run
# time: the bundled example and licence notice, the brand PNGs and OFL fonts, the
# vendored Lucide SVGs and their licence, and the help text under resources/help/
# (the Help browser reads it through importlib.resources, so no module import can
# pull it in). No separate datas entry is needed for any of them.
datas += collect_data_files("curvestudio_desktop", include_py_files=False)
# reportlab's font files are plain data — its PyInstaller hook collects only the
# font-metric *modules*, so the Type1/TTF files (pdfbase's standard fonts) ship
# only through an explicit collect
datas += collect_data_files("reportlab")

a = Analysis(
    [os.path.join(SPECPATH, "launch.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
    # The shared workspace venv also installs the server and test stacks, and static
    # analysis reaches them from modules the app does import. Nothing here imports them
    # at run time, so they are excluded from the bundle (checked against the frozen
    # module graph, xref-curvestudio-desktop.html):
    # * sqlalchemy — only pandas.io.sql references it, under TYPE_CHECKING and lazily
    #   via import_optional_dependency; no desktop code path calls the pandas SQL API
    # * psycopg / psycopg2 / greenlet — imported only by sqlalchemy (its PostgreSQL
    #   dialect and its async bridge), so they drop with sqlalchemy
    # * hypothesis — only pydantic's legacy v1 hypothesis plugin imports it
    #   (pydantic.v1._hypothesis_plugin, which pydantic's PyInstaller hook collects
    #   via collect_submodules); the app imports pydantic v2 models only
    # PIL/pillow is deliberately NOT excluded: matplotlib 3.11 imports it at module
    # level (matplotlib/colors.py, matplotlib/backends/backend_agg.py), and the PDF
    # reports import matplotlib.
    excludes=[
        "tkinter",
        "sqlalchemy",
        "psycopg",
        "psycopg_binary",
        "greenlet",
        "hypothesis",
    ],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="curvestudio-desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Windows ships as a GUI app (no console window); elsewhere the console stays
    # on so `--selftest` output is capturable
    console=sys.platform != "win32",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # generated by make_icon.py (build.py); unused (None) on Linux
    icon=os.environ.get("CURVESTUDIO_ICON"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="curvestudio-desktop",
)
