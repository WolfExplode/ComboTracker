# PyInstaller spec for ComboTracker
# Build: pyinstaller ComboTracker.spec
# Output: dist/ComboTracker/ComboTracker.exe (one-folder bundle)
#
# Keep this as a one-folder, non-UPX build. Self-extracting one-file executables
# that install global input hooks are prone to antivirus heuristic detections.
#
# Windows: uac_admin=True embeds a manifest so the exe requests Administrator elevation.
# That matches elevated games so global keyboard/mouse capture (pynput) works in-game.

import sys

block_cipher = None

static_src = "static"
static_datas = (static_src, "static")

a = Analysis(
    ['ui_server.py'],
    pathex=[],
    binaries=[],
    datas=[static_datas],
    hiddenimports=['pynput.keyboard._win32', 'pynput.mouse._win32'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_exe_options = dict(
    name="ComboTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if sys.platform == "win32":
    _exe_options["uac_admin"] = True

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    **_exe_options,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ComboTracker",
)
