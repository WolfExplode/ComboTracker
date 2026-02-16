# PyInstaller spec for ComboTracker
# Build: pyinstaller ComboTracker.spec
# Output: dist/ComboTracker/ComboTracker.exe (one-folder) or dist/ComboTracker.exe (one-file, slower startup)

import sys

block_cipher = None

# When running, static files must be in the same folder as the script in the bundle
static_src = 'static'
if sys.platform == 'win32':
    # Windows: PyInstaller uses semicolon for path separator in datas
    static_datas = (static_src, 'static')
else:
    static_datas = (static_src, 'static')

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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ComboTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console so user sees server URL and can Ctrl+C
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
