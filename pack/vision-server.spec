# PyInstaller spec for the Windows PLAYER build.
# Run this ON Windows, from python-vision-server, with the venv active:
#   pyinstaller --noconfirm pack/vision-server.spec
#
# Output: dist/VisionServer/vision-server.exe  (+ sibling DLLs)
# Copy the whole VisionServer folder next to the Unity game .exe.
#
# Does NOT include training data, eval scripts, or a console window.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe", "tensorflow", "keras", "cv2"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += [("models/escape_gestures.keras", "models")]

a = Analysis(
    ["scripts/run_server.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "sklearn",
        "matplotlib",
        "vision_server.training",
        "vision_server.recording",
        "vision_server.evaluation",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vision-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VisionServer",
)
