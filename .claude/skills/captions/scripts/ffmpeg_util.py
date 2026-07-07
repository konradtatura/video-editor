import os
import shutil
import sys


def find_ffmpeg(binary="ffmpeg"):
    """Looks up ffmpeg/ffprobe on PATH. Also checks FFMPEG_BIN_DIR (an
    optional env var pointing at the install's bin/ directory) for setups
    where the installer doesn't add it to PATH -- e.g. a fresh winget
    install on Windows before the shell has been restarted."""
    path = shutil.which(binary)
    if path:
        return path
    bin_dir = os.environ.get("FFMPEG_BIN_DIR")
    if bin_dir:
        ext = ".exe" if os.name == "nt" else ""
        candidate = os.path.join(bin_dir, f"{binary}{ext}")
        if os.path.exists(candidate):
            return candidate
    print(
        f"{binary} not found on PATH. Install it (macOS: `brew install ffmpeg`, "
        f"Windows: `winget install Gyan.FFmpeg` then restart the shell) or set "
        f"FFMPEG_BIN_DIR to its bin/ directory.",
        file=sys.stderr,
    )
    sys.exit(1)
