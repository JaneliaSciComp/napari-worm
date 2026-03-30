#!/bin/bash
# napari-worm: One-click installer + launcher
#
# Usage (paste in Terminal):
#   curl -fsSL https://raw.githubusercontent.com/JaneliaSciComp/napari-worm/main/install.sh | bash
#
# Prerequisites:
#   - macOS with NRS drive mounted (Finder → Cmd+K → smb://nrs.janelia.org/shroff)
#   - Git (pre-installed on macOS)

set -e

INSTALL_DIR="$HOME/napari-worm"
DATA_PATH="/Volumes/shroff/data_internal/celegans_mipav_data/2025_12_16/efn-2/120222_efn-2/Pos1/SPIMB/Reg_Sample/For_Tracking/RegB/"

echo "============================================"
echo "  napari-worm installer"
echo "============================================"
echo ""

# 1. Install pixi if not present
if ! command -v pixi &>/dev/null; then
    echo "[1/3] Installing pixi package manager..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
    echo "  pixi installed."
else
    echo "[1/3] pixi already installed ($(pixi --version))"
fi

# 2. Clone or update repository
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[2/3] Updating napari-worm..."
    cd "$INSTALL_DIR" && git pull --ff-only
else
    echo "[2/3] Cloning napari-worm..."
    git clone https://github.com/JaneliaSciComp/napari-worm.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 3. Install Python dependencies
echo "[3/3] Installing dependencies (this may take a few minutes on first run)..."
pixi install

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "To launch napari-worm:"
echo ""
echo "  cd $INSTALL_DIR"
echo "  pixi run python napari_worm.py \"$DATA_PATH\" --start 100"
echo ""
echo "Make sure the NRS drive is mounted first:"
echo "  Finder → Cmd+K → smb://nrs.janelia.org/shroff"
echo ""
echo "Keyboard shortcuts:"
echo "  Cmd+Click      = annotate / place lattice point"
echo "  L              = toggle lattice mode"
echo "  S              = save all"
echo "  Cmd+Z          = undo"
echo "  Right/Left     = navigate timepoints"
echo "  W              = wireframe mesh"
echo "============================================"
