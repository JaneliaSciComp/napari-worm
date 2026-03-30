# napari-worm: One-click installer + launcher for Windows
#
# Usage (paste in PowerShell):
#   irm https://raw.githubusercontent.com/JaneliaSciComp/napari-worm/main/install.ps1 | iex
#
# Prerequisites:
#   - Windows with NRS drive mapped (\\nrs.janelia.org\shroff)
#   - Git (https://git-scm.com/download/win)

$ErrorActionPreference = "Stop"
$INSTALL_DIR = "$HOME\napari-worm"
$DATA_PATH = "\\nrs.janelia.org\shroff\data_internal\celegans_mipav_data\2025_12_16\efn-2\120222_efn-2\Pos1\SPIMB\Reg_Sample\For_Tracking\RegB\"

Write-Host "============================================"
Write-Host "  napari-worm installer (Windows)"
Write-Host "============================================"
Write-Host ""

# 1. Install pixi if not present
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    Write-Host "[1/3] Installing pixi package manager..."
    iwr -useb https://pixi.sh/install.ps1 | iex
    # Refresh PATH
    $env:PATH = "$HOME\.pixi\bin;$env:PATH"
    Write-Host "  pixi installed."
} else {
    $ver = pixi --version
    Write-Host "[1/3] pixi already installed ($ver)"
}

# 2. Check git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: Git is not installed."
    Write-Host "  Download from: https://git-scm.com/download/win"
    Write-Host "  Then re-run this script."
    exit 1
}

# 3. Clone or update repository
if (Test-Path "$INSTALL_DIR\.git") {
    Write-Host "[2/3] Updating napari-worm..."
    Set-Location $INSTALL_DIR
    git pull --ff-only
} else {
    Write-Host "[2/3] Cloning napari-worm..."
    git clone https://github.com/JaneliaSciComp/napari-worm.git $INSTALL_DIR
    Set-Location $INSTALL_DIR
}

# 4. Install Python dependencies
Write-Host "[3/3] Installing dependencies (this may take a few minutes on first run)..."
pixi install

Write-Host ""
Write-Host "============================================"
Write-Host "  Installation complete!"
Write-Host "============================================"
Write-Host ""
Write-Host "To launch napari-worm:"
Write-Host ""
Write-Host "  cd $INSTALL_DIR"
Write-Host "  pixi run python napari_worm.py `"$DATA_PATH`" --start 100"
Write-Host ""
Write-Host "Make sure the NRS drive is mapped first:"
Write-Host "  File Explorer -> Map Network Drive -> \\nrs.janelia.org\shroff"
Write-Host "  Or in PowerShell: net use Z: \\nrs.janelia.org\shroff"
Write-Host ""
Write-Host "If using a mapped drive letter (e.g. Z:), use:"
Write-Host "  pixi run python napari_worm.py `"Z:\data_internal\...\RegB\`" --start 100"
Write-Host ""
Write-Host "Keyboard shortcuts:"
Write-Host "  Ctrl+Click     = annotate / place lattice point"
Write-Host "  L              = toggle lattice mode"
Write-Host "  S              = save all"
Write-Host "  Ctrl+Z         = undo"
Write-Host "  Right/Left     = navigate timepoints"
Write-Host "  W              = wireframe mesh"
Write-Host "============================================"
