#!/usr/bin/env bash
set -euo pipefail

# Installs system dependencies required for:
# - Legacy .doc conversion (LibreOffice -> soffice)
# - Scanned PDF OCR (Tesseract)
#
# Supported distros:
# - Debian/Ubuntu (apt)
# - RHEL/CentOS/Rocky/Alma (dnf/yum)
#
# Usage:
#   bash deploy_scripts/install-ocr-and-doc-conversion.sh

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libreoffice tesseract-ocr
  echo "Installed via apt: libreoffice, tesseract-ocr"
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y libreoffice tesseract
  echo "Installed via dnf: libreoffice, tesseract"
  exit 0
fi

if command -v yum >/dev/null 2>&1; then
  sudo yum install -y libreoffice tesseract
  echo "Installed via yum: libreoffice, tesseract"
  exit 0
fi

echo "Unsupported Linux distro: couldn't find apt-get, dnf, or yum."
echo "Please install system packages manually: LibreOffice + Tesseract OCR."
exit 1

