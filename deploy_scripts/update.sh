#!/bin/bash
# Update script for HR Assistant Suite
# Usage: bash update.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

echo "============================================"
echo "HR Assistant Suite - Update Script"
echo "============================================"
echo ""

APP_DIR=$(pwd)

print_info "Stopping service..."
sudo systemctl stop hr-assistant
print_success "Service stopped"
echo ""

print_info "Updating dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements_for_server.txt
print_success "Dependencies updated"
echo ""

print_info "Downloading latest NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
print_success "NLTK data updated"
echo ""

print_info "Starting service..."
sudo systemctl start hr-assistant
print_success "Service started"
echo ""

print_info "Checking service status..."
sudo systemctl status hr-assistant --no-pager
echo ""

echo "============================================"
echo -e "${GREEN}Update Complete!${NC}"
echo "============================================"
echo ""
echo "View logs with: sudo journalctl -u hr-assistant -f"
echo ""

