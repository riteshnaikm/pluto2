#!/bin/bash
# Installation script for HR Assistant Suite on Ubuntu/Debian
# Usage: bash install.sh

set -e

echo "============================================"
echo "HR Assistant Suite - Installation Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please do not run this script as root"
    exit 1
fi

print_info "Starting installation..."
echo ""

# Update system packages
print_info "Updating system packages..."
sudo apt update
sudo apt upgrade -y
print_success "System packages updated"
echo ""

# Install Python 3.10
print_info "Installing Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip
print_success "Python 3.10 installed"
echo ""

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev poppler-utils git htop
print_success "System dependencies installed"
echo ""

# Get current directory
APP_DIR=$(pwd)
print_info "Application directory: $APP_DIR"
echo ""

# Create virtual environment
print_info "Creating virtual environment..."
python3.10 -m venv venv
print_success "Virtual environment created"
echo ""

# Activate virtual environment and install dependencies
print_info "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_for_server.txt
print_success "Python dependencies installed"
echo ""

# Download NLTK data
print_info "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
print_success "NLTK data downloaded"
echo ""

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p uploads HR_docs
chmod 755 uploads HR_docs
print_success "Directories created"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_info "Creating .env file..."
    cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
EOF
    chmod 600 .env
    print_success ".env file created"
    print_error "⚠️  IMPORTANT: Edit .env file and add your actual API keys!"
else
    print_success ".env file already exists"
fi
echo ""

# Create systemd service file
print_info "Creating systemd service..."
USERNAME=$(whoami)
SERVICE_FILE="/etc/systemd/system/hr-assistant.service"

sudo bash -c "cat > $SERVICE_FILE" << EOF
[Unit]
Description=HR Assistant Suite
After=network.target

[Service]
Type=simple
User=$USERNAME
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/python3 $APP_DIR/run_production.py
Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal
SyslogIdentifier=hr-assistant

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service created"
echo ""

# Reload systemd
print_info "Reloading systemd..."
sudo systemctl daemon-reload
print_success "Systemd reloaded"
echo ""

# Enable service
print_info "Enabling HR Assistant service..."
sudo systemctl enable hr-assistant
print_success "Service enabled"
echo ""

# Configure firewall
print_info "Configuring firewall..."
sudo apt install -y ufw
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 5000/tcp  # Application port
print_success "Firewall configured"
echo ""

print_info "Would you like to enable the firewall now? (y/n)"
read -r ENABLE_UFW
if [ "$ENABLE_UFW" = "y" ] || [ "$ENABLE_UFW" = "Y" ]; then
    sudo ufw --force enable
    print_success "Firewall enabled"
else
    print_info "Firewall not enabled. You can enable it later with: sudo ufw enable"
fi
echo ""

# Create backup script
print_info "Creating backup script..."
cat > ~/backup-hr-assistant.sh << 'EOFBACKUP'
#!/bin/bash
BACKUP_DIR="$HOME/backups"
APP_DIR="$HOME/hr-assistant"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
if [ -f "$APP_DIR/combined_db.db" ]; then
    cp $APP_DIR/combined_db.db $BACKUP_DIR/combined_db_$DATE.db
fi

# Backup uploads
if [ -d "$APP_DIR/uploads" ]; then
    tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz -C $APP_DIR uploads/
fi

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t combined_db_*.db 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null
ls -t uploads_*.tar.gz 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null

echo "Backup completed: $DATE"
EOFBACKUP

chmod +x ~/backup-hr-assistant.sh
print_success "Backup script created at ~/backup-hr-assistant.sh"
echo ""

# Create monitoring script
print_info "Creating monitoring script..."
cat > ~/monitor-hr-assistant.sh << 'EOFMONITOR'
#!/bin/bash
echo "=== HR Assistant Status ==="
sudo systemctl status hr-assistant --no-pager
echo ""
echo "=== Memory Usage ==="
ps aux | grep "python.*run_production.py" | grep -v grep
echo ""
echo "=== Recent Logs ==="
sudo journalctl -u hr-assistant -n 20 --no-pager
EOFMONITOR

chmod +x ~/monitor-hr-assistant.sh
print_success "Monitoring script created at ~/monitor-hr-assistant.sh"
echo ""

# Installation complete
echo ""
echo "============================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Edit the .env file and add your API keys:"
echo "   nano .env"
echo ""
echo "2. Add your HR documents to the HR_docs folder"
echo ""
echo "3. Start the service:"
echo "   sudo systemctl start hr-assistant"
echo ""
echo "4. Check the service status:"
echo "   sudo systemctl status hr-assistant"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u hr-assistant -f"
echo ""
echo "6. Access the application at:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Useful scripts created:"
echo "  - ~/backup-hr-assistant.sh (backup database and uploads)"
echo "  - ~/monitor-hr-assistant.sh (monitor application status)"
echo ""
echo "============================================"

