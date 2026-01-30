#!/bin/bash
# Nginx setup script for HR Assistant Suite
# Usage: bash nginx-setup.sh your_domain.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if domain provided
if [ -z "$1" ]; then
    print_error "Please provide a domain name or server IP"
    echo "Usage: bash nginx-setup.sh your_domain.com"
    exit 1
fi

DOMAIN=$1
USERNAME=$(whoami)
APP_DIR=$(pwd)

echo "============================================"
echo "Nginx Reverse Proxy Setup"
echo "============================================"
echo ""
print_info "Domain/IP: $DOMAIN"
print_info "Application directory: $APP_DIR"
echo ""

# Install Nginx
print_info "Installing Nginx..."
sudo apt update
sudo apt install -y nginx
print_success "Nginx installed"
echo ""

# Create Nginx configuration
print_info "Creating Nginx configuration..."
sudo bash -c "cat > /etc/nginx/sites-available/hr-assistant" << EOF
server {
    listen 80;
    server_name $DOMAIN;

    # Increase timeouts for long-running requests
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static/ {
        alias $APP_DIR/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

print_success "Nginx configuration created"
echo ""

# Enable the site
print_info "Enabling site..."
sudo ln -sf /etc/nginx/sites-available/hr-assistant /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
print_success "Site enabled"
echo ""

# Test Nginx configuration
print_info "Testing Nginx configuration..."
sudo nginx -t
print_success "Nginx configuration is valid"
echo ""

# Update firewall
print_info "Updating firewall rules..."
sudo ufw allow 'Nginx Full'
print_success "Firewall rules updated"
echo ""

# Restart Nginx
print_info "Restarting Nginx..."
sudo systemctl restart nginx
sudo systemctl enable nginx
print_success "Nginx restarted and enabled"
echo ""

# Update run_production.py to bind to localhost only
print_info "Updating application to bind to localhost..."
sed -i 's/config.bind = \["0.0.0.0:5000"\]/config.bind = ["127.0.0.1:5000"]/' run_production.py
print_success "Application configuration updated"
echo ""

# Restart application
print_info "Restarting HR Assistant service..."
sudo systemctl restart hr-assistant
print_success "Service restarted"
echo ""

echo "============================================"
echo -e "${GREEN}Nginx Setup Complete!${NC}"
echo "============================================"
echo ""
echo "Your application should now be accessible at:"
echo "  http://$DOMAIN"
echo ""
echo "To enable HTTPS with Let's Encrypt:"
echo "  1. sudo apt install certbot python3-certbot-nginx"
echo "  2. sudo certbot --nginx -d $DOMAIN"
echo ""
echo "============================================"

