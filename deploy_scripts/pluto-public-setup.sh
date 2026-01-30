#!/bin/bash
# Simple setup script for pluto.peoplelogic.in (PUBLIC - no IP restrictions)
# Usage: bash pluto-public-setup.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

echo "============================================"
echo "Pluto Subdomain Setup (PUBLIC ACCESS)"
echo "============================================"
echo ""
print_info "This will make pluto.peoplelogic.in publicly accessible"
print_info "You can add IP restrictions later if needed"
echo ""

# Install Nginx if not installed
if ! command -v nginx &> /dev/null; then
    print_info "Installing Nginx..."
    sudo apt update
    sudo apt install -y nginx
    print_success "Nginx installed"
else
    print_success "Nginx is already installed"
fi
echo ""

# Create Nginx config (NO IP restrictions)
print_info "Creating Nginx configuration..."
sudo bash -c "cat > /etc/nginx/sites-available/pluto.peoplelogic.in" << 'NGINXEOF'
server {
    listen 80;
    server_name pluto.peoplelogic.in;

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # File upload size
        client_max_body_size 50M;
    }
}
NGINXEOF

print_success "Nginx configuration created"
echo ""

# Enable site
print_info "Enabling site..."
sudo ln -sf /etc/nginx/sites-available/pluto.peoplelogic.in /etc/nginx/sites-enabled/pluto.peoplelogic.in
print_success "Site enabled"
echo ""

# Test Nginx config
print_info "Testing Nginx configuration..."
if sudo nginx -t; then
    print_success "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors!"
    exit 1
fi
echo ""

# Reload Nginx
print_info "Reloading Nginx..."
sudo systemctl reload nginx
print_success "Nginx reloaded"
echo ""

# Setup SSL
print_info "Setting up SSL certificate..."
if command -v certbot &> /dev/null; then
    print_success "Certbot is already installed"
else
    print_info "Installing Certbot..."
    sudo apt install -y certbot python3-certbot-nginx
    print_success "Certbot installed"
fi
echo ""

print_info "Obtaining SSL certificate..."
echo "This will prompt for email and agreement to terms."
echo ""
read -p "Enter your email for SSL certificate notifications: " EMAIL
if [ -z "$EMAIL" ]; then
    EMAIL="admin@peoplelogic.in"
fi

print_info "Getting SSL certificate for pluto.peoplelogic.in..."
sudo certbot --nginx -d pluto.peoplelogic.in --non-interactive --agree-tos --email "$EMAIL" --redirect 2>&1 || {
    print_error "SSL setup failed. You can run manually:"
    echo "  sudo certbot --nginx -d pluto.peoplelogic.in"
    echo ""
    print_info "Continuing without SSL (HTTP only)..."
}

# Update firewall
print_info "Updating firewall rules..."
sudo ufw allow 80/tcp 2>/dev/null || true
sudo ufw allow 443/tcp 2>/dev/null || true
print_success "Firewall rules updated"
echo ""

echo "============================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================"
echo ""
echo "Your application is now accessible at:"
if sudo certbot certificates 2>/dev/null | grep -q "pluto.peoplelogic.in"; then
    echo "  https://pluto.peoplelogic.in ✅ (HTTPS)"
    echo "  http://pluto.peoplelogic.in (redirects to HTTPS)"
else
    echo "  http://pluto.peoplelogic.in"
    echo ""
    echo "To enable HTTPS later, run:"
    echo "  sudo certbot --nginx -d pluto.peoplelogic.in"
fi
echo ""
echo "⚠️  NOTE: This is currently PUBLIC (no IP restrictions)"
echo ""
echo "To add IP restrictions later, run:"
echo "  bash deploy_scripts/pluto-subdomain-setup.sh"
echo ""
echo "============================================"

