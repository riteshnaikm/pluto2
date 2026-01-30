#!/bin/bash
# Setup script for pluto.peoplelogic.in subdomain with IP whitelist
# Usage: bash pluto-subdomain-setup.sh

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
echo "Pluto Subdomain Setup with IP Whitelist"
echo "============================================"
echo ""

# Get current IP (for reference)
CURRENT_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")
print_info "Your current public IP: $CURRENT_IP"
echo ""

# Get office IPs
echo "Enter your office IP addresses (one per line, press Enter twice when done):"
echo "Example: 123.45.67.89"
echo "For subnets, use CIDR notation: 10.0.0.0/24"
echo ""
OFFICE_IPS=()
while IFS= read -r line; do
    [ -z "$line" ] && break
    OFFICE_IPS+=("$line")
done
echo ""

# Get VPN subnet
echo "Enter your VPN subnet (CIDR notation, e.g., 10.0.0.0/8 or 100.64.0.0/10):"
echo "Press Enter to skip if no VPN:"
read -r VPN_SUBNET
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

# Create Nginx config
print_info "Creating Nginx configuration..."
sudo bash -c "cat > /etc/nginx/sites-available/pluto.peoplelogic.in" << 'NGINXEOF'
# IP Whitelist Configuration
geo $allowed_ip {
    default 0;
    127.0.0.1 1;
    ::1 1;
NGINXEOF

# Add office IPs to geo block
for ip in "${OFFICE_IPS[@]}"; do
    echo "    $ip 1;" | sudo tee -a /etc/nginx/sites-available/pluto.peoplelogic.in > /dev/null
done

# Add VPN subnet if provided
if [ -n "$VPN_SUBNET" ]; then
    echo "    $VPN_SUBNET 1;" | sudo tee -a /etc/nginx/sites-available/pluto.peoplelogic.in > /dev/null
fi

# Complete the geo block and add server config
sudo bash -c "cat >> /etc/nginx/sites-available/pluto.peoplelogic.in" << 'NGINXEOF'
}

server {
    listen 80;
    server_name pluto.peoplelogic.in;

    # Block all requests unless from whitelist
    if ($allowed_ip != 1) {
        return 403;
    }

    # Health check endpoint (allow localhost)
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
sudo certbot --nginx -d pluto.peoplelogic.in --non-interactive --agree-tos --email admin@peoplelogic.in --redirect 2>&1 || {
    print_error "SSL setup failed. You can run manually: sudo certbot --nginx -d pluto.peoplelogic.in"
}

# Update firewall
print_info "Updating firewall rules..."
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
print_success "Firewall rules updated"
echo ""

echo "============================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================"
echo ""
echo "Your application is now accessible at:"
echo "  https://pluto.peoplelogic.in"
echo ""
echo "IP Whitelist configured:"
for ip in "${OFFICE_IPS[@]}"; do
    echo "  - $ip"
done
if [ -n "$VPN_SUBNET" ]; then
    echo "  - $VPN_SUBNET (VPN subnet)"
fi
echo ""
echo "To add more IPs later, edit:"
echo "  sudo nano /etc/nginx/sites-available/pluto.peoplelogic.in"
echo "  Then run: sudo nginx -t && sudo systemctl reload nginx"
echo ""
echo "============================================"

