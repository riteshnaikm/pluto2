#!/bin/bash
# Verification script for HR Assistant Suite deployment
# Usage: bash verify-deployment.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

ERRORS=0
WARNINGS=0

echo "================================================"
echo "HR Assistant Suite - Deployment Verification"
echo "================================================"

# Check if running from correct directory
if [ ! -f "app.py" ]; then
    print_error "app.py not found! Please run this script from the application directory."
    exit 1
fi

print_header "1. System Requirements"

# Check Python version
if command -v python3.10 &> /dev/null; then
    PYTHON_VERSION=$(python3.10 --version)
    print_success "Python 3.10 installed: $PYTHON_VERSION"
else
    print_error "Python 3.10 not found"
    ((ERRORS++))
fi

# Check pip
if command -v pip &> /dev/null; then
    print_success "pip is installed"
else
    print_error "pip not found"
    ((ERRORS++))
fi

# Check virtual environment
if [ -d "venv" ]; then
    print_success "Virtual environment exists"
    
    # Check if venv has packages
    if [ -d "venv/lib/python3.10/site-packages" ]; then
        PKG_COUNT=$(ls -1 venv/lib/python3.10/site-packages | wc -l)
        print_success "Virtual environment has $PKG_COUNT packages"
    else
        print_error "Virtual environment appears empty"
        ((ERRORS++))
    fi
else
    print_error "Virtual environment not found"
    ((ERRORS++))
fi

print_header "2. Required Files"

# Check critical files
FILES=("app.py" "run_production.py" ".env" "requirements_for_server.txt")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file missing"
        ((ERRORS++))
    fi
done

# Check directories
DIRS=("static" "templates" "HR_docs" "uploads" "deploy_scripts")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "$dir/ directory exists"
    else
        print_error "$dir/ directory missing"
        ((ERRORS++))
    fi
done

print_header "3. Environment Configuration"

# Check .env file
if [ -f ".env" ]; then
    if grep -q "your_.*_api_key_here" .env; then
        print_error "⚠️  .env file contains placeholder keys - please update with real API keys"
        ((WARNINGS++))
    else
        print_success ".env file appears to be configured"
    fi
    
    # Check .env permissions
    PERMS=$(stat -c %a .env 2>/dev/null || stat -f %A .env)
    if [ "$PERMS" = "600" ]; then
        print_success ".env has correct permissions (600)"
    else
        print_error ".env has insecure permissions ($PERMS) - should be 600"
        ((WARNINGS++))
    fi
else
    print_error ".env file not found"
    ((ERRORS++))
fi

print_header "4. HR Documents"

# Check HR_docs folder
if [ -d "HR_docs" ]; then
    PDF_COUNT=$(find HR_docs -name "*.pdf" -type f | wc -l)
    if [ $PDF_COUNT -gt 0 ]; then
        print_success "Found $PDF_COUNT PDF files in HR_docs/"
    else
        print_error "No PDF files found in HR_docs/"
        ((WARNINGS++))
    fi
else
    print_error "HR_docs directory not found"
    ((ERRORS++))
fi

print_header "5. Systemd Service"

# Check if service file exists
if [ -f "/etc/systemd/system/hr-assistant.service" ]; then
    print_success "Systemd service file exists"
    
    # Check service status
    if systemctl is-enabled hr-assistant &> /dev/null; then
        print_success "Service is enabled (will start on boot)"
    else
        print_error "Service is not enabled"
        ((WARNINGS++))
    fi
    
    # Check if service is running
    if systemctl is-active hr-assistant &> /dev/null; then
        print_success "Service is running"
    else
        print_error "Service is not running"
        echo "         Try: sudo systemctl start hr-assistant"
        ((WARNINGS++))
    fi
else
    print_error "Systemd service file not found"
    echo "         Run: bash deploy_scripts/install.sh"
    ((ERRORS++))
fi

print_header "6. Network & Firewall"

# Check if port 5000 is listening
if command -v netstat &> /dev/null; then
    if netstat -tlnp 2>/dev/null | grep -q ":5000"; then
        print_success "Port 5000 is listening"
    else
        print_error "Port 5000 is not listening"
        ((WARNINGS++))
    fi
elif command -v ss &> /dev/null; then
    if ss -tlnp 2>/dev/null | grep -q ":5000"; then
        print_success "Port 5000 is listening"
    else
        print_error "Port 5000 is not listening"
        ((WARNINGS++))
    fi
else
    print_info "Cannot check port status (netstat/ss not found)"
fi

# Check firewall
if command -v ufw &> /dev/null; then
    if ufw status | grep -q "Status: active"; then
        print_success "UFW firewall is active"
        
        if ufw status | grep -q "5000"; then
            print_success "Port 5000 is allowed in firewall"
        else
            print_error "Port 5000 is not allowed in firewall"
            echo "         Try: sudo ufw allow 5000/tcp"
            ((WARNINGS++))
        fi
    else
        print_info "UFW firewall is not active"
    fi
else
    print_info "UFW not installed"
fi

print_header "7. Dependencies"

# Activate venv and check key packages
if [ -d "venv" ]; then
    source venv/bin/activate
    
    PACKAGES=("flask" "langchain" "pinecone" "groq" "nltk")
    for pkg in "${PACKAGES[@]}"; do
        if pip show $pkg &> /dev/null; then
            print_success "$pkg is installed"
        else
            print_error "$pkg is not installed"
            ((ERRORS++))
        fi
    done
    
    deactivate
fi

print_header "8. Application Test"

# Try to import the app
if [ -d "venv" ]; then
    source venv/bin/activate
    
    if python3 -c "import app" 2>/dev/null; then
        print_success "app.py can be imported successfully"
    else
        print_error "app.py cannot be imported - there may be syntax errors"
        ((ERRORS++))
    fi
    
    deactivate
fi

# Test HTTP endpoint if service is running
if systemctl is-active hr-assistant &> /dev/null; then
    if curl -s http://localhost:5000 > /dev/null; then
        print_success "Application responds to HTTP requests"
    else
        print_error "Application does not respond to HTTP requests"
        ((ERRORS++))
    fi
fi

print_header "9. Disk Space"

# Check disk space
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    print_success "Disk space is adequate ($DISK_USAGE% used)"
else
    print_error "Disk space is low ($DISK_USAGE% used)"
    ((WARNINGS++))
fi

print_header "10. Memory"

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2 {print $7}')
if [ $AVAILABLE_MEM -gt 500 ]; then
    print_success "Sufficient memory available (${AVAILABLE_MEM}MB)"
else
    print_error "Low memory available (${AVAILABLE_MEM}MB)"
    ((WARNINGS++))
fi

print_header "11. Helper Scripts"

# Check if helper scripts exist
if [ -f "$HOME/backup-hr-assistant.sh" ]; then
    print_success "Backup script exists"
else
    print_info "Backup script not found (optional)"
fi

if [ -f "$HOME/monitor-hr-assistant.sh" ]; then
    print_success "Monitoring script exists"
else
    print_info "Monitoring script not found (optional)"
fi

print_header "12. Nginx (Optional)"

# Check if Nginx is installed
if command -v nginx &> /dev/null; then
    print_success "Nginx is installed"
    
    if systemctl is-active nginx &> /dev/null; then
        print_success "Nginx is running"
    else
        print_info "Nginx is installed but not running"
    fi
    
    if [ -f "/etc/nginx/sites-enabled/hr-assistant" ]; then
        print_success "Nginx configuration for hr-assistant exists"
    else
        print_info "Nginx configuration for hr-assistant not found"
    fi
else
    print_info "Nginx is not installed (optional)"
fi

# Summary
echo ""
echo "================================================"
echo "Verification Summary"
echo "================================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your deployment looks good!${NC}"
    echo ""
    echo "You can access your application at:"
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo "  http://$SERVER_IP:5000"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Deployment is functional but has $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please review the warnings above."
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before using the application."
    echo ""
    echo "Common fixes:"
    echo "  - Run: bash deploy_scripts/install.sh"
    echo "  - Update .env with real API keys"
    echo "  - Start service: sudo systemctl start hr-assistant"
    exit 1
fi

