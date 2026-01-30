# Deployment Scripts for HR Assistant Suite

This directory contains automated scripts to help you deploy the HR Assistant Suite on your Linux server.

## 📋 Available Scripts

### 1. `install.sh` - Full Installation
Automates the complete installation process including:
- System updates
- Python 3.10 installation
- Virtual environment setup
- Dependencies installation
- Systemd service creation
- Firewall configuration
- Backup and monitoring scripts

**Usage:**
```bash
cd /path/to/hr-assistant
bash deploy_scripts/install.sh
```

**After installation:**
1. Edit `.env` file with your API keys
2. Add HR documents to `HR_docs/` folder
3. Start the service: `sudo systemctl start hr-assistant`

---

### 2. `nginx-setup.sh` - Nginx Reverse Proxy Setup
Sets up Nginx as a reverse proxy for better security and performance.

**Usage:**
```bash
cd /path/to/hr-assistant
bash deploy_scripts/nginx-setup.sh your_domain.com
# Or use your server IP:
bash deploy_scripts/nginx-setup.sh 192.168.1.100
```

**This script:**
- Installs and configures Nginx
- Creates virtual host configuration
- Updates firewall rules
- Restarts services

---

### 3. `update.sh` - Update Application
Updates the application dependencies and restarts the service.

**Usage:**
```bash
cd /path/to/hr-assistant
bash deploy_scripts/update.sh
```

---

## 🚀 Quick Start Guide

### Step 1: Transfer Files to Server

**Option A: Using SCP from Windows**
```powershell
scp -r "C:\Users\Ritesh\Desktop\Cursor 1" username@server_ip:/home/username/hr-assistant
```

**Option B: Using SFTP client**
- Use FileZilla, WinSCP, or similar
- Upload all files to `/home/username/hr-assistant`

### Step 2: Connect to Server
```bash
ssh username@server_ip
cd hr-assistant
```

### Step 3: Run Installation Script
```bash
bash deploy_scripts/install.sh
```

### Step 4: Configure API Keys
```bash
nano .env
```
Add your actual API keys and save (Ctrl+X, Y, Enter)

### Step 5: Add HR Documents
Transfer your PDF files to the `HR_docs/` folder

### Step 6: Start the Service
```bash
sudo systemctl start hr-assistant
sudo systemctl status hr-assistant
```

### Step 7: Access the Application
Open browser and navigate to:
```
http://your_server_ip:5000
```

---

## 🌐 Optional: Setup Nginx (Recommended for Production)

After the basic installation is complete:

```bash
bash deploy_scripts/nginx-setup.sh your_domain.com
```

Then access via:
```
http://your_domain.com
```

---

## 📊 Managing the Service

### Start/Stop/Restart
```bash
sudo systemctl start hr-assistant
sudo systemctl stop hr-assistant
sudo systemctl restart hr-assistant
```

### View Status
```bash
sudo systemctl status hr-assistant
```

### View Logs (Real-time)
```bash
sudo journalctl -u hr-assistant -f
```

### View Last 100 Log Lines
```bash
sudo journalctl -u hr-assistant -n 100
```

---

## 🔄 Updating the Application

When you need to update dependencies or code:

```bash
# If you made code changes, transfer new files first
# Then run:
bash deploy_scripts/update.sh
```

---

## 💾 Backup and Monitoring

After installation, you'll have these helper scripts in your home directory:

### Backup Script
```bash
~/backup-hr-assistant.sh
```
Backs up database and uploads folder. Keeps last 7 backups.

**Schedule daily backups:**
```bash
crontab -e
# Add this line:
0 2 * * * /home/username/backup-hr-assistant.sh >> /home/username/backup.log 2>&1
```

### Monitoring Script
```bash
~/monitor-hr-assistant.sh
```
Shows service status, memory usage, and recent logs.

---

## 🔒 Security Checklist

- [ ] Firewall enabled: `sudo ufw status`
- [ ] API keys secured: `chmod 600 .env`
- [ ] Running as non-root user
- [ ] Nginx configured (for production)
- [ ] SSL/HTTPS enabled (optional but recommended)
- [ ] Regular backups scheduled
- [ ] Server regularly updated: `sudo apt update && sudo apt upgrade`

---

## 🐛 Troubleshooting

### Service won't start
```bash
# Check logs for errors
sudo journalctl -u hr-assistant -n 50

# Check if port is already in use
sudo lsof -i :5000

# Verify virtual environment
source venv/bin/activate
python3 --version
```

### Can't access from browser
```bash
# Check if service is running
sudo systemctl status hr-assistant

# Check firewall
sudo ufw status

# Test locally first
curl http://localhost:5000
```

### Permission errors
```bash
# Fix ownership
sudo chown -R $USER:$USER /home/$USER/hr-assistant

# Fix permissions
chmod -R 755 /home/$USER/hr-assistant
chmod 755 uploads HR_docs
chmod 600 .env
```

### Out of memory
```bash
# Check memory usage
free -h

# Reduce workers in run_production.py
nano run_production.py
# Change: config.workers = 4  to  config.workers = 2
sudo systemctl restart hr-assistant
```

---

## 📁 File Permissions Reference

```
.env                    600 (read/write owner only)
*.sh scripts            755 (executable)
uploads/                755 (rwxr-xr-x)
HR_docs/                755 (rwxr-xr-x)
combined_db.db          644 (rw-r--r--)
Application files       644 (rw-r--r--)
```

---

## 🔗 Useful Commands

```bash
# Check server resources
htop

# Check disk space
df -h

# Check application process
ps aux | grep python

# Check network connections
sudo netstat -tlnp | grep :5000

# Follow application logs
tail -f hr_assistant.log

# Test application endpoint
curl http://localhost:5000/health  # if health endpoint exists
```

---

## 📞 Support

For issues or questions:
1. Check the full deployment guide: `LINUX_DEPLOYMENT_GUIDE.md`
2. Review service logs: `sudo journalctl -u hr-assistant -n 100`
3. Check system resources: `htop` and `df -h`
4. Verify API keys are correct in `.env` file

---

## 📝 Notes

- All scripts assume you're running as a non-root user with sudo privileges
- Scripts are tested on Ubuntu 20.04/22.04 and Debian 11/12
- Default application port is 5000 (can be changed in `run_production.py`)
- Service runs as your user account (not root)
- Logs are stored in systemd journal and `hr_assistant.log`

---

**Last Updated**: October 27, 2025  
**Compatible OS**: Ubuntu 20.04+, Debian 11+

