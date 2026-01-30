# Deployment Guide (Python 3.10)

## Prerequisites
- Ubuntu/Debian server with sudo access
- Python 3.10, git, and (optionally) nginx

Install basics:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev git nginx
```

## Get the Code (folder: peoplebotv2)
```bash
sudo mkdir -p /opt/peoplebotv2 && sudo chown $USER:$USER /opt/peoplebotv2
git clone https://github.com/riteshnaik77/peoplebotV2.git /opt/peoplebotv2
cd /opt/peoplebotv2
```

## Virtual Env (Python 3.10) and Dependencies
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # or requirements_for_server.txt if applicable
```

## Environment Variables
Create `/opt/peoplebotv2/.env` with your keys:
```
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.5-flash
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=high
FLASK_ENV=production
FLASK_APP=app.py
```

## Smoke Test (Manual)
```bash
source venv/bin/activate
python app.py   # or: flask run --host 0.0.0.0 --port 8000
# then visit http://SERVER_IP:8000
```

## Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## systemd Service (auto-start on reboot)
```bash
sudo tee /etc/systemd/system/peoplebotv2.service <<'EOF'
[Unit]
Description=PeopleBot V2 Flask App
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/peoplebotv2
Environment="PATH=/opt/peoplebotv2/venv/bin"
EnvironmentFile=/opt/peoplebotv2/.env
ExecStart=/opt/peoplebotv2/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 app:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable peoplebotv2
sudo systemctl start peoplebotv2
sudo systemctl status peoplebotv2
```

## Nginx Reverse Proxy (optional but recommended)
```bash
sudo tee /etc/nginx/sites-available/peoplebotv2.conf <<'EOF'
server {
    listen 80;
    server_name your.domain.com;  # or _

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/peoplebotv2.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## SSL (if you have a domain)
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.com
```

## Logs & Troubleshooting
- App logs: `journalctl -u peoplebotv2 -f`
- Nginx logs: `/var/log/nginx/error.log`
- Restart app: `sudo systemctl restart peoplebotv2`

## Updating the App
```bash
cd /opt/peoplebotv2
git pull
source venv/bin/activate && pip install -r requirements.txt
sudo systemctl restart peoplebotv2
```



