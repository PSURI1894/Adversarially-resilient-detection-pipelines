# Deployment Guide

## Local Development

### Prerequisites

| Tool | Version |
|---|---|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| Docker & Docker Compose | ≥ 24 |
| Kafka (via Docker) | 3.x |

### Setup

```bash
# Clone & install
git clone <repo-url>
cd Adversarially-resilient-detection-pipelines

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start infrastructure (Kafka + Zookeeper)
docker compose up kafka zookeeper -d

# Run the backend
uvicorn src.api.server:app --reload --port 8000

# Run the dashboard (new terminal)
cd dashboard
npm install
npm run dev
```

---

## Docker (Full Stack)

`docker-compose.yml` spins up:
- `zookeeper` — Kafka coordination
- `kafka` — Message broker (port 9092)
- `api` — FastAPI backend (port 8000)
- `dashboard` — React + Vite (port 5173)

```bash
docker compose up --build

# To run in background:
docker compose up -d

# View logs:
docker compose logs -f api

# Stop:
docker compose down
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka broker address |
| `KAFKA_TOPIC` | `net.flows` | Flow ingestion topic |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow backend URI |
| `ARDP_API_KEY` | (none) | API auth token for protected endpoints |
| `ARDP_MODE` | `production` | `demo` \| `production` |
| `CONFORMAL_ALPHA` | `0.05` | Target miscoverage rate |
| `RSCP_SIGMA` | `0.1` | Gaussian smoothing σ |
| `DRIFT_CONSENSUS` | `2` | Min detectors agreeing to trigger retrain |

---

## AWS Free-Tier Deployment

All components fit within AWS Free Tier limits.

### Architecture

| Service | AWS Component | Free Tier Limit |
|---|---|---|
| Backend API + ML | EC2 `t2.micro` | 750 hrs/month |
| React Dashboard | S3 static hosting | 5 GB, 15 GB bandwidth |
| Drift health checks | Lambda | 1M requests/month |
| Data storage | S3 | 5 GB |

### Steps

```bash
# 1. Provision EC2 (us-east-1, t2.micro, Amazon Linux 2023)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t2.micro \
  --key-name ardp-key \
  --security-group-ids sg-xxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ardp-backend}]'

# 2. SSH in and deploy
ssh -i ardp-key.pem ec2-user@<public-ip>
sudo yum install -y docker git
sudo systemctl start docker
git clone <repo-url>
cd Adversarially-resilient-detection-pipelines

# Build and run (no Kafka in free-tier mode; use demo mode)
docker build -t ardp-api .
docker run -d -p 8000:8000 \
  -e ARDP_MODE=demo \
  -e ARDP_API_KEY=changeme \
  ardp-api

# 3. Deploy dashboard to S3
cd dashboard
npm run build
aws s3 sync dist/ s3://ardp-dashboard --acl public-read
aws s3 website s3://ardp-dashboard --index-document index.html

# 4. Point dashboard API URL to EC2 public IP
# Edit dashboard/src/config.ts: API_BASE_URL = "http://<ec2-ip>:8000"
```

### Security Group Rules

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22 | TCP | Your IP | SSH |
| 8000 | TCP | 0.0.0.0/0 | API |
| 80 | TCP | 0.0.0.0/0 | HTTP (nginx) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (nginx) |

### nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Production Hardening Checklist

- [ ] Set `ARDP_API_KEY` to a strong random value
- [ ] Enable TLS on nginx (use Let's Encrypt / `certbot`)
- [ ] Restrict security group ports to known IP ranges
- [ ] Set `MLFLOW_TRACKING_URI` to a persistent volume (not ephemeral EC2 storage)
- [ ] Configure Kafka topic replication factor ≥ 2 for production
- [ ] Enable CloudWatch log forwarding: `awslogs` Docker log driver
- [ ] Schedule Lambda health check every 5 minutes (ping `/health`)
- [ ] Set up SNS alert if `/health` returns non-200

---

## Memory & CPU Budget (t2.micro)

| Component | RAM | CPU |
|---|---|---|
| FastAPI + model | ~400 MB | 0.4 vCPU burst |
| MLflow (embedded) | ~80 MB | minimal |
| In-memory feature store | ~50 MB | minimal |
| **Total** | **~530 MB** | well within 1 GB limit |

> The deep ensemble is loaded in inference-only mode (no GPU required).
> Training must be done offline; only weights are deployed.
