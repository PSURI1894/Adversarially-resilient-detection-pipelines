#!/usr/bin/env bash
# ==============================================================================
# ARDP — AWS Free-Tier Deployment Script
# ==============================================================================
# Provisions EC2 t2.micro, deploys the Dockerized backend, uploads the
# React dashboard to S3 static hosting, and registers the Lambda health check.
#
# Prerequisites:
#   - AWS CLI v2 configured: aws configure
#   - An existing EC2 key pair (set KEY_NAME below)
#   - Docker installed locally (for image build/push)
#   - Node.js ≥ 18 installed locally (for dashboard build)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh [--region us-east-1] [--key mykey] [--destroy] [--force]
#
#   --force   Terminate the existing EC2 instance and deploy fresh (use when
#             the instance is in a broken state)
# ==============================================================================

set -euo pipefail

# ── Auto-fix Windows CRLF line endings (Git Bash on Windows) ──
# Uses grep -cP which is available in Git Bash; 'file' is not reliable on Windows
if grep -qP '\r' "$0" 2>/dev/null; then
  sed -i 's/\r//g' "$0"
  exec bash "$0" "$@"
fi

# ── Configurable defaults ──────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-ardp-key}"
INSTANCE_TYPE="t2.micro"
AMI_ID="ami-0c02fb55956c7d316"          # Amazon Linux 2023 (us-east-1)
S3_BUCKET="ardp-dashboard-$(date +%s)"  # unique bucket name
DASHBOARD_DIR="../../dashboard"
API_PORT=8000
SG_NAME="ardp-sg"
TAG_NAME="ardp-backend"
LAMBDA_NAME="ardp-health-check"

# ── Colours ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse args ────────────────────────────────────────────────
DESTROY=false
FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)   REGION="$2";   shift 2 ;;
    --key)      KEY_NAME="$2"; shift 2 ;;
    --destroy)  DESTROY=true;  shift ;;
    --force)    FORCE=true;    shift ;;   # terminate old instance and redeploy fresh
    *)          error "Unknown argument: $1" ;;
  esac
done

export AWS_DEFAULT_REGION="$REGION"

# ── Destroy mode ──────────────────────────────────────────────
if $DESTROY; then
  warn "Destroying all ARDP infrastructure in $REGION..."
  INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running,stopped" \
    --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || echo "None")
  [[ "$INSTANCE_ID" != "None" ]] && aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" && info "Terminated $INSTANCE_ID"

  aws s3 rb "s3://$S3_BUCKET" --force 2>/dev/null && info "Deleted S3 bucket" || true
  aws lambda delete-function --function-name "$LAMBDA_NAME" 2>/dev/null && info "Deleted Lambda" || true
  info "Destroy complete."
  exit 0
fi

# ── Force mode: terminate existing instance so we start clean ─
if $FORCE; then
  warn "--force: terminating any existing ARDP instance..."
  OLD_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running,stopped" \
    --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || echo "None")
  if [[ "$OLD_ID" != "None" && -n "$OLD_ID" ]]; then
    aws ec2 terminate-instances --instance-ids "$OLD_ID" > /dev/null
    aws ec2 wait instance-terminated --instance-ids "$OLD_ID"
    info "Terminated $OLD_ID — starting fresh"
  fi
fi

# ── Cleanup trap: if THIS run created a new instance but fails ─
NEWLY_CREATED_INSTANCE=""
cleanup_on_error() {
  local exit_code=$?
  if [[ $exit_code -ne 0 && -n "$NEWLY_CREATED_INSTANCE" ]]; then
    warn "Deploy failed (exit $exit_code). Terminating instance $NEWLY_CREATED_INSTANCE to avoid charges..."
    aws ec2 terminate-instances --instance-ids "$NEWLY_CREATED_INSTANCE" > /dev/null 2>&1 || true
    warn "Instance terminated. Fix the error and re-run ./deploy.sh"
  fi
}
trap cleanup_on_error EXIT

# ==============================================================================
# STEP 1 — Security Group
# ==============================================================================
info "Step 1/7: Creating security group..."

VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "ARDP backend security group" \
    --vpc-id "$VPC_ID" \
    --query "GroupId" --output text)
  info "Created security group: $SG_ID"

  # Ingress rules
  for PORT in 22 80 443 $API_PORT; do
    aws ec2 authorize-security-group-ingress \
      --group-id "$SG_ID" \
      --protocol tcp --port "$PORT" --cidr "0.0.0.0/0" 2>/dev/null || true
  done
  info "Opened ports 22, 80, 443, $API_PORT"
else
  info "Reusing existing security group: $SG_ID"
fi

# ==============================================================================
# STEP 2 — EC2 Instance
# ==============================================================================
info "Step 2/7: Launching EC2 t2.micro..."

INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || echo "None")

if [[ "$INSTANCE_ID" == "None" || -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":20,\"VolumeType\":\"gp2\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
    --user-data file://ec2_userdata.sh \
    --query "Instances[0].InstanceId" --output text)
  NEWLY_CREATED_INSTANCE="$INSTANCE_ID"   # trap will clean up if deploy fails
  info "Launched instance: $INSTANCE_ID"

  info "Waiting for instance to reach running state..."
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
else
  info "Reusing running instance: $INSTANCE_ID"
fi

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
info "Public IP: $PUBLIC_IP"

# ==============================================================================
# STEP 3 — Deploy application to EC2
# ==============================================================================
info "Step 3/7: Deploying ARDP backend to EC2..."

# Wait for SSH to become available
for i in $(seq 1 20); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    -i "${KEY_NAME}.pem" "ec2-user@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  warn "SSH not ready yet (attempt $i/20), retrying in 10s..."
  sleep 10
done

# Copy source code via scp (create tar locally, upload, then extract)
info "Packaging source code..."
tar --exclude='./.git' --exclude='./data/raw' --exclude='./__pycache__' \
  --exclude='./*.pyc' --exclude='./.venv' --exclude='./venv' \
  --exclude='./tf_env' --exclude='./node_modules' \
  --exclude='./dashboard/node_modules' \
  --exclude='./infrastructure/aws/ardp-key.pem' \
  --exclude='./*.db' --exclude='./mlflow.db' \
  -czf /tmp/ardp-deploy.tar.gz -C ../../ .

info "Uploading to EC2..."
scp -o StrictHostKeyChecking=no -i "${KEY_NAME}.pem" \
  /tmp/ardp-deploy.tar.gz "ec2-user@$PUBLIC_IP:/home/ec2-user/ardp-deploy.tar.gz"

info "Extracting on EC2..."
ssh -o StrictHostKeyChecking=no -i "${KEY_NAME}.pem" "ec2-user@$PUBLIC_IP" \
  "mkdir -p /home/ec2-user/ardp && tar -xzf /home/ec2-user/ardp-deploy.tar.gz -C /home/ec2-user/ardp/ && rm /home/ec2-user/ardp-deploy.tar.gz"

rm -f /tmp/ardp-deploy.tar.gz

# Start the application via docker compose
ssh -o StrictHostKeyChecking=no -i "${KEY_NAME}.pem" "ec2-user@$PUBLIC_IP" << 'REMOTE'
  set -e
  cd /home/ec2-user/ardp

  # Check disk space (need at least 5 GB free)
  FREE_KB=$(df / | awk 'NR==2 {print $4}')
  if [[ "$FREE_KB" -lt 5242880 ]]; then
    echo "[ERROR] Less than 5 GB free on EC2 (${FREE_KB} KB). Cleaning Docker cache..."
    sudo docker system prune -af --volumes 2>/dev/null || true
    FREE_KB=$(df / | awk 'NR==2 {print $4}')
    if [[ "$FREE_KB" -lt 3145728 ]]; then
      echo "[ERROR] Still less than 3 GB free after prune. Aborting."
      exit 1
    fi
  fi
  echo "[INFO] Disk free: $(df -h / | awk 'NR==2 {print $4}')"

  # Wait for user-data to finish installing docker (up to 5 min)
  echo "Waiting for Docker to become available..."
  for i in $(seq 1 30); do
    command -v docker &>/dev/null && sudo docker info &>/dev/null && break
    echo "  Docker not ready yet (attempt $i/30), waiting 10s..."
    sleep 10
  done

  # If user-data failed or AMI doesn't have dnf, install manually
  if ! command -v docker &>/dev/null; then
    echo "Docker not installed by user-data, installing manually..."
    if command -v dnf &>/dev/null; then
      sudo dnf install -y docker
    else
      sudo yum install -y docker
    fi
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
  fi

  # Install docker compose plugin if missing
  if ! sudo docker compose version &>/dev/null 2>&1; then
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  fi

  # Use ONLY the production compose — do NOT use override (it merges with
  # the base docker-compose.yml which includes Kafka/Zookeeper/Grafana).
  # --project-directory ensures build context '.' resolves to the project root,
  # not relative to the compose file's own directory.
  sudo docker compose \
    --project-directory /home/ec2-user/ardp \
    -f /home/ec2-user/ardp/infrastructure/aws/docker-compose.prod.yml \
    up -d --build

  echo "Backend running. API: http://$(curl -s ifconfig.me):8000"
REMOTE

info "Backend deployed at http://$PUBLIC_IP:$API_PORT"

# ==============================================================================
# STEP 4 — Build & Deploy dashboard to S3
# ==============================================================================
info "Step 4/7: Building React dashboard..."

pushd "$DASHBOARD_DIR" > /dev/null
# Patch API URL to point to EC2
sed -i.bak "s|http://localhost:8000|http://$PUBLIC_IP:$API_PORT|g" src/config.ts 2>/dev/null \
  || sed -i.bak "s|http://localhost:8000|http://$PUBLIC_IP:$API_PORT|g" src/config.js 2>/dev/null \
  || true
npm ci --silent
npm run build
popd > /dev/null

info "Step 5/7: Uploading dashboard to S3..."

# Create bucket (ignore error if already exists)
aws s3 mb "s3://$S3_BUCKET" 2>/dev/null || true

# Remove public access blocks so static hosting works
aws s3api put-public-access-block \
  --bucket "$S3_BUCKET" \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false" 2>/dev/null || true

aws s3api put-bucket-policy --bucket "$S3_BUCKET" --policy "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"PublicRead\",
    \"Effect\": \"Allow\",
    \"Principal\": \"*\",
    \"Action\": \"s3:GetObject\",
    \"Resource\": \"arn:aws:s3:::$S3_BUCKET/*\"
  }]
}"

aws s3 sync "$DASHBOARD_DIR/dist/" "s3://$S3_BUCKET" --delete
aws s3 website "s3://$S3_BUCKET" --index-document index.html --error-document index.html

DASHBOARD_URL="http://$S3_BUCKET.s3-website-$REGION.amazonaws.com"
info "Dashboard live: $DASHBOARD_URL"

# ==============================================================================
# STEP 5 — Lambda health check
# ==============================================================================
info "Step 6/7: Deploying Lambda health check..."

# Package lambda (use python zipfile — zip not available on Windows Git Bash)
LAMBDA_DIR="$(dirname "$0")"
cd "$LAMBDA_DIR"
python3 -c "
import zipfile, os
with zipfile.ZipFile('lambda_health_check.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('lambda_health_check.py', 'lambda_health_check.py')
" 2>/dev/null || python -c "
import zipfile
with zipfile.ZipFile('lambda_health_check.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('lambda_health_check.py', 'lambda_health_check.py')
"

# Create IAM role for Lambda (if not exists)
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name ardp-lambda-role \
  --query "Role.Arn" --output text 2>/dev/null || echo "")

if [[ -z "$LAMBDA_ROLE_ARN" ]]; then
  LAMBDA_ROLE_ARN=$(aws iam create-role \
    --role-name ardp-lambda-role \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' \
    --query "Role.Arn" --output text)
  aws iam attach-role-policy \
    --role-name ardp-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10   # IAM propagation delay
fi

# Create or update Lambda
if aws lambda get-function --function-name "$LAMBDA_NAME" &>/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --zip-file fileb://lambda_health_check.zip > /dev/null
  info "Lambda updated: $LAMBDA_NAME"
else
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" \
    --runtime python3.11 \
    --role "$LAMBDA_ROLE_ARN" \
    --handler lambda_health_check.handler \
    --zip-file fileb://lambda_health_check.zip \
    --timeout 15 \
    --memory-size 128 \
    --environment "Variables={ARDP_API_URL=http://$PUBLIC_IP:$API_PORT}" > /dev/null
  info "Lambda created: $LAMBDA_NAME"
fi

# Schedule every 5 minutes via EventBridge
RULE_ARN=$(aws events put-rule \
  --name ardp-health-schedule \
  --schedule-expression "rate(5 minutes)" \
  --state ENABLED \
  --query "RuleArn" --output text)

LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_NAME" \
  --query "Configuration.FunctionArn" --output text)

aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id ardp-schedule-permission \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN" 2>/dev/null || true

aws events put-targets \
  --rule ardp-health-schedule \
  --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\"}]" > /dev/null

info "Health check scheduled every 5 minutes"
rm -f lambda_health_check.zip

# ==============================================================================
# STEP 6 — Summary
# ==============================================================================
trap - EXIT   # clear cleanup trap — deploy succeeded, keep the instance
info "Step 7/7: Deployment complete!"
echo ""
echo "============================================================"
echo "  ARDP Deployment Summary"
echo "============================================================"
echo "  EC2 Instance ID : $INSTANCE_ID"
echo "  Public IP       : $PUBLIC_IP"
echo "  API             : http://$PUBLIC_IP:$API_PORT"
echo "  API Docs        : http://$PUBLIC_IP:$API_PORT/docs"
echo "  Dashboard (S3)  : $DASHBOARD_URL"
echo "  Lambda HC       : $LAMBDA_NAME (every 5 min)"
echo "  Region          : $REGION"
echo "============================================================"
echo ""
echo "  SSH access:"
echo "  ssh -i ${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "  To destroy all resources:"
echo "  ./deploy.sh --destroy"
echo "============================================================"
