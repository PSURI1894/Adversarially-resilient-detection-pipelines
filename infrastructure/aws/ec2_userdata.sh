#!/usr/bin/env bash
# EC2 user-data: runs once on first boot to install Docker + Docker Compose
set -euo pipefail

dnf update -y
dnf install -y docker git rsync

systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Docker Compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Swap file (1 GB) to help t2.micro survive model loading
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
