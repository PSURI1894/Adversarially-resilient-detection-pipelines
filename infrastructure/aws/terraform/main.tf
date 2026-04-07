# ==============================================================================
# Terraform — ARDP AWS Free-Tier Infrastructure
# ==============================================================================
# Provisions:
#   - EC2 t2.micro (backend API + ML inference)
#   - S3 bucket (React dashboard static hosting)
#   - Lambda function (health check, every 5 min)
#   - EventBridge rule (Lambda schedule)
#   - Security group (ports 22, 80, 443, 8000)
#   - IAM roles for Lambda
#
# Usage:
#   terraform init
#   terraform plan -var="key_name=ardp-key"
#   terraform apply -var="key_name=ardp-key"
#   terraform destroy
# ==============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

# ── Variables ──────────────────────────────────────────────────
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "key_name" {
  description = "EC2 key pair name"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type (free tier eligible)"
  type        = string
  default     = "t2.micro"
}

variable "ami_id" {
  description = "Amazon Linux 2023 AMI (us-east-1)"
  type        = string
  default     = "ami-0c02fb55956c7d316"
}

variable "coverage_min" {
  description = "Minimum conformal coverage threshold for Lambda alert"
  type        = string
  default     = "0.85"
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for health-check alerts (optional)"
  type        = string
  default     = ""
}

# ── Provider ───────────────────────────────────────────────────
provider "aws" {
  region = var.region
}

# ── Data: default VPC ──────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

# ── Security Group ─────────────────────────────────────────────
resource "aws_security_group" "ardp" {
  name        = "ardp-sg"
  description = "ARDP backend security group"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = [22, 80, 443, 8000]
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "ardp-sg", Project = "ARDP" }
}

# ── EC2 Instance ───────────────────────────────────────────────
resource "aws_instance" "backend" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.ardp.id]
  user_data              = file("${path.module}/../ec2_userdata.sh")

  # t2.micro root volume (8 GB — fits within free tier)
  root_block_device {
    volume_size           = 8
    volume_type           = "gp2"
    delete_on_termination = true
  }

  tags = { Name = "ardp-backend", Project = "ARDP" }
}

# ── S3 Dashboard Bucket ────────────────────────────────────────
resource "aws_s3_bucket" "dashboard" {
  bucket        = "ardp-dashboard-${random_id.suffix.hex}"
  force_destroy = true
  tags          = { Project = "ARDP" }
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_public_access_block" "dashboard" {
  bucket                  = aws_s3_bucket.dashboard.id
  block_public_acls       = false
  ignore_public_acls      = false
  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_website_configuration" "dashboard" {
  bucket = aws_s3_bucket.dashboard.id

  index_document { suffix = "index.html" }
  error_document { key    = "index.html" }
}

resource "aws_s3_bucket_policy" "dashboard" {
  bucket = aws_s3_bucket.dashboard.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicRead"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.dashboard.arn}/*"
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.dashboard]
}

# ── IAM Role for Lambda ────────────────────────────────────────
resource "aws_iam_role" "lambda" {
  name = "ardp-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = { Project = "ARDP" }
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_cloudwatch" {
  name = "ardp-lambda-cw"
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = var.sns_topic_arn != "" ? [var.sns_topic_arn] : ["arn:aws:sns:*:*:*"]
      }
    ]
  })
}

# ── Lambda Function ────────────────────────────────────────────
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda_health_check.py"
  output_path = "${path.module}/lambda_health_check.zip"
}

resource "aws_lambda_function" "health_check" {
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  function_name    = "ardp-health-check"
  role             = aws_iam_role.lambda.arn
  handler          = "lambda_health_check.handler"
  runtime          = "python3.11"
  timeout          = 15
  memory_size      = 128

  environment {
    variables = {
      ARDP_API_URL  = "http://${aws_instance.backend.public_ip}:8000"
      COVERAGE_MIN  = var.coverage_min
      SNS_TOPIC_ARN = var.sns_topic_arn
    }
  }

  tags = { Project = "ARDP" }
}

# ── EventBridge Schedule (every 5 minutes) ────────────────────
resource "aws_cloudwatch_event_rule" "health_schedule" {
  name                = "ardp-health-schedule"
  description         = "Trigger ARDP health check every 5 minutes"
  schedule_expression = "rate(5 minutes)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.health_schedule.name
  target_id = "ardp-health-check"
  arn       = aws_lambda_function.health_check.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.health_check.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.health_schedule.arn
}

# ── Outputs ────────────────────────────────────────────────────
output "ec2_public_ip" {
  value       = aws_instance.backend.public_ip
  description = "EC2 public IP for SSH and API access"
}

output "api_url" {
  value       = "http://${aws_instance.backend.public_ip}:8000"
  description = "ARDP FastAPI base URL"
}

output "dashboard_url" {
  value       = "http://${aws_s3_bucket_website_configuration.dashboard.website_endpoint}"
  description = "React dashboard URL (S3 static site)"
}

output "ssh_command" {
  value       = "ssh -i ${var.key_name}.pem ec2-user@${aws_instance.backend.public_ip}"
  description = "SSH command for EC2 access"
}
