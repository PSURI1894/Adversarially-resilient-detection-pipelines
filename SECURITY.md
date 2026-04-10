# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | Yes |
| older tags | No — please update to `main` |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately via one of the following:

1. **GitHub Security Advisories** — use the "Report a vulnerability" button on the
   [Security tab](../../security/advisories/new) of this repository.
2. **Email** — send details to the repository owner. You can find contact information
   in the GitHub profile linked to this repository.

### What to include

- A clear description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept code if applicable)
- The version / commit SHA where you observed it
- Any suggested mitigations

### Response timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | 48 hours |
| Initial assessment | 5 business days |
| Fix / advisory | Coordinated with reporter |

We follow responsible disclosure: vulnerabilities are disclosed publicly only after
a fix has been released or a reasonable remediation period has passed.

## Scope

This project is a research pipeline for adversarial ML / IDS. Security issues
of particular interest include:

- Dependency vulnerabilities in `requirements*.txt` that affect production deployments
- Authentication / authorisation bypasses in the FastAPI server (`src/api/server.py`)
- Docker / infrastructure misconfigurations in `infrastructure/aws/`
- Data-poisoning or model-integrity issues that could subvert the detection pipeline
  in a deployed setting
