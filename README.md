# Vision AI

Unified computer vision API for object detection, OCR, face recognition, and image analysis.

## Production Deployment

This repository is wired to the shared LangGraph deployment orchestration agent:

- Pre-production validation
- Docker build validation
- Production hardening checks
- Deployment orchestration
- Post-production validation
- Deployment reporting
- Parallel deployment support
- Time and cost budget enforcement

Deployment flow:

```text
main branch merge
→ deployment agent validation
→ provider deployment
→ live validation
→ deployment report generation
```

Shared deployment agent:

- `AGenNext/code-deploy`

## Where This Agent Deploys

Current deployment mode:

| Field | Value |
|---|---|
| Provider | `webhook` |
| Runtime target | Coolify, Dokploy, Cloud Run trigger, AWS trigger, or any deployment webhook target |
| Deploy trigger | `DEPLOY_WEBHOOK_URL` GitHub secret |
| Production URL | `DEPLOYED_BASE_URL` GitHub secret |
| Compose file | `docker-compose.deploy.yml` |
| Dockerfile | `Dockerfile` |
| Runtime API | `app:app` |
| Health endpoint | `/health` |

To deploy through Coolify:

```text
DEPLOY_WEBHOOK_URL=<coolify deploy webhook>
DEPLOYED_BASE_URL=<public deployed URL>
```

## Required GitHub Secrets

| Secret | Description |
|---|---|
| `DEPLOY_WEBHOOK_URL` | Deployment webhook URL |
| `DEPLOYED_BASE_URL` | Public deployment URL used for validation |

## Runtime

A lightweight FastAPI runtime surface is included for deployment validation.

Run locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
GET /health
```

## Features

- **Object Detection** - Detect and classify objects in images
- **OCR (Optical Character Recognition)** - Extract text from images
- **Face Detection** - Detect and analyze faces
- **Image Classification** - Categorize images
- **Scene Understanding** - Analyze image context

## API Endpoint

```text
POST /api/v1/vision/detect
POST /api/v1/vision/ocr
POST /api/v1/vision/faces
POST /api/v1/vision/classify
```

## Quick Start

```bash
# Using OpenAutonomyX CLI
autonomyx vision detect --image photo.jpg

# Using API
curl -X POST https://api.openautonomyx.com/api/v1/vision/detect \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg"
```

## Production Files

| File | Purpose |
|---|---|
| `Dockerfile` | Production container runtime |
| `docker-compose.deploy.yml` | Production deployment compose |
| `.github/workflows/deploy.yml` | Shared deployment-agent workflow |
| `requirements.txt` | Python runtime dependencies |
| `app.py` | FastAPI deployment runtime |

## Documentation

- [API Reference](docs/api.md)
- [Models Guide](docs/models.md)
- [Integration Examples](docs/examples.md)

---

**Repository:** [openautonomyx/vision-ai](https://github.com/openautonomyx/vision-ai)  
**License:** MIT
