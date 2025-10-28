# Face Attendance Server (FastAPI + PostgreSQL + pgvector)

## Quickstart (Docker)

```bash
cd face-attendance-server
docker-compose up -d --build
# wait ~10-20s for db/app to be ready
curl http://localhost:8000/health
```

## API

OpenAPI docs: http://localhost:8000/docs

### Create employee
```bash
curl -X POST http://localhost:8000/v1/employees \
  -H "Content-Type: application/json" \
  -d '{"emp_id":"E0001","full_name":"Nguyen Van A","dept":"R&D"}'
```

### Enroll embeddings
```bash
curl -X POST http://localhost:8000/v1/enroll/E0001 \
  -H "Content-Type: application/json" \
  -d '{"model":"mobilefacenet","embeddings":[[0.01, -0.02, 0.03, ... 128 floats ...]]}'
```

### Recognize
```bash
curl -X POST http://localhost:8000/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "device_id":"pi-entrance-1",
    "ts":"2025-10-21T02:00:00Z",
    "embedding":[0.01, -0.02, 0.03, ...],
    "liveness":0.82,
    "quality":0.74,
    "options":{"save_event_face":true,"mode":"auto"}
  }'
```

### List attendance
```bash
curl "http://localhost:8000/v1/attendance?from_ts=&to_ts=&emp_id="
```

## Notes
- Uses `ankane/pgvector` image with `CREATE EXTENSION vector;`
- Default threshold: `0.60`, top_k: `5` (env-configurable)
- Attendance images are not stored in this skeleton (add multipart endpoint later)
