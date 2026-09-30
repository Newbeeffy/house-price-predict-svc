# Housing Price Prediction Model Service

A FastAPI service that predicts housing prices using a scikit-learn linear
regression model. Containerised and deployed on k3s/k8s.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict price(s) — accepts a single house or a list |
| `/model-info` | GET | Model coefficients (unstandardised scale) + metrics |

### Example: single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"square_footage":1850,"bedrooms":3,"bathrooms":2,"year_built":1998,
       "lot_size":7500,"distance_to_city_center":5.6,"school_rating":8.2}'
# -> {"prediction": 265000.0}
```

### Example: batch prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{...house1...}, {...house2...}]'
# -> {"predictions": [265000.0, 186000.0]}
```

## Project Layout

```
app/            Inference code (packaged into the image)
scripts/        Offline training script (NOT packaged)
models/         Trained artifacts: scaler.joblib, model.joblib, metrics.json
data/           Dataset (NOT packaged)
tests/          API tests
charts/         Helm chart (k3s deployment)
```

Training and serving are separated: `scripts/train.py` is run once to produce
the artifacts under `models/`, and only those artifacts + `app/` are baked into
the serving image.

## Development

```bash
# 1. Install dev deps (includes pandas/pytest)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Train (produces models/*)
python scripts/train.py

# 3. Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Run tests
python -m pytest
```

## Build & Deploy (k3s/k8s)

```bash
# Build the image
docker build -f Dockerfile -t interview/house-price-predict-svc:latest .

# Import into k3s' containerd
docker save interview/house-price-predict-svc:latest -o house-price-predict-svc.tar
sudo k3s ctr images import house-price-predict-svc.tar

# Deploy via Helm
helm install house-price-predict-svc ./charts

# Override values as needed, e.g.
# helm install house-price-predict-svc ./charts --set image.tag=v1.2.3 --set replicaCount=2

# Verify
kubectl get pods
kubectl port-forward svc/house-price-predict-svc 8080:80
curl http://localhost:8080/health
```
