docker build -t wine_predictor:quality -f Dockerfile.app .
docker run -dp 0.0.0.0:8000:8001 wine_predictor:quality
