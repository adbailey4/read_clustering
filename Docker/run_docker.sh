cp Docker/Dockerfile .
docker build --progress=plain --target runtime -t read_clustering:latest .  || true
rm Dockerfile