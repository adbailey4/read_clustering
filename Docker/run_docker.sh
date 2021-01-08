cp Docker/Dockerfile .
docker build --progress=plain --target build -t read_clustering .  || true
rm Dockerfile