docker build --platform=linux/amd64 \
    --build-arg HTTP_PROXY=http://10.10.43.220:7890 \
    --build-arg HTTPS_PROXY=http://10.10.43.220:7890 \
    -t open-reasoner-zero .

docker build --platform=linux/amd64 \
    -t open-reasoner-zero .


