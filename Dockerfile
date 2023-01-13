FROM python:3.10-slim

COPY main.py /usr/src

RUN apt update && apt install git && \
    git clone https://github.com/openvinotoolkit/open_model_zoo.git && \
    pip install open_model_zoo/demos/object_detection_demo/python && \
    pip install open_model_zoo/tools/model_tools && \
    pip install openvino-dev && \
    cp open_model_zoo/demos/common/python/openvino/model_zoo/model_api /usr/src/ && \
    mkdir -p models/alexeyab && cd models/alexeyab && \
    curl -JOk https://objects.githubusercontent.com/github-production-release-asset-2e65be/75388965/9bb2e8b0-ffab-435f-9c49-97e353558735?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230112%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230112T082607Z&X-Amz-Expires=300&X-Amz-Signature=139848c341207f154316a5f1e50321c1764bc51d1d87ed6ee175b1721ca5d459&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=75388965&response-content-disposition=attachment%3B%20filename%3Dyolov4-tiny.weights&response-content-type=application%2Foctet-stream && \
    cd - && \
    curl -JOk omz_downloader --name person-detection-0200 --output_dir models

WORKDIR /usr/src

ENTRYPOINT /usr/bin/python3

CMD main.py