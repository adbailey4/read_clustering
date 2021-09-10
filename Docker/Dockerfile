# syntax = docker/dockerfile:1.0-experimental
FROM ubuntu:18.04 AS build
MAINTAINER Andrew Bailey, andbaile@ucsc.edu

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.7-dev python3-pip python3.7-venv && \
    apt-get clean && \
    apt-get purge

ENV VIRTUAL_ENV=/opt/venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.7 -m pip -q install wheel setuptools cython
RUN apt-get install -y --no-install-recommends build-essential

COPY . /home/read_clustering
RUN cd /home/read_clustering && \
    python3.7 -m pip install . && \
    mkdir /root/read_clustering && \
    cp /home/read_clustering/Docker/run_wrapper.sh /root/read_clustering/run_wrapper.sh


FROM ubuntu:18.04 as runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.7 python3.7-venv && \
    apt-get clean && \
    apt-get purge

COPY --from=build /root/read_clustering/ /root/read_clustering/
COPY --from=build /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/root/read_clustering/run_wrapper.sh"]

FROM runtime as test
#RUN apt-get -qq install -y --no-install-recommends python3-distutils
#RUN python3.7 -m pip -q install pytest
COPY ./tests /root/read_clustering/tests
#COPY ./pytest.ini /root/read_clustering/pytest.ini
ENV VIRTUAL_ENV=/opt/venv
RUN cd /root/read_clustering && python3.7 tests/variant_call_tests.py
WORKDIR /root/read_clustering
