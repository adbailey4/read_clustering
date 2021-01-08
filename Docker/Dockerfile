# syntax = docker/dockerfile:1.0-experimental
FROM ubuntu:18.04 AS build
MAINTAINER Andrew Bailey, andbaile@ucsc.edu

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.7 python3.7-venv && \
    apt-get clean && \
    apt-get purge

ENV VIRTUAL_ENV=/opt/venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.7 -m pip -q install wheel

COPY . /home/signalAlign
RUN mkdir /home/signalAlign/build && \
    cd /home/signalAlign/sonLib && \
    make clean && \
    cd /home/signalAlign/externalTools && \
    make clean && \
    cd /home/signalAlign/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=. -DCMAKE_VERBOSE_MAKEFILE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=RELEASE && \
    make -j 4 && \
    cd /home/signalAlign && \
    python3.7 -m pip install . && \
    mkdir /root/signalAlign && mkdir /root/signalAlign/build && \
    cp -r /home/signalAlign/build /root/signalAlign && \
    mkdir /root/signalAlign/bin && \
    cp -r /home/signalAlign/bin /root/signalAlign && \
    cp -r /home/signalAlign/Docker/run_wrapper.sh /root/signalAlign/run_wrapper.sh


FROM ubuntu:18.04 as runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.7 python3.7-venv libgomp1 libcurl4-openssl-dev samtools hdf5-tools && \
    apt-get clean && \
    apt-get purge


COPY --from=build /root/signalAlign/ /root/signalAlign/
COPY --from=build /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/root/signalAlign/bin:$PATH"

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/root/signalAlign/run_wrapper.sh"]

FROM runtime as test
RUN apt-get -qq install -y --no-install-recommends python3-distutils
RUN python3.7 -m pip -q install pytest
COPY ./tests /root/signalAlign/tests
COPY ./models /root/signalAlign/models
COPY ./pytest.ini /root/signalAlign/pytest.ini

COPY ./src/signalalign/tests /root/signalAlign/src/signalalign/tests

RUN cd /root/signalAlign && python3.7 -m pytest
WORKDIR /root/signalAlign
