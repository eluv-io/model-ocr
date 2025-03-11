FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n ocr python=3.8 -y

SHELL ["conda", "run", "-n", "ocr", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl && \
    conda install -y -c conda-forge ffmpeg-python

RUN mkdir ocr
COPY setup.py .

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/ocr/bin/pip install .

COPY models ./models

COPY ocr ./ocr
COPY config.yml run.py setup.py config.py .

ENTRYPOINT ["/opt/conda/envs/ocr/bin/python", "run.py"]