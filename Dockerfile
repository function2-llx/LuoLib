# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch

RUN apt-get update
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm environment.yml /opt/algorithm/
RUN conda env create -n umei
RUN conda activate umei

# download pre-trained model for lungmask
RUN python -c 'from lungmask.mask import get_model; get_model("unet", "R231")'

COPY --chown=algorithm:algorithm test_stoic2021.py /opt/algorithm/
# COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
# COPY --chown=algorithm:algorithm algorithm/ /opt/algorithm/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=STOICAlgorithm
