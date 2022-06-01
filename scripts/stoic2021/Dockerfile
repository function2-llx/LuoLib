# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch

RUN apt update
RUN apt install -y git
RUN conda update conda

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm environment.yml /opt/algorithm/
COPY --chown=algorithm:algorithm third-party /opt/algorithm/third-party
RUN conda env create -n umei
SHELL ["conda", "run", "-n", "umei", "/bin/bash", "-c"]
# download pre-trained model for lungmask
RUN python -c 'from lungmask.mask import get_model; get_model("unet", "R231")'
COPY --chown=algorithm:algorithm submit /opt/algorithm/submit
COPY --chown=algorithm:algorithm test /opt/algorithm/test
RUN pip install itk==5.2.1.post1
COPY --chown=algorithm:algorithm conf /opt/algorithm/conf
COPY --chown=algorithm:algorithm umei /opt/algorithm/umei
COPY --chown=algorithm:algorithm process_stoic2021.py /opt/algorithm/process.py

ENTRYPOINT ["conda", "run", "-n", "umei", "python", "-m", "process", "conf/stoic2021/infer.yml"]
#ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=STOICAlgorithm
