# hash:sha256:34b4701acc1898ea4e40b3bae58696d9d2ea827803c3363b407e48e6c5326e58
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && apt-get purge -y --autoremove software-properties-common \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        dssp=3.0.0-2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=https://levinthal:paradox@conda.graylab.jhu.edu --channel=conda-forge --channel=pytorch --channel=dglteam \
        biopython==1.75 \
        dgl==0.4.3 \
        numpy==1.19.4 \
        pyrosetta==2020.50+release.1295438 \
        pytorch==1.2.0 \
    && conda clean -ya
