FROM inseefrlab/onyxia-vscode-python:py3.10.4

RUN git clone https://github.com/ThomasFaria/DT-RN-chapitre3.git && \
    cd DT-RN-chapitre3 && \
    pip install -r requirements.txt && \
    chown -R ${USERNAME}:${GROUPNAME} ${HOME}