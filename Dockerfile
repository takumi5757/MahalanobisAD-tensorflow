FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update \
        && apt-get install -y \
        mesa-utils \
        cmake \
        build-essential \
        python3 \
        python3-pip
        
RUN mkdir -p $HOME/.keras/models
RUN cd $HOME/.keras/models \
        && curl -O https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5

RUN mkdir -p /opt/python_env
WORKDIR /opt/python_env
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install jupyter

RUN pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable highlight_selected_word/main &&\
    jupyter nbextension enable hinterland/hinterland && \
    jupyter nbextension enable toc2/main
    
RUN pip install hydra-core --upgrade
RUN pip install jedi==0.17.2

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
