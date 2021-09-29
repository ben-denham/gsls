# We will run the dev_image for local development.
# You may want to choose a different base image from:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM jupyter/tensorflow-notebook:4a112c0f11eb as dev_image


# Uncomment to install any dependencies (install these before
# performing steps with files that may change frequently so that
# layers may be cached). Best practice for installing apt dependencies
# without saving apt-cache:
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

USER root
RUN apt-get update && apt-get install -y \
        # libfuse2 and xvfb are dependencies of orca, which is a
        # dependency of plotly image saving.
        libfuse2 \
        fuse \
        xvfb \
        && rm -rf /var/lib/apt/lists/*
USER jovyan

COPY bin/orca-1.2.1-x86_64.AppImage /usr/local/bin/orca

# A section like the following can be useful when your base image runs
# as the root user (not necessary for Jupyter Docker images). This
# will create a user with the same uid and gid as our host user,
# meaning files created in the image after this step and during
# container will be owned by our user on the host.

ARG GROUP_ID=1000
ARG USER_ID=1000
# RUN groupadd --gid $GROUP_ID coder
# RUN useradd --uid $USER_ID --gid coder --shell /bin/bash --create-home coder
# USER coder


# Pip packages will be installed in a directory we can mount from the
# host as a volume.
ENV PYTHONUSERBASE=/home/jovyan/work/.pip-packages
ENV PATH=$PATH:/home/jovyan/work/.pip-packages/bin

RUN touch /home/jovyan/CREATE_FILES_IN_WORK_DIRECTORY__FILES_CREATED_HERE_WILL_BE_LOST
COPY ./jupyter_notebook_config.py /home/jovyan/.jupyter/jupyter_notebook_config.py
