FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

RUN apt-get update -y && apt-get upgrade -y

# install uv
RUN apt-get install -y --no-install-recommends python3 python3-pip
RUN pip install --no-cache-dir uv

# install project
WORKDIR /app/
COPY .python-version .python-version
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN uv venv

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY external/mujoco_menagerie/unitree_go2 external/mujoco_menagerie/unitree_go2
COPY src src
COPY train.py train.py
COPY conf conf

RUN uv build

ENTRYPOINT [ "/bin/bash" ]