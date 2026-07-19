FROM python:3.10-slim@sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        make \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements-lock.txt

COPY . /workspace/finauth_audit
WORKDIR /workspace

CMD ["python", "-m", "pytest", "-q", "finauth_audit/tests"]
