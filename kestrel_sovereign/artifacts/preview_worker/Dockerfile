# Immutable multi-architecture Python base; apt metadata is frozen as well.
FROM python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf '%s\n' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261026T000000Z trixie main' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20261026T000000Z trixie-security main' > /etc/apt/sources.list && \
    apt-get update -qq && \
    apt-get install --no-install-recommends -y poppler-utils=25.03.0-5+deb13u4 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /renderer/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /renderer/requirements.txt
COPY worker.py /renderer/worker.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/nonexistent
USER 65532:65532
ENTRYPOINT ["python", "-I", "/renderer/worker.py"]
