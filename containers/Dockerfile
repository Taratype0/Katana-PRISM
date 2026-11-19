FROM python:3.12-slim

LABEL org.opencontainers.image.title="Katana-PRISM" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Standalone & nf-core downstream runtime for SpliceManifold, SpliceNet, SpliceTrace, scSpliceNet, scSpliceTrace, and Katana Studio"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
        samtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/katana-prism
COPY packages/ /opt/katana-prism/packages/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numba>=0.59 \
        umap-learn>=0.5.5 \
        /opt/katana-prism/packages/splicemanifold \
        /opt/katana-prism/packages/splicenet \
        /opt/katana-prism/packages/splicetrace \
        /opt/katana-prism/packages/scsplicenet \
        /opt/katana-prism/packages/scsplicetrace \
        /opt/katana-prism/packages/katana_studio

WORKDIR /workspace
