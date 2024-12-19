# Use the official Python base image
FROM python:3.10

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
WORKDIR /workspace
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip and install essential Python tools
RUN pip install --upgrade pip wheel setuptools

# Install PyRadiomics dependencies
RUN pip install numpy scipy SimpleITK

# Install PyRadiomics
RUN pip install pyradiomics

# Verify installation
RUN python -c "import radiomics; print('PyRadiomics version:', radiomics.__version__)"

# Default command to start the container
CMD ["/bin/bash"]
