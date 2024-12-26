# Use the official Python base image
FROM python:3.10

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /callfromgithub
RUN chmod 755 /callfromgithub
COPY downloadcodefromgithub.sh /callfromgithub/

# Install system dependencies
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
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Create and activate a virtual environment
WORKDIR /workspace
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip and install essential Python tools
RUN pip install --upgrade pip wheel setuptools

# Install required Python packages
RUN pip install numpy scipy SimpleITK pandas nibabel requests xmltodict argparse

# Optional: If you need `pydicom`, uncomment the next line
# RUN pip install pydicom

# Install PyRadiomics
RUN pip install pyradiomics

# Verify installation
RUN python -c "import radiomics; print('PyRadiomics version:', radiomics.__version__)"

# Install other required Python libraries
RUN pip install \
    pathlib \
    glob2 \
    h5py \
    PyGithub \
    scikit-image \
    opencv-python \
    python-dateutil \
    mysql-connector-python==8.0.27


# Default command to start the container
CMD ["/bin/bash"]
