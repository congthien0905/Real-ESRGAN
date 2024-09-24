# Use the base image with CUDA 11.4.2 and cuDNN 8
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git

# Create symlink for python3.8 as python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python3

# Create application directory
WORKDIR /app/src

# Copy source code into the container
COPY ./ /app/src

# Install Python libraries
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install realesrgan

# Run the Python application
CMD [ "python3", "-u", "server.py" ]
