# Use a Python base image
FROM python:3.10-slim

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc-11 \
    g++-11 \
    git \
    make \
    tar \
    libboost-all-dev \
    libc-dev \
    curl \
    ninja-build \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Set CXX and CC environment variables for the compilers
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Set working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install \
    llama-cpp-python
    # --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# Copy the application code
COPY . /app

# Expose Flask port
EXPOSE 5000

# Command to run the app
CMD ["python", "main.py"]
