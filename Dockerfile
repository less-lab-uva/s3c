# Base Image
FROM continuumio/miniconda3

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl  -y

# Copy project folder
RUN mkdir /s3c
COPY ./requirements.txt /s3c/requirements.txt
WORKDIR /s3c
ENV S3C_BASE=/s3c

# Create sg environment
RUN conda create -n sg python=3.9 -y

# Activate the 'sg' environment and install requirements
RUN echo "conda activate sg" >> ~/.bashrc
RUN conda run -n sg pip install -r requirements.txt

# Set the default command to activate the 'sg' environment
CMD ["conda", "run", "-n", "sg", "/bin/bash"]