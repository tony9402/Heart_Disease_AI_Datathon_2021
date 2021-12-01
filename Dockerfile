FROM nvcr.io/nvidia/pytorch:21.10-py3

WORKDIR /workspace
COPY ./requirements.txt requirements.txt

RUN apt update -y
RUN apt install libgl1-mesa-glx -y

RUN pip install -r requirements.txt
