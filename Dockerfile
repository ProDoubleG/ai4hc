FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
MAINTAINER junsuk

COPY members.txt ./
RUN cat members.txt >> /etc/passwd


RUN pip install numpy matplotlib pytorch-ignite monai==0.8.1 nibabel tqdm
# RUN python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"
# RUN python -c "import matplotlib" || pip install -q matplotlib
RUN pip install ipykernel


WORKDIR /home
