FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime as base

RUN apt-get update && apt-get -y install git


ENV HOME=/exp/fromage



WORKDIR /exp/fromage
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install -r ./requirements.txt
RUN python -m pip install --upgrade Jinja2
RUN python -m pip install gradio

COPY . .
RUN chmod -R a+rwX .

CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "7860"]
