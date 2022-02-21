FROM python:3.8

WORKDIR /WORKDIR

COPY requirements.txt .
RUN mkdir fire_dataset
COPY fire_dataset ./fire_dataset
COPY train.py .
COPY predict.jpg .
COPY nofire.png .

RUN pip install -r requirements.txt

CMD ["python","train.py"]