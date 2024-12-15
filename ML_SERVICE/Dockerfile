FROM ultralytics/ultralytics:latest
WORKDIR /ultralytics/ultralytics
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "mlapp.py"]