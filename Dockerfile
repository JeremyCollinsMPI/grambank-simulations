FROM tensorflow/tensorflow:1.15.0-py3
RUN pip install python-nexus geocoder scipy
RUN pip install --upgrade pandas
RUN mkdir /src
WORKDIR /src