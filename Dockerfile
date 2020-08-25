# base image
FROM python:3.7.2-slim

# set working directory
WORKDIR /usr/src/tensorsignatures

# add and install minumum requirements
RUN pip install --upgrade pip setuptools wheel
COPY ./requirements-docker.txt /usr/src/tensorsignatures/requirements-docker.txt
RUN pip install -r requirements-docker.txt

# add tensorsignatures and install
COPY . /usr/src/tensorsignatures

# interchange commands for dev mode
RUN cd /usr/src/tensorsignatures && python setup.py install

# run server
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=./mount", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
