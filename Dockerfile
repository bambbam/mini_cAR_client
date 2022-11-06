FROM python:3.9

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

RUN pip3 install poetry
COPY . /code
WORKDIR /code
RUN poetry install --no-root
CMD ["poetry", "run", "client"]
