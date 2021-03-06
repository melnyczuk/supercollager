FROM python:3.8.5-slim
RUN pip install pipenv
ADD Pipfile Pipfile.lock ./
RUN pipenv install
RUN mkdir ./dump
COPY src ./src
EXPOSE 8080:8080
CMD waitress-serve src.server:server
