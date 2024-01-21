FROM python:3.9.16-slim-buster

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

ENV PROJECT_DIR /usr/local/src/webapp

WORKDIR ${PROJECT_DIR}

COPY Pipfile Pipfile.lock . ${PROJECT_DIR}/

RUN pipenv install --system --deploy

EXPOSE 8080
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]