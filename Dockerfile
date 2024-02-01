FROM python:3.9.16-slim-buster

# Install pipenv and compilation dependencies
RUN pip install pipenv
#RUN apt-get update && apt-get install -y --no-install-recommends gcc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV REPO_URL=pipenv_docker

WORKDIR $REPO_URL

COPY . .
RUN pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
