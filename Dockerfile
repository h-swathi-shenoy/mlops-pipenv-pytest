FROM python:3.9.16-slim-buster as base
RUN mkdir -p /app
WORKDIR /app
COPY . /app
RUN pip install pipenv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

FROM base AS runtime

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

RUN pipenv install --system --deploy --ignore pipfile
EXPOSE 8080
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]