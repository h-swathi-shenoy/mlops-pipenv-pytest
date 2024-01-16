ensure-pip:
	pip install --upgrade pip
	pip install pipenv 
	pip --version
	pipenv --version

setup-env: mk-venv
	pipenv install -r requirements.txt

mk-venv:
	rm -rf .venv
	mkdir -p .venv

lint:
	pylint --disable=R,C --exit-zero *.py src

# test:
# 	python -m pytest -vvv --cov=src test_*.py

format:
	black *.py src/*.py

all: ensure-pip setup-env lint format #test format