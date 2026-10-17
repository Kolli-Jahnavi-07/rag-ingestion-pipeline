PY ?= python
export PYTHONPATH := src

.PHONY: install test lint demo eval

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

demo:
	$(PY) -m ragpipe demo

eval:
	$(PY) -m ragpipe corpus --version 2 --out data/docs
	$(PY) -m ragpipe sync data/docs
	$(PY) -m ragpipe eval --version 2 --min-recall5 0.97
