PYTHONBIN = python3.7
VENV      = pyvenv

export VIRTUAL_ENV := $(abspath ${VENV})
export PATH := ${VIRTUAL_ENV}/bin:${PATH}

${VENV}:
	${PYTHONBIN} -m venv $@

python-reqs: requirements.txt | ${VENV}
	pip install --upgrade -r requirements.txt

setup: ${VENV} python-reqs
	jupyter contrib nbextension install --user
	jupyter nbextension enable autosavetime/main

start-jupyter:
	jupyter notebook --config jupyter_notebook_config.py

CLEANUP = *.pyc

clean:
	rm -rf ${CLEANUP}
