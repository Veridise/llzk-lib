#!/usr/bin/env bash

VENV_DIR=./.venv

echo "Setting up virtualenv in $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR"/bin/pip install -U pip poetry
else
    echo "virtualenv already exists, nothing to do."
fi

echo "Installing requirements.."
source "$VENV_DIR"/bin/activate && pip install -r $(dirname "$0")/requirements.txt
