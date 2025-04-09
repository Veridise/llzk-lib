#!/usr/bin/env bash

set -e

PROJECT_PATH="${PROJECT_PATH:-"."}"
CHANGELOG_INPUT="${CHANGELOG_INPUT:-"changelogs/unreleased"}"
YAML_TEMPLATE=$(dirname "$0")/template.yaml

cd ${PROJECT_PATH}

branch=$(git symbolic-ref --short -q HEAD | sed -r 's/\//__/g')
echo Creating changelog file for branch $branch

NEW_FILE="${CHANGELOG_INPUT}/${branch}.yaml"
if [ -f ${NEW_FILE} ];
then
    echo Changelog file "${NEW_FILE}" already exists..
    exit 0
fi

mkdir -p "${CHANGELOG_INPUT}"
if [[ $1 == "--empty" ]]; then
    touch "${NEW_FILE}"
    echo Empty file ${NEW_FILE} created.
else
    cp "${YAML_TEMPLATE}" "${NEW_FILE}"
    echo File ${NEW_FILE} created. Please edit according to your current work.
fi
