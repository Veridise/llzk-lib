#!/usr/bin/env bash

set -e

RELEASE_MARKDOWN="${RELEASE_MARKDOWN:-"changelogs/PENDING.md"}"

if [ -f ${RELEASE_MARKDOWN} ];
then
    echo "Removing pre-release temp files.."
    rm ${RELEASE_MARKDOWN}
    rm .pre_release.counter
fi
