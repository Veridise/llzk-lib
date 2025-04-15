#!/usr/bin/env bash

set -e

SEMANTIC_VERSION=$1
RELEASE_CHANGELOG="${RELEASE_CHANGELOG:-"changelogs/PENDING.md"}"

echo "Validate version format.."
if [[ $SEMANTIC_VERSION =~ v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Valid release version."
else
    echo "Invalid release version '${SEMANTIC_VERSION}'. Select a release_version that matches semantic versioning format."
    exit 1
fi


if [ ! -f ${RELEASE_CHANGELOG} ];
then
    echo -n 0 > .pre_release.counter
    echo "Creating ${RELEASE_CHANGELOG} with pre-release changes.."

    CHANGELOG_APP=$(dirname "$0")/../changelog_updater/generate_changelog.py
    touch ${RELEASE_CHANGELOG}
    # The date of the release will be updated when executing the prependReleaseChanges.sh
    CHANGELOG_MARKDOWN=${RELEASE_CHANGELOG} python3 "${CHANGELOG_APP}" ${SEMANTIC_VERSION} --save --cleanup
fi
