#!/usr/bin/env bash

set -e

CHANGELOG_MARKDOWN="${CHANGELOG_MARKDOWN:-"CHANGELOG.md"}"
RELEASE_MARKDOWN="${RELEASE_MARKDOWN:-"changelogs/PENDING.md"}"

# Replace heading with current date
RELEASE_HEADING=$(echo $(cat $RELEASE_MARKDOWN | head -1 | cut -d "-" -f 1) - $(date '+%Y-%m-%d'))
echo $RELEASE_HEADING > temp.md
cat $RELEASE_MARKDOWN | tail -n +2 >> temp.md
mv temp.md $RELEASE_MARKDOWN

# Add RELEASE_MARKDOWN at the top of CHANGELOG_MARKDOWN
cat $RELEASE_MARKDOWN $CHANGELOG_MARKDOWN > temp.md
mv temp.md $CHANGELOG_MARKDOWN
