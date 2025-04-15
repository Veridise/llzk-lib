#!/usr/bin/env bash

CHANGELOG_INPUT="${CHANGELOG_INPUT:-"changelogs/unreleased"}"
CHANGELOG_APP=$(dirname "$0")/../changelog_updater/generate_changelog.py
MDX_VALIDATOR_FOLDER=$(dirname "$0")/../mdx-validate
MDX_TEMP_FILENAME="TEST.md"
MDX_TEMP_FILE="${MDX_VALIDATOR_FOLDER}/${MDX_TEMP_FILENAME}"
YAML_TEMPLATE=$(dirname "$0")/template.yaml

BRANCH_NAME=$(git symbolic-ref --short -q HEAD | sed -r 's/\//__/g' )
CHANGELOG_FILE="${CHANGELOG_INPUT}/${BRANCH_NAME}.yaml"

echo "Searching if changelog file ${CHANGELOG_FILE} exists..."
test -f ${CHANGELOG_FILE}
exit_code=$?
if [ $exit_code -eq  1 ]; then
    echo "File ${CHANGELOG_FILE} does not exist."
    echo "Run './release_helpers/scripts/createChangelogEntry.sh' to create it."
    exit 1
fi
echo "Check if ${CHANGELOG_FILE} has been edited..."
diff ${CHANGELOG_FILE} ${YAML_TEMPLATE} > /dev/null
exit_code=$?
if [ $exit_code -eq  0 ]; then
    echo "File has not been edited."
    echo "Describe your work on ${CHANGELOG_FILE}"
    exit 1
fi
echo "Check if unreleased changelog entries are formated properly..."
python3 "${CHANGELOG_APP}" --ignore-empty  test > "${MDX_TEMP_FILE}"
exit_code=$?
if [ $exit_code -eq  1 ]; then
    echo "The format of ${CHANGELOG_FILE} is not correct."
    echo "Check the available options from 'release_helpers/scripts/template.yaml'"
    rm "${MDX_TEMP_FILE}"
    exit 1
fi
echo "Check if generated file is mdx compliant..."
pushd "${MDX_VALIDATOR_FOLDER}"
npx nbb -m validate "${MDX_TEMP_FILENAME}"
exit_code=$?
if [ $exit_code -eq  1 ]; then
    echo "The format of ${CHANGELOG_FILE} is not correct for mdx format. Check the error output above."
    rm "${MDX_TEMP_FILENAME}"
    exit 1
fi
popd
rm "${MDX_TEMP_FILE}"
echo "Changelog entry file validated."
