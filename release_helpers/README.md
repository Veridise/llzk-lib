#  Description

Python utility to update CHANGELOG.md from multiple special formatted yaml files

Initial work from https://github.com/jdipierro/changelog-generator
Modified to work with python3.9 and above

# How to use as git submodule

## Create the submodule
```
git submodule add git@github.com:Veridise/release_helpers.git release_helpers
./release_helpers/setup.sh
```

## Create changelogEntry file
```
./release_helpers/scripts/createChangelogEntry.sh
```
Now edit changelogs/unreleased/<branch_name>.yaml according to your changes in this branch
> **_NOTE:_**  Always run the script from the root folder of your project.

To create an empty file provide the `--empty` option
```
./release_helpers/scripts/createChangelogEntry.sh --empty
```

## Validate changelogEntry file of current branch
```
./release_helpers/scripts/validateChangelogEntry.sh
```
> **_NOTE:_**  Always run the script from the root folder of your project.

## Create pre-release files
```
./release_helpers/scripts/createPreReleaseFiles.sh v0.0.1
```
> **_NOTE:_**  Always run the script from the root folder of your project.

## Create final CHANGELOG.md and remove pre-release files
```
./release_helpers/scripts/updateChangelog.sh v0.0.1
```

> **_NOTE:_**  Always run the script from the root folder of your project.