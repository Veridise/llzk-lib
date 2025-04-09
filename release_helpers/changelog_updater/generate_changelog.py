#!/usr/bin/env python

# Edit  https://github.com/jdipierro/changelog-generator to:
# 1. work with python 3.9 or higher
# 2. add '--move' option of changelog files to release folder, when changes are saved to changelog.md file
# Modified version by Michalis Konstantopoulos

#  MIT License
#
#  Copyright (c) 2018 Justin DiPierro
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

__doc__ = """
Yaml Changelog Generator

Environment Variables:
  CHANGELOG_INPUT            Folder of yaml changelogs. [default: ./changelogs/unreleased]
  CHANGELOG_MARKDOWN         Path to the CHANGELOG.md file to prepend the new section to, when saving. [default: ./CHANGELOG.md]
  CHANGELOG_RELEASES         Path to the CHANGELOG.md file to prepend the new section to, when saving. [default: ./changelogs/releases]   

Usage:
  generate_changelog <version_num> [<yamls_folder> <markdown_file>] [options]

Options:
  -h --help                 Show this screen.
  version_num               The version number being written.
  --codename=<codename>     Optional codename to display in the header for this version.
  --save                    Writes the changelog, also prints to stdoutd.
  --ignore-empty            Do not exit with error if no changres are present
  --cleanup                 Delete yaml files when finished.
  --move                    Move yaml files to release folder when finished.
"""


import yaml
from glob import iglob
from jinja2 import Template
from docopt import docopt
from datetime import date
from os import remove, rename, mkdir
import os

CHANGELOG_SECTIONS = [
    "added",
    "changed",
    "fixed",
    "deprecated",
    "removed",
    "security",
    "upgraded",
]
CHANGELOG_TEMPLATE = Template(
    """
{%- macro section(possible_changes, title) %}
{%- if possible_changes is defined and possible_changes|length > 0 %}
### {{ title }}
{{ possible_changes }}
{%- endif %}{%- endmacro %}

## {{ version_num }}{% if version_codename is defined %} "{{ version_codename }}"{%- endif %} - {{ release_date }}
{{- section(added, "Added") }}
{{- section(changed, "Changed") }}
{{- section(deprecated, "Deprecated") }}
{{- section(fixed, "Fixed") }}
{{- section(removed, "Removed") }}
{{- section(security, "Security") }}
{{- section(upgraded, "Upgraded") }}
"""
)


class Changeset:
    def __init__(self, input_dir, version_num, version_codename, release_dir):
        self.input_dir = input_dir
        self.version_num = version_num
        self.version_codename = version_codename
        self.release_dir = os.path.join(release_dir, version_num)
        print("Creating changelog for release:", version_num)
        print(
            "When --move option is provided, unreleased files will be moved to:",
            self.release_dir,
        )
        self.input_files = []
        self._rendered = None

        # Sections
        self.added = []
        self.changed = []
        self.fixed = []
        self.deprecated = []
        self.removed = []
        self.security = []
        self.upgraded = []

    def generate(self):
        """Process each yaml file in the input directory"""
        glob_pattern = "{}/*.y*ml".format(self.input_dir)
        for clog_path in iglob(glob_pattern):
            self.add_file(clog_path)

    def add_file(self, filepath):
        self.input_files.append(filepath)
        with open(filepath) as target_file:
            contents = yaml.load(target_file, Loader=yaml.FullLoader)
        if contents:
            self.add(contents)

    def add(self, clog_dict):
        """Add a set of changes"""
        for clog_section in clog_dict.keys():
            section = str.lower(clog_section)
            if section not in CHANGELOG_SECTIONS:
                raise Exception(
                    "Don't know what to do with changelog section '{}'. Valid sections are: {}".format(
                        clog_section, CHANGELOG_SECTIONS
                    )
                )
            section_changes = getattr(self, section)
            section_changes += self._parse_section(clog_dict.get(clog_section))
            setattr(self, section, section_changes)

    @staticmethod
    def _parse_section(clog_section_yaml):
        """Transform the section for display if necessary"""
        if clog_section_yaml is None:
            # Nothing provided for this section
            return []
        if isinstance(clog_section_yaml, list):
            # Print it as it is
            return clog_section_yaml
        if isinstance(clog_section_yaml, str):
            # Print the sting as list with a single element
            return [clog_section_yaml]
        if not isinstance(clog_section_yaml, dict):
            raise Exception("Error in this section: \n{}".format(clog_section_yaml))

        # Convert a subsections into a list of dicts so each line displays as "subsection: change"
        section_changes = []

        for subsection, changes in clog_section_yaml.items():
            if isinstance(changes, list):
                section_changes.extend([{subsection: change} for change in changes])
            else:
                section_changes.append({subsection: changes})
        return section_changes

    def _sort(self):
        for section in CHANGELOG_SECTIONS:
            section_changes = getattr(self, section)

            setattr(
                self,
                section,
                sorted(
                    section_changes,
                    key=lambda ele: (
                        (0, list(ele.keys())[0]) if isinstance(ele, dict) else (1, ele)
                    ),
                ),
            )

            # setattr(self, section, sorted(section_changes))

    def render(self, ignore_empty: bool):
        if self._rendered is not None:
            return self._rendered
        yamlfmt = lambda lst: yaml.dump(lst, default_flow_style=False, width=1024)
        self._sort()

        # Build the context for the jinja template
        jinja_args = dict(
            release_date=date.today().isoformat(),
            version_num=self.version_num,
        )
        if self.version_codename:
            jinja_args["version_codename"] = self.version_codename

        have_changes = False
        for section in CHANGELOG_SECTIONS:
            section_attr = getattr(self, section)
            if len(section_attr) > 0:
                have_changes = True
                jinja_args[section] = yamlfmt(section_attr)

        if not ignore_empty and not have_changes:
            print(
                "No changes found. "
                + "Please ensure properly formatted yaml files are present in: {}".format(
                    self.input_dir
                )
            )
            exit(1)

        # Render the jinja template
        self._rendered = CHANGELOG_TEMPLATE.render(**jinja_args).strip()
        return self._rendered

    def save(self, markdown_file, ignore_empty: bool):
        # Load old changelog data
        with open(markdown_file, "r") as original_changelog:
            old_changelog = original_changelog.read()

        # Write new data to the top of the file
        with open(markdown_file, "w") as master_changelog:
            master_changelog.write(
                "{}\n\n{}".format(self.render(ignore_empty), old_changelog)
            )

    def cleanup(self):
        """Delete the files at the specified paths"""
        for input_file in self.input_files:
            remove(input_file)

    def copy(self):
        """Add all files under the releases folder and the specified version"""
        mkdir(self.release_dir)
        for input_file in self.input_files:
            new_path = os.path.join(self.release_dir, os.path.basename(input_file))
            rename(input_file, new_path)
            print(f"{input_file} ---->  {new_path}")


def main():
    cli_args = docopt(__doc__)

    # this is the project path within a gitlab runner
    default_project_location = os.environ.get("CI_PROJECT_DIR", "")
    input_dir = os.environ.get(
        "CHANGELOG_INPUT",
        os.path.join(default_project_location, "changelogs/unreleased"),
    )
    release_dir = os.environ.get(
        "CHANGELOG_RELEASES",
        os.path.join(default_project_location, "changelogs/releases"),
    )
    markdown_file = os.environ.get(
        "CHANGELOG_MARKDOWN", os.path.join(default_project_location, "CHANGELOG.md")
    )
    version = cli_args["<version_num>"]
    codename = cli_args["--codename"]
    ignore_empty = cli_args["--ignore-empty"]

    # Generate new changelog section
    changes = Changeset(input_dir, version, codename, release_dir)
    changes.generate()
    print("{}\n\n".format(changes.render(ignore_empty)))

    if cli_args["--save"]:
        changes.save(markdown_file, ignore_empty)
        print(f"{markdown_file} updated")
        if cli_args["--move"]:
            changes.copy()
        if cli_args["--cleanup"]:
            changes.cleanup()
            print("Yaml files deleted")


if __name__ == "__main__":
    main()
