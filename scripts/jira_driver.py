#! /usr/bin/env python3

"""
This script is designed to be used from various GitHub workflows that are
set up to sync information from GitHub issues on this repo to Veridise's
Jira instance.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import requests
import requests.auth

# Environment (set by GitHub Workflow)

# - Jira
JIRA_BASE_URL = os.environ["JIRA_BASE_URL"]
JIRA_USER_EMAIL = os.environ["JIRA_USER_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]

# - GitHub
GITHUB_API_URL = os.environ["GITHUB_API_URL"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY_OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]


class GitHubCli:
    """A wrapper for GitHub API calls."""

    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.repo_base_url = (
            f"{GITHUB_API_URL}/repos/{GITHUB_REPOSITORY_OWNER}/{GITHUB_REPOSITORY}"
        )

    def get_issue(self, issue: str | int) -> dict:
        """Get all issue data in JSON form.
        Reference: https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#get-an-issue
        """
        resp = requests.get(
            f"{self.repo_base_url}/issues/{issue}", headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()

    def get_issues(self, labels: List[str] = []) -> List[dict]:
        """Get all issues, optionally restricted to issues with a given label.
        Reference: https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#list-repository-issues
        """
        if labels:
            params = {"labels": ",".join(labels)}
        else:
            params = None
        resp = requests.get(
            f"{self.repo_base_url}/issues", headers=self.headers, params=params
        )
        resp.raise_for_status()
        return resp.json()

    def get_labels(self, issue: str | int) -> List[str]:
        """Query and return all label names attached to the given issue.
        Reference: https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#list-labels-for-an-issue
        """
        resp = requests.get(
            f"{self.repo_base_url}/issues/{issue}/labels", headers=self.headers
        )
        resp.raise_for_status()
        return [label["name"] for label in resp.json()]

    def add_label(self, issue: str, label: str):
        """Add an existing label to the given issue.
        Reference: https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#add-labels-to-an-issue
        """
        resp = requests.post(
            f"{self.repo_base_url}/issues/{issue}/labels",
            headers=self.headers,
            data=json.dumps({"labels": [label]}),
        )
        resp.raise_for_status()

    def remove_label(self, issue: str | int, label: str):
        """Remove the specified label from an issue.
        Reference: https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#remove-a-label-from-an-issue
        """
        resp = requests.delete(
            f"{self.repo_base_url}/issues/{issue}/labels/{label}", headers=self.headers
        )
        resp.raise_for_status()

    def create_label(self, label_name: str, description: str):
        """Create a new label for the repository.
        Reference: https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#create-a-label
        """
        resp = requests.post(
            f"{self.repo_base_url}/labels",
            headers=self.headers,
            data=json.dumps(
                {
                    "name": label_name,
                    "description": description,
                    "color": "0362fc",
                }
            ),
        )
        resp.raise_for_status()

    def delete_label(self, label_name: str):
        """Delete a label from the repository.
        Reference: https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#delete-a-label
        """
        resp = requests.delete(
            f"{self.repo_base_url}/labels/{label_name}", headers=self.headers
        )
        resp.raise_for_status()


class JiraCli:
    """A wrapper for Jira API calls."""

    def __init__(self):
        self.jira_auth = requests.auth.HTTPBasicAuth(JIRA_USER_EMAIL, JIRA_API_TOKEN)
        self.jira_headers = {"Content-Type": "application/json"}
        self.project_id = self._get_project_id()
        self.issue_type = self._get_issue_type()
        self.user_id = self._get_user_id()

    def _get_project_id(self) -> str:
        """Get the Jira Project ID for the environment-specified project key.
        Required for creating new issues.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-projects/#api-rest-api-3-project-projectidorkey-get
        """
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/project/{JIRA_PROJECT_KEY}",
            auth=self.jira_auth,
            headers=self.jira_headers,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _get_issue_type(self) -> str:
        """Get the ID of the issue type for Jira issues synced with GitHub.
        Required for creating new issues.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-types/#api-rest-api-3-issuetype-project-get
        """
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/issuetype/project",
            auth=self.jira_auth,
            headers=self.jira_headers,
            params={"projectId": self.project_id},
        )
        resp.raise_for_status()
        issue_types = resp.json()
        github_issue_type = [i for i in issue_types if i["name"] == "GitHub Issue"]
        if len(github_issue_type) == 0:
            raise Exception(
                f"Could not find issue type 'GitHub Issue' in Jira Project '{JIRA_PROJECT_KEY}'"
            )
        if len(github_issue_type) > 1:
            raise Exception(
                f"Found multiple issues matching 'GitHub Issue' in Jira Project '{JIRA_PROJECT_KEY}'"
            )
        return github_issue_type[0]["id"]

    def _get_user_id(self) -> str:
        """Get the user ID from the environment-set Jira email.
        Used to assign new issues to a user for tracking.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-myself/#api-rest-api-3-myself-get
        """
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/myself",
            auth=self.jira_auth,
            headers=self.jira_headers,
        )
        resp.raise_for_status()
        return resp.json()["accountId"]

    def _get_issue_response(self, issue_key: str) -> requests.Response:
        """Get all issue information.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-issueidorkey-get
        """
        return requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}",
            auth=self.jira_auth,
            headers=self.jira_headers,
        )

    def get_issue(self, issue_key: str) -> dict:
        """Get the issue from a given issue key."""
        resp = self._get_issue_response(issue_key)
        resp.raise_for_status()
        return resp.json()

    def try_get_issue(self, issue_key: str) -> Optional[dict]:
        """Check if an issue exists, returning it if it does,
        returning None otherwise."""
        resp = self._get_issue_response(issue_key)
        return resp.json() if resp.status_code == requests.codes.ok else None

    def get_existing_issues(self, issue_keys: List[str]) -> List[dict]:
        """For each key in issue_keys, attempt to fetch an issue. If no such
        issue exists, ignore the issue key."""
        return [
            issue
            for key in issue_keys
            if (issue := self.try_get_issue(key)) is not None
        ]

    def create_issue(self, github_issue: dict) -> str:
        """Create a new Jira issue and link it to a GitHub issue.
        Returns the issue key of the newly created issue.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-post
        """
        description_obj = github_issue["body"]
        if github_issue["body"] is not None:
            description_obj = {"text": github_issue["body"], "type": "text"}
        else:
            description_obj = {
                "text": "No description added.",
                "type": "text",
                "marks": [{"type": "strong"}],
            }
        link_objs = [
            {"text": "\n\nCreated from ", "type": "text", "marks": [{"type": "em"}]},
            {
                "text": github_issue["html_url"],
                "type": "text",
                "marks": [
                    {"type": "em"},
                    {"type": "link", "attrs": {"href": github_issue["html_url"]}},
                ],
            },
            {"text": ".", "type": "text", "marks": [{"type": "em"}]},
        ]
        title_text = f"[GitHub Issue] {github_issue['title']}"
        payload = json.dumps(
            {
                "fields": {
                    "project": {"id": self.project_id},
                    "issuetype": {"id": self.issue_type},
                    "assignee": {"id": self.user_id},
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [description_obj, *link_objs],
                            }
                        ],
                    },
                    "summary": title_text,
                }
            }
        )
        resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/3/issue",
            auth=self.jira_auth,
            headers=self.jira_headers,
            data=payload,
        )
        resp.raise_for_status()
        return resp.json()["key"]

    def get_issue_transitions(self, issue_key: str) -> Dict[str, dict]:
        """Get the list of supported transitions for the given issue.
        Returns transitions as a mapping of transition name -> transition object.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-issueidorkey-transitions-get
        """
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
            headers=self.jira_headers,
            auth=self.jira_auth,
        )
        resp.raise_for_status()
        return {t["name"]: t for t in resp.json()["transitions"]}

    def transition_issue(self, issue_key: str, transition_id: str):
        """Change issue status using the given transition.
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-issueidorkey-transitions-post
        """
        payload = json.dumps(
            {
                "transition": {"id": transition_id},
            }
        )
        resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
            auth=self.jira_auth,
            headers=self.jira_headers,
            data=payload,
        )
        resp.raise_for_status()

    def post_comment(self, issue_key: str, text: str, github_issue_url: str):
        """Post a comment on a given issue, from a given GitHub Issue
        (denoted by the URL).
        Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-comments/#api-rest-api-3-issue-issueidorkey-comment-post
        """
        payload = json.dumps(
            {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"text": f"{text}\n\n", "type": "text"},
                                {
                                    "text": f"Generated from {github_issue_url}.",
                                    "type": "text",
                                    "marks": [
                                        {"type": "em"},
                                        {
                                            "type": "link",
                                            "attrs": {"href": github_issue_url},
                                        },
                                    ],
                                },
                            ],
                        }
                    ],
                }
            }
        )
        resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment",
            auth=self.jira_auth,
            headers=self.jira_headers,
            data=payload,
        )
        resp.raise_for_status()


# Commands


def link_issue(args: argparse.Namespace):
    """Link GitHub issue to Jira."""
    jira = JiraCli()
    github = GitHubCli()

    issue_num = args.issue

    # Get the GitHub issue.
    github_issue = github.get_issue(issue_num)
    # Check if the issue already has a Jira ticket label.
    existing_jira_issues = jira.get_existing_issues(github.get_labels(issue_num))
    if existing_jira_issues:
        # Post a comment on each issue stating it is now tracking this github issue
        for jira_issue in existing_jira_issues:
            jira.post_comment(
                jira_issue["key"],
                f"Now tracking GitHub issue #{github_issue['number']}",
                github_issue["html_url"],
            )
    else:
        # Create a new Jira ticket
        jira_issue_key = jira.create_issue(github_issue)
        # Create a GitHub label that corresponds to the ticket
        github.create_label(
            jira_issue_key,
            f"Synced with Jira Issue {JIRA_BASE_URL}/browse/{jira_issue_key}",
        )
        # Add a label to the github issue that corrsponds to the Jira issue key
        github.add_label(issue_num, jira_issue_key)


def unlink_issue(args: argparse.Namespace):
    """Unlink GitHub issue from Jira."""
    jira = JiraCli()
    github = GitHubCli()

    issue_num: int = args.issue

    # Get the GitHub issue.
    github_issue = github.get_issue(issue_num)
    # Find Jira ticket labels
    existing_jira_issues = jira.get_existing_issues(github.get_labels(issue_num))
    # For each Jira ticket label:
    for jira_issue in existing_jira_issues:
        jira_issue_key = jira_issue["key"]
        # Post a comment stating it is no longer tracking this github issue
        jira.post_comment(
            jira_issue_key,
            f"No longer tracking GitHub issue #{github_issue['number']}.",
            github_issue["html_url"],
        )
        # Remove the label from the GitHub issue
        github.remove_label(issue_num, jira_issue_key)
        # Check to see if any other issues are using this label. If not, delete it to avoid clutter.
        if not github.get_issues([jira_issue_key]):
            github.delete_label(jira_issue_key)


def update_issue(args: argparse.Namespace):
    """Update linked Jira issue based on GitHub issue updates.
    This posts a comment on the Jira issue with the given update summary."""

    jira = JiraCli()
    github = GitHubCli()

    issue_num: int = args.issue
    update_summary: str = args.summary

    # Get the GitHub issue.
    github_issue = github.get_issue(issue_num)
    # Find Jira ticket labels
    existing_jira_issues = jira.get_existing_issues(github.get_labels(issue_num))
    # For each Jira ticket label:
    for jira_issue in existing_jira_issues:
        # Post a comment with the update
        jira.post_comment(
            jira_issue["key"],
            f"Issue update from GitHub:\n{update_summary}",
            github_issue["html_url"],
        )


def transition_issue(args: argparse.Namespace):
    """Transitions the state of a linked Jira issue as the GitHub issue is updated."""

    jira = JiraCli()
    github = GitHubCli()

    issue_num: int = args.issue
    transition_action: str = args.action

    # Get the GitHub issue.
    github_issue = github.get_issue(issue_num)
    # Find Jira ticket labels
    existing_jira_issues = jira.get_existing_issues(github.get_labels(issue_num))
    # For each Jira ticket label:
    for jira_issue in existing_jira_issues:
        jira_issue_key = jira_issue["key"]
        # Post a comment with the update
        jira.post_comment(
            jira_issue_key,
            f'Issue transitioned to "{transition_action}" on GitHub.',
            github_issue["html_url"],
        )
        # Transition the issue, if there is an equivalent on Jira
        jira_transitions = jira.get_issue_transitions(jira_issue_key)
        # Note: the "Closed (1)" is because it is the "closing transition" to the "Closed" state,
        # so it avoids a name collision
        if transition_action == "closed" and "Closed (1)" in jira_transitions:
            jira.transition_issue(jira_issue_key, jira_transitions["Closed (1)"]["id"])


# Script args
argparser = argparse.ArgumentParser()
argparser.add_argument("--issue", type=int, help="GitHub issue number")
subparsers = argparser.add_subparsers(help="subcommand help")

link_cmd = subparsers.add_parser("link")
link_cmd.set_defaults(cmd=link_issue)

unlink_cmd = subparsers.add_parser("unlink")
unlink_cmd.set_defaults(cmd=unlink_issue)

update_cmd = subparsers.add_parser("update")
update_cmd.set_defaults(cmd=update_issue)
update_cmd.add_argument(
    "--summary", type=str, help="A summary of the update being performed"
)

transition_cmd = subparsers.add_parser("transition")
transition_cmd.set_defaults(cmd=transition_issue)
transition_cmd.add_argument(
    "--action", type=str, help="The type of transition being performed"
)


def main():
    args = argparser.parse_args()
    args.cmd(args)


if __name__ == "__main__":
    main()
