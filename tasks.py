"""Command-line tools to facilitate the development of NumS."""
from invoke import task

from nums.core.version import __version__


@task
def tag(c):
    """Tag the current version of NumS and push the tag upstream."""
    result = c.run("git tag", hide=True)
    versions = result.stdout.splitlines()
    current_version = "v" + __version__
    if current_version in versions:
        if not accepts(f"{current_version} is already tagged. Force update?"):
            return
        c.run(f"git tag {current_version} -f")
        c.run("git push --tags -f")
    else:
        if not accepts(f"Tag {current_version} and push upstream?"):
            return
        c.run(f"git tag {current_version}")
        c.run("git push --tags")


def accepts(message: str) -> bool:
    """Ask the user to respond 'y' or 'n' to the specified prompt.

    If the user supplies an invalid response (i.e., neither 'y' or 'n'), then
    the user is re-asked the question.

    Args:
        message: The question to ask the user.

    Returns:
        True if the user responds 'y' and false if the user responds 'n'.
    """
    response = None
    while response not in {"y", "n"}:
        print(f"{message} (y/n)", end=" ")
        response = input()

    assert response in {"y", "n"}
    return response == "y"
