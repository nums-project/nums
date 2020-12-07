import subprocess
import os
import getpass

from nums.core.version import __version__


def runproc(*args):
    subproc_env = os.environ.copy()
    return subprocess.Popen(args,
                            env=subproc_env,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def communicate(*args):
    p = runproc(*args)
    return tuple(map(lambda x: x.decode("utf-8"), p.communicate()))


def execute():

    out, err = communicate("conda", "env", "list")
    if err:
        raise Exception(err)
    for row in out.split("\n"):
        if "*" in row:
            env_dir = row.split("*")[-1].strip()
            print("Running subproc in conda environment", env_dir)
            break

    out, err = communicate("python", "-m", "pip", "install", "--upgrade", "pip")
    if err:
        raise Exception(err)
    print("out", out)

    out, err = communicate("pip", "install", "setuptools", "wheel", "twine")
    if err:
        raise Exception(err)
    print("out", out)

    # Build it.
    out, err = communicate("python", "setup.py", "sdist", "bdist_wheel")
    if err:
        lines = err.split("\n")
        print(lines)
        for line in lines:
            if not ("warn" in line or "UserWarning" in line or line == ""):
                raise Exception(err)
    print(out)

    r = input("Release to test.pypi (y/n)? ")
    release_cmd = ["twine", "upload"]
    username = input("Username? ")
    password = getpass.getpass(prompt='Password? ', stream=None)

    release_cmd += ["--username", username, "--password", password]
    assert r in ("y", "n")
    if r != "y":
        release_cmd += ["dist/*"]
    else:
        release_cmd += ["-r", "testpypi", "dist/*"]

    out, err = communicate(*release_cmd)
    print("out", out)
    print("err", err)


if __name__ == "__main__":
    execute()
