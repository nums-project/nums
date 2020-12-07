import subprocess
import os
import getpass
import shutil


__version__ = None


with open('nums/core/version.py') as f:
    # pylint: disable=exec-used
    exec(f.read(), globals())


pj = lambda *paths: os.path.abspath(os.path.expanduser(os.path.join(*paths)))


def project_root():
    return os.path.abspath(os.path.dirname(__file__))


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
    s = " upgrade pip"
    print("-"*(50 - len(s)) + s)
    print("out", out)
    print("-"*50)

    out, err = communicate("pip", "install", "setuptools", "wheel", "twine")
    if err:
        raise Exception(err)
    s = " install release deps"
    print("-"*(50 - len(s)) + s)
    print("out", out)
    print("-"*50)

    # Remove old build.
    build_dir = pj(project_root(), "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    dist_dir = pj(project_root(), "dist")
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)

    # Build it.
    out, err = communicate("python", "setup.py", "sdist", "bdist_wheel")
    if err:
        lines = err.split("\n")
        print(lines)
        for line in lines:
            if not ("warn" in line or "UserWarning" in line or line == ""):
                raise Exception(err)
    s = " build %s" % __version__
    print("-"*(50 - len(s)) + s)
    print(out)
    print("-"*50)

    repo_name = input("Release %s to pypi or test.pypi (pypi/test.pypi)? " % __version__)
    assert repo_name in ("pypi", "test.pypi")

    release_cmd = ["twine", "upload"]
    username = input("Username? ")
    password = getpass.getpass(prompt='Password? ', stream=None)

    release_cmd += ["--username", username, "--password", password]
    if repo_name == "pypi":
        release_cmd += ["dist/*"]
    elif repo_name == "test.pypi":
        release_cmd += ["-r", "testpypi", "dist/*"]
    else:
        raise Exception("Unknown repository %s" % repo_name)

    out, err = communicate(*release_cmd)
    s = " release %s" % __version__
    print("-"*(50 - len(s)) + s)
    print("out", out)
    print("err", err)


if __name__ == "__main__":
    execute()
