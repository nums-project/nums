import subprocess

from nums.core.version import __version__


def runproc(*args):
    print(" ".join(args))
    return subprocess.Popen(args,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def communicate(*args):
    p = runproc(*args)
    return tuple(map(lambda x: x.decode("utf-8"), p.communicate()))


def execute():
    input_handler = input

    out, err = communicate("git", "version")
    if err:
        raise Exception(err)

    out, err = communicate("git", "tag")
    versions = list(map(lambda x: x.strip("\r"), out.strip("\n\r").split("\n")))

    print("")
    print("tagged versions:")
    for version in versions:
        print(version)
    print("")

    # Prefix versions with "v"
    v = "v" + __version__
    if v in versions:
        r = input_handler("%s already tagged, force update (y/n)?" % v)
        if r != "y":
            return
        out, err = communicate("git", "tag", v, "-f")
        print(out)
        print(err)
        out, err = communicate("git", "push", "--tags", "-f")
        print(out)
        print(err)
    else:
        r = input_handler("tag %s (y/n)?" % v)
        if r != "y":
            return
        out, err = communicate("git", "tag", v)
        print(out)
        print(err)
        out, err = communicate("git", "push", "--tags")
        print(out)
        print(err)


if __name__ == "__main__":
    execute()
