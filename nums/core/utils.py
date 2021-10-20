# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import inspect

# From project Dask: https://github.com/dask/dask/blob/main/dask/utils.py


# TODO (bcp): decorator maybe causing issues with parsing: https://stackoverflow.com/questions/51248994/python-3-6-inspect-signature-doesnt-show-arguments
def get_named_args(func):
    """Get all non ``*args/**kwargs`` arguments for a function"""
    s = inspect.signature(func)
    return [
        n
        for n, p in s.parameters.items()
        if p.kind in [p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY]
    ]


def _skip_doctest(line):
    # NumPy docstring contains cursor and comment only example
    stripped = line.strip()
    if stripped == ">>>" or stripped.startswith(">>> #"):
        return line
    elif ">>>" in stripped and "+SKIP" not in stripped:
        if "# doctest:" in line:
            return line + ", +SKIP"
        else:
            return line + "  # doctest: +SKIP"
    else:
        return line


def skip_doctest(doc):
    if doc is None:
        return ""
    return "\n".join([_skip_doctest(line) for line in doc.split("\n")])


def extra_titles(doc):
    lines = doc.split("\n")
    titles = {
        i: lines[i].strip()
        for i in range(len(lines) - 1)
        if lines[i + 1].strip() and all(c == "-" for c in lines[i + 1].strip())
    }

    seen = set()
    for i, title in sorted(titles.items()):
        if title in seen:
            new_title = "Extra " + title
            lines[i] = lines[i].replace(title, new_title)
            lines[i + 1] = lines[i + 1].replace("-" * len(title), "-" * len(new_title))
        else:
            seen.add(title)

    return "\n".join(lines)


def ignore_warning(doc, cls, name, extra="", skipblocks=0):
    """Expand docstring by adding disclaimer and extra text"""
    import inspect

    if inspect.isclass(cls):
        l1 = "This docstring was copied from %s.%s.%s.\n\n" % (
            cls.__module__,
            cls.__name__,
            name,
        )
    else:
        l1 = "This docstring was copied from %s.%s.\n\n" % (cls.__name__, name)
    l2 = "Some inconsistencies with the NumS version may exist."

    i = doc.find("\n\n")
    if i != -1:
        # Insert our warning
        head = doc[: i + 2]
        tail = doc[i + 2 :]
        while skipblocks > 0:
            i = tail.find("\n\n")
            head = tail[: i + 2]
            tail = tail[i + 2 :]
            skipblocks -= 1
        # Indentation of next line
        indent = re.match(r"\s*", tail).group(0)
        # Insert the warning, indented, with a blank line before and after
        if extra:
            more = [indent, extra.rstrip("\n") + "\n\n"]
        else:
            more = []
        bits = [head, indent, l1, indent, l2, "\n\n"] + more + [tail]
        doc = "".join(bits)

    i = doc.find("Examples\n")
    if i != -1:
        docstring_warning = "The doctests shown below are copied from NumPy. They won’t show the correct result until you operate ``.get()``."
        head = doc[: i + 22]
        tail = doc[i + 22 :]
        bits = [head, indent, docstring_warning, "\n\n"] + [tail]
        doc = "".join(bits)

    return doc


def unsupported_arguments(doc, args):
    """Mark unsupported arguments with a disclaimer"""
    lines = doc.split("\n")
    for arg in args:
        subset = [
            (i, line)
            for i, line in enumerate(lines)
            if re.match(r"^\s*" + arg + " ?:", line)
        ]
        if len(subset) == 1:
            [(i, line)] = subset
            lines[i] = line + "  (Not supported in NumS)"
    return "\n".join(lines)


def _derived_from(cls, method, ua_args=[], extra="", skipblocks=0, doctest=True):
    """Helper function for derived_from to ease testing"""
    # do not use wraps here, as it hides keyword arguments displayed
    # in the doc
    original_method = getattr(cls, method.__name__)

    if isinstance(original_method, property):
        # some things like SeriesGroupBy.unique are generated.
        original_method = original_method.fget

    doc = original_method.__doc__

    if doc is None:
        doc = ""

    # Insert disclaimer that this is a copied docstring
    if doc:
        doc = ignore_warning(
            doc, cls, method.__name__, extra=extra, skipblocks=skipblocks
        )
    elif extra:
        doc += extra.rstrip("\n") + "\n\n"

    # Mark unsupported arguments
    try:
        method_args = get_named_args(method)
        original_args = get_named_args(original_method)
        not_supported = [m for m in original_args if m not in method_args]
    except ValueError:
        not_supported = []
    if len(ua_args) > 0:
        not_supported.extend(ua_args)
    if len(not_supported) > 0:
        doc = unsupported_arguments(doc, not_supported)

    doc = skip_doctest(doc)
    doc = extra_titles(doc)

    # TODO: Make regex handle edge cases.
    doc = re.sub(r"np\.", "nps.", doc)
    doc = re.sub(": ndarray", ": BlockArray", doc)
    doc = re.sub(": array_like", ": BlockArray", doc)

    # TODO: Buggy implementation with multiple lines, or assignment to
    # two or more variables.
    # Appends a get() to each numpy function.
    if doctest:
        fns = find_nums_functions(doc)
        for fn in fns:
            doc = doc.replace(fn, fn + ".get()")

    return doc


def find_nums_functions(doc):
    """
    Returns a list of string that includes all the instances of nums function
    in the form of nps.<function>()
    This is done, so we can easily append .get() in docstring tests
    """
    # TODO: Polish the format of this docstring
    return re.findall(r"nps\..+\(.+\)", doc)


def derived_from(
    original_klass, version="1.20.0", ua_args=[], skipblocks=0, doctest=True
):
    """Decorator to attach original class's docstring to the wrapped method.
    The output structure will be: top line of docstring, disclaimer about this
    being auto-derived, any extra text associated with the method being patched,
    the body of the docstring and finally, the list of keywords that exist in
    the original method but not in the dask version.
    Parameters
    ----------
    original_klass: type
        Original class which the method is derived from
    version : str
        Original package version which supports the wrapped method
    ua_args : list
        List of keywords which NumS doesn't support. Keywords existing in
        original but not in NumS will automatically be added.
    skipblocks : int
        How many text blocks (paragraphs) to skip from the start of the
        docstring. Useful for cases where the target has extra front-matter.
    """

    def wrapper(method):
        try:
            extra = getattr(method, "__doc__", None) or ""
            method.__doc__ = _derived_from(
                original_klass,
                method,
                ua_args=ua_args,
                extra=extra,
                skipblocks=skipblocks,
                doctest=doctest,
            )
            return method

        except AttributeError:
            module_name = original_klass.__module__.split(".")[0]

            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                msg = "Base package doesn't support '{0}'.".format(method.__name__)
                if version is not None:
                    msg2 = " Use {0} {1} or later to use this method."
                    msg += msg2.format(module_name, version)
                raise NotImplementedError(msg)

            return wrapped

    return wrapper


if __name__ == "__main__":
    # TODO: Test doc for debugging, delete once finished
    from nums import numpy as nps

    print(nps.loadtxt.__doc__)
