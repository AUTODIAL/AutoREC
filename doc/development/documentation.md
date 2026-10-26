# Building the documentation

The documentation uses Sphinx, the Furo theme, and MyST Markdown. Its dependencies
are separate from AutoREC's scientific environment. The build does not install or
import AutoREC, initialize Julia, or run notebooks or training workloads.

## Create a documentation environment

Use Python 3.12. From the root of an AutoREC checkout, create a dedicated virtual
environment:

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r doc/requirements.txt
```

On Windows, use PowerShell:

```powershell
py -3.12 -m venv .venv-docs
.venv-docs\Scripts\Activate.ps1
python -m pip install -r doc/requirements.txt
```

Keep this environment separate from the one used to run AutoREC. Documentation
builds do not require the package's root `requirements.txt`, Conda environment,
notebook extra, or kernel guard.

## Build HTML

With the documentation environment activated, run from the repository root:

```bash
python -m sphinx -n -W --keep-going -b html doc doc/_build/html
```

The flags report unresolved references and make warnings fail the build while
allowing Sphinx to report all problems in one run. The output starts at
`doc/_build/html/index.html`.

The Makefile runs the same strict checks:

```bash
make -C doc html
```

On Windows, the equivalent wrapper is:

```powershell
.\doc\make.bat html
```

For a full rebuild that ignores cached Sphinx state, add `-E -a` to the direct
Python command. Both `.venv-docs/` and `doc/_build/` are ignored by Git.

## Preview locally

From the repository root, serve the generated HTML:

```bash
python -m http.server --bind 127.0.0.1 --directory doc/_build/html 8000
```

Open <http://127.0.0.1:8000/>. Rebuild after editing sources, then refresh the
browser. Stop the server with `Ctrl+C`.

## Edit and review pages

Write narrative pages in MyST Markdown (`.md`). Use reStructuredText (`.rst`) when
it makes Sphinx directives clearer. Add new pages to a toctree so readers can find
them through the navigation. For example, the homepage includes this page in its
Development toctree.

Before submitting a documentation change:

1. Run the strict HTML build.
2. Check the changed pages, navigation, links, and code blocks in the local preview.
3. Review the diff and keep generated HTML and virtual environments out of commits.

Keep `conf.py` independent of AutoREC imports and scientific dependencies. The
documentation uses a `development` release label until release-aware builds are
introduced. Notebook rendering, API extraction, and automated deployment will be
added separately.

`doc/requirements.txt` pins both the three documentation tools and their resolved
dependencies. When updating them, resolve the tools in a fresh Python 3.12
environment, use `python -m pip freeze` to refresh the resolved pins, and preserve
the Windows-only `colorama` requirement. Run `python -m pip check` as well as the
strict build, and record the versions and validation results in the pull request.
