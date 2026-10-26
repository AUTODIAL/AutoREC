# Building the documentation

The documentation uses Sphinx, the Furo theme, MyST Markdown, and Sphinx
Copybutton. Its dependencies are separate from AutoREC's scientific environment.
The build does not install or import AutoREC, initialize Julia, or run notebooks
or training workloads.

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

## Automated builds and review artifacts

The `Documentation` GitHub Actions workflow in `.github/workflows/docs.yml` runs
on pull requests, pushes to `main`, and manual dispatch. It uses Python 3.12 in a
dedicated virtual environment, installs `doc/requirements.txt`, checks dependency
compatibility, and runs the same strict HTML build described above. Sphinx
warnings fail the build.

After a successful build, the workflow uploads `doc/_build/html/` as the
`docs-html` artifact, retained for seven days. To review it:

1. Open the pull request's **Build documentation** check, or find its run under
   **Actions → Documentation** in the repository.
2. Open the workflow run summary and download **docs-html** from **Artifacts**.
   GitHub requires you to sign in to download artifacts.
3. Extract the archive. From the extracted directory containing `index.html`, run:

   ```bash
   python -m http.server --bind 127.0.0.1 8000
   ```

4. Open <http://127.0.0.1:8000/> to review the built pages.

The workflow has read-only repository permissions and does not deploy a website.
An artifact is available only when the build and upload steps succeed. Subsequent
updates to the same pull request or branch cancel an older in-progress build.

## Edit and review pages

Write narrative pages in MyST Markdown (`.md`). Use reStructuredText (`.rst`) when
it makes Sphinx directives clearer. Add new pages to a toctree so readers can find
them through the navigation. For example, the homepage includes this page in its
Development toctree.

Before submitting a documentation change:

1. Run the strict HTML build.
2. Check the changed pages, navigation, links, and code blocks in the local preview.
3. Review the diff and keep generated HTML and virtual environments out of commits.

## Appearance and navigation

`doc/conf.py` selects the existing repository-level `Logo.png` for both color
modes and defines the teal/light and aqua/dark accent colors. The visible project
name stays below the logo. Furo supplies the system fonts, search, theme toggle,
and responsive sidebar; no external font service or copied AutoEIS assets are
required. Small layout and keyboard-focus adjustments live in
`doc/_static/custom.css`.

The homepage's toctrees group navigation into Getting started, Development, and
Project. User guides currently link to the existing repository content. Replace
those links with local pages as the guides are migrated, and add further groups
only when they contain useful content.

For appearance changes, preview both pages in light and dark modes at desktop
and narrow mobile widths. Check the sidebar, visible keyboard focus, and code
copy buttons. Long code blocks and tables should scroll within their containers
without widening the whole page; long navigation labels should wrap.

Keep `conf.py` independent of AutoREC imports and scientific dependencies. The
documentation uses a `development` release label until release-aware builds are
introduced. Notebook rendering, API extraction, and automated deployment will be
added separately.

`doc/requirements.txt` pins both the documentation tools and their resolved
dependencies. When updating them, resolve the tools in a fresh Python 3.12
environment, use `python -m pip freeze` to refresh the resolved pins, and preserve
the Windows-only `colorama` requirement. Run `python -m pip check` as well as the
strict build, and record the versions and validation results in the pull request.
