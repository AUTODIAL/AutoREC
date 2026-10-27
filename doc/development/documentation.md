# Building the documentation

The documentation uses Sphinx, the Furo theme, MyST Markdown, Sphinx AutoAPI,
nbsphinx, nbsphinx-link, and Sphinx Copybutton. Its dependencies are separate
from AutoREC's scientific environment. The build does not install or import
AutoREC, initialize Julia, execute notebooks, or run training workloads.

## Create a documentation environment

Use Python 3.12. From the root of an AutoREC checkout, create a dedicated virtual
environment:

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r doc/requirements.txt
```

Notebook conversion also requires Pandoc. The documentation is validated with
Pandoc 3.1.3. On Ubuntu or Debian, install the distribution package and confirm
that the executable is available:

```bash
sudo apt-get install pandoc
pandoc --version
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

## Automated builds, review artifacts, and deployment

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

The build job has read-only repository permissions. An artifact is available
only when the build and upload steps succeed. Subsequent updates to the same pull
request or branch cancel an older in-progress build.

Pull requests validate and upload the review artifact but never deploy. A
successful push to `main`, or a manual run on `main`, also packages the same HTML
as a GitHub Pages artifact. A separate deployment job has only the `pages: write`
and OpenID Connect permissions required by GitHub Pages and publishes through
the protected `github-pages` environment.

Repository administrators must select **GitHub Actions** as the Pages source in
**Settings → Pages** before the first deployment. The deployment step reports
the authoritative public URL. The organization Pages domain currently redirects
the repository URL to <https://autodiallab.ca/AutoREC/>; that path will return
404 until a successful deployment is available on the default branch.

## Edit and review pages

Write narrative pages in MyST Markdown (`.md`). Use reStructuredText (`.rst`) when
it makes Sphinx directives clearer. Add new pages to a toctree so readers can find
them through the navigation. For example, the homepage includes this page in its
Development toctree.

Before submitting a documentation change:

1. Run the strict HTML build.
2. Check the changed pages, navigation, links, and code blocks in the local preview.
3. Run the external link check when links were added or changed:

   ```bash
   python -m sphinx -E -a -W --keep-going -b linkcheck doc doc/_build/linkcheck
   ```

   Review failures individually. External services can rate-limit or reject
   automated requests even when a browser can open the page.
4. Review the diff and keep generated HTML and virtual environments out of commits.

## Content ownership and synchronization

Narrative guides under `doc/` are the canonical location for detailed user and
developer instructions. After the documentation website is deployed, the root
README should remain a concise project introduction that links to those guides.
Avoid maintaining full configuration tables, CLI manuals, or tutorials in both
places.

Use these ownership boundaries:

- constructor signatures and parameter-level behavior belong in source
  docstrings and the generated API reference;
- workflows, lifecycle constraints, and cross-component explanations belong in
  narrative guides;
- executable walkthroughs belong in the repository-level notebooks, linked into
  the site with `.nblink` files;
- example-specific cluster or experiment instructions remain beside that
  example, with an overview link from the website.

When behavior changes, search the README, `doc/`, notebook sources, example
configuration comments, and docstrings for affected statements. Validate prose
against implementation and tests instead of treating an existing example as
authoritative. Qualify scientific, portability, reproducibility, and
performance claims to the conditions actually tested.

## Maintain notebook tutorials

The `.nblink` files under `doc/tutorials/` point to the three authoritative
notebooks under the repository-level `tutorials/` directory. Do not copy the
notebooks into `doc/`. nbsphinx renders their saved outputs with
`nbsphinx_execute = "never"`; documentation builds do not need an IPython kernel
or AutoREC's scientific dependencies.

Before publishing notebook changes, check that notebook-level `nbsphinx`
metadata does not override the global no-execution setting. Clear accidental
error outputs, retain only useful and current display outputs, and review any
performance or scientific claims separately. A successful documentation build
shows that the saved notebook is renderable; it does not validate the numerical
results.

The linked notebooks assume `tutorials/` is the working directory when users run
them locally. Keep input and output paths on the tutorials index synchronized
with the notebook cells. The download and source links shown on every rendered
notebook are configured in `doc/conf.py`.

## Maintain the API reference

Sphinx AutoAPI parses the Python files under `src/autorec/` without importing
the package. The curated pages under `doc/api/` use explicit `autoapi`
directives, and `autoapi_generate_api_docs = False` prevents an exhaustive
module tree from being generated. Keep each Python object on one canonical API
page so Sphinx does not create duplicate index entries.

The API pages intentionally document the primary workflow classes, public
factory exports, runtime setup, and a small set of user-facing utilities. Add an
object only when it is useful as a supported interface. Prefer improving its
source docstring in a separately reviewed source change; use nearby prose on the
API page for lifecycle rules, side effects, or other contract details that do
not belong in the signature.

`doc/conf.py` contains a narrow `autodoc-process-docstring` hook for known
formatting problems, stale default text, and model-export terminology in the
current source docstrings. Keep such replacements exact and temporary so they
cannot silently rewrite unrelated documentation; remove them when the source
docstrings are corrected in an approved source change.

Do not replace the `autoapi` directives with import-based `automodule`,
`autoclass`, or `autofunction` directives. A strict API build must continue to
pass in the isolated documentation environment without AutoREC, TensorFlow,
JAX, AutoEIS, or Julia installed.

## Appearance and navigation

`doc/conf.py` selects the existing repository-level `Logo.png` for both color
modes and defines the teal/light and aqua/dark accent colors. The visible project
name stays below the logo. Furo supplies the system fonts, search, theme toggle,
and responsive sidebar; no external font service or copied AutoEIS assets are
required. Small layout and keyboard-focus adjustments live in
`doc/_static/custom.css`.

The homepage's toctrees group navigation into Getting started, User guide,
Reference, Development, and Project. Add a page only when it has a clear owner
and distinct purpose; prefer cross-references over duplicated explanations.

For appearance changes, preview both pages in light and dark modes at desktop
and narrow mobile widths. Check the sidebar, visible keyboard focus, and code
copy buttons. Long code blocks and tables should scroll within their containers
without widening the whole page; long navigation labels should wrap.

Keep `conf.py` independent of AutoREC imports and scientific dependencies. The
documentation uses a `development` release label until release-aware builds are
introduced.

`doc/requirements.txt` pins both the documentation tools and their resolved
dependencies. When updating them, resolve the tools in a fresh Python 3.12
environment, use `python -m pip freeze` to refresh the resolved pins, and preserve
the Windows-only `colorama` requirement. Run `python -m pip check` as well as the
strict build, and record the versions and validation results in the pull request.
