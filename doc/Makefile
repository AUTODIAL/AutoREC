# Run from the repository root with: make -C doc html
SPHINXBUILD ?= python -m sphinx
SPHINXOPTS ?= -n -W --keep-going
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help Makefile

help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
