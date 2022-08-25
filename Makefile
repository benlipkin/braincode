SHELL := /usr/bin/env bash
EXEC = python
PACKAGE = braincode
INSTALL = pip install -e .
ACTIVATE = source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin main

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@conda create -yn $(PACKAGE) $(EXEC)
	@$(ACTIVATE) ; $(INSTALL)

## setup     : download prerequisite files, e.g. neural data, models.
.PHONY : setup
setup : env $(PACKAGE)/inputs/
$(PACKAGE)/inputs/ : setup/setup.sh
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F)

## test      : run testing pipeline.
.PHONY : test
test : pylint mypy
pylint : env html/pylint/index.html
mypy : env html/mypy/index.html
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(PACKAGE)/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(PACKAGE) --output-format=colorized,json:$@ || pylint-exit $$?
html/mypy/index.html : $(PACKAGE)/*.py
	@$(ACTIVATE) ; mypy --ignore-missing-import -p $(PACKAGE) --html-report $(@D)