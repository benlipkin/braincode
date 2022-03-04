SHELL := /usr/bin/env bash
PACKAGE = braincode
ACTIVATE = @source activate $(PACKAGE)
PIPELINE = python $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py
	$(ACTIVATE) ; pip install -e .
setup.py :
ifeq "$(shell conda info --envs | grep $(PACKAGE) | wc -l)" "0"
	@conda create -yn $(PACKAGE) python=3.7
endif

## setup     : download prerequisite files, e.g. neural data, models.
.PHONY : setup
setup : $(PACKAGE)/inputs/
$(PACKAGE)/inputs/ : $(PACKAGE).egg-info/ setup/setup.sh
	$(ACTIVATE) ; cd setup/ ; bash setup.sh


## analysis  : run core analyses to replicate paper.
.PHONY : analysis
analysis : $(PACKAGE)/outputs/
$(PACKAGE)/outputs/ : $(PACKAGE)/inputs/ $(PACKAGE)/*.py
	$(ACTIVATE) ; $(PIPELINE) mvpa
	$(ACTIVATE) ; $(PIPELINE) rsa
	$(ACTIVATE) ; $(PIPELINE) vwea
	$(ACTIVATE) ; $(PIPELINE) nlea
	$(ACTIVATE) ; $(PIPELINE) prda

## paper     : run scripts to generate final plots and tables.
.PHONY : paper
paper : paper/plots/
paper/plots/ : $(PACKAGE)/outputs/ paper/scripts/*.py $(PACKAGE)/.cache/scores/**
	$(ACTIVATE) ; cd paper/scripts ; bash run.sh
