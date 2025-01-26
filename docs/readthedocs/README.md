# NuMojo-Docs

Documentation and Base for documentation website for NuMojo.

## How to run checks and change code

Clone this repository

`pip install mdutils mkdocs mkdocs-material`

cd into NuMojo and run `mkdocs serve` in the terminal.

## Generating MD files from Mojo docs jsons

In the folder containing the NuMojo package run `mojo doc numojo/ -o docs.json` then move docs.json into the directory that you cloned NuMojo-Docs into. Then run docs.py, and it will generate the documentation and put it in the correct place.

## Structure
* NuMojo
    * docs
        * docs # the automatically generated docs don't put things here it will be deleted on each run of doc_pages or final
        * getting_started #Installation and tutorial stuff
        * index.html # the front page
* docs.json # mojo doc generated docs
* docs.py # code to generate markdown for the docs pages
* mkdocs.yml # site documentation config

At the root of the project is `.readthedocs.yaml` it performs the setup for the read the docs site
