.PHONY=watch compile clean open init

watch: main.tex compile
	fswatch *.tex {chapters,assets}/*.tex | xargs -n1 make compile
	#> /dev/null 2>&1

compile: main.tex
	latex-compile main; make open

open: 
	open-file-in-skim $$(pwd)/main.pdf

clean: 
	rm -f *.aux *.bbl *.bcf *.blg *.fmt *.log *.out *.xml *.toc

init:
	pdftex -ini -jobname="main" "&pdflatex" mylatexformat.ltx main.tex
