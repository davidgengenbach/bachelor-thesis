.PHONY=watch compile clean

watch: main.tex compile
	fswatch *.tex {chapters,assets}/*.tex | xargs -n1 make compile
	#> /dev/null 2>&1

compile: main.tex
	latex-compile main; open-file-in-skim $$(pwd)/main.pdf

clean: 
	rm -f *.aux *.bbl *.bcf *.blg *.fmt *.log *.out *.xml *.toc

init:
	pdftex -ini -jobname="main" "&pdflatex" mylatexformat.ltx main.tex
