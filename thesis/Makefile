.PHONY=watch compile

watch: main.tex
	fswatch *.tex {chapters,assets}/*.tex | xargs -n1 make compile
	#> /dev/null 2>&1

compile: main.tex
	latex-compile main; open-file-in-skim $$(pwd)/main.pdf