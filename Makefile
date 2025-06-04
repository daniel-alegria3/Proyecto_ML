# Se necesita tener tectonic installado

OUTPUT_DEPS = main.tex _preamble.tex _postamble.tex

output: ${OUTPUT_DEPS}
	tectonic -X compile main.tex

clean:
	rm -f main.aux main.log main.pyg main.xdv main.out

.PHONY: clean

