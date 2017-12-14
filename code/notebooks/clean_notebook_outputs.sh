nbstrip_jq() {
    jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = [] | (.cells[] | select(has("execution_count")) | .execution_count) = null | .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} | .cells[].metadata = {}' $1
}

for nb in *.ipynb; do
	echo "Processing: $nb"
	nbstrip_jq $nb | sponge $nb
done