mkdir -p data/html
cd data/html
rsync -a --info=progress2 pe:ted-talks/data/html/ .
ls | wc -l
