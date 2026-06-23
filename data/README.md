Download the myScheme dataset from one of:

- https://huggingface.co/datasets/shrijayan/gov_myscheme
- https://www.kaggle.com/datasets/jainamgada45/indian-government-schemes

Save it here as `schemes.json` or `schemes.csv`.

Before running scripts/ingest.py, open the file and check its column
headers against FIELD_MAP / NAME_COLS / LINK_COLS in that script - the
two sources don't use identical column names. Adjust the lists if a
field isn't showing up.
