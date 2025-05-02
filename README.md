```
python3 -m venv env
./env/bin/pip install pygit2
./mkmono.py monorepo.git upstream --fetch
./git push origin upstream/develop
```
