# make-a-monorepo

Given a set of upstream repositories and their default branches,
this tool will rewrite the branches so they are all inside subdirectories,
then merge the branches by interweaving the first parent commits,
as well as propagating the new subdirectories across the new history.

In other words, it creates a monorepo from multiple repos.

For example: Given two repositories `UpstreamA/main` and `UpstreamB/main`,
this tool creates a new `main` branch containing all of the original commits,
interweaved amongst the first-parent history of the original `main` branches.

It also creates 'UpstreamA/*' and 'UpstreamB/*' branches for all of the
related branches on the upstream repositories. Branches without any
initial commits in common are ignored.

This allows you to copy across feature branches from the upstream repositories to the new monorepo.
If the tool is run again, it builds whatever new commits have
arrived onto the existing tree, assuming the settings have not changed.

In other words, you can run the tool incrementally.

Create a new monorepo branch at `upstream/main`, branch it off
to get your ci/cd working, re-run the tool, rebase your branch (or merge),
and keep going until you're ready to make the switch-over.

## Quick-Start

Install pygit2

```
$ python3 -m venv env
$ ./env/bin/pip install pygit2
```

Write your config file `monorepo.json`, and run the tool:

```
$ ./gitgraph.py build monorepo
```

Run the tool again, forcing it to update the local copies:

```
$ ./gitgraph.py run monorepo --fetch
```

## Configuration

A config file is a json object made up of build steps.
The key is the name of the step, and optionally the name of the branch that gets created as a result.
The value is the configuration for the build step.

Every step has a `step` key defining what the step does, and the other entries in the object are
passed as configuration.

### Create a branch (with an initial empty commit)

```
"init": {
    "step": "start_branch",
    "first_commit": {
        "name": "GitHub",
        "email": "noreply@github.com",
        "timestamp": "2001-01-01",
        "message": "feat: initial commit"
    }
},
```

The initial empty commit comes in useful if you ever use `gh-pages` or similar branch tricks.

### Fetch an upstream branch

```
"upstream-a": {
    "step": "fetch_branch",
    "default_branch": "develop",
    "remote": "git@github.com:username/upstream-a.git",
    "bad_files": "bad_files.json",
    "replace_names": "replace_names.json",
    "named_heads": {
        "develop-mergepoint": "5381cd5dfdfdf434545454545455454545554edc",
        "init": "5d8ddd0454bsfgdfgdfdfgdfdgfgt68d70f98199"
    },
    "include_branches": "**",
    "exclude_branches": "dependabot**"
},
```

- `bad_files` is which files to strip from the repository, a nested dict of `{bad: True, subdir: {file: True}}` or name of json file
- `replace_names` is a dict of {new: [old, old, old} or the name of json file
- `exclude_branches` / `include_branches` are a recursive glob
- `named_heads` allows you to name commits and those names are carried through rewrites

### Interweave a set of branches

```
    "merged-branches": {
        "step": "merge_branches",
        "merge_strategy": "first-parent",
        "branches": {
            "subdirectory_a": "upstream-a",
            "subdirectory_b": "upstream-b",
        },
        "merge_named_heads": [
            "develop-mergepoint"
        ],
        "prefix_message": "conventional-commit"
    },
```

### Append branches together

```
    "main": {
        "step": "append_branches",
        "branches": [
            "init",
            "merged-branches"
        ]
    },
```

### Write branches out

```
    "output": {
        "step": "write_branch",
        "branch": "main",
        "prefix": "upstream",
        "include_branches": "**",
        "exclude_branches": "**/init"
    },
```

### Print Report

```
"report": {
    "step": "show_branch",
    "branch": "main",
    "named_heads": [
        "develop-mergepoint",
        "upstream-a/main",
        "upstream-b/main",
    ]
},
```

### Write names found in branch

```
"write_branch_names": {
    "step": "write_branch_names",
    "branch": "develop",
    "output_filename": "names.txt"
}
```


### Add a remote to the repository

This is useful for when you want to push the output upstream

```
"origin": {
    "step": "add_remote",
    "url": "git@github.com:username/monorepo.git",
    "remote_name": "origin"
},
```
