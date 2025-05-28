# `ratking`: tie git repositories together


Let's say you have two repositories, `UpstreamA` and `UpstreamB`.  `UpstreamA` has several feature branches open `A1`, `A2`, ... and similiarly `UpstreamB` has `B1`, `B2`, etc.

After running `ratking`, you now have one repository, and one `main` branch:

- The new `main` branch consists of all of the commits from the upstream 'main' branches, interweaved by commit date.
- Inside each commit is a `UpstreamA` subdirectory, and a `UpstreamB` subdirectory.
- Commits that branch off `main` also contain `UpstreamA` and `UpstreamB`
- There's also `UpstreamA/A1` branches, and `UpstreamB/B1` branches created alongside the new `main` branch.

If you make some changes upstream, and run `ratking` again, all the new commits
and branches are carried over, atop of the already patched commits. If you don't change
the settings, it produces the same commits as output each time.

Unlike "Just do a merge commit", `ratking` will preserve file history so tools like `git blame` continue to work.

It also means you don't have to do everything at once.

- Create a new monorepo, and use `ratking` to populate the history to `upstream/main`.
- Create a new `main` branch based off `upstream/main` and get your CI/Integrations working.
- Re-run `ratking` to add new commits to `upstream/main`, and `git rebase` or `git merge` your working branch as normal.
- When you're ready to make the switch, reopen PRs based on the migrated branches, like `upstream/UpstreamA/A1`. 

## Algorithm notes

The merge is 'best effort':

- It pulls out a 'linear history' from each branch, i.e the first-parent of each commit
- It merges them by date, preserving the existing ordering of commits
- It then throws out any commits that would cause a conflict when merging


## Caveats

I have only tested this code in production settings. There is no test suite beyond the repos I have used this on at work.

However, there is an unholy amount of defensive code, the graph is checked for correctness before and after several transformations, and I'm pretty sure it will work well enough if you give it a bit of encouragement.

Still, I cannot be responsible for bugs, faults, or problems that result. 

## Quick-Start

Install pygit2

```
$ python3 -m venv env
$ ./env/bin/pip install pygit2
```

Write your config file `monorepo.json`, and run the tool.
This creates a `monorepo.git` subdirectory, containing the insides of a git repo.

```
$ ./ratking.py build monorepo.json
```

Run the tool again, forcing it to update the local copies:

```
$ ./ratking.py build monorepo.json --fetch
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


### Using the underlying python libraries

There's the main classes for 'doing things'

- GitRepo for getting branches, rewriting branches
- GitWriter for grafting a set of branches together, or writing out a changed branch

Then there's the underlying data structures for storing things:

- GitBranch 
- GitGraph
- GitCommit
- GitSignature
- GitTree

There's also a little make-like processor that loads and runs the configuration file:

- GitBuilder


```
from ratking import GitRepo

r = GitRepo("filename", bare=True)

r.add_remote(...)

r.get_remote_branch(...)

r.interweave_branches(name, {"prefix": branch}, ....)

```



