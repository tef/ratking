#!env/bin/python3

import functools
import glob
import json
import os.path
import re
import subprocess

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import translate as glob_to_regex

import pygit2

GIT_DIR_MODE = 0o040_000
GIT_FILE_MODE = 0o100_644
GIT_FILE_MODE2 = 0o100_664
GIT_EXEC_MODE = 0o100_755
GIT_LINK_MODE = 0o120_000
GIT_GITLINK_MODE = 0o160_000  # actually a submodule, blegh
GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # Wow, isn't git amazing.


class Bug(Exception):
    pass


class Error(Exception):
    pass


class TimeTravel(Exception):
    pass


class NonLinear(Exception):
    pass


class Callback:
    def __init__(self):
        self.callbacks = {}

    def get(self, name):
        return self.callbacks.get(name, name)

    def add(self, name):
        def _add(fn):
            self.callbacks[name] = fn
            return fn

        return _add


@functools.cache
def compile_pattern(pattern):
    regex = glob.translate(pattern, recursive=True)
    return re.compile(regex)


def glob_match(pattern, string):
    if not pattern:
        return False
    if pattern is True:
        return True
    rx = compile_pattern(pattern)
    return rx.match(string) is not None


prefix_message_callbacks = Callback()


cc_prefixes = (
    "build:",
    "chore:",
    "ci:",
    "docs:",
    "feat:",
    "fix:",
    "perf:",
    "refactor:",
    "revert:",
    "style:",
    "test:",
)

cc_prefixes2 = tuple(c.replace(":", "\x28") for c in cc_prefixes)


@prefix_message_callbacks.add("conventional-commit")
def prefix_with_conventional_commit(c, prefix):
    message = c.message
    if len(c.parents) > 1:  # skip merges
        pass
    elif message.startswith(cc_prefixes2):  # feat(...):
        pass
    elif message.startswith(cc_prefixes):
        kind, tail = message.split(":", 1)
        message = f"{kind}({prefix}): {message}"
    else:
        message = f"feat({prefix}): {message}"
    return message


@dataclass
class GitSignature:
    name: str
    email: str
    time: int
    offset: int

    def __eq__(self, other):
        return all(
            (
                self.name == other.name,
                self.email == other.email,
                self.time == other.time,
                self.offset == other.offset,
            )
        )

    def replace(self, name=None, email=None, time=None, offset=None):
        return GitSignature(
            name=name if name else self.name,
            email=email if email else self.email,
            time=time if time else self.time,
            offset=offset if offset else self.offset,
        )

    @property
    def date(self):
        tz = timezone(timedelta(minutes=self.offset))
        date = datetime.fromtimestamp(float(self.time), tz)
        return date

    @date.setter
    def date(self, date):
        self.time = int(date.timestamp())
        self.offset = int(date.tzinfo.utcoffset(None).total_seconds()) // 60


    def to_pygit(self):
        return pygit2.Signature(
            name=self.name, email=self.email, time=self.time, offset=self.offset
        )

    def __str__(self):
        return f"{self.name} <{self.email}>"

    @classmethod
    def from_pygit(self, obj):
        return GitSignature(
            name=obj.name, email=obj.email, time=obj.time, offset=obj.offset
        )


@dataclass
class GitCommit:
    tree: str
    parents: list
    author_date: object
    committer_date: object
    max_date: object
    author: str
    committer: str
    message: str

    def __eq__(self, other):
        return all(
            (
                self.tree == other.tree,
                self.parents == other.parents,
                self.max_date == other.max_date,
                self.author == other.author,
                self.committer == other.committer,
                self.message == other.message,
            )
        )

    def clone(self):
        return GitCommit(
            tree=self.tree,
            parents=list(self.parents),
            author=self.author,
            committer=self.committer,
            message=self.message,
            max_date=self.max_date,
            author_date=self.author_date,
            committer_date=self.committer_date,
        )


@dataclass
class GitTree:
    entries: list


@dataclass
class GitGraph:
    commits: dict
    tails: set
    heads: set
    parents: dict
    parent_count: dict
    children: dict
    child_count: dict
    fragments: set

    @classmethod
    def new(self):
        return GitGraph(
            commits={},
            tails=set(),
            heads=set(),
            parents={},
            parent_count={},
            children={},
            child_count={},
            fragments=set(),
        )

    def clone(self):
        return GitGraph(
            commits=dict(self.commits),
            tails=set(self.tails),
            heads=set(self.heads),
            children={k: set(v) for k, v in self.children.items()},
            parents={k: list(v) for k, v in self.parents.items()},
            parent_count=dict(parent_count),
            child_count=dict(child_count),
            fragments=set(self.fragments),
        )

    def walk_children(self):
        search = list(self.tails)
        counts = dict(self.parent_count)

        search.sort(key=lambda x: self.commits[x].max_date)

        while search:
            c = search.pop(0)
            yield c

            for i in sorted(self.children[c], key=lambda x: self.commits[x].max_date):
                counts[i] -= 1
                if counts[i] == 0:
                    search.append(i)

    def walk_parents(self):
        search = list(self.heads)
        counts = dict(self.child_count)

        while search:
            c = search.pop(0)
            yield c

            for i in self.parents[c]:
                counts[i] -= 1
                if counts[i] == 0:
                    search.append(i)

    def first_parents(self, head):
        history = [head]
        n = head

        while n is not None:
            p = self.parents.get(n)
            p = p[0] if p else None
            if p is None:
                break
            history.append(p)
            n = p

        history.reverse()
        return history

    def get_all_names(self):
        names = {}

        def add_name(i, n):
            if n not in names:
                names[n] = set()
            names[n].add(i)

        for i, c in self.commits.items():
            add_name(i, str(c.author))
            add_name(i, str(c.committer))

            for line in c.message.splitlines():
                if "Co-authored-by: " in line:
                    _, name = line.rsplit("Co-authored-by: ", 1)
                    add_name(i, name.strip())
        return names

    def add_commit(self, idx, c):
        c_parents = c.parents

        self.commits[idx] = c
        self.parent_count[idx] = len(c.parents)
        self.parents[idx] = list(c_parents)

        if not c_parents:
            self.tails.add(idx)

        if idx not in self.children:
            self.children[idx] = set()
            self.child_count[idx] = 0

        for pidx in c_parents:
            if pidx in self.heads:
                self.heads.remove(pidx)

            if pidx not in self.children:
                self.children[pidx] = set()
            self.children[pidx].add(idx)
            self.child_count[pidx] = len(self.children[pidx])

    def add_graph(self, other):
        for idx in other.commits:
            if idx not in self.commits or idx in self.fragments:
                c = other.commits[idx].clone()
                self.add_commit(idx, c)
                if idx in other.fragments:
                    self.fragments.add(idx)
                elif idx in self.fragments:
                    self.fragments.remove(idx)
            else:
                if idx not in other.fragments:
                    if self.commits[idx] != other.commits[idx]:
                        raise Bug(
                            "Two commits with same hash and different values cannot be added to the same graph"
                        )
                    if self.parents[idx] != other.parents[idx]:
                        raise Bug(
                            "Two commits with same hash and different graph values cannot be added to the same graph"
                        )
                    continue

                if idx not in self.children:
                    self.children[idx] = set()
                    self.child_count[idx] = 0

                for c in other.children[idx]:
                    self.children[idx].add(c)
                    self.child_count[idx] = len(self.children[idx])

        for f in other.tails:
            # xxx don't merge in graphs with new init commits
            # xxx we could allow this? but for now it may prevent bugs

            if not self.parents[f] and f not in self.tails:
                raise Bug("Tail commit shouldn't have any parents")

        for l in other.heads:
            if not self.children[l]:
                self.heads.add(l)

        return self

    @classmethod
    def union(cls, graphs):
        all_graphs = cls.new()

        for name, graph in graphs.items():
            all_graphs.add_graph(graph)

        return all_graphs

    def validate(self):
        # all tails items have no parents

        for f in self.tails:
            fp = self.parents[f]
            cp = self.commits[f].parents
            if cp:
                raise Bug("Graph tail has parents in commit")
            if fp:
                raise Bug("Graph tail has parent in graph")

        for f in self.fragments:
            if f not in self.tails:
                raise Bug("Graph has fragment that's not in tails")
            if self.commits[f].parents:
                raise Bug("Graph has fragment with parents in commit")
            if self.parents[f]:
                raise Bug("Graph has fragment with parents in graph")

            # fragment can have children if it's found from branch head
            # or no children if it's just the branch head

        # validate children

        inverted_children = {}

        for idx in self.commits:
            children = self.children[idx]
            if not children:
                if idx not in self.heads:
                    raise Bug("Graph with untracked head")
            for child in children:
                if child not in inverted_children:
                    inverted_children[child] = set()
                inverted_children[child].add(idx)

                m = [i for i in self.parents[child] if i == idx]
                if len(m) != 1 or m[0] != idx:
                    raise Bug("Graph has commit with children, but not in parents")

        # validate parents

        for c in self.commits:
            commit_parents = self.commits[c].parents
            graph_parents = self.parents[c]
            child_parents = inverted_children.get(c, set())

            if commit_parents != graph_parents:
                raise Bug(f"Graph has commit with parents: {c}")

            if child_parents != set(graph_parents):
                raise Bug(f"Graph has commit with missing children: {c}")

            if len(set(graph_parents)) != len(graph_parents):
                raise Error("Graph has commit with dupe parent")

            if self.parent_count[c] != len(self.parents[c]):
                raise Bug("Graph has inconsistent parent count")

            if self.child_count[c] != len(self.children[c]):
                raise Bug("Graph has inconsistent child count")

        # walk backwards from heads

        found_tails = set()
        walked = set()

        for c in self.walk_parents():
            walked.add(c)
            if not self.parents[c]:
                found_tails.add(c)

        if found_tails != self.tails:
            raise Bug("Graph has inconsistent tails")

        if walked != set(self.commits):
            raise Bug("Graph has unreachable commits from heads")

        # walk forward from talks
        # validate complete walk through children

        walked = set()
        heads = set()

        for i in self.walk_children():
            walked.add(i)
            if not self.children[i]:
                heads.add(i)

        if heads != self.heads:
            raise Exception("Graph has inconsistent heads")

        if walked != set(self.commits):
            raise Bug("Graph has unreachable commits from tails")

    def to_branch(self, name, head, named_heads, original):
        history = self.first_parents(head)

        date = self.commits[history[0]].max_date
        for i in history[1:]:
            new_date = self.commits[i].max_date
            if new_date < date:
                raise TimeTravel("Graph has commits out of date order")

        return GitBranch(
            name=name,
            head=head,
            graph=self,
            tail=history[0],
            named_heads=named_heads,
            original=original,
        )


@dataclass
class GitBranch:
    name: str
    head: str
    graph: object
    named_heads: dict
    tail: str
    original: dict

    def clone(self):
        return GitBranch(
            name=str(name),
            head=str(self.head),
            graph=graph.clone(),
            tail=str(self.tail),
            named_heads=dict(self.named_heads),
            original=dict(self.original),
        )

    def common_ancestor(self, other):
        left_linear = self.graph.first_parents(self.head)
        right_linear = other.graph.first_parents(other.head)
        left_children, right_children = self.graph.children, other.graph.children

        before, after = None, None
        for x, y in zip(left_linear, right_linear):
            if x != y:
                after = y
                break

            if left_children[x] != right_children[y]:
                after = y
                break

            before = x
        return before, after

    def validate(self):
        self.graph.validate()
        graph = self.graph

        for name, idx in self.named_heads.items():
            if idx not in graph.commits:
                raise Bug(f"Graph has named head with no commit: {name}, {idx}")

        if self.head not in graph.commits:
            raise Bug(f"Graph has head with no commit: {self.head}")

        if self.tail not in graph.commits:
            raise Bug(f"Graph has tail with no commit: {self.tail}")

        history = graph.first_parents(self.head)

        if history[-1] != self.head:
            raise Exception("First parent history does not start at head")
        if history[0] != self.tail:
            raise Exception("First parent history does not end at tail")

    def add_named_fragment(self, name, head, other):
        graph = self.graph
        graph.add_graph(other)
        self.named_heads[name] = head

    @staticmethod
    def make_linear_parent(history, tails, children):
        linear_parent = {c: n for n, c in enumerate(history, 1)}

        for lc in reversed(history):
            n = linear_parent[lc]
            search = list(children.get(lc, ()))
            while search:
                c = search.pop(0)
                if c not in linear_parent:
                    linear_parent[c] = n
                    search.extend(children.get(c, ()))

        for f in tails:
            n = linear_parent.get(f, 0)
            linear_parent[f] = n
            search = list(children.get(f, ()))
            while search:
                c = search.pop(0)
                if c not in linear_parent:
                    linear_parent[c] = n
                    search.extend(children.get(c, ()))

        return linear_parent

    @classmethod
    def merge_linear_history(self, branches):
        graphs = {}
        branch_history = {}
        branch_linear_parent = {}

        for name, branch in branches.items():
            graph = branch.graph
            graphs[name] = graph

            history = graph.first_parents(branch.head)
            branch_history[name] = history
            branch_linear_parent[name] = GitBranch.make_linear_parent(
                history, graph.tails, graph.children
            )

        merged_graph = GitGraph.union(graphs)

        history = [list(h) for h in branch_history.values()]
        new_history = []

        while history:
            next_head = [h[-1] for h in history]

            next_head.sort(key=lambda i: merged_graph.commits[i].max_date)

            c = next_head[-1]
            new_history.append(c)

            for h in history:
                if h[-1] == c:
                    h.pop()

            if any(not h for h in history):
                history = [h for h in history if h]

        new_history.reverse()

        # validate new history

        seen = set()
        new_history2 = []

        for h in branch_history.values():
            new_history2.extend(x for x in h if x not in seen)
            seen.update(h)

        new_history2.sort(key=lambda idx: merged_graph.commits[idx].max_date)

        if new_history != new_history2:
            raise TimeTravel("Merged branch somehow out of date order")

        head = new_history[-1]
        tail = new_history[0]

        if merged_graph.commits[tail].parents:
            raise Bug("Merged graph has tail with parent commits")
        if merged_graph.parents[tail]:
            raise Bug("Merged graph has tail with parent commits in graph")

        for name, branch in branches.items():
            h = list(branch_history[name])
            h.reverse()
            g = branch.graph

            for idx in new_history:
                if idx in g.commits:
                    i = h.pop()
                    if idx != i:
                        raise NonLinear(
                            "Merged linear history does not respect individual branch linear history"
                        )
            if h:
                raise NonLinear(
                    "Merged linear history does include individual branch linear history"
                )

        return new_history, branch_history, branch_linear_parent, merged_graph

    @classmethod
    def interweave(cls, new_name, branches, named_heads=None, merge_named_heads=()):

        # create a new linear history
        history, branch_history, branch_linear_parent, merged_graph = (
            cls.merge_linear_history(branches)
        )

        head, tail = history[-1], history[0]

        # rewrite the parents
        # note: could remove original linear parent, and not create a merge commit

        prev = tail
        for idx in history[1:]:
            old_parents = merged_graph.commits[idx].parents
            new_parents = [prev] + [o for o in old_parents if o != prev]

            merged_graph.parents[idx] = list(new_parents)
            merged_graph.commits[idx].parents = list(new_parents)
            merged_graph.parent_count[idx] = len(new_parents)

            merged_graph.children[prev].add(idx)
            merged_graph.child_count[prev] = len(merged_graph.children[prev])

            if idx in merged_graph.tails:
                merged_graph.tails.remove(idx)
            if idx != head and idx in merged_graph.heads:
                merged_graph.heads.remove(idx)

            prev = idx

        merged_graph.validate()

        # validate rewrite

        prev = history[0]
        date = merged_graph.commits[history[0]].max_date
        for idx in history[1:]:
            old_parents = merged_graph.commits[idx].parents
            if prev not in old_parents:
                raise Bug("Rewritten commit has missing parent")

            new_date = merged_graph.commits[idx].max_date
            if new_date < date:
                raise TimeTravel("New history is out of date order")

            prev = idx
            date = new_date

        # fill out linear parents

        linear_parent = GitBranch.make_linear_parent(
            history, merged_graph.tails, merged_graph.children
        )

        # validate linear parents

        linear_depth = [linear_parent[x] for x in history]

        if linear_depth != sorted(linear_depth):
            raise NonLinear("First parent history is not in order")

        if set(merged_graph.commits) != set(linear_parent):
            raise Error("Linear parent does not cover all graph")

        # ensure linear ordering of source branches is preserved in merged branch

        for name, branch in branches.items():
            graph = branch.graph
            history_depth = [linear_parent[x] for x in branch_history[name]]
            if history_depth != sorted(history_depth):
                raise Bug("Linear history of source branch is out of order")
            for c in graph.commits:
                if branch_linear_parent[name][c] == 0:
                    if linear_parent[c] != 0:
                        # xxx - we exclude extra tails and shouldn't fold them in
                        # xxx - and we error elsewhere about it
                        raise NonLinear(
                            "Commit in original branch has new linear parent in merged branch"
                        )
                lp = branch_linear_parent[name][c]
                lp_idx = branch_history[name][lp - 1]
                nlp = linear_parent[lp_idx]
                if nlp != linear_parent[c]:
                    raise NonLinear(
                        "Commit in original branch has new linear parent in merged branch"
                    )

        # create the named heads for merged branch
        # find all merge points passed in, passed { "named head" : {"upstream name":"commit id"}}

        all_named_heads = {}

        for name, branch in branches.items():
            for k, v in branch.named_heads.items():
                all_named_heads[f"{name}/{k}"] = v

        named_heads = {} if named_heads is None else named_heads
        merge_named_heads = () if merge_named_heads is None else merge_named_heads

        for point_name in merge_named_heads:
            mp = {}
            for name, branch in branches.items():
                if point_name in branch.named_heads:
                    mp[name] = branch.named_heads[point_name]
            named_heads[point_name] = mp

        for point_name, merge_points in named_heads.items():

            all_merge_points = [
                (idx, name, branch_linear_parent[name][idx], linear_parent[idx])
                for name, idx in merge_points.items()
            ]
            all_merge_points.sort(key=lambda x: x[3])

            merge_point = all_merge_points[-1][0]
            merge_point_time = all_merge_points[-1][-1]

            for idx, name, old_lp, new_lp in all_merge_points:
                if old_lp == len(branch_history[name]):
                    pass  # last commit, no worries
                else:
                    next_c = branch_history[name][old_lp]
                    next_lp = linear_parent[next_c]
                    if next_lp <= merge_point_time:
                        raise Error(f"Cannot merge named head {name} in merged branch")

            all_named_heads[point_name] = merge_point

        all_original = {}
        for name, branch in branches.items():
            for k, v in branch.original.items():
                if k not in all_original:
                    all_original[k] = v
                elif v != all_original[k]:
                    raise Bug("Conflicting rewrites")

        branch = GitBranch(
            name=new_name,
            graph=merged_graph,
            head=head,
            tail=tail,
            named_heads=all_named_heads,
            original=all_original,
        )

        branch.validate()
        return branch, linear_parent


# used for fetch
class AuthCallbacks(pygit2.RemoteCallbacks):
    def credentials(self, url, username_from_url, allowed_types):
        if allowed_types & pygit2.enums.CredentialType.USERNAME:
            return pygit2.Username("git")
        elif allowed_types & pygit2.enums.CredentialType.SSH_KEY:
            x = os.path.expanduser("~/.ssh/id_ed25519.pub")
            y = os.path.expanduser("~/.ssh/id_ed25519")
            return pygit2.Keypair("git", x, y, "")
        else:
            return None


class GitRepo:
    def __init__(self, repo_dir):
        if not repo_dir.endswith(".git"):
            repo_dir += ".git"
        self.git = pygit2.init_repository(repo_dir, bare=True)
        self._all_remotes = None

    def add_remote(self, rname, url):
        names = list(self.git.remotes.names())

        if rname not in names:
            o = self.git.remotes.create(rname, url)
            return True
        else:
            o = self.git.remotes[rname]
            return False

    def fetch_remote(self, rname):
        o = self.git.remotes[rname]
        o.fetch(callbacks=AuthCallbacks())
        self._all_remotes = None

    def remote_branches(self, remote_name):
        remote = self.git.remotes[remote_name]
        remote_refs = remote.ls_remotes(callbacks=AuthCallbacks())
        HEAD = None
        branches = []
        for ref in remote_refs:
            if ref["name"] == "HEAD":
                HEAD = ref["symref_target"].split("refs/heads/", 1)[1]
            elif ref["name"].startswith("refs/heads/"):
                name = ref["name"].split("refs/heads/", 1)[1]
                branches.append(name)
        return HEAD, branches

    def all_remote_branch_names(self, refresh=False):
        if self._all_remotes and not refresh:
            return self._all_remotes
        all_branches = list(self.git.branches.remote)
        out = {}
        for b in all_branches:
            prefix, name = b.split("/", 1)
            if prefix not in out:
                out[prefix] = {}
            out[prefix][name] = self.get_remote_branch_head(prefix, name)
        self._all_remotes = out
        return out

    def get_commit(self, addr):
        obj = self.git.get(addr)

        a_tz = timezone(timedelta(minutes=obj.author.offset))
        a_date = datetime.fromtimestamp(float(obj.author.time), a_tz).astimezone(
            timezone.utc
        )

        c_tz = timezone(timedelta(minutes=obj.committer.offset))
        c_date = datetime.fromtimestamp(float(obj.committer.time), c_tz).astimezone(
            timezone.utc
        )

        tree = str(obj.tree_id)

        author = GitSignature.from_pygit(obj.author)
        committer = GitSignature.from_pygit(obj.committer)
        message = obj.message

        return GitCommit(
            tree=tree,
            parents=list(str(x) for x in obj.parent_ids),
            author=author,
            committer=committer,
            message=message,
            max_date=max(a_date, c_date),
            author_date=a_date,
            committer_date=c_date,
        )

    def get_tree(self, addr):
        if isinstance(addr, GitTree):
            addr = self.repo.write_tree(addr)

        elif not isinstance(addr, str):
            raise Bug("Tree address must be string")

        obj = self.git.get(addr)
        entries = []
        for i in obj:
            e = (int(i.filemode), i.name, str(i.id))
            entries.append(e)
        return addr, GitTree(entries)

    def write_commit(self, c):
        tree = c.tree
        if isinstance(c, GitTree):
            tree = self.write_tree(tree)
        elif isinstance(c.tree, str):
            tree = pygit2.Oid(hex=tree)
        else:
            raise Bug("Passed bad tree")
        parents = [pygit2.Oid(hex=p) for p in c.parents]
        author = c.author.to_pygit()
        committer = c.committer.to_pygit()
        out = self.git.create_commit(None, author, committer, c.message, tree, parents)
        if not out:
            raise Error("Couldn't write commit")
        return str(out)

    def write_tree(self, t):
        tb = self.git.TreeBuilder()
        t.entries.sort(key=lambda x: x[1] if x[0] != GIT_DIR_MODE else x[1] + "/")
        for mode, name, addr in t.entries:
            if isinstance(addr, GitTree):
                i = self.write_tree(addr)
            elif isinstance(addr, str):
                i = pygit2.Oid(hex=addr)
            else:
                raise Bug("bad tree passed")
            tb.insert(name, i, mode)

        out = tb.write()
        return str(out)

    def start_branch(self, branch_name, name, email, timestamp, message):
        if timestamp.tzinfo is None:
            raise Bug("Commit datetime needs timestamp")
        ts = timestamp.astimezone(timezone.utc)
        signature = GitSignature(name, email, int(ts.timestamp()), 0)
        c = GitCommit(
            tree=GIT_EMPTY_TREE,
            parents=[],
            author=signature,
            committer=signature,
            message=message,
            max_date=ts,
            author_date=ts,
            committer_date=ts,
        )

        head = self.write_commit(c)

        graph = GitGraph.new()
        graph.add_commit(head, c)
        graph.heads = set([head])
        named_heads = {branch_name: head}

        branch = graph.to_branch(branch_name, head, named_heads, {})
        branch.validate()

        return branch

    def get_branch_head(self, name):
        if name in self.git.branches:
            return str(self.git.branches[name].target)

    def write_branch_head(self, name, addr):
        c = self.git.revparse_single(addr)
        self.git.branches.create(name, c, force=True)

    def get_remote_branch_head(self, rname, name):
        return str(self.git.branches.remote[f"{rname}/{name}"].target)

    def new_branch(self, name, head, graph=None, named_heads=None):
        graph = graph or self.get_graph(head)
        named_heads = dict(named_heads) if named_heads else {}
        named_heads.update(graph.named_heads)
        return graph.to_branch(name, head, named_heads, {})

    def get_branch(self, branch_name, include="*", exclude=None, replace_parents=None):
        branch_head = self.get_branch_head(branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}
        branch = graph.to_branch(name, head, named_heads, {})
        branch.validate()

        return branch

    def get_remote_branch(
        self, rname, branch_name, include=None, exclude=None, replace_parents=None
    ):

        branch_head = self.get_remote_branch_head(rname, branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}

        branch = branch_graph.to_branch(
            name=branch_name,
            head=branch_head,
            named_heads=named_heads,
            original={},
        )

        if include:
            all_remote_branches = self.all_remote_branch_names()
            remote_branches = all_remote_branches.get(rname, {})

            for name, idx in remote_branches.items():
                if name == branch_name:
                    continue
                if not glob_match(include, name) or glob_match(exclude, name):
                    continue

                graph = self.get_graph(idx, replace_parents, known=branch_graph.commits)
                graph.validate()

                if all(f in branch_graph.commits for f in graph.tails):
                    branch.add_named_fragment(name, idx, graph)
                else:
                    pass  # orphan branch or new tail commit

        branch.validate()
        return branch

    def get_graph(self, head, replace_parents=None, known=None, report=print):
        if replace_parents is None:
            replace_parents = {}
        if known is None:
            known = {}

        graph = GitGraph.new()
        graph.heads = set([head])

        search = [head]
        walked = set([head])

        while search:
            idx = search.pop(0)

            c = self.get_commit(idx)

            if idx in known:
                c.parents = []
                graph.fragments.add(idx)

            elif idx in replace_parents:
                c.parents = list(replace_parents[idx])
                report("    > replaced", c.parents)

            graph.add_commit(idx, c)

            for pidx in c.parents:
                if pidx not in walked:
                    search.append(pidx)
                    walked.add(pidx)

        missing = set(walked) - set(graph.parents)
        if missing:
            raise Bug("Missing elements")

        return graph

    def get_branch_names(self, branch):
        return branch.graph.get_all_names()

    def clean_tree(self, addr, old_tree, bad_files):
        if bad_files == None:
            return addr, old_tree

        entries = []
        dropped = False
        for i in old_tree.entries:
            name = i[1]
            bad = bad_files.get(name, False)
            if bad is False:
                entries.append(i)
            elif callable(bad):
                out = bad(i[0], i[1], i[2])
                if out:
                    entries.append(out)
                    if out != i:
                        dropped = True
                else:
                    dropped = True
            elif isinstance(bad, dict):
                sub_addr, sub_tree = self.get_tree(i[2])
                new_addr, tree_obj = self.clean_tree(sub_addr, sub_tree, bad)
                if new_addr != i[2]:
                    entries.append((i[0], i[1], new_addr))
                    dropped = True
            elif bad is True:
                dropped = True
                pass  # delete it, if it's an empty hash
            else:
                raise Bug("Bad value in bad_files")
        if not dropped:
            return addr, old_tree
        new_tree = GitTree(entries)
        return self.write_tree(new_tree), new_tree

    def prefix_tree(self, tree, prefix):
        entries = []
        for p in prefix:
            e = (GIT_DIR_MODE, p, tree)
            entries.append(e)
        t = GitTree(entries)
        return self.repo.write_tree(t), t

    def rewrite_branch(self, branch, *, bad_files=None, replace_names=None):
        branch.validate()

        def fix_tree(writer, idx, commit, ctree):
            tree, ctree = self.clean_tree(commit.tree, ctree, bad_files)
            commit.tree = tree
            return commit, ctree

        def fix_names(writer, idx, commit, ctree):
            author, committer, message = commit.author, commit.committer, commit.message

            new_author = replace_names.get(str(author))
            if new_author:
                name, email = new_author[:-1].rsplit(" <")
                commit.author = author.replace(name=name, email=email)

            new_committer = replace_names.get(str(committer))
            if new_committer:
                name, email = new_committer[:-1].rsplit(" <")
                commit.committer = committer.replace(name=name, email=email)

            if "Co-authored-by: " in message:
                lines = []
                for line in message.splitlines():
                    if "Co-authored-by: " in line:
                        line_start, name = line.rsplit("Co-authored-by: ", 1)
                        if name in replace_names:
                            name_email = replace_names.get(name)
                            name, email = name_email[:-1].rsplit(" <")
                            name = f"{name} <{email}>"
                        line = f"{line_start}Co-authored-by: {name}"
                        if line_start or len(lines) == 0 or line != lines[-1]:
                            lines.append(line)

                    else:
                        lines.append(line)

                commit.message = "\n".join(lines)

            return commit, ctree

        rewrite = []
        if bad_files:
            rewrite.append(fix_tree)
        if replace_names:
            rewrite.append(fix_names)

        if not rewrite:
            return branch

        writer = GitWriter(self, branch.name)
        writer.graft(branch, rewrite=rewrite)

        new_branch = writer.to_branch()

        for x, y in zip(branch.graph.walk_children(), new_branch.graph.walk_children()):
            if writer.grafted(x) != y:
                raise Bug("Grafted branch out of sync with input branch")

        return new_branch

    def prefix_branch(self, branch, prefix):
        writer = GitWriter(self, branch.name)

        def fix_tree(writer, idx, commit, ctree):
            tree, ctree = self.prefix_tree(commit.tree, ctree, [prefix])
            commit.tree = tree
            return commit, ctree

        writer.graft(branch, rewrite=(fix_tree,))

        new_branch = writer.to_branch()
        for x, y in zip(branch.graph.walk_children(), new_branch.graph.walk_children()):
            if writer.grafted(x) != y:
                raise Bug("Grafted branch out of sync with input branch")
        return new_branch

    def interweave_branch_heads(self, branches, *, rewrite=()):
        heads = []
        for name, branch in branches.items():
            head = branch.head
            c = branch.graph.commits[head].clone()
            prefix = [name]
            heads.append((head, c, prefix))

        heads.sort(key=lambda x: x[1].max_date)
        entries = []

        prev = None
        for head, commit, prefix in heads:

            tree_idx, old_tree = self.get_tree(commit.tree)

            entries = [e for e in entries if e[1] not in prefix]

            for p in prefix:
                e = (GIT_DIR_MODE, p, tree_idx)
                entries.append(e)

            t = GitTree(entries)
            tidx = self.write_tree(t)

            for callback in rewrite:
                commit, t = callback(self, head, commit, t)

            c = GitCommit(
                tree=tidx,
                parents=([prev] if prev else []),
                author=commit.author,
                committer=commit.committer,
                message=commit.message,
                max_date=commit.max_date,
                author_date=commit.author_date,
                committer_date=commit.committer_date,
            )

            prev = self.write_commit(c)
        return prev

    def interweave_branches(
        self,
        new_name,
        branches,
        named_heads=None,
        merge_named_heads=None,
        prefix_message=None,
    ):
        graph_prefix = {}
        for name, branch in branches.items():
            for c in branch.graph.commits:
                if c not in graph_prefix:
                    graph_prefix[c] = set()
                graph_prefix[c].add(name)

        merged_branch, linear_parent = GitBranch.interweave(
            new_name,
            branches,
            named_heads=named_heads,
            merge_named_heads=merge_named_heads,
        )

        writer = GitWriter(self, new_name)
        start_tree = GitTree([])
        graph = merged_branch.graph
        grafted_trees = {}

        def merge_tree(prev_tree, tree, prefix):
            entries = [e for e in prev_tree.entries if e[1] not in prefix]
            for p in prefix:
                e = (GIT_DIR_MODE, p, tree)
                entries.append(e)
            t = GitTree(entries)
            return self.write_tree(t), t

        def prefix_tree(writer, idx, commit, ctree):
            tree = commit.tree
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Bug("bad prefix, must be set or dict of set")

            if idx in graph.tails:
                tree, ctree = merge_tree(start_tree, tree, prefix)
            else:
                max_parent = max(graph.parents[idx], key=linear_parent.get)
                max_tree = grafted_trees[max_parent]
                tree, ctree = merge_tree(max_tree, tree, prefix)

            grafted_trees[idx] = ctree
            commit.tree = tree

            return commit, ctree

        prefix_message = prefix_message_callbacks.get(prefix_message)

        def prefix_commit(writer, idx, commit, ctree):
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Bug("bad prefix, must be set or dict of set")
            commit.message = prefix_message(commit, ", ".join(sorted(prefix)))
            return commit, ctree

        rewrites = [prefix_tree]

        if prefix_message:
            rewrites.append(prefix_commit)

        writer.graft(merged_branch, rewrite=rewrites)

        new_branch = writer.to_branch()

        for x, y in zip(
            merged_branch.graph.walk_children(), new_branch.graph.walk_children()
        ):
            if writer.grafted(x) != y:
                raise Bug("Grafted branch out of sync with input branch")

        rewrites = [prefix_commit] if prefix_message else []

        check_branch = self.interweave_branch_heads(branches, rewrite=rewrites)
        shallow_c = self.get_commit(check_branch)
        output_c = self.get_commit(new_branch.head)

        if shallow_c.author != output_c.author:
            raise Bug("Branch head inconsistent")
        if shallow_c.committer != output_c.committer:
            raise Bug("Branch head inconsistent")
        if shallow_c.tree != output_c.tree:
            raise Bug("Branch head inconsistent")
        if shallow_c.max_date != output_c.max_date:
            raise Bug("Branch head inconsistent")
        if shallow_c.message != output_c.message:
            raise Bug("Branch head inconsistent")
        # parents will not match
        return new_branch

    def Writer(self, name):
        return GitWriter(self, name)


class GitWriter:
    def __init__(self, repo, name):
        self.repo = repo
        self.name = name
        self.head = None  # maybe support multiple heads as parents
        self.named_heads = {}
        self.grafts = {}
        self.replaces = {}
        self.original = {}
        self.graph = GitGraph.new()

    def grafted(self, idx):
        return self.grafts[idx]

    def save_grafts(self, path):
        with open(path, "w+") as fh:
            out = {k: v.idx for k, v in self.grafts.items}
            json.dump(out, fh, sort_keys=True, indent=2)

    def to_branch(self):
        named_heads = dict(self.named_heads)
        named_heads[self.name] = self.head
        branch = self.graph.to_branch(
            self.name, self.head, named_heads, original=self.original
        )
        branch.named_heads["init"] = branch.tail
        branch.validate()
        return branch

    def graft_commit(self, idx, rewrite=()):
        start_parents = [self.head] if self.head else []

        c = self.repo.get_commit(idx)
        c.tree, ctree = self.repo.get_tree(c.tree)

        if not c.parents:
            c.parents = start_parents
        else:
            c.parents = [self.grafts[p] for p in c.parents]

        for callback in rewrite:
            c, ctree = callback(self, idx, c, ctree)

        cidx = self.repo.write_commit(c)
        self.grafts[idx] = cidx
        self.replaces[cidx] = idx
        self.head = cidx

        self.graph.add_commit(cidx, c)
        self.graph.heads = set([self.head])
        return cidx

    def graft(self, branch, *, rewrite=(), report=print):
        start_parents = [self.head] if self.head else []

        graph = branch.graph
        graph_total = len(graph.commits)
        graph_count = 0

        for idx in graph.walk_children():
            if idx not in self.grafts:
                if idx in graph.fragments:
                    raise Bug("Cannot graft fragment")

                c = graph.commits[idx].clone()
                c.tree, ctree = self.repo.get_tree(c.tree)

                if not c.parents:
                    c.parents = start_parents
                else:
                    c.parents = [self.grafts[p] for p in graph.parents[idx]]

                for callback in rewrite:
                    c, ctree = callback(self, idx, c, ctree)

                cidx = self.repo.write_commit(c)
                self.grafts[idx] = cidx
                self.replaces[cidx] = idx
                self.graph.add_commit(cidx, c)
                self.original[cidx] = branch.original.get(idx, idx)

                if idx in graph.heads:
                    self.graph.heads.add(cidx)

            graph_count += 1

            if graph_count & 512 == 0:
                per = graph_count / graph_total
                report(
                    f"\r    > progress {per:.2%} {graph_count} of {graph_total}", end=""
                )

        per = graph_count / graph_total
        report(f"\r    > progress {per:.2%} {graph_count} of {graph_total}")

        for x in graph.commits:
            if x not in self.grafts:
                raise Bug("Cannot graft fragment")

        self.head = self.grafts[branch.head]
        self.named_heads.update(
            {
                k: self.grafts[v]
                for k, v in branch.named_heads.items()
                if k != branch.name
            }
        )

        for k, v in self.named_heads.items():
            self.repo.get_commit(v)
        return self.head


class GitBuilder:
    def __init__(self, repo, report=print):
        self.repo = repo
        self.fetched = set()
        self.branches = {}
        self.report = report
        self.loaded = {}
        self.replace_names = {}

        # XXX self.stdout
        # XXX def report(self, ...)

    def sort_steps(self, steps):

        deps = {}

        number = {name: n for n, name in enumerate(steps)}

        for name, config in steps.items():
            d = set()
            step = config["step"]
            if step in ("show_branch", "write_branch", "write_branch_names"):
                d.add(config["branch"])
            elif step == "append_branches":
                d.update(config["branches"])
            elif step == "merge_branches":
                for branch_name, branch in config["branches"].items():
                    d.add(branch_name)

            deps[name] = d

        pred = {name: set() for name in steps}

        for name, dset in deps.items():
            for d in dset:
                pred[d].add(name)

        count = {k: len(v) for k, v in deps.items()}
        search = []

        for name in steps:
            if count[name] == 0:
                search.append((number[name], name))

        output = {}

        while search:
            n = min(range(len(search)), key=lambda x: search[x])
            _, name = search.pop(n)
            output[name] = steps[name]

            for d in pred[name]:
                count[d] -= 1
                if count[d] == 0:
                    search.append((number[d], d))

        if set(steps) != set(output):
            raise Bug("Steps missing after sorting")

        if list(steps) == list(output):
            self.report("running steps in given order:", ", ".join(output))
        else:
            self.report("running steps in dependency order:", ", ".join(output))
        self.report()

        return output.items()

    def run(self, steps, refresh=False):
        if refresh:
            self.fetched = set()

        # XXX -toposort
        for name, config in self.sort_steps(steps):
            step = config["step"]
            if step == "add_remote":
                config["refresh"] = refresh
                self.add_remote(name, config)
            elif step == "fetch_branch":
                config["refresh"] = refresh
                self.fetch_branch(name, config)
            elif step == "start_branch":
                self.start_branch(name, config)
            elif step == "merge_branches":
                self.merge_branches(name, config)
            elif step == "append_branches":
                self.append_branches(name, config)
            elif step == "show_branch":
                self.show_branch(name, config)
            elif step == "write_branch":
                self.write_branch(name, config)
            elif step == "write_branch_names":
                self.write_branch_names(name, config)
            else:
                raise Bug(f"Bad step for {name}: {step}")
            self.report()

        return self.branches

    def load_json(self, filename):
        if filename in self.loaded:
            return self.loaded[filename]
        with open(filename, "r+") as fh:
            out = json.load(fh)

        self.loaded[filename] = out
        return out

    def load_badfiles(self, bad_files):
        if bad_files is None:
            return None
        if isinstance(bad_files, dict):
            return bad_files
        elif isinstance(bad_files, str):
            out = self.load_json(bad_files)
            return out
        else:
            raise Bug("unsupported bad_files")

    def load_replacement_names(self, replace_names):
        if replace_names is None:
            return None
        elif isinstance(replace_names, dict):
            pass
        elif isinstance(replace_names, str):
            replace_names = self.load_json(replace_names)
        else:
            raise Bug("unsupported replace_names")

        if id(replace_names) in self.replace_names:
            return self.replace_names[id(replace_names)]

        replace = {}
        for name, r in replace_names.items():
            for x in r:
                replace[x] = name

        self.replace_names[id(replace_names)] = replace

        return replace

    def add_remote(self, name, config):
        remote_name = config.get("remote_name", name)
        url = config["url"]
        refresh = config["refresh"]

        self.report("adding remote", f"{remote_name}")
        created = self.repo.add_remote(remote_name, url)
        # if created or refresh:
        #    self.report(f"    fetching {remote_name} from {url}", end="")
        #    if url not in self.fetched:
        #        # self.repo.fetch_remote(remote_name)
        #        self.fetched.add(url)
        #    self.report()
        # else:
        #    self.report(f"    already fetched {remote_name} from {url}")

    def fetch_branch(self, name, config):
        branch_name = config["default_branch"]
        url = config["remote"]
        refresh = config["refresh"]
        remote_name = f"{name}-origin"

        self.report("loading branch", f"{name}/{branch_name}")

        if self.repo.add_remote(remote_name, url) or refresh:
            self.report(f"    fetching {name} from {url}", end="")

            if url not in self.fetched:
                self.repo.fetch_remote(remote_name)
                self.fetched.add(url)
            self.report()
        else:
            self.report(f"    already fetched {name} from {url}")

        replace = config.get("replace_parents")
        include = config.get("include_branches", True)
        exclude = config.get("exclude_branches", False)

        branch = self.repo.get_remote_branch(
            remote_name,
            branch_name,
            replace_parents=replace,
            include=include,
            exclude=exclude,
        )

        self.report(
            "    >",
            name,
            "has",
            len(branch.named_heads) - 1,
            "related branches",
            end=" ",
        )

        # xxx - maybe don't pre-gen init tags
        branch.named_heads["init"] = branch.tail

        for ref_name, ref_head in config.get("named_heads", {}).items():
            branch.named_heads[ref_name] = ref_head

        self.report(len(branch.graph.commits), "total commits")

        bad_files = config.get("bad_files")
        replace_names = config.get("replace_names")

        if bad_files or replace_names:
            bad_files = self.load_badfiles(bad_files)
            replace_names = self.load_replacement_names(replace_names)
            actions = []
            if bad_files:
                actions.append("removing bad files")
            if replace_names:
                actions.append("replacing names")
            self.report("    rewriting branch:", ", ".join(actions))
            branch = self.repo.rewrite_branch(
                branch, bad_files=bad_files, replace_names=replace_names
            )

        self.branches[name] = branch

    def start_branch(self, name, config):
        first_commit = config["first_commit"]

        timestamp = first_commit.pop("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp).astimezone(timezone.utc)

        self.report("starting empty branch:", name)
        init = self.repo.start_branch(name, timestamp=timestamp, **first_commit)

        writer = self.repo.Writer(name)
        writer.graft(init)

        c = init.graph.commits[init.head]

        branch = writer.to_branch()
        branch.named_heads["head"] = branch.head
        branch.named_heads["init"] = branch.tail
        self.branches[name] = branch

    def merge_branches(self, name, config):
        branches = config["branches"]
        prefix_message = config.get("prefix_message")
        merge_named_heads = config.get("merge_named_heads")
        branches = {k: self.branches[v] for k, v in branches.items()}
        strategy = config.get("merge_strategy", "first-parent")

        if strategy != "first-parent":
            raise Error("only first-parent merges are implemented")

        self.report("creating merged branch:", name, "from", len(branches), "branches")
        branch = self.repo.interweave_branches(
            name,
            branches,
            merge_named_heads=merge_named_heads,
            prefix_message=prefix_message,
        )
        self.branches[name] = branch

    def append_branches(self, name, config):
        branches = config["branches"]
        writer = self.repo.Writer(name)
        self.report("creating new branch:", name, "from ", len(branches), "branches")
        for branch_name in branches:
            self.report("   ", "appending", branch_name)
            writer.graft(self.branches[branch_name])

        branch = writer.to_branch()
        self.branches[name] = branch

    def show_branch(self, name, config):
        branch = self.branches[config["branch"]]
        named_heads = config["named_heads"]

        report = []

        init = branch.tail
        date = branch.graph.commits[init].max_date
        report.append((date, init, "first commit"))

        for name in named_heads:
            new = branch.named_heads[name]
            date = branch.graph.commits[new].max_date
            old = branch.original.get(new, new)
            text = f"{name} (was {old})" if old else name
            report.append((date, new, text))

        self.report(f"branch: {branch.name} (includes {len(branch.named_heads)} heads)")
        self.report()
        for date, nidx, text in sorted(report):
            self.report("   ", nidx, text)
        self.report()
        self.report("   ", "new head:", branch.head)

    def write_branch(self, name, config):
        branch_name = config["branch"]
        branch = self.branches[branch_name]
        prefix = config.get("prefix") + "/" if "prefix" in config else ""

        self.repo.write_branch_head(f"{prefix}/{branch_name}", branch.head)

        named_heads = config.get("named_heads", None)
        include = config.get("include_branches", True)
        exclude = config.get("exclude_branches", False)

        count = 0
        skipped = 0

        if named_heads is None:
            named_heads = {}
            for name, head in branch.named_heads.items():
                if not glob_match(include, name) or glob_match(exclude, name):
                    skipped += 1
                    continue
                if name == branch_name:
                    continue

                named_heads[name] = head

        for name, head in named_heads.items():
            self.repo.write_branch_head(f"{prefix}/{name}", head)
            count += 1

        self.report(f"writing branch '{branch_name}' to '{prefix}{branch_name}'")
        if count or skipped:
            self.report("   ", f"plus {count} branches, skipping {skipped}")

    def write_branch_names(self, name, config):
        branch_name = config["branch"]
        branch = self.branches[branch_name]

        output_filename = config["output_filename"]
        self.report("writing name list", end="")

        names = self.repo.get_branch_names(branch)
        name_list = names.keys()

        self.report(",", len(name_list), "unique names", end="")

        with open(output_filename, "w") as fh:
            for name in sorted(name_list):
                fh.write(f"{name}\n")

        self.report()


def main(name):
    if name != "__main__":
        return

    # cmd run config.file --fetch --skip="..."/
    # cmd git commmand
    # cmd / cmd help

    import sys

    if len(sys.argv) <= 1:
        print(sys.argv[0], "run <...>")
        return

    arg = sys.argv[1]

    if arg == "run":
        name = sys.argv[2]
        refresh = any(x == "--fetch" for x in sys.argv[3:])

        with open(f"{name}.json", "r+") as fh:
            builder_config = json.load(fh)

        git_repo = GitRepo(name)

        builder = GitBuilder(git_repo)

        builder.run(builder_config, refresh=refresh)


main(__name__)

#### speculative work
#
# xxx - general idea of finer grained merges, file based or subdirectory based
#
#       i.e. non_linear_depth[x] = {project:n, project:n}
#       build by walk up from roots, and store a dict of the last 'linear' version of a subtree was
#
# xxx - merging with other histories other than linear
#       and maybe not merging by replacing the first parent
#
# xxx - merging into /file.mine /file.theirs rather than /mine/file, /theirs/file
#
# xxx - merging and allowing a root subdirectory
#
# xxx - GitGraph properties
#       graph.properties = set([monotonic, monotonic-author, monotonic-committer])
#
# xxx - preserving weird git headers (?) or change-ids
