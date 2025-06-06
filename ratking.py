#!env/bin/python3

import functools
import glob
import json
import os.path
import re
import subprocess
import sys

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import translate as glob_to_regex
from heapq import heappush, heappop

try:
    import pygit2
except ImportError:
    if sys.version_info.major < 3 or sys.version_info.minor < 13:
        raise Exception("python3.13 and pygit2 must be installed")
    else:
        raise Exception("pygit2 must be installed")
else:
    if sys.version_info.major < 3 or sys.version_info.minor < 13:
        raise Exception("python 3.13 is required")


## constants

INCLUDE_ORPHANS = True # include branches with orphan init commits
TIME_TRAVEL = False

GIT_DIR_MODE = 0o040_000
GIT_FILE_MODE = 0o100_644
GIT_FILE_MODE2 = 0o100_664
GIT_EXEC_MODE = 0o100_755
GIT_LINK_MODE = 0o120_000
GIT_GITLINK_MODE = 0o160_000  # actually a submodule, blegh
GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # Wow, isn't git amazing.


## errors

class Bug(Exception):
    pass


class Error(Exception):
    pass


class TimeTravel(Exception):
    pass


class MergeError(Exception):
    pass

# helpers

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


def sibling(base_file, filename):
    return os.path.join(os.path.dirname(base_file), filename)


# ratking library: Signature, Graph, Branch, and Repo objects


@dataclass
class GitSignature:
    """A name, email, timestamp tuple that represents a Committer
    or Author header inside a git commit"""

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
    tree: str  # can also be GitTree
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
    """A set of connected commits"""

    commits: dict
    tails: set
    heads: set
    parents: dict
    children: dict
    fragments: set
    parent_count: dict
    child_count: dict

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

        # walk forward from tails
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

    def min_linear_parent(self, history):
        """ youngest commit this descends from on the given history """
        linear_parent = {c: n for n, c in enumerate(history, 1)}
        return self.walk_linear_parent(history, linear_parent)

    def max_linear_parent(self, history):
        linear_parent = {c: n for n, c in enumerate(history, 1)}
        return self.walk_linear_parent(reversed(history), linear_parent)

    def walk_linear_parent(self, history, linear_parent):
        for lc in history:
            n = linear_parent[lc]
            search = list(self.children.get(lc, ()))
            while search:
                c = search.pop(0)
                if c not in linear_parent:
                    linear_parent[c] = n
                    search.extend(self.children.get(c, ()))

        for f in self.tails:
            n = linear_parent.get(f, 0)
            linear_parent[f] = n
            search = list(self.children.get(f, ()))
            while search:
                c = search.pop(0)
                if c not in linear_parent:
                    linear_parent[c] = n
                    search.extend(self.children.get(c, ()))

        return linear_parent

    def min_linear_children(self, history):
        linear_children = {c: n for n, c in enumerate(history, 1)}
        return self.walk_linear_children(history, linear_children)

    def max_linear_children(self, history):
        linear_children = {c: n for n, c in enumerate(history, 1)}
        return self.walk_linear_children(reversed(history), linear_children)

    def walk_linear_children(self, history, linear_children):
        for lc in history:
            n = linear_children[lc]
            search = list(self.parents.get(lc, ()))
            while search:
                c = search.pop(0)
                if c not in linear_children:
                    linear_children[c] = n
                    search.extend(self.parents.get(c, ()))

        for f in self.heads:
            n = linear_children.get(f, 0)
            linear_children[f] = n
            search = list(self.parents.get(f, ()))
            while search:
                c = search.pop(0)
                if c not in linear_children:
                    linear_children[c] = n
                    search.extend(self.parents.get(c, ()))

        return linear_children


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

    def first_parents(self):
        return self.graph.first_parents(self.head)

    @classmethod
    def merge_linear_history(self,  branches, report):
        graphs = {}
        branch_history = {}
        branch_linear_parent = {}

        for name, branch in branches.items():
            graph = branch.graph
            graphs[name] = graph

            history = graph.first_parents(branch.head)
            branch_history[name] = history
            branch_linear_parent[name] = graph.max_linear_parent(history)

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

        if not TIME_TRAVEL:
            seen = set()
            new_history1 = []

            for h in branch_history.values():
                new_history1.extend(x for x in h if x not in seen)
                seen.update(h)

            # n.b. python sort is stable
            new_history1.sort(key=lambda idx: merged_graph.commits[idx].max_date)

            if new_history != new_history1:
                raise TimeTravel("bad")


        # scrap bad commits
        skipped = set()
        skipped_orig = {}

        for name, branch in branches.items():
            graph = branch.graph
            history = branch_history[name]
            history_set = set(history)
            linear_parent = branch_linear_parent[name]
            linear_children = graph.max_linear_children(history)

            conflicts = []
            
            # any commit in the merged history that exists in the branch graph
            # but not on the branch's linear history is a conflict

            for h in new_history:
                if h in graph.commits and h not in history_set:
                    lc = linear_children[h]
                    lp = linear_parent[h]
                    conflicts.append((h, lp, lc))
                    # h has newest ancestor lp, and earliest descendent lc on the history
                    # so any x on the history lp < x < lc is in conflict


            count = 0
            for h, lp, lc in conflicts:
                for x in range(lp+1, lc):
                    c = branch_history[name][x-1]
                    if c not in skipped:
                        count+=1
                        skipped.add(c)
                        skipped_orig[c] = branch.original.get(c, c)

            if count > 0:
                report("   ", name, "has", count, "commits with conflict with", len(conflicts), "commits shared with other branches")

        for name, branch in branches.items():
            if branch.head in skipped:
                raise MergeError(f"branch head {name} in conflict")

        new_history = [h for h in new_history if h not in skipped]

        if skipped:
            report("   ", "skipped", ", ".join(s for s in skipped_orig.values()))

        head = new_history[-1]
        tail = new_history[0]

        if merged_graph.commits[tail].parents:
            raise Bug("Merged graph has tail with parent commits")
        if merged_graph.parents[tail]:
            raise Bug("Merged graph has tail with parent commits in graph")


        # final linear history check

        for name, branch in branches.items():
            h = [x for x in branch_history[name] if x not in skipped]
            h.reverse()
            g = branch.graph

            for idx in new_history:
                if idx in g.commits:
                    if idx == h[-1]:
                        h.pop()
            if h:
                raise MergeError(
                    f"New merged history does not contain all commits from {name} to be merged"
                )

        new_parents = {}
        prev = new_history[0]
        for idx in new_history[1:]:
            new_parents[idx] = prev
            prev=idx

        new_tails = set(new_history)

        for name, branch in branches.items():
            graph = branch.graph
            graphs[name] = graph

            for idx in graph.tails:
                if idx in new_tails:
                    continue

                p = None
                i_date = graph.commits[idx].max_date

                for h in new_history:
                    if h not in graph.commits:
                        continue
                    h_date =  graph.commits[h].max_date
                    if h_date < i_date:
                        p = h
                    else:
                        break
                if p is not None:
                    report("   ",'folding tail', branch.original.get(idx,idx), "after", branch.original.get(p,p))
                    new_parents[idx] = p
                    new_tails.add(idx)

        return new_history, branch_history, branch_linear_parent, merged_graph, new_parents

    @classmethod
    def interweave(
        cls,
        new_name,
        branches,
        commit_prefix,
        *,
        fix_message=None,
        named_heads=None,
        merge_named_heads=(),
        report=None
    ):

        # create a new linear history
        history, branch_history, branch_linear_parent, merged_graph, new_parents = (
            cls.merge_linear_history(branches, report)
        )

        head, tail = history[-1], history[0]

        # rewrite the parents
        # note: could remove original linear parent, and not create a merge commit

        for idx, prev in new_parents.items():
            old_parents = merged_graph.commits[idx].parents
            new_parents = [prev] + [o for o in old_parents if o != prev]

            merged_graph.parents[idx] = list(new_parents)
            merged_graph.commits[idx].parents = list(new_parents)
            merged_graph.parent_count[idx] = len(new_parents)

            merged_graph.children[prev].add(idx)
            merged_graph.child_count[prev] = len(merged_graph.children[prev])

            if idx in merged_graph.tails:
                merged_graph.tails.remove(idx)

            if prev in merged_graph.heads:
                merged_graph.heads.remove(prev)


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

        linear_parent = merged_graph.max_linear_parent(history)

        # validate linear parents

        linear_depth = [linear_parent[x] for x in history]

        if linear_depth != sorted(linear_depth):
            raise Bug("New merged history is numbered out of order")

        if set(merged_graph.commits) != set(linear_parent):
            raise Bug("linear parent for graph does not contain all graph elements")

        # ensure linear ordering of source branches is preserved in merged branch

        # XXX - missing
        # issue was that some items dropped from history
        # but should check that "max_parent" is the same

        # fix commits

        def merge_tree(prev_tree, tree, prefix):
            entries = [e for e in prev_tree.entries if e[1] not in prefix]
            for p in prefix:
                e = (GIT_DIR_MODE, p, tree)
                entries.append(e)
            return GitTree(entries)

        start_tree = GitTree([])
        grafted_trees = {}

        for idx in merged_graph.walk_children():
            prefix = commit_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Bug("bad prefix, must be set or dict of set")

            commit = merged_graph.commits[idx]

            if idx in merged_graph.tails:
                commit.tree = merge_tree(start_tree, commit.tree, prefix)
            else:
                max_parent = max(merged_graph.parents[idx], key=linear_parent.get)
                max_tree = grafted_trees[max_parent]
                commit.tree = merge_tree(max_tree, commit.tree, prefix)

            grafted_trees[idx] = commit.tree

            if fix_message:
                commit.message = fix_message(commit, ", ".join(sorted(prefix)))

        # create the branch:
        # - named heads
        # - original

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
        return branch


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
    def __init__(self, repo_dir, *, bare=True, report=None):
        if os.path.exists(repo_dir):
            self.git = pygit2.Repository(repo_dir)
        else:
            self.git = pygit2.init_repository(repo_dir, bare=bare)
        self.path = self.git.path
        self._all_remotes = None
        self.report = report if report else lambda *a, **k: None

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
            addr = self.write_tree(addr)

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

    def get_branch(self, branch_name, include=None, exclude=None, replace_parents=None):
        branch_head = self.get_branch_head(branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}
        branch = branch_graph.to_branch(branch_name, branch_head, named_heads, {})
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

                if INCLUDE_ORPHANS:
                    if any(f in branch_graph.commits for f in graph.tails):
                        branch.add_named_fragment(name, idx, graph)
                else:
                    if all(f in branch_graph.commits for f in graph.tails):
                        branch.add_named_fragment(name, idx, graph)

        branch.validate()
        return branch

    def get_graph(self, head, replace_parents=None, known=None):
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
                self.report("    > replaced", c.parents)

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

    def reparent_branch(self, branch):
        self.report("   ", "reparenting branch")
        reachable = {}

        for idx in branch.graph.walk_children():
            r = set()

            p = branch.graph.parents[idx]
            if p:
                r.update(reachable[p[0]])
            r.add(idx)
            reachable[idx] = r

        old_history = branch.graph.first_parents(branch.head)

        writer = GitWriter(self, branch.name)

        bump = 0

        def reparent(writer, idx, commit, ctree):
            nonlocal bump
            if len(commit.parents) > 1:
                first = commit.parents[0]
                f = writer.replaces[first]

                out = []
                for x in commit.parents[1:]:
                    y = writer.replaces[x]
                    if f in reachable[y]:
                        out.append(first)
                        first, f = x, y
                    else:
                        out.append(x)
                new_parents = [first] + out
                if new_parents != commit.parents:
                    bump +=1
                commit.parents = new_parents
            return commit, ctree

        writer.graft(branch, rewrite=(reparent,))
        self.report("   ", "changed", bump, "commits")

        new_branch = writer.to_branch()

        for x, y in zip(branch.graph.walk_children(), new_branch.graph.walk_children()):
            if writer.grafted(x) != y:
                raise Bug("Grafted branch out of sync with input branch")

        new_history = new_branch.graph.first_parents(new_branch.head)

        self.report("   ", "history was", len(old_history), "now", len(new_history))

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

    def interweave_branch_heads(self, branches, *, rewrite=(), fix_message=None):
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

            parents = [prev] if prev else []
            parents.extend(p for p in commit.parents if p != prev)

            commit.parents = parents

            if fix_message:
                commit.message = fix_message(commit, ", ".join(sorted(prefix)))

            for callback in rewrite:
                commit, t = callback(self, head, commit, t)

            c = GitCommit(
                tree=tidx,
                parents=parents,
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
        fix_message=None,
    ):

        # XXX - fix message not applied properly to commits that are interwoven
        #       as they obtain a new parent and sometimes turn into merge commits

        graph_prefix = {}

        branch_tails = set()
        for name, branch in branches.items():
            for c in branch.graph.commits:
                if c not in graph_prefix:
                    graph_prefix[c] = set()
                graph_prefix[c].add(name)

            branch_tails.add(branch.tail)

        merged_branch = GitBranch.interweave(
            new_name,
            branches,
            graph_prefix,
            fix_message=fix_message,
            named_heads=named_heads,
            merge_named_heads=merge_named_heads,
            report=self.report,
        )

        writer = GitWriter(self, new_name)
        rewrites = []

        writer.graft(merged_branch, rewrite=rewrites)

        new_branch = writer.to_branch()

        for x, y in zip(
            merged_branch.graph.walk_children(), new_branch.graph.walk_children()
        ):
            if writer.grafted(x) != y:
                raise Bug("Grafted branch out of sync with input branch")

        check_branch = self.interweave_branch_heads(branches, fix_message=fix_message)
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
        if self.name:
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

    def graft(self, branch, *, rewrite=()):
        report = self.repo.report
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


# ratking cli program

@dataclass
class Step:
    name: str
    cmd: str
    config: dict

    def load_dependencies(self, result):
        config =  dict(self.config)
        cmd = self.cmd.name
        if cmd in ("show_branch", "write_branch", "write_branch_names", "reparent_branch"):
            config["branch"] = result[config["branch"]]
        elif cmd == "append_branches":
            config["branches"] = [result[x] for x in config["branches"]]
        elif cmd == "merge_branches":
            config["branches"] = {k:result[x] for k,x in config["branches"].items()}

        return config

    def dependencies(self):
        deps = set()
        cmd, config = self.cmd.name, self.config
        if cmd in ("show_branch", "write_branch", "write_branch_names", "reparent_branch"):
            deps.add(config["branch"])
        elif cmd == "append_branches":
            deps.update(config["branches"])
        elif cmd == "merge_branches":
            for branch_prefix, branch in config["branches"].items():
                deps.add(branch)
        return deps


    @classmethod
    def make_step_config(cls, config_dir, config):
        """ load json, update filenames """
        replace = {k[:-5]:v for k,v in config.items() if k.endswith(".json")}

        loaded = {}

        for name, filename in replace.items():
            filename = os.path.join(config_dir, filename)
            if filename not in loaded:
                with open(filename, "r+") as fh:
                    out = json.load(fh)
                loaded[filename] = out
            config[name] = loaded[filename]
            del config[f"{name}.json"]

        replace = {k:v for k,v in config.items() if k in ("path", "filename", "output_filename")}

        for name, value in replace.items():
            config[name] = os.path.join(config_dir, value)

        return config

    @classmethod
    def load_steps(cls, filename, commands):
        config_dir = os.path.dirname(filename)

        with open(filename, "r+") as fh:
            raw_config = json.load(fh)

        steps = []

        if isinstance(raw_config, list):
            for step_config in raw_config:
                name = step_config.pop("name", None)
                cmd = commands[step_config.pop("cmd")]
                step_config = cls.make_step_config(config_dir, step_config)
                steps.append( Step(name, cmd, step_config))

        elif isinstance(raw_config, dict):
            for name, step_config in raw_config.step_config():
                cmd = commands[step_config.pop("cmd")]
                step_config = cls.make_step_config(config_dir, step_config)
                steps.append( Step(name, cmd, step_config))
        else:
            raise Error("builder config is of unsupported type")

        return steps

    @classmethod
    def sort_steps(cls, steps):
        """Given a list of build_steps, this method returns
        a new list in a stable topologically sorted order. If the
        list is already in order, it will return the same value.
        """

        search = []
        numbers = {}
        after = defaultdict(set)
        count = {}

        numbers = {s.name:n for n,s in enumerate(steps) if s.name}

        for n, step in enumerate(steps):
            deps = step.dependencies()
            count[n] = len(deps)
            if count[n] == 0:
                search.append(n)
            for name in deps:
                after[numbers[name]].add(n)

        output = []

        while search:
            # of the work ready to go, i.e no dependencies left
            # we pick the step that appeared earliest in the file
            idx = heappop(search)
            output.append(steps[idx])

            for d in after[idx]:
                count[d] -= 1
                if count[d] == 0:
                    heappush(search, d)

        if len(steps) != len(output):
            raise Bug("Steps missing after sorting")

        return output

    @classmethod
    def run_steps(cls, repo,  steps, refresh=False, report=None):
        fetched = set()
        report = report if report else lambda *a, **k: None

        sorted_steps = Step.sort_steps(steps)
        names = ", ".join(s.name or s.cmd.name for s in sorted_steps)

        if steps == sorted_steps:
            report("running steps in given order:", (names))
        else:
            report("running steps in dependency order:", (names))
        report()

        result = {}
        for step in sorted_steps:

            callback = step.cmd
            config = step.load_dependencies(result)
            config["refresh"] = refresh
            config["fetched"] = fetched
            config["report"] = report

            output = callback(repo, step.name, config)
            report()
            if step.cmd == "load_repository":
                repo = output
            elif step.name:
                result[step.name] = output

        return repo, result


class Callbacks:
    def __init__(self):
        self.callbacks = {}

    def canon(self, name):
        return name.replace(".", "_").replace("-", "_")

    def __contains__(self, name):
        return self.canon(name) in self.callbacks

    def get(self, name, default=None):
        return self.callbacks.get(self.canon(name), default)

    def __getitem__(self, name):
        return self.callbacks[self.canon(name)]

    def add(self, name):
        def _add(fn):
            c = self.canon(name)
            self.callbacks[c] = fn
            fn.name = c
            return fn

        return _add

class App:
    commands = Callbacks()

    @classmethod
    def add_command(self, name):
        return self.commands.add(name)

    @classmethod
    def main(cls, name=__name__):
        if name != "__main__":
            return

        def report(*ar, **kw):
            print(*ar, **kw, flush=True)

        # cmd run config.file --fetch --skip="..."/
        # cmd git commmand
        # cmd / cmd help

        arg = sys.argv[1] if len(sys.argv) > 0 else None
        git_repo = None

        if not arg:
            report(sys.argv[0], "build <...>")
            sys.exit(-1)
        elif arg == "build":
            filename = sys.argv[2]
            refresh = any(x == "--fetch" for x in sys.argv[3:])

            if not filename.endswith(".json"):
                filename = f"{filename}.json"

            steps = Step.load_steps(filename, cls.commands)

            repo_step = any(a.cmd == "load_repository" for a in steps)

            if not repo_step:
                fn = filename.replace(".json", ".git")
                git_repo = GitRepo(fn, report=report)
                git_repo.report("opened default repo:", git_repo.git.path, end="\n\n")

            Step.run_steps(git_repo, steps, refresh=refresh, report=report)
        elif arg == "merge_branches":
            path = None
            cwd = os.getcwd()
            while cwd:
                if os.path.exists(os.path.join(cwd, ".git")):
                    path = cwd
                    break

                cwd, last = os.path.split(cwd)
                if not last:
                    break

            if path != cwd:
                print(path, cwd)
                raise Exception("Run inside the root directory of the repository, thanks.")

            git_repo = GitRepo(path, report=report)

            steps = []

            raw_config = sys.argv[2:]
            target = branches.pop()

            for name in branches:
                steps.append(Step(name, "load_branch", {}, set()))

            args = {"branches": {n: n for n in branches}}
            steps.append( Step(target, cls.commands["merge_branches"], args, set(branches)))
            steps.append( Step(
                f"{target}-output",
                cls.command["write_branch"],
                {"branch": target, "include_branches": False},
                set(branches),
            ))
            Step.run_steps(git_repo, steps,  report=report)



prefix_message_callbacks = Callbacks()


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


def invert_replacement_names(replace_names):
    replace = {}
    for name, r in replace_names.items():
        for x in r:
            replace[x] = name

    return replace

@App.add_command("load_repository")
def load_repository(repo, name, config):
    path = config.get("filename", name)
    bare = config.get("bare", True)
    report = config["report"]
    repo = GitRepo(path, bare=bare, report=report)
    report("opened repo from config:", repo.path)
    return repo

@App.add_command("add_remote")
def add_remote(repo, name, config):
    remote_name = config.get("remote_name", name)
    url = config["url"]
    refresh = config["refresh"]

    repo.report("adding remote:", f"{remote_name}")
    created = repo.add_remote(remote_name, url)

@App.add_command("load_branch")
def fetch_branch(repo, name, config):
    branch_name = config.get("branch_name", name)

    repo.report("loading branch", f"{branch_name}")

    replace = config.get("replace_parents")
    include = config.get("include_branches", True)
    exclude = config.get("exclude_branches", False)

    branch = repo.get_branch(
        branch_name,
        replace_parents=replace,
        include=include,
        exclude=exclude,
    )

    repo.report(
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

    repo.report(len(branch.graph.commits), "total commits")

    bad_files = config.get("bad_files")
    replace_names = config.get("replace_names")

    if bad_files or replace_names:
        replace_names = invert_replacement_names(replace_names)
        actions = []
        if bad_files:
            actions.append("removing bad files")
        if replace_names:
            actions.append("replacing names")
        repo.report("    rewriting branch:", ", ".join(actions))
        branch = repo.rewrite_branch(
            branch, bad_files=bad_files, replace_names=replace_names
        )

    return branch

@App.add_command("fetch_remote_branch")
def fetch_remote_branch(repo, name, config):
    branch_name = config["default_branch"]
    url = config["remote"]
    refresh = config["refresh"]
    remote_name = f"{name}-origin"

    repo.report("loading branch", f"{name}/{branch_name}")

    if repo.add_remote(remote_name, url) or refresh:
        repo.report(f"    fetching {name} from {url}", end="")

        if url not in config["fetched"]:
            repo.fetch_remote(remote_name)
            config["fetched"].add(url)
        repo.report()
    else:
        repo.report(f"    already fetched {name} from {url}")

    replace = config.get("replace_parents")
    include = config.get("include_branches", True)
    exclude = config.get("exclude_branches", False)

    branch = repo.get_remote_branch(
        remote_name,
        branch_name,
        replace_parents=replace,
        include=include,
        exclude=exclude,
    )

    repo.report(
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

    repo.report(len(branch.graph.commits), "total commits")

    bad_files = config.get("bad_files")
    replace_names = config.get("replace_names")

    if bad_files or replace_names:
        replace_names = invert_replacement_names(replace_names)
        actions = []
        if bad_files:
            actions.append("removing bad files")
        if replace_names:
            actions.append("replacing names")
        repo.report("    rewriting branch:", ", ".join(actions))
        branch = repo.rewrite_branch(
            branch, bad_files=bad_files, replace_names=replace_names
        )

    if config.get("reparent_branch"):
        branch = repo.reparent_branch(branch)

    count = sum(len(x) > 1 for x in branch.graph.parents.values())
    repo.report("   ", "total commits", len(branch.graph.commits), "total merge commits", count)

    return branch

@App.add_command("reparent_branch")
def reparent_branch(repo, name, config):
    branch= config["branch"]
    repo.report("reparenting branch", branch.name)

    branch = repo.reparent_branch(
        branch,
    )
    return branch


@App.add_command("start_branch")
def start_branch(repo, name, config):
    first_commit = config["first_commit"]

    timestamp = first_commit.pop("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp).astimezone(timezone.utc)

    repo.report("starting empty branch:", name)
    init = repo.start_branch(name, timestamp=timestamp, **first_commit)

    writer = repo.Writer(name)
    writer.graft(init)

    c = init.graph.commits[init.head]

    branch = writer.to_branch()
    # branch.named_heads["head"] = branch.head
    branch.named_heads["init"] = branch.tail
    return branch

@App.add_command("merge_branches")
def merge_branches(repo, name, config):
    branches = config["branches"]
    prefix_message = config.get("prefix_message")
    merge_named_heads = config.get("merge_named_heads")
    strategy = config.get("merge_strategy", "first-parent")

    if strategy != "first-parent":
        raise Error("only first-parent merges are implemented")

    fix_message = None
    if prefix_message:
        fix_message = prefix_message_callbacks.get(prefix_message, None)

    repo.report("creating merged branch:", name, "from", len(branches), "branches")
    branch = repo.interweave_branches(
        name,
        branches,
        merge_named_heads=merge_named_heads,
        fix_message=fix_message,
    )
    return branch

@App.add_command("append_branches")
def append_branches(repo, name, config):
    branches = config["branches"]
    writer = repo.Writer(name)
    repo.report("creating new branch:", name, "from ", len(branches), "branches")
    for branch in branches:
        repo.report("   ", "appending", branch.name)
        writer.graft(branch)

    branch = writer.to_branch()
    return branch

@App.add_command("show_branch")
def show_branch(repo, name, config):
    branch = config["branch"]
    named_heads = config["named_heads"]
    graph = branch.graph

    report = []

    init = branch.tail
    date = branch.graph.commits[init].max_date
    report.append((date, init, "first commit"))

    for name in named_heads:
        new = branch.named_heads[name]
        date = graph.commits[new].max_date
        old = branch.original.get(new, new)
        text = f"{name} (was {old})" if old else name
        report.append((date, new, text))

    repo.report(f"branch: {branch.name} (includes {len(branch.named_heads)} heads)")
    repo.report()
    for date, nidx, text in sorted(report):
        repo.report("   ", nidx, text)
    repo.report()
    repo.report("   ", "new head:", branch.head)
    repo.report()
    count = sum(len(x) > 1 for x in graph.parents.values())
    repo.report("   ", "total commits", len(graph.commits), "total merge commits", count)

@App.add_command("write_branch")
def write_branch(repo, name, config):
    branch = config["branch"]
    prefix = config["prefix"] + "/" if "prefix" in config else ""

    repo.report(f"writing branch '{branch.name}' to '{prefix}{branch.name}'")
    repo.write_branch_head(f"{prefix}{branch.name}", branch.head)

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
            if name == branch.name:
                continue

            named_heads[name] = head

    for name, head in named_heads.items():
        repo.write_branch_head(f"{prefix}/{name}", head)
        count += 1

    if count or skipped:
        repo.report("   ", f"plus {count} branches, skipping {skipped}")

@App.add_command("write_branch_names")
def write_branch_names(repo, name, config):
    branch = config["branch"]

    output_filename = config["output_filename"]
    repo.report("writing name list", end="")

    names = repo.get_branch_names(branch)
    name_list = names.keys()

    repo.report(",", len(name_list), "unique names", end="")

    with open(output_filename, "w") as fh:
        for name in sorted(name_list):
            fh.write(f"{name}\n")

    repo.report()


App.main()
