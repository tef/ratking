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
GIT_GITLINK_MODE = 0o160_000 # actually a submodule, blegh
GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904" # Wow, isn't git amazing.


@functools.cache
def compile_pattern(pattern):
    regex = glob.translate(pattern, recursive=True)
    return re.compile(regex)

def glob_match(pattern, string):
    if pattern is None:
        return None
    rx = compile_pattern(pattern)
    return rx.match(string) is not None


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
        return all((
            self.tree == other.tree,
            self.parents == other.parents,
            self.max_date == other.max_date,
            self.author == other.author,
            self.committer == other.committer,
            self.message == other.message,
        ))

@dataclass
class GitTree:
    entries: list

@dataclass
class GitGraph:
    commits: set
    tails: set
    heads: set
    parents: dict
    parent_count: dict
    children: dict
    child_count: dict
    fragments: set

    def clone(self):
        return GitGraph(
            commits = dict(self.commits),
            tails = set(self.tails),
            heads = set(self.heads),
            children = {k:set(v) for k,v in self.children.items()},
            parents = {k:list(v) for k,v in self.parents.items()},
            parent_count = dict(parent_count),
            child_count = dict(child_count),
            fragments = set(self.fragments),
        )

    def walk_tails(self):
        search = list(self.tails)
        counts = dict(self.parent_count)

        while search:
            c = search.pop(0)
            yield c

            for i in self.children[c]:
                counts[i] -= 1
                if counts[i] == 0:
                    search.append(i)

    def walk_heads(self):
        search = list(self.heads)
        counts = dict(self.child_count)

        while search:
            c = search.pop(0)
            yield c

            for i in self.parents[c]:
                counts[i] -= 1
                if counts[i] == 0:
                    search.append(i)

    def add_fragment(self, other):
        for idx in other.commits:
            if idx not in self.commits:
                c = other.commits[idx]
                self.commits[idx] = c
                self.parents[idx] = other.parents[idx]
                self.parent_count[idx] = other.parent_count[idx]
                for p in self.parents[idx]:
                    if p in self.heads:
                        self.heads.remove(p)
            else:
                if idx not in other.fragments:
                    raise Exception("nope")

            if idx not in self.children:
                self.children[idx] = set()
                self.child_count[idx] = 0

            for c in other.children[idx]:
                self.children[idx].add(c)
                self.child_count[idx] = len(self.children[idx])

        for f in other.tails:
            # don't merge in graphs with new init commits
            # XXX we could allow this? but for now it may prevent bugs
            if not self.parents[f] and f not in self.tails:
                raise Exception("error")

        for l in other.heads:
            if not self.children[l]:
                self.heads.add(l)

        return self

    def validate(self):
        # all tails items have no parents

        for f in self.tails:
            fp = self.parents[f]
            cp = self.commits[f].parents
            if cp:
                raise Exception("bad: tails has parent in commit")
            if fp:
                raise Exception("bad: tails has parent[] ... ")

        for f in self.fragments:
            if f not in self.tails:
                raise Exception("fragment not in tails")
            if self.commits[f].parents:
                raise Exception("fragment with parents")
            if self.parents[f]:
                raise Exception("fragment with parents")
            # fragment can have children if it's found from branch head
            # or no children if it's just the branch head

        # validate children

        inverted_children = {}

        for idx in self.commits:
            children = self.children[idx]
            if not children:
                if idx not in self.heads:
                    raise("untracked head")
            for child in children:
                if child not in inverted_children:
                    inverted_children[child] = set()
                inverted_children[child].add(idx)

                m = [i for i in self.parents[child] if i == idx]
                if len(m) != 1 or m[0] != idx:
                    #print(c.parents, m)
                    print("parent commit", idx)
                    print("child commit", child, "parents", c.parents)
                    raise Exception("child listed, not in parent")

        # validate parents

        for c in self.commits:
            commit_parents = self.commits[c].parents
            graph_parents = self.parents[c]
            child_parents  = inverted_children.get(c, set())

            if commit_parents != graph_parents:
                raise Exception(f"bad parents: {c}")

            if child_parents != set(graph_parents):
                raise Exception(f"missing children: {c}")

            if len(set(graph_parents)) != len(graph_parents):
                raise Exception("dupe parent")

            if self.parent_count[c] != len(self.parents[c]):
                raise Exception("bad count")

            if self.child_count[c] != len(self.children[c]):
                raise Exception("bad count")

        # walk backwards from heads

        found_tails = set()
        walked = set()

        for c in self.walk_heads():
            walked.add(c)
            if not self.parents[c]:
                found_tails.add(c)

        if found_tails != self.tails:
            raise Error("missing tails")

        if walked != set(self.commits):
            raise Exception("missing commits")

        # walk forward from talks
        # validate complete walk through children

        walked = set()
        heads = set()

        for i in self.walk_tails():
            walked.add(i)
            if not self.children[i]:
                heads.add(i)

        if heads != self.heads:
            print("extra", heads-self.heads)
            print("missing", self.heads-heads)

            raise Exception("heads is not correct")

        if walked != set(self.commits):
            print("commits", len(self.commits), "walked", len(walked))
            raise Exception("missing commits")


    @classmethod
    def union(cls, graphs):
        all_commits = {}
        all_tails = set()
        all_heads = set()
        all_children = {}
        all_parents = {}
        all_parent_count = dict()
        all_child_count = dict()
        all_fragments = set()

        for name, graph in graphs.items():
            for idx, c in graph.commits.items():
                if idx in graph.fragments: # skip fake commit
                    if idx not in all_commits:
                        all_fragments.add(idx)
                    continue

                if idx in all_fragments:
                    all_fragments.remove(idx)

                if idx not in all_commits:
                    all_commits[idx] = c
                    all_parents[idx] = graph.parents[idx]
                    all_parent_count[idx] = graph.parent_count[idx]

                else:
                    o = all_commits[idx]
                    if c != o:
                        raise Exception("dupe??")
                    if all_parents[idx] != graph.parents[idx]:
                        raise Exception("dupe??")

                if idx not in all_children:
                    all_children[idx] = set()
                    all_child_count[idx] = 0

                if graph.children[idx]:
                    all_children[idx].update(graph.children[idx])
                    all_child_count[idx] = len(all_children[idx])
                    if idx in all_heads:
                        all_heads.remove(idx)

            all_tails.update(f for f in graph.tails if f not in graph.fragments)
            all_heads.update(t for t in graph.heads if not all_children[t])

        return cls(
            commits = all_commits,
            tails = all_tails,
            heads = all_heads,
            children = all_children,
            parents = all_parents,
            parent_count = all_parent_count,
            child_count = all_child_count,
            fragments = all_fragments,
        )

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

    def Branch(self, name, head, named_heads):
        history = self.first_parents(head)

        date = self.commits[history[0]].max_date
        for i in history[1:]:
            new_date = self.commits[i].max_date
            if new_date < date:
                raise Exception("time travel")

        linear_parent = GitBranch.make_linear_parent(history, self.tails, self.children)

        if set(self.commits) != set(linear_parent):
            raise Exception(f'bad {len(self.commits)} {len(linear_parent)}')

        return GitBranch(
            name = name,
            head = head,
            graph = self,
            tail = history[0],
            linear = history,
            linear_parent = linear_parent,
            named_heads = named_heads,
        )



@dataclass
class GitBranch:
    name: str
    head: str
    graph: object
    named_heads: dict
    tail: str
    linear: dict
    linear_parent: dict

    def clone(self):
        return GitBranch(
            name = str(name),
            head = str(self.head),
            graph = graph.clone(),
            tail = str(self.tail),
            named_heads = dict(self.named_heads),
            linear = list(self.linear),
            linear_parent = dict(self.linear_parent),
        )

    def common_ancestor(self, right):
        left = self
        left_children, right_children = left.graph.children, right.graph.children

        before, after = None, None
        for x, y in zip(left.linear, right.linear):
            if x != y:
                after = y
                break

            if left_children[x] != right_children[y]:
                after = y
                break

            before = x
            # XXX: skip consolidating, as more branch history
        return before, after


    def validate(self):
        self.graph.validate()
        graph = self.graph

        for name, idx in self.named_heads.items():
            if idx not in graph.commits:
                raise Exception("missing head")

        if self.head not in graph.commits:
            raise Exception("missing head")

        if self.tail not in graph.commits:
            raise Exception("missing tail")

        history = graph.first_parents(self.head)

        if history[0] not in graph.tails:
            raise Exception("what")
        if history[-1] != self.head:
            raise Exception("how")

        if history != self.linear:
            raise Exception("wrong linear")

        history.sort(key=lambda idx: self.graph.commits[idx].max_date)

        if history != self.linear:
            raise Exception("time travel")

        linear_parent = GitBranch.make_linear_parent(self.linear, graph.tails, graph.children)

        if linear_parent != self.linear_parent:
            print("extra keys",set(linear_parent) - set(self.linear_parent))
            print("missing keys",set(self.linear_parent) - set(linear_parent))

            raise Exception("missing linear parent")

        # validate linear_depth through counts

        linear_depth = {f: self.linear_parent[f] for f in self.linear}
        linear_depth.update({f: self.linear_parent[f] for f in graph.tails})

        for i in graph.walk_tails():
            if i not in linear_depth:
                linear_depth[i] = max(linear_depth[p] for p in graph.parents[i])

        if linear_depth != self.linear_parent:
            print(len(linear_depth), len(self.linear_parent))
            for x,y in linear_depth.items():
                if self.linear_parent[x] != y:
                    print(x, y, self.linear_parent[x])
            raise Exception("welp")

    def add_named_fragment(self, name, head, other):
        graph = self.graph
        graph.add_fragment(other)

        start = [(self.linear_parent.get(x), x) for x in other.tails]
        start.sort(reverse=True, key=lambda x: x[0])

        new_linear_parent = dict(self.linear_parent)

        for n, idx in start:
            search = []
            search.extend(c for c in graph.children[idx] if c not in new_linear_parent)
            while search:
                top = search.pop(0)
                if top not in new_linear_parent:
                    new_linear_parent[top] = n
                    search.extend(c for c in graph.children[top] if c not in new_linear_parent)

        self.linear_parent = new_linear_parent
        self.named_heads[name] = head

        ## validation

        missing = set(graph.commits) - set(new_linear_parent)

        if missing:
            raise Exception("what")

        ## extra check

        new_linear_parent2 = GitBranch.make_linear_parent(self.linear, graph.tails, graph.children)
        if new_linear_parent != new_linear_parent2:
            for k, v in new_linear_parent.items():
                v2 = new_linear_parent2[k]
                if v != v2:
                    print(v,v2)
            raise Exception("no")

        for k, v in self.linear_parent.items():
            if new_linear_parent[k] != v:
                raise Exception("error")
            if new_linear_parent2[k] != v:
                raise Exception("error")


    @staticmethod
    def make_linear_parent(history, tails, children):
        linear_parent = {c:n for n,c in enumerate(history,1)}

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
    def merge_history(self, all_history, merged_graph):
        history = [list(h) for h in all_history]
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

        for h in all_history:
            new_history2.extend(x for x in h if x not in seen)
            seen.update(h)

        new_history2.sort(key=lambda idx: merged_graph.commits[idx].max_date)

        if new_history != new_history2:
            raise Exception("welp")

        tail = new_history[0]

        if tail not in merged_graph.tails:
            raise Exception("bad")
        if merged_graph.commits[tail].parents:
            raise Exception("bad")
        if merged_graph.parents[tail]:
            raise Exception("bad")

        return new_history

    @classmethod
    def interweave(cls, name, branches, named_heads=None):
        if named_heads is None:
            named_heads = {}

        graphs = {k:v.graph for k,v in branches.items()}
        merged_graph = GitGraph.union(graphs)

        branch_heads = set()
        branch_tails = set()
        all_linear = list()
        all_named_heads = {}

        for name, branch in branches.items():
            all_linear.append(list(branch.linear))
            branch_heads.add(branch.head)
            branch_tails.add(branch.tail)

            for k, v in branch.named_heads.items():
                all_named_heads[f"{name}/{k}"] = v

        # create a new linear history
        linear = cls.merge_history(all_linear, merged_graph)

        head, tail = linear[-1], linear[0]

        if head not in branch_heads or tail not in branch_tails:
            raise Exception("bad history")

        # rewrite the parents
        # note: could remove original linear parent, and not create a merge commit

        prev = tail
        for idx in linear[1:]:
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

        prev = linear[0]
        date = merged_graph.commits[linear[0]].max_date
        for idx in linear[1:]:
            old_parents = merged_graph.commits[idx].parents
            if prev not in old_parents:
                raise Exception('bad merge', prev, idx)

            new_date = merged_graph.commits[idx].max_date
            if new_date < date:
                raise Exception("time travel")

            prev = idx
            date = new_date

        # fill out linear parents

        linear_parent = GitBranch.make_linear_parent(linear, merged_graph.tails, merged_graph.children)

        # validate linear parents

        linear_depth = [linear_parent[x] for x in linear]

        if linear_depth != sorted(linear_depth):
            raise Exception("bad linear depth")

        if set(merged_graph.commits) != set(linear_parent):
            missing = set(merged_graph.commits) - set(linear_parent)
            print(missing)
            for m in missing:
                if m in linear:
                    print(m, "found in linear history")
                else:
                    print(m, "not found in linear history")
            raise Exception(f'bad {len(merged_graph.commits)} {len(linear_parent)}')

        # ensure linear ordering of source branches is preserved in merged branch

        for name, branch in branches.items():
            graph = branch.graph
            history_depth = [linear_parent[x] for x in branch.linear]
            if history_depth != sorted(history_depth):
                raise Exception("branch history not preserved")
            for c in graph.commits:
                if branch.linear_parent[c] == 0:
                    if linear_parent[c] != 0:
                        # XXX - we exclude extra tails and shouldn't fold them in
                        raise Exception("bad")
                lp = branch.linear_parent[c]
                lp_idx = branch.linear[lp-1]
                nlp = linear_parent[lp_idx]
                if nlp != linear_parent[c]:
                    raise Exception("bad")

        # create the merged branch

        # find all merge points passed in, passed { "name of commit" : {"upstream name":"commit id"}}

        for point_name, merge_points in named_heads.items():

            all_merge_points =  [(idx, name, branches[name].linear_parent[idx], linear_parent[idx]) for name, idx in merge_points.items()]
            all_merge_points.sort(key=lambda x:x[3])

            merge_point = all_merge_points[-1][0]
            merge_point_time = all_merge_points[-1][-1]

            for idx, name, old_lp, new_lp in all_merge_points:
                if old_lp == len(branches[name].linear):
                    pass # last commit, no worries
                else:
                    next_c = branches[name].linear[old_lp]
                    next_lp = linear_parent[next_c]
                    if next_lp <= merge_point_time:
                        raise Exception("can't make merge point")

            all_named_heads[point_name] = merge_point

        # create map of commit prefixes

        prefix = {}
        for name, branch in graphs.items():
            for c in branch.commits:
                if c not in prefix:
                    prefix[c] = set()
                prefix[c].add(name)

        branch = GitBranch(
                name = name,
                graph = merged_graph,
                head = head,
                tail = tail,
                linear = linear,
                linear_parent = linear_parent,
                named_heads=all_named_heads
        )

        branch.validate()
        return branch, prefix

# used for fetch
class AuthCallbacks(pygit2.RemoteCallbacks):
    def credentials(self, url, username_from_url, allowed_types):
        if allowed_types & pygit2.enums.CredentialType.USERNAME:
            return pygit2.Username("git")
        elif allowed_types & pygit2.enums.CredentialType.SSH_KEY:
            x = os.path.expanduser("~/.ssh/id_ed25519.pub")
            y = os.path.expanduser("~/.ssh/id_ed25519")
            return pygit2.Keypair("git",x,y, "")
        else:
            return None


class GitRepo:
    def __init__(self, repo_dir):
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

    def all_remote_branches(self, refresh=False):
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
        a_date = datetime.fromtimestamp(float(obj.author.time), a_tz).astimezone(timezone.utc)

        c_tz = timezone(timedelta(minutes=obj.committer.offset))
        c_date = datetime.fromtimestamp(float(obj.committer.time), c_tz).astimezone(timezone.utc)

        tree = str(obj.tree_id)

        return GitCommit(
                tree = tree,
                parents = list(str(x) for x in obj.parent_ids),
                author=obj.author,
                committer=obj.committer,
                message = obj.message,
                max_date = max(a_date, c_date),
                author_date = a_date,
                committer_date = c_date,
        )


    def get_tree(self, addr):
        if isinstance(addr, GitTree):
            return addr
        elif not isinstance(addr, str):
            raise Exception("bad")

        obj = self.git.get(addr)
        entries = []
        for i in obj:
            e = (int(i.filemode), i.name, str(i.id))
            entries.append(e)
        return GitTree(entries)

    def write_commit(self, c):
        tree = c.tree
        if isinstance(c, GitTree):
            tree = self.write_tree(tree)
        elif isinstance(c.tree, str):
            tree = pygit2.Oid(hex=tree)
        else:
            raise Exception('bad')
        parents = [pygit2.Oid(hex=p) for p in c.parents]
        out = self.git.create_commit(None, c.author, c.committer, c.message, tree, parents)
        return str(out)

    def write_tree(self, t):
        tb = self.git.TreeBuilder()
        t.entries.sort(key=lambda x: x[1] if x[0] != GIT_DIR_MODE else x[1]+'/')
        for mode, name, addr in t.entries:
            if isinstance(addr, GitTree):
                i = self.write_tree(addr)
            elif isinstance(addr, str):
                i = pygit2.Oid(hex=addr)
            else:
                raise Exception("bad")
            tb.insert(name, i, mode)

        out = tb.write()
        return str(out)

    def write_empty_commit(self, name, email, timestamp, message):
        signature = pygit2.Signature(name, email, int(timestamp.timestamp()), 0, "utf-8")
        ts = timestamp.astimezone(timezone.utc),
        c = GitCommit(
                tree = GIT_EMPTY_TREE,
                parents=[],
                author=signature,
                committer=signature,
                message=message,
                max_date = ts,
                author_date = ts,
                committer_date = ts,
        )

        out = self.write_commit(c)
        return out

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
        return graph.Branch(name, head, named_heads)


    def get_branch(self, branch_name, include="*", exclude=None, replace_parents=None):
        branch_head = self.get_branch_head(branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}
        branch = graph.Branch(name, head, named_heads)
        branch.validate()

        return branch

    def get_remote_branch(self, rname, branch_name, include=None, exclude=None, replace_parents=None):

        branch_head = self.get_remote_branch_head(rname, branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}

        branch = branch_graph.Branch(name=f"{rname}/{branch_name}", head=branch_head, named_heads=named_heads)

        if include:
            all_remote_branches = self.all_remote_branches()
            remote_branches = all_remote_branches.get(rname,{})

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
                    pass # orphan branch or new tail commit

        branch.validate()
        return branch

    def get_fragment(self, head):
        init = self.get_commit(head)
        return GitGraph(
            commits = {head: init},
            heads = set([head]),
            tails = set([head]),
            children = {head: set()},
            parents =  {head: set()},
            parent_count = {head: 0},
            child_count = {head: 0},
            fragments = set([head]),
        )


    def get_graph(self, head, replace_parents=None, known=None):
        if replace_parents is None:
            replace_parents = {}
        if known is None:
            known = {}

        start = self.get_commit(head)

        commits = {head: start}
        tails = set()
        children = {}
        parents = {}
        parent_count = {}
        child_count = {}

        old_parents = {}
        search = [head]
        while search:
            idx = search.pop(0)

            c = commits[idx]
            if idx in known:
                c.parents = []
            elif idx in replace_parents:
                c.parents = list(replace_parents[idx])
                print("    > replaced", c.parents)
            c_parents = c.parents

            parent_count[idx] = len(c.parents)
            parents[idx] = list(c_parents)

            if not c_parents:
                tails.add(idx)

            if idx not in children:
                children[idx] = set()
                child_count[idx] = 0

            for pidx in c_parents:
                if pidx not in children:
                    children[pidx] = set()
                children[pidx].add(idx)
                child_count[pidx] = len(children[pidx])

                if pidx not in commits:
                    p = self.get_commit(pidx)
                    search.append(pidx)
                    commits[pidx] = p

        missing = set(commits) - set(parents)
        if missing:
            print(len(parents), len(commits))
            print(list(commits)[:10], list(parents)[:10])
            raise Exception()

        missing = set(commits) - set(children)
        if missing:
            print('c',missing)
            raise Exception()

        return GitGraph(
            commits = commits,
            tails = tails,
            children = children,
            parents = parents,
            parent_count = parent_count,
            child_count = child_count,
            heads = set([head]),
            fragments = set(f for f in tails if f in known),
        )

    def get_graph_names(self, graph):
        names = {}

        def add_name(i,n):
            if n not in names:
                names[n] = set()
            names[n].add(i)

        for i, c in graph.commits.items():
            add_name(i, str(c.author))
            add_name(i, str(c.committer))

            for line in c.message.splitlines():
                if "Co-authored-by: " in line:
                    _, name = line.rsplit("Co-authored-by: ", 1)
                    add_name(i, name.strip())
        return names


    def clean_branches(self, branch, bad_files):
        writer = GitWriter(repo, None)

        def fix_tree(idx, tree, ctree):
            tree, ctree = self.clean_tree(tree, ctree, bad_files)
            return tree, ctree

        writer.graft(branch, fix_tree=fix_tree, fix_commit=None)

        return writer.to_branch(branch.name)

    def interweave_branches(self, name, branches, named_heads=None, fix_commit=None, bad_files=None):
        branch, prefix = GitBranch.interweave(name, branches, named_heads=named_heads)

        # XXX
        # writer= self.Writer(None)
        # writer.graft_prefix()

        return branch, prefix


    def Writer(self, init, tree=None, named_heads=None):
        return GitWriter(self, init, tree, named_heads)



@dataclass
class Graft:
    idx: str
    commit: object
    root: str
    tree: object


class GitWriter:
    def __init__(self, repo, head, tree=None, named_heads=None):
        self.repo = repo
        self.head = head
        self.tree = tree
        self.named_heads = named_heads if named_heads else {}
        self.grafts = {}

    def grafted(self, idx):
        return self.grafts[idx].idx

    def save_grafts(self, path):
        with open(path, "w+") as fh:
            out = {k:v.idx for k,v in self.grafts.items}
            json.dump(out, fh, sort_keys=True, indent=2)

    def to_branch(self, name):
        graph = self.repo.get_graph(self.head)
        return graph.Branch(name, self.head, self.named_heads)

    def clean_tree(self, addr, old_tree, bad_files):
        entries = []
        dropped = False
        for i in old_tree.entries:
            name = i[1]
            bad = bad_files.get(name, None)
            if bad is None:
                entries.append(i)
            elif callable(bad):
                out = bad(i[0],i[1],i[2])
                if out:
                    entries.append(out)
                    if out != i:
                        dropped=True
                else:
                    dropped=True
            elif isinstance(bad, dict):
                sub_tree = self.repo.get_tree(i[2])
                new_addr, tree_obj = self.clean_tree(i[2], sub_tree, bad)
                if new_addr != i[2]:
                    entries.append((i[0], i[1], new_addr))
                    dropped = True
            else:
                dropped = True
                pass # delete it, if it's an empty hash
        if not dropped:
            return addr, old_tree
        new_tree = GitTree(entries)
        return self.repo.write_tree(new_tree), new_tree

    def prefix_tree(self, tree, prefix):
        entries = []
        for p in prefix:
            e = (GIT_DIR_MODE, p, tree)
            entries.append(e)
        t = GitTree(entries)
        return self.repo.write_tree(t), t

    def merge_tree(self, prev_tree, tree, prefix):
        entries = [e for e in prev_tree.entries if e[1] not in prefix]
        for p in prefix:
            e = (GIT_DIR_MODE, p, tree)
            entries.append(e)
        t = GitTree(entries)
        return self.repo.write_tree(t), t

    def shallow_merge(self, branches, bad_files, fix_commit):
        init = self.head
        heads = []

        for name, branch in branches.items():
            head = branch.head
            c = branch.graph.commits[head]
            prefix = [name]
            heads.append((head, c, prefix))

        heads.sort(key=lambda x:x[1].max_date)
        entries = []

        prev = init
        for head, commit, prefix in heads:
            old_tree = self.repo.get_tree(commit.tree)
            tree_idx, tree = self.clean_tree(commit.tree, old_tree, bad_files)

            entries = [e for e in entries if e[1] not in prefix]

            for p in prefix:
                e = (GIT_DIR_MODE, p, tree_idx)
                entries.append(e)

            t = GitTree(entries)
            tidx = self.repo.write_tree(t)

            if fix_commit is not None:
                author, committer, message = fix_commit(commit,  ", ".join(sorted(prefix)))
            else:
                author, committer, message = commit.author, commit.committer, commit.message

            c1 = GitCommit(
                    tree=tidx,
                    parents=[prev],
                    author=author,
                    committer=committer,
                    message = message,
                    max_date = commit.max_date,
                    author_date = commit.author_date,
                    committer_date = commit.committer_date,
            )

            prev = self.repo.write_commit(c1)
        return prev

    def graft_prefix(self, branch, graph_prefix, bad_files, fix_commit):
        start_tree = self.tree if self.tree else GitTree([])
        graph = branch.graph

        def prefix_tree(idx, tree, ctree):
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Exception("bad prefix, must be set or dict of set")

            tree, ctree = self.clean_tree(tree, ctree, bad_files)

            if idx in graph.tails:
                tree, ctree = self.merge_tree(start_tree, tree, prefix)
            else:
                max_parent = max(graph.parents[idx], key=branch.linear_parent.get)
                max_tree = self.grafts[max_parent].tree
                tree, ctree = self.merge_tree(max_tree, tree, prefix)

            return tree, ctree

        def prefix_commit(idx, commit):
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Exception("bad prefix, must be set or dict of set")
            return fix_commit(commit, ", ".join(sorted(prefix)))

        return self.graft(branch, prefix_tree, prefix_commit)

    def graft(self, branch, fix_tree, fix_commit):
        start_parents = [self.head] if self.head else None

        graph = branch.graph
        new_heads = {}

        graph_total = len(graph.commits)
        total = len(branch.linear)
        graph_count = 0
        depth = 0


        for idx in graph.walk_tails():
            if idx not in self.grafts:
                if idx in graph.fragments:
                    raise Exception("fragment missing")


                c1 = graph.commits[idx]
                ctree = self.repo.get_tree(c1.tree)

                if idx in graph.tails:
                    c1.parents = start_parents
                else:
                    c1.parents = [self.grafts[p].idx for p in graph.parents[idx]]
                
                if fix_tree is not None:
                    c1.tree, ctree = fix_tree(idx, c1.tree, ctree)

                if fix_commit is not None:
                    c1.author, c1.committer, c1.message = fix_commit(idx, c1)

                c2 = self.repo.write_commit(c1)
                self.grafts[idx] = Graft(c2, c1, c1.tree, ctree)

            else:
                graft = self.grafts[idx]
                c1 = graft.commit
                c2 = graft.idx
                ctree = graft.tree

            graph_count += 1

            if not graph.children[idx]:
                new_heads[idx] = c2

            c_depth = branch.linear_parent[idx]
            if c_depth > depth:
                depth = c_depth
                per = graph_count/graph_total
                print(f"\r    progress {per:.2%} {graph_count} of {graph_total}", end="")

        per = graph_count/graph_total
        print(f"\r    progress {per:.2%} {graph_count} of {graph_total}")

        for x in graph.commits:
            if x not in self.grafts:
                raise Exception("missing")

        self.head = self.grafts[branch.head].idx
        self.named_heads.update({k: self.grafts[v].idx for k,v in branch.named_heads.items()})
        return self.head

####
#   xxx: moving merge tree logic out of graft and into interweave
#   - does interweave return a written branch, or a branch with mutated commits and trees
#  
#
#
#



#### todo
#       repo.clean_branch(branch, bad_files)      
#       
#       repo.interweave(branches, bad_files, fix_message) 
#           calls branch interweave, then calls graft
#           branch.interweave creates prefixed trees 
#           ?? writer.graft(..., fix_trees)
#           repo.rewrite(branch, fix_commit=...)
#           writer(none).write(...)
#
#       branch.new_head()
#       branch.new_tail(tail, head)
#
#       repo.clean_branch, repo.prefix_branch, repo.interweave
#           take and return a new branch, saved
#       
#       clean_tree/prefix take and return GitTree??
#
# xxx - shallow merge is a proper writer and stores grafts
#       maybe calls branch.interweave_heads(....)
#
# xxx - named_heads to interweave can take named_heads and not just commits
#       as to merge various points
#
# xxx - Processor()
#       fold mkrepo.py up into more general class
# xxx - GitGraph
#       graph.properties = set([monotonic, monotonic-author, monotonic-committer])
#
# xxx - preserving old commit names in headers / changes
#
# xxx - graph.trees like graph.commits?


#### merging thoughts
# 
# xxx - move max parent logic out of graft
#       - write the trees and commits and create a proper branch before grafting
#       - store the trees inside graph.trees and nest them
#           write_tree recurses and dumps all
#           load_tree can recurse=True
#       - pass in a callback to graft fix_tree(commit, tree)
#       - calculate a parent_tree dict and pass it in, etc
#
#       fix tree allows me to preserve behaviour / push logic into script
#       so what, it's branch.intersect, and repo.intersect writes the actual commits
#       maybe graph carries "base_parent" and "prefix" 
#           so that graft knows which parent to inherit from
#       or pass in base_parent=lambda idx: base_parent[idx]}

#       
# xxx - general idea of finer grained merges, file based or subdirectory based
#
#       non linear parents? i.e i tag each tail with which repo it comes from
#       and inherit that, i.e n linear parents across all graphs
#       and i can combine "this subdirectory from this root"
#
#       i.e. non_linear_depth[x] = {project:n, project:n}
#       build by walk up from roots, and store a dict of the last 'linear' version of a subtree was
#
#
# xxx - merging into /file.mine /file.theirs rather than /mine/file, /theirs/file

# xxx - merging with other histories other than linear
#       and maybe not merging by replacing the first parent
