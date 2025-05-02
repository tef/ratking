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
class GitSignature:
    name: str
    email: str
    time: int
    offset: int

    def replace(self, name=None, email=None, time=None, offset=None):
        return GitSignature(
            name = name if name else self.name,
            email = email if email else self.email,
            time = time if time else self.time,
            offset = offset if offset else self.offset,
        )

    def to_pygit(self):
        #time = int(self.date.timestamp())
        #offset = int(self.date.tzinfo.utcoffset(None).total_seconds()) // 60
        return pygit2.Signature(name=self.name, email=self.email, time=self.time, offset=self.offset)

    def __str__(self):
        return f"{self.name} <{self.email}>"

    @classmethod
    def from_pygit(self, obj):
        #tz = timezone(timedelta(minutes=obj.offset))
        #date = datetime.fromtimestamp(float(obj.time), tz)
        return GitSignature(name=obj.name, email=obj.email, time=obj.time, offset=obj.offset)

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

    def clone(self):
        return GitCommit(
            tree = self.tree,
            parents = list(self.parents),
            author = self.author,
            committer = self.committer,
            message = self.message,

            max_date = self.max_date,
            author_date = self.author_date,
            committer_date = self.committer_date,
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
            fragments=set()
        )

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

    def walk_children(self):
        search = list(self.tails)
        counts = dict(self.parent_count)

        search.sort(key=lambda x:self.commits[x].max_date)

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
                c = other.commits[idx]
                self.add_commit(idx, c)
                if idx in other.fragments:
                    self.fragments.add(idx)
                elif idx in self.fragments:
                    self.fragments.remove(idx)
            else:
                if idx not in other.fragments:
                    if self.commits[idx] != other.commits[idx]:
                        raise Exception("nope")
                    if self.parents[idx] != other.parents[idx]:
                        raise Exception("nope")
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
                raise Exception("error")

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
                    print("child commit", child, "parents", self.parents[idx])
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

        for c in self.walk_parents():
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

        for i in self.walk_children():
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

    def to_branch(self, name, head, named_heads):
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
            named_heads = named_heads,
        )



@dataclass
class GitBranch:
    name: str
    head: str
    graph: object
    named_heads: dict
    tail: str

    def clone(self):
        return GitBranch(
            name = str(name),
            head = str(self.head),
            graph = graph.clone(),
            tail = str(self.tail),
            named_heads = dict(self.named_heads),
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
                raise Exception(f"missing head: {name}")

        if self.head not in graph.commits:
            raise Exception("missing head")

        if self.tail not in graph.commits:
            raise Exception("missing tail")

        history = graph.first_parents(self.head)

        if history[0] not in graph.tails:
            raise Exception("what")
        if history[-1] != self.head:
            raise Exception("how")


    def add_named_fragment(self, name, head, other):
        graph = self.graph
        graph.add_graph(other)
        self.named_heads[name] = head


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
    def merge_linear_history(self, branches):
        graphs = {}
        branch_history = {}
        branch_linear_parent = {}

        for name, branch in branches.items():
            graph = branch.graph
            graphs[name] = graph

            history = graph.first_parents(branch.head)
            branch_history[name] = history
            branch_linear_parent[name] = GitBranch.make_linear_parent(history, graph.tails, graph.children)
            
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
            raise Exception("welp")

        head = new_history[-1]
        tail = new_history[0]

        if merged_graph.commits[tail].parents:
            raise Exception("bad")
        if merged_graph.parents[tail]:
            raise Exception("bad")

        for name, branch in branches.items():
            h = list(branch_history[name])
            h.reverse()
            g = branch.graph
            
            for idx in new_history:
                if idx in g.commits:
                    i = h.pop()
                    if idx != i:
                        raise Exception("disorder")
            if h:
                raise Exception("missing")

        return new_history, branch_history, branch_linear_parent, merged_graph

    @classmethod
    def interweave(cls, name, branches, named_heads=None, merge_named_heads=()):

        # create a new linear history
        history, branch_history, branch_linear_parent, merged_graph = cls.merge_linear_history(branches)

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
                raise Exception('bad merge', prev, idx)

            new_date = merged_graph.commits[idx].max_date
            if new_date < date:
                raise Exception("time travel")

            prev = idx
            date = new_date

        # fill out linear parents

        linear_parent = GitBranch.make_linear_parent(history, merged_graph.tails, merged_graph.children)

        # validate linear parents

        linear_depth = [linear_parent[x] for x in history]

        if linear_depth != sorted(linear_depth):
            raise Exception("bad linear depth")

        if set(merged_graph.commits) != set(linear_parent):
            missing = set(merged_graph.commits) - set(linear_parent)
            print(missing)
            for m in missing:
                if m in history:
                    print(m, "found in linear history")
                else:
                    print(m, "not found in linear history")
            raise Exception(f'bad {len(merged_graph.commits)} {len(linear_parent)}')

        # ensure linear ordering of source branches is preserved in merged branch

        for name, branch in branches.items():
            graph = branch.graph
            history_depth = [linear_parent[x] for x in branch_history[name]]
            if history_depth != sorted(history_depth):
                raise Exception("branch history not preserved")
            for c in graph.commits:
                if branch_linear_parent[name][c] == 0:
                    if linear_parent[c] != 0:
                        # xxx - we exclude extra tails and shouldn't fold them in
                        # xxx - and we error elsewhere about it
                        raise Exception("bad")
                lp = branch_linear_parent[name][c]
                lp_idx = branch_history[name][lp-1]
                nlp = linear_parent[lp_idx]
                if nlp != linear_parent[c]:
                    raise Exception("bad")

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

            all_merge_points =  [(idx, name, branch_linear_parent[name][idx], linear_parent[idx]) for name, idx in merge_points.items()]
            all_merge_points.sort(key=lambda x:x[3])

            merge_point = all_merge_points[-1][0]
            merge_point_time = all_merge_points[-1][-1]

            for idx, name, old_lp, new_lp in all_merge_points:
                if old_lp == len(branch_history[name]):
                    pass # last commit, no worries
                else:
                    next_c = branch_history[name][old_lp]
                    next_lp = linear_parent[next_c]
                    if next_lp <= merge_point_time:
                        raise Exception("can't make merge point")

            all_named_heads[point_name] = merge_point

        branch = GitBranch(
                name = name,
                graph = merged_graph,
                head = head,
                tail = tail,
                named_heads=all_named_heads
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

    def remote_branches(self, remote_name):
        remote = self.git.remotes[remote_name]
        remote_refs = remote.ls_remotes(callbacks=AuthCallbacks())
        HEAD = None
        branches = []
        for ref in remote_refs:
            if ref['name'] == "HEAD":
                HEAD = ref['symref_target'].split("refs/heads/",1)[1]
            elif ref['name'].startswith("refs/heads/"):
                name = ref['name'].split("refs/heads/",1)[1]
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
        a_date = datetime.fromtimestamp(float(obj.author.time), a_tz).astimezone(timezone.utc)

        c_tz = timezone(timedelta(minutes=obj.committer.offset))
        c_date = datetime.fromtimestamp(float(obj.committer.time), c_tz).astimezone(timezone.utc)

        tree = str(obj.tree_id)

        author = GitSignature.from_pygit(obj.author)
        committer = GitSignature.from_pygit(obj.committer)
        message = obj.message

        if author.to_pygit() != obj.author:
            raise Exception("wait")

        return GitCommit(
                tree = tree,
                parents = list(str(x) for x in obj.parent_ids),
                author=author,
                committer=committer,
                message = message,
                max_date = max(a_date, c_date),
                author_date = a_date,
                committer_date = c_date,
        )


    def get_tree(self, addr):
        if isinstance(addr, GitTree):
            addr = self.repo.write_tree(addr)

        elif not isinstance(addr, str):
            raise Exception("bad")

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
            raise Exception('bad')
        parents = [pygit2.Oid(hex=p) for p in c.parents]
        author = c.author.to_pygit()
        committer = c.committer.to_pygit()
        out = self.git.create_commit(None, author, committer, c.message, tree, parents)
        if not out:
            raise Exception("what")
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

    def start_branch(self, branch_name, name, email, timestamp, message):
        if timestamp.tzinfo is None:
            raise Exception("needs timestamp")
        ts = timestamp.astimezone(timezone.utc)
        signature = GitSignature(name, email, int(ts.timestamp()), 0)
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

        head = self.write_commit(c)
        graph = self.get_graph(head) # XXX - build a graph from the commit
        named_heads = {branch_name: head}

        branch = graph.to_branch(branch_name, head, named_heads)
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
        return graph.to_branch(name, head, named_heads)


    def get_branch(self, branch_name, include="*", exclude=None, replace_parents=None):
        branch_head = self.get_branch_head(branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}
        branch = graph.to_branch(name, head, named_heads)
        branch.validate()

        return branch

    def get_remote_branch(self, rname, branch_name, include=None, exclude=None, replace_parents=None):

        branch_head = self.get_remote_branch_head(rname, branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)

        named_heads = {branch_name: branch_head}

        branch = branch_graph.to_branch(name=f"{rname}/{branch_name}", head=branch_head, named_heads=named_heads)

        if include:
            all_remote_branches = self.all_remote_branch_names()
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
                print("    > replaced", c.parents)

            graph.add_commit(idx, c)

            for pidx in c.parents:
                if pidx not in walked:
                    search.append(pidx)
                    walked.add(pidx)

        missing = set(walked) - set(graph.parents)
        if missing:
            raise Exception("Missing elements")

        return graph

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

    def clean_tree(self, addr, old_tree, bad_files):
        if bad_files == None:
            return addr, old_tree

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
                sub_addr, sub_tree = self.get_tree(i[2])
                new_addr, tree_obj = self.clean_tree(sub_addr, sub_tree, bad)
                if new_addr != i[2]:
                    entries.append((i[0], i[1], new_addr))
                    dropped = True
            else:
                dropped = True
                pass # delete it, if it's an empty hash
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


    def clean_branch(self, branch, bad_files):
        writer = GitWriter(self, branch.name)
        branch.validate()

        def fix_tree(writer, idx, tree, ctree):
            tree, ctree = self.clean_tree(tree, ctree, bad_files)
            return tree, ctree

        writer.graft(branch, fix_tree=fix_tree, fix_commit=None)

        new_branch = writer.to_branch()
        # print("XXX", len(new_branch.graph.commits), len(branch.graph.commits), len(writer.grafts))

        for x, y in zip(branch.graph.walk_children(), new_branch.graph.walk_children()):
            if writer.grafted(x) != y:
                print('oooo')
        return new_branch

    def prefix_branch(self, branch, prefix):
        writer = GitWriter(self, branch.name)

        def fix_tree(writer, idx, tree, ctree):
            tree, ctree = self.prefix_tree(tree, ctree, [prefix])
            return tree, ctree

        writer.graft(branch, fix_tree=fix_tree, fix_commit=None)

        branch = writer.to_branch()
        return branch

    def interweave_branch_heads(self, branches, bad_files, fix_commit):
        heads = []
        for name, branch in branches.items():
            head = branch.head
            c = branch.graph.commits[head]
            prefix = [name]
            heads.append((head, c, prefix))

        heads.sort(key=lambda x:x[1].max_date)
        entries = []

        prev = None
        for head, commit, prefix in heads:
            tree_idx, old_tree = self.get_tree(commit.tree)
            tree_idx, tree = self.clean_tree(commit.tree, old_tree, bad_files)

            entries = [e for e in entries if e[1] not in prefix]

            for p in prefix:
                e = (GIT_DIR_MODE, p, tree_idx)
                entries.append(e)

            t = GitTree(entries)
            tidx = self.write_tree(t)

            if fix_commit is not None:
                author, committer, message = fix_commit(commit,  ", ".join(sorted(prefix)))
            else:
                author, committer, message = commit.author, commit.committer, commit.message

            c1 = GitCommit(
                    tree=tidx,
                    parents=([prev] if prev else []),
                    author=author,
                    committer=committer,
                    message = message,
                    max_date = commit.max_date,
                    author_date = commit.author_date,
                    committer_date = commit.committer_date,
            )

            prev = self.write_commit(c1)
        return prev


    def interweave_branches(self, name, branches, named_heads=None, merge_named_heads=None, fix_commit=None):
        graph_prefix = {}
        for name, branch in branches.items():
            for c in branch.graph.commits:
                if c not in graph_prefix:
                    graph_prefix[c] = set()
                graph_prefix[c].add(name)

        merged_branch, linear_parent = GitBranch.interweave(name, branches, named_heads=named_heads, merge_named_heads=merge_named_heads)

        writer = GitWriter(self, name)
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


        def prefix_tree(writer, idx, tree, ctree):
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Exception("bad prefix, must be set or dict of set")

            if idx in graph.tails:
                tree, ctree = merge_tree(start_tree, tree, prefix)
            else:
                max_parent = max(graph.parents[idx], key=linear_parent.get)
                max_tree = grafted_trees[max_parent]
                tree, ctree = merge_tree(max_tree, tree, prefix)

            grafted_trees[idx] = ctree

            return tree, ctree

        def prefix_commit(writer, idx, commit):
            prefix = graph_prefix
            if isinstance(prefix, dict):
                prefix = prefix[idx]
            if prefix and not isinstance(prefix, set):
                raise Exception("bad prefix, must be set or dict of set")
            return fix_commit(commit, ", ".join(sorted(prefix)))

        writer.graft(merged_branch, fix_tree=prefix_tree, fix_commit=prefix_commit)

        new_branch = writer.to_branch()

        for x, y in zip(merged_branch.graph.walk_children(), new_branch.graph.walk_children()):
            if writer.grafted(x) != y:
                print('oooo')
        return new_branch


    def Writer(self, name):
        return GitWriter(self, name)




class GitWriter:
    def __init__(self, repo, name):
        self.repo = repo
        self.name = name
        self.head = None # maybe support multiple heads as parents
        self.named_heads = {}
        self.grafts = {}
        self.replaces = {}
        self.graph = GitGraph.new()

    def grafted(self, idx):
        return self.grafts[idx]

    def save_grafts(self, path):
        with open(path, "w+") as fh:
            out = {k:v.idx for k,v in self.grafts.items}
            json.dump(out, fh, sort_keys=True, indent=2)

    def to_branch(self):
        # XXX -  BUILD A GRAPH
        graph = self.repo.get_graph(self.head)

        for k,v in self.named_heads.items():
            fragment = self.repo.get_graph(v, known=graph.commits)
            graph.add_graph(fragment)

        # print("XXX", "new graph has ", len(graph.commits), "built graph", len(self.graph.commits))

        self.graph.validate()
        branch = self.graph.to_branch(self.name, self.head, dict(self.named_heads))
        branch.validate()
        return branch

    def graft_commit(self, idx, fix_tree=None, fix_commit=None):
        start_parents = [self.head] if self.head else []

        c1 = self.repo.get_commit(idx).clone()
        c1.tree, ctree = self.repo.get_tree(c1.tree)

        if not c1.parents:
            c1.parents = start_parents
        else:
            c1.parents = [self.grafts[p] for p in c1.parents[idx]]
        
        if fix_tree is not None:
            c1.tree, ctree = fix_tree(self, idx, c1.tree, ctree)

        if fix_commit is not None:
            c1.author, c1.committer, c1.message = fix_commit(self, idx, c1)

        c2 = self.repo.write_commit(c1)
        self.grafts[idx] = c2
        self.replaces[c2] = idx
        self.head = c2

        self.graph.add_commit(c2, c1)
        self.graph.heads = set([self.head])
        return c2

    def graft(self, branch, *, fix_tree=None, fix_commit=None):
        start_parents = [self.head] if self.head else []

        graph = branch.graph
        graph_total = len(graph.commits)
        graph_count = 0


        for idx in graph.walk_children():
            if idx not in self.grafts:
                if idx in graph.fragments:
                    raise Exception("fragment missing")


                c1 = graph.commits[idx].clone()
                c1.tree, ctree = self.repo.get_tree(c1.tree)

                if not c1.parents:
                    c1.parents = start_parents
                else:
                    c1.parents = [self.grafts[p] for p in graph.parents[idx]]
                
                if fix_tree is not None:
                    c1.tree, ctree = fix_tree(self, idx, c1.tree, ctree)

                if fix_commit is not None:
                    c1.author, c1.committer, c1.message = fix_commit(self, idx, c1)

                c2 = self.repo.write_commit(c1)
                self.grafts[idx] = c2
                self.replaces[c2] = idx
                self.graph.add_commit(c2, c1)

                if idx in graph.heads:
                    self.graph.heads.add(c2)

            graph_count += 1


            if graph_count & 512 == 0:
                per = graph_count/graph_total
                print(f"\r    > progress {per:.2%} {graph_count} of {graph_total}", end="")

        per = graph_count/graph_total
        print(f"\r    > progress {per:.2%} {graph_count} of {graph_total}")

        for x in graph.commits:
            if x not in self.grafts:
                raise Exception("missing")

        self.head = self.grafts[branch.head]
        self.named_heads.update({k: self.grafts[v] for k,v in branch.named_heads.items()})

        for k,v in self.named_heads.items():
            self.repo.get_commit(v)
        return self.head

#### future thoughts
# 
# xxx - datetime handling in Signature - @property 
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
# xxx - merging into /file.mine /file.theirs rather than /mine/file, /theirs/file
#
# xxx - merging with other histories other than linear
#       and maybe not merging by replacing the first parent
#
# xxx - Processor()
#       fold mkrepo.py up into more general class
#
# xxx - GitGraph
#       graph.properties = set([monotonic, monotonic-author, monotonic-committer])
#
# xxx - preserving old commit names in headers / changes

