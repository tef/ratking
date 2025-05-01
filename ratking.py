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

@functools.cache
def compile_pattern(pattern):
    regex = glob.translate(pattern, recursive=True)
    return re.compile(regex)

def glob_match(pattern, string):
    if pattern is None:
        return None
    rx = compile_pattern(pattern)
    return rx.match(string) is not None


import pygit2

GIT_DIR_MODE = 0o040_000
GIT_FILE_MODE = 0o100_644
GIT_FILE_MODE2 = 0o100_664
GIT_EXEC_MODE = 0o100_755
GIT_LINK_MODE = 0o120_000
GIT_GITLINK_MODE = 0o160_000 # actually a submodule, blegh
GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904" # Wow, isn't git amazing.

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
    head: str
    tail: str
    tails: set
    heads: set
    named_heads: set
    parents: dict
    parent_count: dict
    children: dict
    fragments: set 

    linear: dict
    linear_parent: dict

    def clone(self):
        return GitGraph(
            commits = dict(self.commits),
            tails = set(self.tails),
            heads = set(self.heads),
            children = {k:set(v) for k,v in self.children.items()},
            parents = {k:list(v) for k,v in self.parents.items()},
            parent_count = dict(parent_count),
            head = str(self.head),
            tail = str(self.tail),
            named_heads = dict(self.named_heads),
            linear = list(self.linear),
            linear_parent = dict(self.linear_parent),
            fragments = set(self.fragments),
        )

    def add_graph_fragment(self, other):
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

            for c in other.children[idx]:
                self.children[idx].add(c)

        for f in other.tails:
            # don't merge in graphs with new init commits
            # XXX we could allow this? but for now it may prevent bugs
            if not self.parents[f] and f not in self.tails:
                raise Exception("error")

        for l in other.heads:
            if not self.children[l]:
                self.heads.add(l)

        start = [(self.linear_parent.get(x), x) for x in other.tails]
        start.sort(reverse=True, key=lambda x: x[0])

        new_linear_parent = dict(self.linear_parent)

        for n, idx in start:
            search = []
            search.extend(c for c in self.children[idx] if c not in new_linear_parent)
            while search:
                top = search.pop(0)
                if top not in new_linear_parent:
                    new_linear_parent[top] = n
                    search.extend(c for c in self.children[top] if c not in new_linear_parent)

        missing = set(self.commits) - set(new_linear_parent)

        if missing:
            raise Exception("what")

        new_linear_parent2 = GitGraph.make_linear_parent(self.linear, self.tails, self.children)
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

        self.linear_parent = new_linear_parent

        return self

    def validate(self):
        head = self.head

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

        # children[x] = set(commits that have x as a parent)
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

            lb = self.linear_parent[idx] -1

            if self.linear[lb] == idx and lb > 0:
                c_prev = self.parents[idx][0]
                l_prev = self.linear[lb-1]
                if l_prev != c_prev:
                    raise Exception("linear mismatch")

        # validate linear path

        history = [head]
        p = self.parents[head]
        while p:
            p = p[0]
            history.append(p)
            p = self.parents[p]

        history.reverse()

        if history[0] not in self.tails:
            raise Exception("what")
        if history[-1] != head:
            raise Exception("how")
        
        if history != self.linear:
            raise Exception("wrong linear")

        history.sort(key=lambda idx: self.commits[idx].max_date)

        if history != self.linear:
            raise Exception("time travel")

        linear_parent = GitGraph.make_linear_parent(self.linear, self.tails, self.children)

        if linear_parent != self.linear_parent:
            print("extra keys",set(linear_parent) - set(self.linear_parent))
            print("missing keys",set(self.linear_parent) - set(linear_parent))

            raise Exception("missing linear parent")

        # walk backwards from heads

        found_tails = set()
        walked = set([head])
        search = [head]
        for h in self.heads:
            if h not in walked:
                walked.add(h)
                search.append(h)

        # all commits parents are correctly indexed
        while search:
            c = search.pop(0)

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

            if graph_parents:
                for p in graph_parents:
                    if p not in walked:
                        walked.add(p)
                        search.append(p)
            else:
                found_tails.add(c)

        if found_tails != self.tails:
            raise Error("missing tails")

        if walked != set(self.commits):
            raise Exception("missing commits")


        # walk forward from talks
        # validate complete walk through children

        search = list(self.tails)
        walked = set(self.tails)
        heads = set()

        while search:
            i = search.pop(0)

            if self.children[i]:
                for c in self.children[i]:
                    if c not in walked:
                        walked.add(c)
                        search.append(c)
            else:
                heads.add(i)

        if heads != self.heads:
            print("extra", heads-self.heads)
            print("missing", self.heads-heads)

            raise Exception("heads is not correct")

        if walked != set(self.commits):
            print("commits", len(self.commits), "walked", len(walked))
            raise Exception("missing commits")


        # validate walk through counts

        for f in self.tails:
            if self.parents[f]: 
                raise Exception("bad")

        walked = set(self.tails)
        search = list(self.tails)
        counts = dict(self.parent_count)
        heads = set()

        linear_depth = {f: self.linear_parent[f] for f in self.linear}
        linear_depth.update({f: self.linear_parent[f] for f in self.tails})

        while search:
            i = search.pop(0)
            
            if counts[i] != 0:
                raise Exception("bad")
            if i not in linear_depth:
                linear_depth[i] = max(linear_depth[p] for p in self.parents[i])

            if self.children[i]:
                for c in self.children[i]:
                    if counts[c] <= 0:
                        raise Exception("bad")

                    counts[c] -= 1
                    if counts[c] == 0:
                        if c not in walked:
                            walked.add(c)
                            search.append(c)
                        else:
                            raise Exception("what")
            else:
                heads.add(i)

        if not heads:
            missing = [x for x in self.commits if x not in walked and counts[x] != self.parent_count[x]]
            print("never walked", len(self.commits)-len(walked))
            print("almost walked", len(missing))
            for m in missing:
                print("missing", m)
            print("total", len(self.commits))

            raise Exception("exited early")

        if heads != self.heads:
            print(heads, self.heads)
            raise Exception("bad head")

        if walked != set(self.commits):
            print(heads, head)
            print(len(walked), len(self.commits), len(self.tails))
            raise Exception("missing commits")

        if linear_depth != self.linear_parent:
            print(len(linear_depth), len(self.linear_parent))
            for x,y in linear_depth.items():
                if self.linear_parent[x] != y:
                    print(x, y, self.linear_parent[x])
            raise Exception("welp")


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
    def interweave(cls, graphs, named_heads=None):
        all_commits = {}
        all_tails = set()
        all_heads = set()
        all_children = {}
        all_parents = {}
        all_linear = list()
        graph_heads = set()
        graph_tails = set()
        all_named_heads = {}
        all_fragments = set()
        all_parent_count = dict()

        if named_heads is None:
            named_heads = {}

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

                if graph.children[idx]:
                    all_children[idx].update(graph.children[idx])
                    if idx in all_heads:
                        all_heads.remove(idx)

            all_linear.append(list(graph.linear))

            graph_heads.add(graph.head)
            graph_tails.add(graph.tail)

            all_tails.update(f for f in graph.tails if f not in graph.fragments)
            all_heads.update(t for t in graph.heads if not all_children[t])

            for k, v in graph.named_heads.items():
                all_named_heads[f"{name}/{k}"] = v
    
        history = [list(h) for h in all_linear]
        new_history = []

        while history:
            next_head = [h[-1] for h in history]

            next_head.sort(key=lambda i: all_commits[i].max_date)

            c = next_head[-1]
            new_history.append(c)

            for h in history:
                if h[-1] == c:
                    h.pop()

            if any(not h for h in history):
                history = [h for h in history if h]

        new_history.reverse()

        seen = set()
        new_history2 = []

        for h in all_linear:
            new_history2.extend(x for x in h if x not in seen)
            seen.update(h)

        new_history2.sort(key=lambda idx: all_commits[idx].max_date)

        if new_history != new_history2:
            raise Exception("welp")

        linear = new_history2
    
        prev = linear[0]

        if prev not in all_tails:
            raise Exception("bad")
        if all_commits[prev].parents:
            raise Exception("bad")
        if all_parents[prev]:
            raise Exception("bad")

        for idx in linear[1:]:
            old_parents = all_commits[idx].parents
            if idx in all_tails:
                all_tails.remove(idx)

            # XXX - could remove original linear parent
            #       and not create a merge commit
            new_parents = [prev] + [o for o in old_parents if o != prev]

            all_parents[idx] = list(new_parents)
            all_commits[idx].parents = list(new_parents)
            all_parent_count[idx] = len(new_parents)

            all_children[prev].add(idx)

            prev = idx

        prev = linear[0]
        date = all_commits[linear[0]].max_date
        for idx in linear[1:]:
            old_parents = all_commits[idx].parents
            if prev not in old_parents:
                raise Exception('bad merge', prev, idx)

            new_date = all_commits[idx].max_date
            if new_date < date:
                raise Exception("time travel")

            prev = idx
            date = new_date

        linear_parent = cls.make_linear_parent(linear, all_tails, all_children)

        linear_depth = [linear_parent[x] for x in linear]
        if linear_depth != sorted(linear_depth):
            raise Exception("bad linear depth")

        if set(all_commits) != set(linear_parent):
            missing = set(all_commits) - set(linear_parent)
            print(missing)
            for m in missing:
                if m in linear:
                    print(m, "found in linear history")
                else:
                    print(m, "not found in linear history")
            raise Exception(f'bad {len(all_commits)} {len(linear_parent)}')

        head = linear[-1]
        tail = linear[0]

        if head not in graph_heads:
            raise Exception("bad")
        if tail not in graph_tails:
            # should be one of the .tail of the graphs
            raise Exception("worse")

        # for each source branch
        #    find new linear history for each point, and that it always goes up
        #    for each commit, check it has the new value 

        for name, graph in graphs.items():
            history_depth = [linear_parent[x] for x in graph.linear]
            if history_depth != sorted(history_depth):
                raise Exception("history not preserved")
            for c in graph.commits:
                if graph.linear_parent[c] == 0:
                    if linear_parent[c] != 0:
                        # XXX - we exclude extra tails and shouldn't fold them in
                        raise Exception("bad")
                lp = graph.linear_parent[c]
                lp_idx = graph.linear[lp-1]
                nlp = linear_parent[lp_idx]
                if nlp != linear_parent[c]:
                    raise Exception("bad")

        all_heads = {l for l in all_heads if not all_children[l]}

        for point_name, merge_points in named_heads.items():
                
            all_merge_points =  [(idx, name, graphs[name].linear_parent[idx], linear_parent[idx]) for name, idx in merge_points.items()]
            all_merge_points.sort(key=lambda x:x[3])

            merge_point = all_merge_points[-1][0]
            merge_point_time = all_merge_points[-1][-1]

            for idx, name, old_lp, new_lp in all_merge_points:
                if old_lp == len(graphs[name].linear):
                    pass # last commit, no worries
                else:
                    next_c = graphs[name].linear[old_lp]
                    next_lp = linear_parent[next_c]
                    if next_lp <= merge_point_time:
                        raise Exception("can't make merge point")

            all_named_heads[point_name] = merge_point

        return cls(
            commits = all_commits,
            tails = all_tails,
            heads = all_heads,
            children = all_children,
            parents = all_parents,
            parent_count = all_parent_count,
            named_heads = all_named_heads,
            head = head,
            tail = tail,
            linear = linear,
            linear_parent = linear_parent,
            fragments = all_fragments,
        )

@dataclass
class GitBranch:
    name: str
    head: str
    graph: object
    named_heads: str

    @classmethod
    def interweave(cls, name, branches, named_heads=None):

        graphs = {k:v.graph for k,v in branches.items()}

        merged_graph = GitGraph.interweave(graphs, named_heads=named_heads)
        merged_graph.validate() # merged

        prefix = {}
        for name, branch in graphs.items():
            for c in branch.commits:
                if c not in prefix:
                    prefix[c] = set()
                prefix[c].add(name)

        branch = GitBranch(name, head=merged_graph.head, graph=merged_graph, named_heads=merged_graph.named_heads)
        return branch, prefix

    @classmethod
    def common_ancestor(self, left, right):
        left, right = left.graph, right.graph

        before, after = None, None
        for x, y in zip(left.linear, right.linear):
            if x != y:
                after = y
                break

            if left.children[x] != right.children[y]:
                after = y
                break

            before = x
            # XXX: skip consolidating, as more branch history
        return before, after



class GitRepo:
    def __init__(self, repo_dir):
        self.git = pygit2.init_repository(repo_dir, bare=True)
        self._all_remotes = None

    def get_branch_head(self, name):
        if name in self.git.branches:
            return str(self.git.branches[name].target)

    def write_branch_head(self, name, addr):
        c = self.git.revparse_single(addr)
        self.git.branches.create(name, c, force=True)

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

    def get_remote_branch_head(self, rname, name):
        return str(self.git.branches.remote[f"{rname}/{name}"].target)

    def get_remote_branch_graph(self, rname, name, replace_parents):
        head = self.get_remote_branch_head(rname, name)
        branch = self.get_graph(head, replace_parents)
        return branch


    def get_commit(self, addr):
        obj = self.git.get(addr)

        a_tz = timezone(timedelta(minutes=obj.author.offset))
        a_date = datetime.fromtimestamp(float(obj.author.time), a_tz).astimezone(timezone.utc)

        c_tz = timezone(timedelta(minutes=obj.committer.offset))
        c_date = datetime.fromtimestamp(float(obj.committer.time), c_tz).astimezone(timezone.utc)

        return GitCommit(
                tree = str(obj.tree_id),
                parents = list(str(x) for x in obj.parent_ids),
                author=obj.author,
                committer=obj.committer,
                message = obj.message,
                max_date = max(a_date, c_date),
                author_date = a_date,
                committer_date = c_date,
        )


    def get_tree(self, addr):
        obj = self.git.get(addr)
        entries = []
        for i in obj:
            e = (int(i.filemode), i.name, str(i.id))
            entries.append(e)
        return GitTree(entries)

    def write_commit(self, c):
        parents = [pygit2.Oid(hex=p) for p in c.parents]
        tree = pygit2.Oid(hex=c.tree)
        out = self.git.create_commit(None, c.author, c.committer, c.message, tree, parents)
        return str(out)

    def write_tree(self, t):
        tb = self.git.TreeBuilder()
        t.entries.sort(key=lambda x: x[1] if x[0] != GIT_DIR_MODE else x[1]+'/')
        for mode, name, addr in t.entries:
            i = pygit2.Oid(hex=addr)
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
                sub_tree = self.get_tree(i[2])
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
        return self.write_tree(new_tree), new_tree

    def prefix_tree(self, tree, prefix):
        entries = []
        for p in prefix:
            e = (GIT_DIR_MODE, p, tree)
            entries.append(e)
        t = GitTree(entries)
        return self.write_tree(t), t

    def merge_tree(self, prev_tree, tree, prefix):
        entries = [e for e in prev_tree.entries if e[1] not in prefix]
        for p in prefix:
            e = (GIT_DIR_MODE, p, tree)
            entries.append(e)
        t = GitTree(entries)
        return self.write_tree(t), t

    def get_fragment(self, head):
        init = self.get_commit(head)
        return GitGraph(
            commits = {head: init},
            tails = set([head]),
            children = {head: set()},
            parents =  {head: set()},
            parent_count = {head: 0},
            head = head,
            tail = head,
            heads = set([head]),
            named_heads = {},
            linear = [head],
            linear_parent = {head:1},
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

            for pidx in c_parents:
                if pidx not in children:
                    children[pidx] = set()
                children[pidx].add(idx)

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

        history = [head]
        n = head

        while n is not None:
            p = parents.get(n)
            p = p[0] if p else None
            if p is None:
                break
            history.append(p)
            n = p

        history.reverse()
        date = commits[history[0]].max_date
        for i in history[1:]:
            new_date = commits[i].max_date
            if new_date < date:
                raise Exception("time travel")

        # XXX set property

        linear_parent = GitGraph.make_linear_parent(history, tails, children)

        if set(commits) != set(linear_parent):
            raise Exception(f'bad {len(commits)} {len(linear_parent)}')

        return GitGraph(
            commits = commits,
            tails = tails,
            children = children,
            parents = parents,
            parent_count = parent_count,
            head = head,
            tail = history[0],
            heads = set([head]),
            named_heads = {},
            linear = history,
            linear_parent = linear_parent,
            fragments = set(f for f in tails if f in known),
        )

    def new_branch(self, name, head, graph=None, named_heads=None):
        graph = graph or self.get_graph(head)
        named_heads = dict(named_heads) if named_heads else {}
        named_heads.update(graph.named_heads)
        return GitBranch(name=name, head=head, graph=graph, named_heads=named_heads)


    def get_branch(self, branch_name, include="*", exclude=None, replace_parents=None):
        branch_head = self.get_branch_head(branch_name)

        branch_graph = self.get_graph(branch_head, replace_parents)
        branch_graph.validate()

        branch_graph.named_heads[branch_name] = branch_head

        return GitBranch(name=branch_name, head=branch_head, graph=branch_graph, named_heads=named_heads)

    def get_remote_branch(self, rname, branch_name, include=None, exclude=None, replace_parents=None):

        branch_head = self.get_remote_branch_head(rname, branch_name)
        branch_graph = self.get_graph(branch_head, replace_parents)
        branch_graph.validate()

        named_heads = {branch_name: branch_head}

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
                    branch_graph.add_graph_fragment(graph)
                    named_heads[name] = graph.head
                else:
                    pass # orphan branch or new tail commit
            branch_graph.validate()


        branch_graph.named_heads.update(named_heads)
        return GitBranch(name=f"{rname}/{branch_name}", head=branch_head, graph=branch_graph, named_heads=named_heads)

    def get_names(self, head):
        names = {}
        graph = self.get_graph(head)

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


    def shallow_merge(self, graphs, bad_files, fix_commit):
        init = self.head
        heads = []

        for name, graph in graphs.items():
            head = graph.head
            c = graph.commits[head]
            prefix = [name]
            heads.append((head, c, prefix))

        heads.sort(key=lambda x:x[1].max_date)
        entries = []
        
        prev = init
        for head, commit, prefix in heads:
            old_tree = self.repo.get_tree(commit.tree)
            tree_idx, tree = self.repo.clean_tree(commit.tree, old_tree, bad_files)

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

    def _graft(self, branch, init_tree, prefix, bad_files, fix_commit):
        self.graph = self.old_graft(self.graph, init_tree, branch, prefix, bad_files, fix_commit)
        self.head = self.graph.head
        self.named_heads.update(self.graph.named_heads)
        return self

    def graft(self, branch, graph_prefix, bad_files, fix_commit):
        init = self.head
        init_tree = self.tree
        graph = branch.graph
        count = dict(graph.parent_count)
        to_graft = []


        graph_total = len(graph.commits)
        graph_count = 0

        for idx in graph.tails:
            if idx not in self.grafts:
                if idx in graph.fragments:
                    raise Exception("fragment missing")
                c1 = graph.commits[idx]
                prefix = graph_prefix
                if isinstance(prefix, dict):
                    prefix = prefix[idx]
                if prefix and not isinstance(prefix, set):
                    raise Exception("bad prefix, must be set or dict of set")

                ctree = self.repo.get_tree(c1.tree)
                tree, ctree = self.repo.clean_tree(c1.tree, ctree, bad_files)
                c1.parents = [init]
                if prefix:
                    c1.tree, ctree = self.repo.merge_tree(init_tree, tree, prefix)
                if fix_commit is not None:
                    c1.author, c1.committer, c1.message = fix_commit(c1,  ", ".join(sorted(prefix)))
                c2 = self.repo.write_commit(c1)

                self.grafts[idx] = Graft(c2, c1, c1.tree, ctree)

            else:
                graft = self.grafts[idx]
                c1 = graft.commit
                c2 = graft.idx
                ctree = graft.tree
                self.grafts[idx] = Graft(c2, c1, c1.tree, ctree)

            graph_count += 1

            for n in graph.children[idx]:
                count[n] -= 1
                if count[n] == 0:
                    to_graft.append(n)

        new_heads = {}

        total = len(graph.linear)
        depth = 0


        while to_graft:
            idx = to_graft.pop(0)
            if idx not in self.grafts:
                prefix = graph_prefix
                if isinstance(prefix, dict):
                    prefix = prefix[idx]
                if prefix and not isinstance(prefix, set):
                    raise Exception("bad prefix, must be set or dict of set")

                c1 = graph.commits[idx]
                ctree = self.repo.get_tree(c1.tree)
                c1.tree, ctree = self.repo.clean_tree(c1.tree, ctree, bad_files)

                c1.parents = [self.grafts[p].idx for p in graph.parents[idx]]

                if prefix:
                    max_parent = max(graph.parents[idx], key=graph.linear_parent.get)
                    max_tree = self.grafts[max_parent].tree
                    c1.tree, ctree = self.repo.merge_tree(max_tree, c1.tree, prefix)

                if fix_commit is not None:
                    c1.author, c1.committer, c1.message = fix_commit(c1,  ", ".join(sorted(prefix)))

                c2 = self.repo.write_commit(c1)
                self.grafts[idx] = Graft(c2, c1, c1.tree, ctree)

            else:
                graft = self.grafts[idx]
                c1 = graft.commit
                c2 = graft.idx
                ctree = graft.tree

            graph_count += 1

            if graph.children[idx]:
                for n in graph.children[idx]:
                    count[n] -= 1
                    if count[n] == 0:
                        to_graft.append(n)
            else:
                new_heads[idx] = c2

            c_depth = graph.linear_parent[idx]
            if c_depth > depth:
                depth = c_depth
                per = graph_count/graph_total
                print(f"\r    progress {per:.2%} {graph_count} of {graph_total}", end="")

        per = graph_count/graph_total
        print(f"\r    progress {per:.2%} {graph_count} of {graph_total}")

        for x in graph.commits:
            if x not in self.grafts:
                raise Exception("missing")
    
        self.head = self.grafts[graph.head].idx
        self.named_heads.update({k: self.grafts[v].idx for k,v in graph.named_heads.items()})
        return self.head

        return fragment

#
# xxx - GitBranch
#
#       has
#           a name, a head, a tail, 
#           named heads
#           a linear history and a linear_parent
#           and contains a Graph
#       
#       branch.new_head()
#       branch.new_tail(tail, head)
#
#       repo.Branch(graph, bad_files=.., fix_message=...)
#       writer.new_head(...)
#
#       writer.graft(..., replace_trees = {init.tree:empty{}})
#
#
# xxx  shallow merge
#       should be GitBranch.shallow_merge() and then a graft?
#       
#
# xxx - GitGraph
#       interweave calls graph.union(graph) on all
#       graft calls graft.add(commit) on all
#       
#       graph.walk_forwards graph walk_backwards - out of validate
#       graph.properties = set([monotonic, monotonic-author, monotonic-committer]) 
#       

# maybe:  a branch contains prefixes 
# xxx - maybe clean branches before merging history
# maybe: interweave writes a branch always, then we graft it
# maybe we clean branches, interweave with prefix, and then graft
# maybe interweave creates a branch, merges the graph, and then calls graft

# xxx - Processor()
#       fold mkrepo.py up into more general class
#
# xxx - preserving old commit names in headers / changes
#
# xxx - named_heads to interweave can take named_heads and not just commits
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

