import git
import call_graph.graph_generator as gg
import numpy as np
import pandas as pd

from git import Git, Repo
from pathlib import Path
from tree_sitter import Language


LANG = "java"
gg.set_language(LANG)

# Cell
class SoftwareRepo:
    def __init__(self, repo_dir, commit, n=None):
        self.repo_dir = repo_dir
        self.repo_name = f"{repo_dir.parent.name}/{repo_dir.name}"
        self.commit = commit
        self.n = n

        # Checkout the commit for the repo
        g = Git(str(self.repo_dir))
        g.checkout(self.commit)
        g.clean(force=True, d=True)

        # Get the edges for the repo, both call graph and class graph
        self.method_df, self.call_edge_df = gg.parse_directory(str(self.repo_dir))
        self.call_edge_df.rename(
            columns={
                "callee_index": "from_id",
                "called_index": "to_id",
            },
            inplace=True,
        )
        self.class_edge_df = self.get_class_edges()
        self.edge_df = pd.concat([self.call_edge_df, self.class_edge_df]).reset_index(
            drop=True
        )

    def get_class_edges(self):
        """
        Generate edges for all methods that belong to the same class.

        Returns:
            list[tuple]: List of edges
        """
        # Get all the classes in the repo
        paths = self.method_df.groupby("path")

        # Make a list of all the edge combinations for each class
        edges = []
        for path in paths.groups:
            res = [
                (a, b)
                for idx, a in enumerate(paths.groups[path])
                for b in paths.groups[path][idx + 1 :]
            ]
            edges.extend(res)

        edges = list(zip(*edges))
        data = {"from_id": edges[0], "to_id": edges[1]}
        return pd.DataFrame(data)

