'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-07 16:10:05
LastEditors: Please set LastEditors
Description: Tree structure
FilePath: /Tree2Seq/Utils/tree.py
'''
from typing import List


class TreeNode:
    def __init__(self,
                 name: str,
                 parent=None,
                 copy_mark: int = 0,
                 type: str = None,
                 schema=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.idx = -1
        self.seq_idx = -1
        self.graph_idx = -1
        self.descendants = []
        self.ancestors = []

        # indicating if the node belonging to a specific column
        self.copy_mark = copy_mark
        self.depth = -1
        self.height = -1
        self.type = type
        self.schema = schema

    def add_child(self, child, front=False):
        if front:
            self.children = [child] + self.children
        else:
            self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, children):
        self.children += children

    def set_idx(self, idx: int):
        self.idx = idx

    def set_col_mark(self, col_mark: int):
        self.col_mark = col_mark

    def set_clique_mark(self, clique_mark: int):
        self.clique_mark = clique_mark

    def set_seq_idx(self, idx):
        self.seq_idx = idx

    def set_graph_idx(self, idx):
        self.graph_idx = idx

    def set_schema(self, schema):
        self.schema = schema

    def display(self):
        # print(
        #     f"{self.name}: {' '.join(map(lambda x: x.name, self.children))}\n")
        print(f"{self.name}: {self.ancestors}")
        for child in self.children:
            child.display()


def pre_order(root: TreeNode,
              start_num: int = 0,
              start_seq_num: int = 0,
              start_graph_num: int = 0,
              k: int = 4):
    cur_num = 0
    cur_seq_num = 0
    cur_graph_num = 0
    root.set_idx(start_num)
    graph = []

    if not root.children:
        seq_ans = [root]
        root.set_seq_idx(start_seq_num)
        cur_seq_num += 1
        start_seq_num += 1
        graph_ans = []
    else:
        seq_ans = []
        root.set_graph_idx(start_graph_num)
        cur_graph_num += 1
        start_graph_num += 1
        graph_ans = [root]
        # add graph edges
        if root.parent is not None:
            graph.append([root.graph_idx, root.parent.graph_idx])
            graph.append([root.parent.graph_idx, root.graph_idx])

    cur_num += 1
    start_num += 1

    for child in root.children:
        sub_seq_nodes, cur_seq_num_, sub_graph_nodes, cur_graph_num_, cur_num_, graph_ = pre_order(
            child, start_num, start_seq_num, start_graph_num, k)

        seq_ans += sub_seq_nodes
        graph_ans += sub_graph_nodes
        cur_num += cur_num_
        start_num += cur_num_
        cur_seq_num += cur_seq_num_
        start_seq_num += cur_seq_num_
        cur_graph_num += cur_graph_num_
        start_graph_num += cur_graph_num_
        graph += graph_
    return seq_ans, cur_seq_num, graph_ans, cur_graph_num, cur_num, graph


def pre_order_(root: TreeNode):
    ans = [root]

    for child in root.children:
        ans += pre_order_(child)
    return ans


def pre_order_non_leaf(root: TreeNode):
    if not root.children:
        return []
    ans = [root]

    for child in root.children:
        ans += pre_order_non_leaf(child)
    return ans


def get_all_nodes(root: TreeNode):
    return pre_order_(root)


def get_all_non_leaf_nodes(root: TreeNode):
    return pre_order_non_leaf(root)


def flatten(root: TreeNode, k: int = 4) -> List[TreeNode]:
    """Flatten the tree in the pre-order and set the idx of all nodes.

    Args:
        root (TreeNode): the root of the tree.

    Returns:
        List[List]: the flattened tree sequence.
    """
    seq_data, _, graph_data, _, _, graph = pre_order(root, k=k)

    set_descendants(root)
    set_ancestors(root)
    depth = get_graph_depth(root, k)
    sql_num = len(seq_data)
    head_mask = get_head_mask(root, k, sql_num)
    up_to_down_mask = get_up_to_down_mask(root)
    down_to_up_mask = get_down_to_up_mask(root)
    return seq_data, graph_data, graph, depth, head_mask, up_to_down_mask, down_to_up_mask


# N, k, leaf_num, dim


def get_head_mask_helper(root, k):
    if not root.children:
        return []
    if root.height > k:
        ans = []
    else:
        ans = [(root.height, root.descendants)]
    for child in root.children:
        ans += get_head_mask_helper(child, k)
    return ans


def get_head_mask(root, k, leaf_num):
    temp = get_head_mask_helper(root, k)
    ans = [[[] for j in range(leaf_num)] for i in range(k)]

    for item in temp:
        height = item[0] - 1
        for leaf in item[1]:
            ans[height][leaf] += item[1]
    return ans


def get_up_to_down_mask(root):
    if not root.children:
        return []
    ans = [root.descendants]

    for child in root.children:
        ans += get_up_to_down_mask(child)

    return ans


def get_down_to_up_mask(root):
    if root.children:
        ans = []
    else:
        ans = [root.ancestors]

    for child in root.children:
        ans += get_down_to_up_mask(child)

    return ans


def set_ancestors(root):
    if root.parent is not None:
        root.ancestors = root.parent.ancestors + [root.parent.graph_idx]

    for child in root.children:
        set_ancestors(child)


def set_descendants(root):
    if not root.children:
        root.descendants = [root.seq_idx]
    else:
        for child in root.children:
            set_descendants(child)
            root.descendants += child.descendants


def get_graph_depth(root: TreeNode, k: int) -> List:
    nodes = get_all_non_leaf_nodes(root)

    depth = []
    for node1 in nodes:
        for node2 in nodes:
            if node2.graph_idx in set(
                    node1.ancestors) or node1.graph_idx in set(
                        node2.ancestors):
                depth.append([
                    node1.graph_idx, node2.graph_idx,
                    min(k, max(-k, node2.depth - node1.depth)) + k
                ])
    return depth


def flatten_(root: TreeNode) -> List[TreeNode]:
    return pre_order(root, 0)[0]


def get_depth(root: TreeNode) -> int:
    """Get the height of the tree.

    Args:
        root (TreeNode): the tree root.

    Returns:
        int: height.
    """
    if root.parent is None:
        root.depth = 0
    else:
        root.depth = 1 + root.parent.depth

    if not root.children:
        return root.depth

    return max([get_depth(child) for child in root.children])


def get_height(root):
    if not root.children:
        root.height = 0
    else:
        for child in root.children:
            root.height = max(root.height, 1 + get_height(child))

    return root.height


def set_max_depth(trees: List[TreeNode]) -> int:
    """Get the maximum height among trees.

    Args:
        trees (List[TreeNode]): trees.

    Returns:
        int: the maximum height.
    """
    [get_height(tree) for tree in trees]
    return max([get_depth(tree) for tree in trees])
