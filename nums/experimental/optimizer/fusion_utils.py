from nums.experimental.optimizer.graph import TreeNode, FunctionNode, Leaf


def traverse_marker(node: TreeNode, marker, inputs={}):
    """
    Recursively traverse this node and return the number of unique blocks.
    If <= max_args, then it's a fusion candidate.
    """
    if isinstance(node, Leaf):
        node.marker = marker + 1
        inputs[node.marker] = node
        return marker + 1, inputs
    new_marker = marker
    for child in node.get_children():
        new_marker, inputs = traverse_marker(child, new_marker, inputs)
    return new_marker, inputs


def print_marker(node):
    """
    Recursively print the marker values of the leaves of a tree.
    """
    if isinstance(node, Leaf):
        print(node, node.marker)
        return
    for child in node.get_children():
        print_marker(child)


def set_using_marker(node, inputs):
    """
    Update the children of a n-nary function node, including FunctionNodes.
    """
    assert isinstance(node, FunctionNode)
    if len(node.get_children()) == 0:
        return node
    children = list(node.get_children())
    for child in children:
        assert isinstance(child, Leaf)
        node.update_child([child], [inputs[child.marker]])
        inputs[child.marker].parent = node
    return node