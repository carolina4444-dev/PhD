def dfs(encoded_tree):
    """
    Perform a DFS on an n-ary tree encoded as a binary string.
    This function assumes '1' represents the start of a new node,
    and '0' represents the end of a node's children (moving back up the tree).

    :param encoded_tree: A binary string encoding of the n-ary tree.
    :return: None
    """

    def dfs_helper(index, depth):
        """
        Helper function for DFS.

        :param index: Current index in the binary string.
        :param depth: Current depth in the tree for visualization.
        :return: The next index to process after completing the current subtree.
        """
        # Base case: If we reach the end of the string, return.
        if index >= len(encoded_tree):
            return index

        # Assuming '1' is a node, print it or process it here.
        print(f"{'  ' * depth}Node at depth {depth}")

        # Move to the next character in the encoded string.
        index += 1

        # Go deeper into the tree as long as we encounter '1's, indicating more children/subtrees.
        while index < len(encoded_tree) and encoded_tree[index] == '1':
            # Recursive call to process each child, updating the index each time.
            index = dfs_helper(index, depth + 1)

        # Once we encounter a '0', it means we are done with this node's children and can go back up.
        return index + 1  # Skip the '0' and move to the next part of the encoded tree.

    # Start the DFS from the beginning of the encoded string and at depth 0.
    dfs_helper(0, 0)

# Example usage:
encoded_tree = "111001000"
dfs(encoded_tree)