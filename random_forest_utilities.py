import sklearn
import numpy as np

def describe_tree(tree: sklearn.tree.DecisionTreeRegressor):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )


def see_rules(tree: sklearn.tree.DecisionTreeRegressor, X: np.ndarray, sample_id: int):

    node_indicator = tree.decision_path(X)
    leaf_id = tree.apply(X)
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature[node_id],
                value=X[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
    )
        

def see_common_nodes(tree: sklearn.tree.DecisionTreeRegressor, X: np.array, sample_ids: list):
    node_indicator = tree.decision_path(X)
    n_nodes = tree.tree_.node_count
    # boolean array indicating the nodes both samples go through
    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    # obtain node ids using position in array
    common_node_id = np.arange(n_nodes)[common_nodes]

    print(
        "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
            samples=sample_ids, nodes=common_node_id
        )
    )
    print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
            
