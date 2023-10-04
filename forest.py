    
from tree import DecisionTree

def random_forest(train_set, attributes, target_attr, sample_size, max_depth=4, min_samples=0, n_trees=64):
    
    trees = []

    for n in range(n_trees):
        
        trees.append(DecisionTree(max_depth=max_depth, min_samples=min_samples))

        # get random sample from train_set with replacement
        aux = train_set.sample(n=sample_size, replace=True)

        # train tree
        trees[n].train(aux, attributes, target_attr)
        print("Tree "+str(n+1)+" complete out of "+str(n_trees))
    return trees


def random_forest_evaluate(trees, test_set):
    results = {}
    for tree in trees:
        result = tree.evaluate(test_set)
        if result not in results:
            results[result] = 0

        results[result] += 1

    return max(results, key=results.get)
