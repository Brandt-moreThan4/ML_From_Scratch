import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from pathlib import Path
import statsmodels.formula.api as smf
import sklearn
from sklearn import datasets



iris_data = datasets.load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Clean up column names
iris_df.columns = [col.replace(' (cm)', '').replace(' ','_') for col in iris_df.columns]

new_indexes = np.random.permutation(iris_df.index)
iris_df2 = iris_df#.reindex(new_indexes) #.reset_index(drop=True)

X = iris_df2.drop('species', axis='columns')
y = iris_df2['species']


def calculate_gini(y:np.ndarray) -> float:
    
    # Get the unique values of y
    unique_vals = np.unique(y)
    n = len(y)
    gini = 1 - np.sum([(np.sum(y == val) / n)**2 for val in unique_vals])
    return gini



class Model:
    
    def _clean_X(self,X,add_ones_col=True) -> np.ndarray:

        X = X.copy()

        if isinstance(X, pd.Series):
            X = X.array
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            X = np.array(X)
        if len(X.shape) == 1:
            # Convert to a 2D array even if it's just one column
            X = X.reshape(-1,1)
        
        X = np.array(X)

        # Add a column of ones to X to represent the intercept term
        if add_ones_col:
            X = np.hstack((np.ones((X.shape[0],1)), X))

        return X
    
    def _clean_y(self,y) -> np.ndarray:
        y = y.copy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        else:
            y = np.array(y)

        if len(y.shape) == 1:
            # Convert y to a 2D array even if it's just one column
            y = y.reshape(-1,1)
        y = np.array(y)

        return y

class BTree(Model):

    feature_names:list = None

    def __init__(self,max_depth:int=None,min_samples_split=2,min_samples_leaf:int=1, min_impurity_decrease=0.00001):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self,X,y) -> None:
        
        self.X = self._clean_X(X,add_ones_col=False)
        self.y = self._clean_y(y)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns


    
        self.root_node = Node(X=self.X,y=self.y, tree=self)
        self.root_node.split()


    def predict(self,X) -> np.ndarray:
        
        X = self._clean_X(X,add_ones_col=False)
        n = X.shape[0]
        predictions = [None] * n
        for i in range(n):
            current_node = self.root_node
            # We need to keep walking down the tree until we reach a leaf node, the grab the prediction
            # that the leaf node has
            while not current_node.is_leaf:
                # Figure out if we should go left or right child node
                x_row = X[i]

                if x_row[current_node.split_variable_index] < current_node.split_value:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            predictions[i] = current_node.prediction
        predictions = np.array(predictions)
        return predictions

    def summary(self):
        pass

    def calculate_model_metrics(self):
        pass

    
    def get_max_depth(self) -> int:
        
        return self.root_node.get_children_depth()
            

    def __repr__(self):
        params = {'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split, 'min_samples_leaf': self.min_samples_leaf}
        fit_stats = {'n': self.n, 'p': self.p, 'max_fit_depth': self.get_max_depth(),'base_prediction': self.root_node.prediction}
        return {**params, **fit_stats}.__repr__()


class Node:
    left_child:"Node" = None
    right_child:"Node" = None
    split_variable_index:int = None
    split_variable_name:str = None
    split_value:float = None
    prediction:str = None
    gini_reduction:float = None

    def __init__(self,X,y,tree:BTree,parent_node:"Node"=None):
        # self.indexes = np.array(indexes)
        self.tree = tree
        self.parent = parent_node       
        self.X = X
        self.y = y
        self.depth = self.calculate_depth()
        self.gini = calculate_gini(self.y)

        # Calculate the prediction for this node (Doesn't handle ties)
        self.val_counts = pd.Series(self.y.flatten()).value_counts()
        self.prediction = self.val_counts.idxmax()


        

    def split(self,recrusive:bool=True) -> None:
        '''Find the variable and value to split this node on.'''
        
        # Validate that we can split this node
        if self.depth == self.tree.max_depth:
            return None

        n = len(self.X)
        if n < self.tree.min_samples_split:
            return None

        gini_reductions = []
        var_count = self.tree.X.shape[1]

        # Loop through all predictors        
        for i in range(var_count):
            # Explore all possible split points for this predictor. I want to test all possible values as the mid-point
            # between the sorted values
            x_vals = sorted(np.unique(self.X[:,i]))
            for j in range(len(x_vals)-1):
                split_val = (x_vals[j] + x_vals[j+1]) / 2
                left_y = self.y[self.X[:,i] < split_val]
                right_y = self.y[self.X[:,i] >= split_val]

                # Check if the split is valid
                if (len(left_y) < self.tree.min_samples_leaf) or (len(right_y) < self.tree.min_samples_leaf):
                    continue

                left_gini = calculate_gini(left_y)
                right_gini = calculate_gini(right_y)
                split_gini = (len(left_y) / n) * left_gini + (len(right_y) / n) * right_gini
                gini_reduction = self.gini - split_gini
                stats = {'split_variable_index': i, 'split_value': split_val, 'gini_reduction': gini_reduction}
                gini_reductions.append(stats)
        
        if len(gini_reductions) == 0:
            print('!!!!!No valid splits found!!!')
            return None
        
        # Find the best split (# Should I bother thinking about tie breakers here?)
        best_split_stats = max(gini_reductions, key=lambda x: x['gini_reduction'])

        if best_split_stats['gini_reduction'] < self.tree.min_impurity_decrease:
            # Don't split if the gini reduction is too small.
            return None

        self.split_variable_index = best_split_stats['split_variable_index']
        self.split_value = best_split_stats['split_value']
        self.gini_reduction = best_split_stats['gini_reduction']
        if self.tree.feature_names is not None:
            self.split_variable_name = self.tree.feature_names[self.split_variable_index]
        
        # Create the left and right child nodes
        left_x = self.X[self.X[:,self.split_variable_index] < self.split_value]
        right_x = self.X[self.X[:,self.split_variable_index] >= self.split_value]
        left_y = self.y[self.X[:,self.split_variable_index] < self.split_value]
        right_y = self.y[self.X[:,self.split_variable_index] >= self.split_value]

        self.left_child = Node(X=left_x, y=left_y, tree=self.tree, parent_node=self)
        self.right_child = Node(X=right_x, y=right_y, tree=self.tree, parent_node=self)

        if recrusive:
            self.left_child.split()
            self.right_child.split()


    def calculate_depth(self) -> int:
        if self.is_root:
            return 0
        else:
            return 1 + self.parent.calculate_depth()

    def get_children_depth(self) -> int:

        # If we are at a leaf, return 0, otherwise return the max depth of the children
        if self.is_leaf:
            return 0
        else:
            left_depth = self.left_child.get_children_depth() if self.left_child is not None else 0
            right_depth = self.right_child.get_children_depth() if self.right_child is not None else 0            
            return  1 + max(left_depth, right_depth)
        

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def __repr__(self):
        interesting_stats = {
            'n': len(self.X),
            'depth': self.depth, 
            'split_var': self.split_variable_name,
            'gini': self.gini,
            'prediction': self.prediction,
            'split_variable_index': self.split_variable_index,
            'split_value': self.split_value}
        
        
        return interesting_stats.__repr__()


    



bmodel = BTree()
bmodel.fit(X,y)
max_depth = bmodel.get_max_depth()
preds = bmodel.predict(X)

print('!Done!')