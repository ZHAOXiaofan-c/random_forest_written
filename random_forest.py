class Node:
  def __init__(self,data_indices,parent):
    #initialize datas, such as data indices and other parameters that is useful in later algorithm
    self.data_indices = data_indices
    self.left = None
    self.right = None
    self.split_value = None
    self.split_feature = None
    self.prob = None
    # here define the transformation from parent to data
    if parent:
      self.depth = parent.depth+1
      self.n_classes = parent.n_classes
      self.X = parent.X
      self.y = parent.y
      # this can be used in later prediction
      class_prob = np.bincount(self.y[data_indices], minlength=self.n_classes)
      self.prob = class_prob / np.sum(class_prob)
def greedy(node, cost_function, d_features): # greedy function
  #cost function: gini or others  #select d features: d
  best_cost = np.inf # define cost as infinite and decrease time by time
  # define best feature and best value which split the data
  best_feature = None
  best_value = None
  
  # randomly select d features from all features
  num_instances, all_features = node.X.shape
  if d_features == "log2":
    d_features = (np.floor(np.log2(all_features))+1).astype(int)
  if d_features == "sqrt":
    d_features = (np.floor(np.sqrt(all_features))+1).astype(int)
  features_idxs = np.random.permutation(all_features)[:d_features]

  # get average of feature value and then from all the values, we will choose the value which minimizes the cost and corresponding feature.
  data_sorted = np.sort(node.X[node.data_indices],axis=0) 
  test_candidates = (data_sorted[:-1] + data_sorted[1:]) / 2
  
  #run through all the features 
  for f in features_idxs:
    data_f = node.X[node.data_indices,f] # get the data
    #run through all the values of the feature f
    for test in test_candidates[:,f]:
      # divide left and right part
      # if a data index<test, then it belongs to the left part
      left = node.data_indices[data_f < test] 
      # if a data index>=test, then it belongs to the right part
      right = node.data_indices[data_f>=test]
      # if left or right = 0, we skip this loop and so cost still equals the cost of last loop
      # this can avoid the calculation error of cost_function 
      if len(left) == 0 or len(right) == 0:
        continue
      # calculate the left cost and right cost
      l_cost = cost_function(node.y[left])
      r_cost = cost_function(node.y[right])
      # get the number of left data and right data
      n_left = left.shape[0]
      n_right = right.shape[0]
      # calculate cost by (n_left*l_cost + n_right*r_cost)/num_instances
      cost = (n_left*l_cost + n_right*r_cost)/num_instances
      # if cost is smaller than best_cost, we update the best_cost, best_feature and best split point
      if cost < best_cost:
        best_cost = cost
        best_feature = f
        best_value = test
  return best_cost, best_feature, best_value

# there are three types of cost function that can be used in the random forest algorithm
# usually we use gini or entropy as cost function for random forest
# the model with misclassification might not be that stable, which means if we change one data point, the model could change a lot
def entropy(y):
  output = np.bincount(y) / y.size
  output = output[output>0]
  output = -np.sum(output*np.log2(output)) 
  return output
def gini(y):
  output = np.bincount(y) / y.size
  output = 1 - np.sum(np.square(output))
  return output
def misclassification(y):
  output = np.bincount(y) / y.size
  output = 1 - np.max(output)
  return output

# below shows the decision tree algorithm, which will be very useful when we define random forest class
class DecisionTree:
  # initialize the class
  def __init__(self, max_depth,min_leaf, cost_function = gini,n_classes=None):
    self.max_depth = max_depth
    self.min_leaf = min_leaf
    self.cost_function = cost_function
    self.n_classes = n_classes
    self.root = None
  # fit function, with the number of selected features, X is the data need to fit, y is the label
  def fit(self, X,y, d_features):
    # here we assume the class is [0,1,...,n-1] in all n classes, so the number of classes is max(y)+1
    if self.n_classes == None:
      self.n_classes = np.max(y)+1
    # if we didn't difine d_features, then d equals to n
    if d_features == None:
      d_features = X.shape[1]
    self.d_features = d_features
    self.X = X
    # if y = [[0],[1],[2],...], we need to flatten y
    if y.shape[0]!=1: 
      y=np.squeeze(y)
    self.y = y
    self.root = Node(np.arange(X.shape[0]), None)
    self.root.X = self.X
    self.root.y = self.y
    self.root.n_classes = self.n_classes
    self.root.depth = 0
    self.fit_tree(self.root) # fit the tree model
    return self
  def fit_tree(self,node):
    # define the stop point, if depth reaches the max_depth or number of leaf reaches the min_leaf, we can stop training
    if node.depth == self.max_depth or len(node.data_indices) == self.min_leaf:
      return
    # get cost and best split feature and split point
    cost, split_feature, split_value = greedy(node, self.cost_function, self.d_features)
    # if no cost, we return the tree directly, and later we will check this in the prediction function
    if np.isinf(cost):
      return
    # get the divide the dataset into left and right
    test = node.X[node.data_indices,split_feature]<split_value
    node.split_feature = split_feature
    node.split_value = split_value
    # get left tree
    left = Node(node.data_indices[test], node)
    self.fit_tree(left)
    # get right tree
    right = Node(node.data_indices[np.logical_not(test)], node)
    self.fit_tree(right)

    node.left = left
    node.right = right
  def predict(self,test_X):
    output = np.zeros((test_X.shape[0], self.n_classes))
    for i in range(len(test_X)):
      node = self.root
      while node.left != None:
        if test_X[i,node.split_feature] < node.split_value:
          node = node.left
        else:
          node = node.right
      output[i,:]= node.prob # this only predict something like this [[1,0,0],[0,1,0],[0,0,1]] 
    #output = np.argmax(output,axis = 1) # this can output the actural prediction class 
    return output
 
class RandomForest:
  def __init__(self, max_depth, min_leaf, n_trees, d_features, sample_size, cost_function, random_state = 1): 
    # initialize the class
    #usually d should be log2(num_of_all_features) or sqrt(num_of_all_features), in different cases, there are different selected value
    self.random_state = random_state
    self.max_depth = max_depth
    self.min_leaf = min_leaf
    self.n_trees = n_trees
    self.d_features = d_features
    self.sample_size = sample_size
    self.cost_function = cost_function
  def fit(self,X,y):
    # fit model
    np.random.seed(self.random_state)
    self.X = X
    self.y = y
    self.n_classes = np.max(self.y)+1
    self.forest = []
    nrows,ncols = X.shape
    for d in range(self.n_trees):
      # bagging / boostrapping
      bagging_i = np.floor(np.random.rand(self.sample_size)*nrows).astype(int)
      bagging_data = self.X[bagging_i,:]
      bagging_labels = self.y[bagging_i]
      tree = DecisionTree(max_depth=self.max_depth, min_leaf=self.min_leaf,cost_function=self.cost_function, n_classes=self.n_classes)
      tree.fit(bagging_data, bagging_labels, self.d_features) # get each tree
      self.forest.append(tree) 
    return self

  def predict(self, test_X):
    y_pred_collect = [[0]*(np.max(self.y)+1) for _ in range(test_X.shape[0])]
    test = []
    probs_test = []
    for tree in self.forest:
      probs_test = tree.predict(test_X)
      if (np.isnan(probs_test[0,0])): # test whether the tree exists
        continue
      y_pred_collect+=probs_test
    return np.argmax(y_pred_collect,axis = 1)
