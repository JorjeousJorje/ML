import numpy as np

class PotentialFunctionClassifier:
    
    
    def __init__(self,  window_size: float, kernel_func=lambda x: 1 / (x + 1)) -> None:
        self.kernel_func = kernel_func
        self.window_size: float = window_size
    
    def __set_classifier_parameters(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x: np.ndarray = train_x
        self.train_y: np.ndarray = train_y
        self.charges: np.ndarray = np.zeros_like(train_y, dtype=np.int32)
        self.classes: np.ndarray = np.unique(train_y)
    
    def __get_reference_samples(self) -> None:
        self.train_x: np.ndarray = self.train_x[self.charges > 0]
        self.train_y: np.ndarray = self.train_y[self.charges > 0]
        self.charges: np.ndarray = self.charges[self.charges > 0]
        
    def fit(self, train_x: np.ndarray, train_y: np.ndarray, num_epoch: int=5) -> None:
        assert train_x.shape[0] == train_y.shape[0]
        
        self.__set_classifier_parameters(train_x, train_y)
        
        for _ in range(num_epoch):
            for i, x_sample in enumerate(self.train_x):
                if self.predict(x_sample) != self.train_y[i]:
                    self.charges[i] += 1
                    
        self.__get_reference_samples()
                    
        
                
    def predict(self, test_x_sample: np.ndarray) -> np.ndarray:
        
        if len(test_x_sample.shape) < 2:
            test_x_sample = test_x_sample.copy()
            test_x_sample = test_x_sample[None, :]
        
        diffs: np.ndarray = test_x_sample[:, None] - self.train_x[None, :]
        assert diffs.shape[0] == test_x_sample.shape[0] and diffs.shape[1] == self.train_x.shape[0]
        assert diffs.shape[2] == test_x_sample.shape[1] and test_x_sample.shape[1] == self.train_x.shape[1]
        
        dists: np.ndarray = np.sqrt(np.sum(diffs ** 2,  axis=-1))
        assert dists.shape[0] == test_x_sample.shape[0] and dists.shape[1] == self.train_x.shape[0]
        
        weight: np.ndarray = self.charges * self.kernel_func(dists / self.window_size)
        assert weight.shape[0] == test_x_sample.shape[0] and weight.shape[1] == self.train_x.shape[0]
        
        result_predictions: np.ndarray = np.zeros((test_x_sample.shape[0], self.classes.size))
        
        for class_ in self.classes:
            result_predictions[:, class_] = np.sum(weight[:, self.train_y == class_], axis=-1)
            
        return np.argmax(result_predictions, axis=-1)
    








































# weight[:, self.train_y == class_] tells classifier what weights
# correspond to specific class. For instance, if class == 0 we will use
# weights that correspond to zero class samples from train_x
# then we summ those weights to tell what class is the closes of the most relevant
# TODO (about weights): for example, test_x_sample has shape of (5, 4) where
# 5 -> num of samples, 4 -> num of features. So, weight will have shape of 
# (4, self.train_x.shape). Weights contain weights for ALL classes!!! (it is really important)
# And after that we summarize weigts that correspont to specific class (weight[:, self.train_y == class_])!
# So, for example, we want to take weights that correspond to 0 class. For that we 
# use weight[:, self.train_y == 0]. self.train_y == 0 is a mask that tells us what weights to take!
# Let's consider that np.sum(self.train_y == 0) == 50 (50 elements of train_x were of 0 class). 
# So, in that situation the shape of weight[:, self.train_y == class_] will be (5, 50)
# And then we summarize along axis = 1 to get array of shape (5, ) that will contain information
# about 0 class affiliation for those 5 samples!
            
            