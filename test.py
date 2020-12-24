import numpy as np

def convert_pred_matrix_to_one_hot_encoding(pred_matrix, y_true):
    pred_matrix_one_hot = np.zeros(pred_matrix.shape)
    for idx, label in enumerate(y_true):
        if float(y_true[idx]) == 1.0:
            idx_max = np.argmax(pred_matrix[idx])
            pred_matrix_one_hot[idx, idx_max] = 1.0
        else:
            idx_min = np.argmin(pred_matrix[idx])
            pred_matrix_one_hot[idx, idx_min] = 1.0
    return pred_matrix_one_hot

y_true = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
pred_matrix = np.asarray([[4.73856926e-06, 0.0],
                          [0.99999785, 1.        ],
                          [0.43797284, 0.        ], 
                          [0.99999738, 1.        ], 
                          [0.99638921, 1.        ], 
                          [0.13892543, 0.        ]])
pred_matrix_one_hot = convert_pred_matrix_to_one_hot_encoding(pred_matrix, y_true)

for i in range(pred_matrix.shape[0]):
    print(pred_matrix[i])
print("")
for i in range(pred_matrix_one_hot.shape[0]):
    print(pred_matrix_one_hot[i])

# OLD one hot conversions:
# target one hot   : [0. 1.]
# target one hot   : [1. 0.]
# target one hot   : [0. 1.]
# target one hot   : [1. 0.]
# target one hot   : [0. 1.]


pred_matrix = np.random.uniform(size=(10,5))
y_true      = np.random.randint(low=0, high=2, size=(10,1))

one_hot =  convert_pred_matrix_to_one_hot_encoding(pred_matrix, y_true)
print(y_true)
for idx in range(pred_matrix.shape[0]):
    print(pred_matrix[idx])
for idx in range(one_hot.shape[0]):
    print(one_hot[idx])