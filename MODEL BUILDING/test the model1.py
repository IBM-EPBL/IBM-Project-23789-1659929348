prediction=model.predict(X_test[:4])
print(prediction)

import numpy as np
print(np.argmax(prediction,axis=1))
print(y_test[:4])