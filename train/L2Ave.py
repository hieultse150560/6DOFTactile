import pickle
import numpy as np
with open('./predictions/L2/singlePerson_0.0001_10_best_dis.p', 'rb') as f:
  dis = pickle.load(f)
  print(type(dis), dis.shape)
  full = np.mean(np.abs(dis), axis = 0)
  print(full)
  print(np.mean(full, axis = 0))
  full = np.mean(dis, axis = 0)
  print(full)
  print(np.mean(full, axis = 0))

