import pickle
import numpy as np
with open('./predictions/L2/singlePeopleFull_2392022_dis.p', 'rb') as f:
  dis = pickle.load(f)
  print("100 epochs: ")
  print(type(dis), dis.shape)
  full = np.mean(np.abs(dis), axis = 0)
  print(full)
  print(np.mean(full, axis = 0))
  full = np.mean(dis, axis = 0)
  print(full)
  print(np.mean(full, axis = 0))
  print()
  

with open('./predictions/L2/singlePeopleOrigin_dis.p', 'rb') as f:
  dis = pickle.load(f)
  print("Origin Evaluation: ")
  print(type(dis), dis.shape)
  full = np.mean(np.abs(dis), axis = 0)
  print(full)
  print(np.mean(full, axis = 0))
  full = np.mean(dis, axis = 0)
  print(full)
  print(np.mean(full, axis = 0))
