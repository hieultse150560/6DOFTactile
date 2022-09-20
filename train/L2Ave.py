import pickle
with open('./predictions/L2/singlePerson_0.0001_10_best_dis.p', 'rb') as f:
  dis = pickle.load(f)
  print(type(dis))
  
