import pickle

with open('raw_smpl/gvhmrmotion.pkl','rb') as f:
    d = pickle.load(f)

print(d.keys())
for k, v in d.items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    else:
        print(k, type(v))
