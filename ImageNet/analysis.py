import pickle

features_data_val_64_1024 = pickle.load(open("/data/chy/feacture_cache/val_features_64x1024.pkl", 'rb'))
features_data_val_256_1024 = pickle.load(open("/data/chy/feacture_cache/val_features_256x1024.pkl", 'rb'))

print(features_data_val_64_1024[0].shape) #torch.Size([64, 1024])
print(features_data_val_256_1024[0].shape) # torch.Size([256, 1024])

