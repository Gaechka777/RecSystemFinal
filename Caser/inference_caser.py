from src.models.caser import CaserModel

import torch
import numpy as np

def make_predict(train_DL, user_ids, ckpt_path, k = 5):
    model = CaserModel(data_dir="data",dataset_name="train_ver2.csv", dataset_name_test="train_ver2.csv")

    print('Resume training from %s' % ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])




    train_dataset = train_DL.dataset
    indexes = [train_dataset.users.tolist().index([x]) for x in user_ids]

    sequence, user, target = train_dataset[indexes][:3]

    target = torch.tensor(np.array([np.arange(0,23), np.arange(0,23)]),dtype=torch.int)

    scores = model.forward(sequence, user, target, for_pred=True)
    
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    print("Predictions for users:")
    result = {user_ids[i]: cut[i] for i in range(len(user_ids))}
    print(result)
    