from src.datamodules.base_datamodule import BertDataModule

if __name__ == "__main__":
    data = BertDataModule(seed=0, bert_max_len=100, bert_mask_prob=0.15,
                          train_negative_sample_size=0, train_negative_sampling_seed=0,
                          test_negative_sample_size=10, test_negative_sampling_seed=98765,
                          train_batch_size=64, val_batch_size=64, test_batch_size=64,
                          num_workers=0, pin_memory=False)
    data.prepare_data()
    data.setup()
    train_DL = data.train_dataloader()
    Data = next(iter(train_DL))
    print(Data)
    print(len(Data))
    print(Data[0].shape)
    print(Data[1].shape)