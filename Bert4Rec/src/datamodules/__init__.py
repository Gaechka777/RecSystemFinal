# from src.datasets import dataset_factory
# from .bert import BertDataloader
#
#
# DATALOADERS = {
#     'bert': BertDataloader
# }
#
#
# def dataloader_factory(seed,
#                        bert_max_len,
#                        bert_mask_prob,
#                        train_negative_sample_size,
#                        train_negative_sampling_seed,
#                        test_negative_sample_size,
#                        test_negative_sampling_seed,
#                        train_batch_size,
#                        val_batch_size,
#                        test_batch_size,
#                        num_items):
#     dataset = dataset_factory()
#     dataloader = DATALOADERS['bert']
#     dataloader = dataloader(seed,
#                             bert_max_len,
#                             bert_mask_prob,
#                             train_negative_sample_size,
#                             train_negative_sampling_seed,
#                             test_negative_sample_size,
#                             test_negative_sampling_seed,
#                             train_batch_size,
#                             val_batch_size,
#                             test_batch_size,
#                             num_items,
#                             dataset)
#     train, val, test = dataloader.get_pytorch_dataloaders()
#     return train, val, test
