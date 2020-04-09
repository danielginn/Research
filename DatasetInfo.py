import sys
def GetDatasetInfo(dataset):

    if (dataset == 'NUbotsField'):
        scene_info = [{'num_train_images': 9268, 'num_test_images': 14975}]

    elif (dataset == '7scenes'):
        scene_info = [{'name':'chess', 'train_sequences': [1, 2, 4, 6], 'test_sequences': [3, 5], 'num_images': 1000},
                  {'name':'fire', 'train_sequences': [1, 2], 'test_sequences': [3, 4], 'num_images': 1000},
                  {'name':'heads', 'train_sequences': [2], 'test_sequences': [1], 'num_images': 1000},
                  {'name':'office', 'train_sequences': [1, 3, 4, 5, 8, 10], 'test_sequences': [2, 6, 7, 9], 'num_images': 1000},
                  {'name':'pumpkin', 'train_sequences': [2, 3, 6, 8], 'test_sequences': [1, 7], 'num_images': 1000},
                  {'name':'redkitchen', 'train_sequences': [1, 2, 5, 7, 8, 11, 13], 'test_sequences': [3, 4, 6, 12, 14], 'num_images': 1000},
                  {'name':'stairs', 'train_sequences': [2, 3, 5, 6], 'test_sequences': [1, 4], 'num_images': 500}]

    else:
        sys.exit('Unknown dataset')

    return scene_info