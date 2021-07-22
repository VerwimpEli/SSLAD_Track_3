from avalanche.benchmarks.utils import AvalancheDataset

import haitain_classification as hc


def create_val_set(root, img_size):
    def val_match_fn_1(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        date = img_annot["date"]
        return not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))

    def val_match_fn_2(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        time = img_annot['time']
        date = img_annot["date"]
        return obj.y == 6 or (obj.y == 2 and date == "20181015" and (time == '152030'
                                                                     or time == '160000'))

    val_set_1 = hc.get_matching_set(root, 'val', val_match_fn_1, img_size=img_size)
    val_set_2 = hc.get_matching_set(root, 'train', val_match_fn_2, img_size=img_size)

    return [AvalancheDataset(val_set_1, targets=val_set_1.targets()),
            AvalancheDataset(val_set_2, targets=val_set_2.targets())]


def create_test_set(root, img_size):
    test_sets = hc.get_haitain_domain_sets(root, 'test', ["period", "weather", "city", "location"],
                                           img_size=img_size)
    test_sets_keys = [ds.meta for ds in test_sets if len(ds) > 0]
    test_sets = [AvalancheDataset(test_set, targets=test_set.targets()) for test_set in test_sets if len(test_set) > 0]
    return [test_sets[0]], test_sets_keys


def create_train_set(root, img_size):
    task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                  {'date': '20191111', 'period': 'Night'},
                  {'date': '20191117', 'period': 'Daytime'},
                  {'date': '20191117', 'period': 'Night'},
                  {'date': '20191120', 'period': 'Daytime'},
                  {'date': '20191121', 'period': 'Night'}, ]

    match_fn = (hc.create_match_dict_fn(td) for td in task_dicts)

    train_sets = [hc.get_matching_set(root, 'val', mf, img_size=img_size) for mf in match_fn]
    for ts in train_sets:
        ts.chronological_sort()

    return [AvalancheDataset(train_set, targets=train_set.targets()) for train_set in train_sets]
