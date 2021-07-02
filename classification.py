import torch
import torchvision.models
from torch.nn import Linear

import haitain
import argparse

from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, class_accuracy_metrics
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from class_strategy import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test',
                        help='Name of the result files')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--root', default="../data",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers to use for dataloading. Recommended to have more than 1')
    args = parser.parse_args()

    ######################################
    #                                    #
    # Editing below this line allowed    #
    #                                    #
    ######################################

    args.root = f"{args.root}/huawei/dataset/labeled"
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Choose between one of the two, test set creates .pkl file, validation set only prints results but is way
    # shorter.
    evaluate = 'test'
    # evaluate = 'val'

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = Linear(2048, 7, bias=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 10

    # Add any additional plugins to be used by Avalanche to this list. A template
    # is provided in class_strategy.py.
    plugins = [ClassStrategy()]

    ######################################
    #                                    #
    # No editing below this line allowed #
    #                                    #
    ######################################

    if batch_size > 10:
        raise ValueError(f"Batch size {batch_size} not allowed, should be less than or equal to 10")

    img_size = 64
    train_sets = _create_train_set(args.root, img_size)

    if evaluate == "val":
        test_sets = _create_val_set(args.root, img_size)
    else:
        test_sets, test_sets_keys = _create_test_set(args.root, img_size)

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_sets, test_datasets=test_sets)

    text_logger = TextLogger(open(f"./{args.name}.log", 'w'))
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True),
        class_accuracy_metrics(stream=True),
        loggers=[text_logger, interactive_logger])

    strategy = Naive(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=256, device=device,
        evaluator=eval_plugin, eval_every=1, plugins=plugins)

    logger = haitain.Logger()
    accuracies_test = []

    print(f"Starting training.")

    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)

        # Shuffle will be passed through to dataloader creator.
        strategy.train(experience, eval_streams=[], shuffle=False, num_workers=args.num_workers)

        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        mean_acc = results['Top1_ClassAcc_Stream/eval_phase/test_stream'].values()
        accuracies_test.append(sum(mean_acc) / len(mean_acc))

        if evaluate == "test":
            haitain.log_avalanche_results(results, test_sets_keys, logger, test_id=i, run_id=0)
            logger.build_df()
            logger.save(f'./{args.name}.pkl')

        break

    print(f"Average mean test accuracy: {sum(accuracies_test) / len(accuracies_test) * 100:.3f}%")
    print(f"Final mean test accuracy: {accuracies_test[-1] * 100:.3f}%")



def _create_val_set(root, img_size):
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

    val_set_1 = haitain.get_matching_set(root, 'val', val_match_fn_1, img_size=img_size)
    val_set_2 = haitain.get_matching_set(root, 'train', val_match_fn_2, img_size=img_size)

    return [AvalancheDataset(val_set_1, targets=val_set_1.targets()),
            AvalancheDataset(val_set_2, targets=val_set_2.targets())]


def _create_test_set(root, img_size):
    test_sets = haitain.get_haitain_domain_sets(root, 'test', ["period", "weather", "city", "location"],
                                                img_size=img_size)
    test_sets_keys = [ds.meta for ds in test_sets if len(ds) > 0]
    test_sets = [AvalancheDataset(test_set, targets=test_set.targets()) for test_set in test_sets if len(test_set) > 0]
    return test_sets, test_sets_keys


def _create_train_set(root, img_size):
    task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                  {'date': '20191111', 'period': 'Night'},
                  {'date': '20191117', 'period': 'Daytime'},
                  {'date': '20191117', 'period': 'Night'},
                  {'date': '20191120', 'period': 'Daytime'},
                  {'date': '20191121', 'period': 'Night'},]

    match_fn = (haitain.create_match_dict_fn(td) for td in task_dicts)

    train_sets = [haitain.get_matching_set(root, 'val', mf, img_size=img_size) for mf in match_fn]
    for ts in train_sets:
        ts.chronological_sort()

    return [AvalancheDataset(train_set, targets=train_set.targets()) for train_set in train_sets]


if __name__ == '__main__':
    main()
