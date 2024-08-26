import os

import audeer
import audiofile
import audmetric as audmetric
import audformat
import numpy as np
import pandas as pd
import torch
import transformers
from omegaconf import DictConfig
import datasets

from src.data.create_data_frames import load_training_data
from trainer import Trainer, EdlTrainer, KlTrainer
from utils.loss_functions import WeightedEdlDigammaLoss, WeightedKlLoss


def train(model, cfg: DictConfig):
    torch_root = audeer.path(cfg.model_root, 'torch')

    if os.path.isdir(torch_root):
        print(f'There is already a trained model under {torch_root}')
        return torch_root

    df_train, df_dev = load_training_data(cfg.data)
    model_root = audeer.mkdir(cfg.model_root)
    important_col_lab = [[c, l] for c, l in cfg.data.important_columns_labels.items()] \
        if cfg.data.important_columns_labels else [[None, None]]
    print("important columns", important_col_lab)

    assert len(important_col_lab) == 1, 'this classification fine tuning works only for exactly 1' \
                                        'important column with class labels'

    column = important_col_lab[0][0]
    labels = sorted(df_train[column].unique())

    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for i, c in enumerate(labels)}

    print('Training Data')
    print(df_train)
    print('Duration: ', audformat.utils.duration(df_train.index))
    print(df_train.columns)
    print(df_train[column].value_counts())

    df_train['targets'] = df_train[column].map(label2id)
    if cfg.loss == 'kl_loss':
        if 'data_source_labeled_train' in cfg.data_ood:
            df_train_ood, df_dev_ood = load_training_data(cfg.data_ood)
            df_train_ood['targets'] = -1
            if len(df_train_ood) * 4 > len(df_train):
                print(
                    f'ood dataset is larger than in domain dataset, set to the same size of {int(len(df_train) / 4)} samples')
                df_train_ood = df_train_ood.sample(n=int(len(df_train) / 4))
            df_train = pd.concat([df_train, df_train_ood])
        else:
            print('No OOD data used for kl training')

    # labeled_df_train.rename(columns={column: 'targets'}, inplace=True)

    # combine target columns, as they are NaN and a value it is fine to sum
    # labeled_df_train = labeled_df_train.groupby(level=0, axis=1).sum()
    if not df_dev.empty and 'targets' not in df_dev.columns:
        df_dev[column] = df_dev[column].map(label2id)
        df_dev.rename(columns={column: 'targets'}, inplace=True)

    def data_collator(data):
        files = [d['file'] for d in data]
        starts = [d['start'] for d in data]
        ends = [d['end'] for d in data]
        labels = [d['targets'] for d in data]

        num_samples = int((cfg.sampling_rate * cfg.max_duration_sec))

        input_values = np.zeros(
            (len(files), num_samples),
            dtype=np.float64,
        )
        attention_mask = np.zeros(
            (len(files), num_samples),
            dtype=np.int32,
        )

        for idx, (file, start, end) in enumerate(zip(files, starts, ends)):
            duration = min(cfg.max_duration_sec, (pd.to_timedelta(end) - pd.to_timedelta(
                start)).total_seconds())

            signal, sampling_rate = audiofile.read(file,
                                                   duration=duration,
                                                   offset=pd.to_timedelta(start).total_seconds())
            assert sampling_rate == cfg.sampling_rate

            input_values[idx, :signal.size] = signal
            attention_mask[idx, :signal.size] = 1

        batch = transformers.BatchFeature()
        batch['input_values'] = torch.tensor(input_values).float()
        batch['attention_mask'] = torch.tensor(attention_mask)
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)

        return batch

    # metrics and criterion for categorical tasks
    metrics = {
        'UAR': audmetric.unweighted_average_recall,
        'UAP': audmetric.unweighted_average_precision,
        'UAF': audmetric.unweighted_average_fscore,
        'MSE': audmetric.mean_squared_error,
        'ACC': audmetric.accuracy,
    }

    if cfg.metric_for_best_model is None:
        cfg.metric_for_best_model = 'UAR'

    def compute_metrics(p: transformers.EvalPrediction):
        truth = p.label_ids
        logits = p.predictions[0]

        preds = np.argmax(logits, axis=1)
        scores = {name: metric(truth, preds) for name, metric in metrics.items()}
        return scores

    targets = pd.Series(df_train['targets'])
    counts = targets.value_counts().sort_index()
    train_weights = (1 / counts)
    train_weights /= train_weights.sum()
    if cfg.loss == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(train_weights).to(cfg.device))
        specific_trainer = Trainer
    elif cfg.loss == 'edl_digamma':
        criterion = WeightedEdlDigammaLoss(num_classes=len(labels),
                                           class_weights=torch.Tensor(train_weights).to(cfg.device))
        # criterion = EdlDigammaLoss(num_classes=len(labels), device = cfg.device)
        specific_trainer = EdlTrainer

    elif cfg.loss == 'kl_loss':
        # targets = pd.Series(df_train['targets']).array
        # targets = np.array([np.array(xi) for xi in targets])
        # targets = targets / targets.sum(axis=1)[:, np.newaxis]
        # counts = targets.mean(axis=0)
        # train_weights = (1 / counts)
        # train_weights /= train_weights.sum()
        if 'reverse' in cfg and not cfg['reverse']:
            criterion = WeightedKlLoss(torch.Tensor(train_weights).to(cfg.device), mode='forward')
        else:
            criterion = WeightedKlLoss(torch.Tensor(train_weights).to(cfg.device))
        specific_trainer = KlTrainer
    else:
        raise NotImplementedError('please specify cross_entropy, kl_loss or edl_digamma as training loss')

    config = transformers.AutoConfig.from_pretrained(
        cfg.model.initial_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        fine_tuning_task=cfg.model.prediction,
        problem_type='multi_label_classification',
    )
    if cfg.model.num_layers is not None:
        config.num_hidden_layers = cfg.model.num_layers
    print("config", config)

    if 'dropout' in cfg.model and cfg.model.dropout != 0.1:
        drop = cfg.model.dropout
        config.activation_dropout = drop
        config.attention_dropout = drop
        config.feat_proj_dropout = drop
        config.final_dropout = drop
        config.hidden_dropout = drop
        config.hidden_dropout_prob = drop
        config.layerdrop = drop
    print("config", config)

    setattr(config, 'sampling_rate', cfg.sampling_rate)

    model = model.from_pretrained(
        cfg.model.initial_model,
        config=config,
    )
    model.wav2vec2.feature_extractor._freeze_parameters()

    model.train()

    training_args = transformers.TrainingArguments(
        output_dir=model_root,
        logging_dir='./runs',
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        evaluation_strategy='steps',
        logging_strategy='steps',
        save_strategy='steps',
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        dataloader_drop_last=True,
        save_total_limit=2,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        load_best_model_at_end=True,
        log_level='error',
        remove_unused_columns=False,
        label_names=['labels']
    )

    dataset = {'train': datasets.Dataset.from_pandas(df_train)}

    # create test set from train set if no dev set is given
    if df_dev.empty:
        num_validation_samples = dataset["train"].num_rows * 10 // 100

        if num_validation_samples == 0:
            raise ValueError(
                "`args.validation_split_percentage` is less than a single sample "
                f"for {len(dataset['train'])} training samples. Increase "
                "`args.num_validation_split_percentage`. "
            )

        dataset['test'] = dataset['train'].select(range(num_validation_samples))
        dataset['train'] = dataset['train'].select(range(num_validation_samples, dataset['train'].num_rows))
    else:
        dataset['test'] = datasets.Dataset.from_pandas(df_dev)

    trainer = specific_trainer(
        criterion,
        args=training_args,
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    trainer.train(resume_from_checkpoint=False)
    torch_root = audeer.path(model_root, 'torch')
    trainer.save_model(torch_root)
    return torch_root
