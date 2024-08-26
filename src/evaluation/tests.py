import os
from typing import Dict, List

import audb
import audinterface
import audmetric
import audplot
import auglib as auglib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from omegaconf import omegaconf, open_dict
from scipy.stats import entropy
import seaborn as sns

from src.evaluation.edl_uncertainty import compute_df_edl_uncertainty


def test_ood_data(df_true, prediction_classes, predict_func, test_cfg_name, test_cfg, cfg):
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'

    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            max_signal_dur=8.0,
            resample=True,
            verbose=True,
        )
        if test_cfg.uncertainty_method == 'mc_dropout':
            if 'mc_dropout_forward_passes' not in test_cfg:
                print('No number of MC Dropout forward passes specified, set it to 20. ')
                with open_dict(test_cfg):
                    test_cfg.mc_dropout_forward_passes = 20

            pred_dfs = []
            # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
            for i in range(test_cfg.mc_dropout_forward_passes):
                print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
                new_df = interface.process_index(df_true.index, preserve_index=True)
                pred_dfs.append(new_df)

            df_pred = pd.concat(pred_dfs)
        else:
            df_pred = interface.process_index(df_true.index, preserve_index=True)

        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)

    if test_cfg.uncertainty_method == 'mc_dropout':
        def calc_var(x):
            return x.var().sum()

        variance = df_pred.groupby(level=df_pred.index.names).apply(calc_var)
        df_pred = df_pred.groupby(level=df_pred.index.names).mean()
        df_pred['variance'] = variance

    scores = {}

    df_uncertain = get_uncertainty(uncertainty_method=test_cfg.uncertainty_method,
                                   classes=prediction_classes,
                                   df_pred=df_pred,
                                   )

    if 'data_source_labeled_test' in test_cfg:
        df_uncertain['db'] = '-'.join([f'{x.name}-{x.version}-{x.table}' for x in test_cfg.data_source_labeled_test])
    elif 'data_source_artificaial_test' in test_cfg:
        df_uncertain['db'] = test_cfg.data_source_artificaial_test
    else:
        raise NotImplementedError('No datasource selected')

    df_uncertain['uncertainty_method'] = f'{test_cfg.uncertainty_method}'
    if 'post_processing' in test_cfg:
        df_uncertain['post_processing'] = f'{test_cfg.post_processing}'
    else:
        df_uncertain['post_processing'] = 'without'

    col_name = test_cfg.important_columns_labels
    df_uncertain[col_name] = df_true[col_name]

    if 'plot_combined' not in test_cfg or test_cfg.plot_combined:
        df_uncertain['plot_combined'] = True
    else:
        df_uncertain['plot_combined'] = False

    print(f'Save {test_cfg_name} to combined evaluation')
    cache_path_combined = f'{cfg.results_root}/{cfg.testing.combined_df}.pkl'
    if not os.path.exists(cache_path_combined):
        df_combined = df_uncertain.copy()
        df_combined.reset_index().to_pickle(cache_path_combined)
    else:
        # read combined df from pickle, update rows if new data is available, add columns if index did not exist yet
        df_combined = pd.read_pickle(cache_path_combined).set_index(
            ['db', 'uncertainty_method', 'post_processing', 'file', 'start', 'end'])
        df_uncertain = df_uncertain.reset_index().set_index(
            ['db', 'uncertainty_method', 'post_processing', 'file', 'start', 'end'])
        df_combined = df_combined.reindex(columns=df_combined.columns.union(df_uncertain.columns))
        df_uncertain = df_uncertain.reindex(columns=df_uncertain.columns.union(df_combined.columns))

        df_combined = pd.concat([df_combined, df_uncertain])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined.reset_index().to_pickle(cache_path_combined)

    print(df_uncertain.columns)
    scores[f'{test_cfg_name}.uncertainty'] = df_uncertain['uncertainty'].mean()
    scores[f'{test_cfg_name}.std'] = df_uncertain['uncertainty'].std()

    if 'precision' in df_uncertain and 'mode_max' in df_uncertain:
        scores[f'{test_cfg_name}.precision.mean'] = df_uncertain['precision'].mean()
        scores[f'{test_cfg_name}.precision.std'] = df_uncertain['precision'].std()
        scores[f'{test_cfg_name}.mode_max.mean'] = df_uncertain['mode_max'].mean()
        scores[f'{test_cfg_name}.mode_max.std'] = df_uncertain['mode_max'].std()

    if 'emotion' in df_uncertain:
        in_emotion_mask = df_uncertain['emotion'].isin(['anger', 'happiness', 'sadness', 'neutral'])
        in_emo = df_uncertain[in_emotion_mask]
        out_emo = df_uncertain[~in_emotion_mask]
        if not in_emo.empty:
            scores[f'{test_cfg_name}.uncertainty.in'] = in_emo['uncertainty'].mean()
            if 'precision' in df_uncertain and 'mode_max' in df_uncertain:
                scores[f'{test_cfg_name}.precision.in'] = in_emo['precision'].mean()
                scores[f'{test_cfg_name}.mode_max.in'] = in_emo['mode_max'].mean()
        if not out_emo.empty:
            scores[f'{test_cfg_name}.uncertainty.out'] = out_emo['uncertainty'].mean()
            if 'precision' in df_uncertain and 'mode_max' in df_uncertain:
                scores[f'{test_cfg_name}.precision.out'] = out_emo['precision'].mean()
                scores[f'{test_cfg_name}.mode_max.out'] = out_emo['mode_max'].mean()

    return scores


def dirichlet_mode_max(parameters):
    a0 = np.sum(parameters) - len(parameters)
    mode = [max((a - 1), 0) / a0 for a in parameters]
    return max(mode)


def get_uncertainty(uncertainty_method,
                    classes,
                    df_pred: pd.DataFrame,
                    ):
    df = pd.DataFrame(index=df_pred.index)

    if uncertainty_method == 'entropy':
        df['uncertainty'] = df_pred.apply(entropy, axis=1)
        df['prediction'] = df_pred.idxmax(axis=1)
        df['precision'] = np.log(df_pred.sum(axis=1))
        df['mode_max'] = df_pred.apply(dirichlet_mode_max, axis=1)
    elif uncertainty_method == 'max_logit':
        df['uncertainty'] = 1 - df_pred.max(axis=1)
        df['prediction'] = df_pred.idxmax(axis=1)
    elif uncertainty_method == 'mc_dropout':
        df['uncertainty'] = df_pred.pop('variance')
        df['prediction'] = df_pred.idxmax(axis=1)
    elif uncertainty_method == 'edl':
        df['uncertainty'] = compute_df_edl_uncertainty(df_pred, classes)
        df['prediction'] = df_pred.idxmax(axis=1)
    else:
        print('No uncertainty prediction method specified')
        return None

    df = df.dropna()
    return df


def test_categorical_noise(df_true, prediction_classes, predict_func, test_cfg_name, test_cfg, cfg):
    test_col_lab = [[c, l] for c, l in test_cfg.important_columns_labels.items()] \
        if test_cfg.important_columns_labels else [[None, None]]
    assert len(test_col_lab) == 1 and test_col_lab[0][1] is not None, \
        'Select one column which should be used for this test'

    snr_dfs = pd.DataFrame()
    scores = {}
    # for snr in [-30, -20, -10, 0, 10, 20, 30]:
    for snr in [-10, -5, 0, 5, 10, 15, 20, 25, 30]:
        if test_cfg.noise == 'white_noise':
            transform = auglib.transform.Mix(
                auglib.transform.WhiteNoiseGaussian(),
                snr_db=snr,
                transform=auglib.transform.BandPass(
                    center=4000,
                    bandwidth=1000,
                ),
            )
        elif test_cfg.noise == 'added_files':

            noise_files = audb.load(
                name=test_cfg.added_db[0].name,
                version=test_cfg.added_db[0].version,
                tables=[test_cfg.added_db[0].table],
                format="wav",
                sampling_rate=cfg.sampling_rate,
                channels=None,
                mixdown=True,
            )

            transform = auglib.transform.Mix(
                auglib.observe.List(noise_files.files, draw=True),
                snr_db=snr,
                loop_aux=True,
            )

        augment = auglib.Augment(transform, num_workers=20)
        df_true_augment = augment.augment(df_true)

        cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.{snr}.pred.pkl'

        if not os.path.exists(cache_path_pred):
            interface = audinterface.Feature(
                prediction_classes,
                process_func=predict_func,
                sampling_rate=cfg.sampling_rate,
                resample=True,
                verbose=True,
            )
            if test_cfg.uncertainty_method == 'mc_dropout':
                if 'mc_dropout_forward_passes' not in test_cfg:
                    print('No number of MC Dropout forward passes specified, set it to 20. ')
                    with open_dict(test_cfg):
                        test_cfg.mc_dropout_forward_passes = 20

                pred_dfs = []
                # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
                for i in range(test_cfg.mc_dropout_forward_passes):
                    print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
                    new_df = interface.process_index(df_true_augment.index, preserve_index=True)
                    pred_dfs.append(new_df)

                df_pred = pd.concat(pred_dfs)
            else:
                df_pred = interface.process_index(df_true_augment.index, preserve_index=True)

            df_pred.to_pickle(cache_path_pred)

        df_pred = pd.read_pickle(cache_path_pred)

        if test_cfg.uncertainty_method == 'mc_dropout':
            def calc_var(x):
                return x.var().sum()

            variance = df_pred.groupby(level=df_pred.index.names).apply(calc_var)
            df_pred = df_pred.groupby(level=df_pred.index.names).mean()
            df_pred['variance'] = variance

        df_snr = df_pred.copy()
        df_snr['group'] = snr
        if snr_dfs.empty:
            snr_dfs = df_snr
        else:
            snr_dfs = pd.concat([snr_dfs, df_snr])

        scores[f'snr[{snr}]'] = eval_correctness_uncertainty(
            name=f'{test_cfg_name}.{snr}',
            uncertainty_method=test_cfg.uncertainty_method,
            classes=prediction_classes,
            df_true=df_true_augment,
            df_pred=df_pred,
            cfg=cfg,
            test_cfg=test_cfg,
            per_class=False,
            for_snr=True,
        )

    create_ecdf_for_groups(
        df_pred=snr_dfs,
        df_true=df_true,
        uncertainty_method=test_cfg.uncertainty_method,
        classes=prediction_classes,
        test_cfg_name=test_cfg_name,
        cfg=cfg,
    )

    return scores


def create_ecdf_for_groups(
        df_pred: pd.DataFrame,
        df_true: pd.DataFrame,
        uncertainty_method,
        classes,
        test_cfg_name,
        cfg,
):
    groups = df_pred.pop('group')
    assert len(df_true.columns) == 1

    df = get_uncertainty(uncertainty_method,
                         classes,
                         df_pred)

    df_true = pd.concat([df_true] * int(len(df) / len(df_true)))
    df['target'] = df_true[df_true.columns[0]].values
    df['group'] = groups
    df = df.dropna()

    sns.ecdfplot(df, x='uncertainty', hue='group')
    plt.savefig(f'{cfg.results_root}/{test_cfg_name}.ecdf.group.png')
    plt.close()

    def df_uar(df):
        uar = audmetric.unweighted_average_recall(df['target'], df['prediction'])
        return uar

    # df['correct']= df['target'] == df['prediction']
    df['correct'] = df.apply(lambda row: 'correct' if row.prediction == row.target else 'false', axis=1)
    uar_df = df.groupby(['group']).apply(df_uar)
    if 'mc' not in test_cfg_name:
        plt.ylim(0, np.log(4))
    plt.close()
    ax = sns.lineplot(df, x='group', y='uncertainty', hue='correct', legend='brief')
    ax.set_xlabel('SNR')
    ax.set_ylabel('uncertainty')
    ax.legend(title='', loc='upper left')
    ax2 = plt.twinx()
    sns.lineplot(data=uar_df, color="g", ax=ax2)
    ax2.set_ylabel('UAR')
    ax2.set_ylim([0.3, 1.0])
    plt.legend(title='', loc='upper right', labels=['UAR'])
    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{test_cfg_name}.snr.lineplot.png')
    plt.close()


def test_categorical(df_true, prediction_classes, predict_func, test_cfg_name, test_cfg, cfg):
    test_col_lab = [[c, l] for c, l in test_cfg.important_columns_labels.items()] \
        if test_cfg.important_columns_labels else [[None, None]]
    assert len(test_col_lab) == 1 and test_col_lab[0][1] is not None, \
        'Select one column which should be used for this test'
    test_task = test_col_lab[0][0]
    test_classes = test_col_lab[0][1]

    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'
    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            resample=True,
            verbose=True,
        )
        df_pred = interface.process_index(df_true.index, preserve_index=True)
        df_pred.to_pickle(cache_path_pred)
    df_pred = pd.read_pickle(cache_path_pred)

    return eval_categorical(
        test_cfg_name,
        test_task,
        test_classes,
        df_true,
        df_pred,
        cfg,
    )


def eval_categorical(name, task, classes, df_true, df_pred, cfg):
    scores = {}

    y_true = df_true[task]
    y_pred = df_pred.idxmax(axis=1)

    scores['UAR'] = audmetric.unweighted_average_recall(y_true, y_pred)
    scores['ACC'] = audmetric.accuracy(y_true, y_pred)

    plt.clf()
    _, ax = plt.subplots(1, 1, figsize=[len(classes) + 1, len(classes)])

    audplot.confusion_matrix(
        y_true,
        y_pred,
        percentage=True,
        show_both=True,
        labels=classes,
        ax=ax,
    )

    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{name}.cm.png')
    plt.close()

    with open(f'{cfg.results_root}/{name}.yaml', 'w') as fp:
        yaml.dump(scores, fp)
    return scores


def test_deviaion(df_true, classes, predict_func, test_cfg_name, test_cfg, cfg):
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'
    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            resample=True,
            verbose=True,
        )
        if 'mc_dropout_forward_passes' not in test_cfg:
            print('No number of MC Dropout forward passes specified, set it to 20. ')
            with open_dict(test_cfg):
                test_cfg.mc_dropout_forward_passes = 20

        pred_dfs = []
        # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
        for i in range(test_cfg.mc_dropout_forward_passes):
            print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
            new_df = interface.process_index(df_true.index, preserve_index=True)
            pred_dfs.append(new_df)

        df_pred = pd.concat(pred_dfs)
        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)

    vars = []
    for c in classes:
        vars.append(df_pred[c].groupby(level=df_pred.index.names).var())

    df_pred = df_pred.groupby(level=df_pred.index.names).mean()

    for i, c in enumerate(classes):
        df_pred[f'{c}-var'] = vars[i]

    return eval_dimensional_deviation(
        name=test_cfg_name,
        uncertainty_method=test_cfg.uncertainty_method,
        classes=classes,
        df_true=df_true,
        df_pred=df_pred,
        cfg=cfg, )


def eval_dimensional_deviation(name,
                               uncertainty_method,
                               classes,
                               df_true: pd.DataFrame,
                               df_pred: pd.DataFrame,
                               cfg: omegaconf.DictConfig,
                               ):
    scores = {}
    true_col = df_true.columns
    df = pd.DataFrame(index=df_pred.index)
    for c in true_col:
        df[f'{c}-target'] = df_true[c]
        df[f'{c}-prediction'] = df_pred[c]
    if uncertainty_method == 'entropy':
        raise NotImplementedError('entropy does not work for dimensional tasks')
    elif uncertainty_method == 'max_logit':
        raise NotImplementedError('max logit does not work for dimensional tasks')
    elif uncertainty_method == 'mc_dropout':
        for c in true_col:
            df[f'{c}-uncertainty'] = df_pred.pop(f'{c}-var')
    elif uncertainty_method == 'edl':
        raise NotImplementedError('edl does not work for dimensional tasks')
    else:
        print('No uncertainty prediction method specified')
        return None
    df = df.dropna()

    steps = 5
    plt.clf()
    _, ax = plt.subplots(len(true_col), steps, figsize=[5 * steps + 1, 5 * len(true_col)])

    for h, c in enumerate(true_col):

        scores[f'{c}-CCC'] = audmetric.concordance_cc(df[f'{c}-target'], df[f'{c}-prediction'])
        scores[f'{c}-MSE'] = audmetric.mean_squared_error(df[f'{c}-target'], df[f'{c}-prediction'])
        scores[f'{c}-Mean Uncertainty'] = df[f'{c}-uncertainty'].mean() * 100
        scores[f'{c}- PCC Uncertainty MSE'] = audmetric.pearson_cc(df[f'{c}-uncertainty'], np.square(
            (df[f'{c}-target'] - df[f'{c}-prediction'])))
        scores[f'{c}- PCC Uncertainty MAE'] = audmetric.pearson_cc(df[f'{c}-uncertainty'],
                                                                   np.abs((df[f'{c}-target'] - df[f'{c}-prediction'])))
        df_task = df.copy()
        steps = 5

        for i in range(steps):
            df2 = df_task.nlargest(int(len(df) / steps), f'{c}-uncertainty')
            df_task.drop(df2.index, inplace=True)
            scores[f'{c}-CCC {(100 / steps) * i}% - {(100 / steps) * (i + 1)}%'] = \
                audmetric.concordance_cc(df2[f'{c}-target'], df2[f'{c}-prediction'])
            scores[f'{c}-MSE {(100 / steps) * i}% - {(100 / steps) * (i + 1)}%'] = \
                audmetric.mean_squared_error(df2[f'{c}-target'], df2[f'{c}-prediction'])

            audplot.scatter(
                df2[f'{c}-target'].astype('float'),
                df2[f'{c}-prediction'].astype('float'),
                ax=ax[h][i],
                fit=True,
            )
            ax[h][i].set_title(f'{c}-{(100 / steps) * i}% - {(100 / steps) * (i + 1)}%')

        plt.tight_layout()
        plt.savefig(f'{cfg.results_root}/{name}.cm.png')
        plt.close()

    with open(f'{cfg.results_root}/{name}.yaml', 'w') as fp:
        yaml.dump(scores, fp)
    return scores


def test_dimensional(df_true, classes, predict_func, test_cfg_name, test_cfg, cfg):
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'
    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            resample=True,
            verbose=True,
        )
        df_pred = interface.process_index(df_true.index, preserve_index=True)
        df_pred.to_pickle(cache_path_pred)
    df_pred = pd.read_pickle(cache_path_pred)

    return eval_dimensional(
        test_cfg_name,
        list(test_cfg.important_columns_labels.keys()),
        df_true,
        df_pred,
        cfg,
    )


def eval_dimensional(name, classes, df_true, df_pred, cfg):
    scores = {}

    plt.clf()
    _, ax = plt.subplots(len(classes), 2, figsize=[15, 10])

    print(classes)
    for i, task in enumerate(classes):
        df_test = pd.DataFrame(index=df_true.index)

        # remove columns with nan values
        df_test['true'] = df_true[task]
        df_test['pred'] = df_pred[task]
        df_test.dropna(inplace=True)
        y_true = df_test['true']
        y_pred = df_test['pred']

        # calculate scores
        scores[f'{task}-CCC'] = audmetric.concordance_cc(y_true, y_pred)
        scores[f'{task}-PCC'] = audmetric.pearson_cc(y_true, y_pred)
        scores[f'{task}-MSE'] = audmetric.mean_squared_error(y_true, y_pred)
        scores[f'{task}-MAE'] = audmetric.mean_absolute_error(y_true, y_pred)

        # create plots
        audplot.distribution(
            y_true,
            y_pred,
            ax=ax[i][0],
        )
        audplot.scatter(
            y_true,
            y_pred,
            ax=ax[i][1],
            # fit=True,
        )
        ax[i][0].set_title(f'{task}-dist')
        ax[i][1].set_title(f'{task}-scatter')

    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{name}.png')
    plt.close()

    with open(f'{cfg.results_root}/{name}.yaml', 'w') as fp:
        yaml.dump(scores, fp)
    return scores


def test_uncertainty(df_true, prediction_classes, predict_func, test_cfg_name, test_cfg, cfg) -> Dict:
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'
    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            resample=True,
            verbose=True,
        )

        df_pred = interface.process_index(df_true.index, preserve_index=True)
        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)
    per_class = False
    if 'per_class_evaluation' in test_cfg and test_cfg.per_class_evaluation:
        per_class = True
    if test_cfg.type == 'agreement':
        return eval_rater_agreement(
            name=test_cfg_name,
            uncertainty_method=test_cfg.uncertainty_method,
            classes=prediction_classes,
            agreement_column=test_cfg.agreement_column,
            df_true=df_true,
            df_pred=df_pred,
            cfg=cfg,
            per_class=per_class,
        )
    elif test_cfg.type == 'correctness':
        return eval_correctness_uncertainty(
            name=test_cfg_name,
            uncertainty_method=test_cfg.uncertainty_method,
            classes=prediction_classes,
            df_true=df_true,
            df_pred=df_pred,
            cfg=cfg,
            test_cfg=test_cfg,
            per_class=per_class,
        )


def test_uncertainty_dropout(df_true, prediction_classes, predict_func, test_cfg_name, test_cfg, cfg) -> Dict:
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'
    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            resample=True,
            verbose=True,
        )
        if 'mc_dropout_forward_passes' not in test_cfg:
            print('No number of MC Dropout forward passes specified, set it to 20. ')
            with open_dict(test_cfg):
                test_cfg.mc_dropout_forward_passes = 20

        pred_dfs = []
        # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
        for i in range(test_cfg.mc_dropout_forward_passes):
            print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
            new_df = interface.process_index(df_true.index, preserve_index=True)
            pred_dfs.append(new_df)

        df_pred = pd.concat(pred_dfs)
        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)

    def calc_var(x):
        return x.var().sum()

    variance = df_pred.groupby(level=df_pred.index.names).apply(calc_var)
    df_pred = df_pred.groupby(level=df_pred.index.names).mean().reindex(df_true.index)

    df_pred['variance'] = variance
    per_class = False
    if 'per_class_evaluation' in test_cfg and test_cfg.per_class_evaluation:
        per_class = True

    if test_cfg.type == 'agreement':
        return eval_rater_agreement(
            name=test_cfg_name,
            uncertainty_method=test_cfg.uncertainty_method,
            classes=prediction_classes,
            agreement_column=test_cfg.agreement_column,
            df_true=df_true,
            df_pred=df_pred,
            cfg=cfg,
            per_class=per_class,
        )
    elif test_cfg.type == 'correctness':
        return eval_correctness_uncertainty(
            name=test_cfg_name,
            uncertainty_method=test_cfg.uncertainty_method,
            classes=prediction_classes,
            df_true=df_true,
            df_pred=df_pred,
            cfg=cfg,
            test_cfg=test_cfg,
            per_class=per_class,
        )


def eval_correctness_uncertainty(name,
                                 uncertainty_method,
                                 classes,
                                 df_true: pd.DataFrame,
                                 df_pred: pd.DataFrame,
                                 cfg: omegaconf.DictConfig,
                                 test_cfg: omegaconf.DictConfig,
                                 per_class: bool = False,
                                 for_snr: bool = False,
                                 ):
    scores = {}

    df = get_uncertainty(uncertainty_method,
                         classes,
                         df_pred)

    df_true = pd.concat([df_true] * int(len(df) / len(df_true)))
    df['target'] = df_true[df_true.columns[0]].values

    df_uncertain = df.copy()
    df_uncertain['db'] = '-'.join([f'{x.name}-{x.version}-{x.table}' for x in test_cfg.data_source_labeled_test])
    df_uncertain['uncertainty_method'] = f'{test_cfg.uncertainty_method}'
    if 'post_processing' in test_cfg:
        df_uncertain['post_processing'] = f'{test_cfg.post_processing}'
    else:
        df_uncertain['post_processing'] = 'without'

    if not for_snr:
        cache_path_combined = f'{cfg.results_root}/{cfg.testing.combined_df}.pkl'
        if not os.path.exists(cache_path_combined):
            df_uncertain.reset_index().to_pickle(cache_path_combined)
            print('save', df_uncertain)
        else:
            # read combined df from pickle, update rows if new data is available, add columns if index did not exist yet
            df_combined = pd.read_pickle(cache_path_combined).set_index(
                ['db', 'uncertainty_method', 'post_processing', 'file', 'start', 'end'])
            df_uncertain = df_uncertain.reset_index().set_index(
                ['db', 'uncertainty_method', 'post_processing', 'file', 'start', 'end'])
            df_combined = df_combined.reindex(columns=df_combined.columns.union(df_uncertain.columns))
            df_uncertain = df_uncertain.reindex(columns=df_uncertain.columns.union(df_combined.columns))

            df_combined = pd.concat([df_combined, df_uncertain])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined.reset_index().to_pickle(cache_path_combined)

    scores['uncertainty.mean'] = df_uncertain['uncertainty'].mean()
    scores['uncertainty.std'] = df_uncertain['uncertainty'].std()

    if 'precision' in df_uncertain:
        scores['precision.mean'] = df_uncertain['precision'].mean()
        scores['precision.std'] = df_uncertain['precision'].std()
        scores['pcc-uncertain-precision'] = audmetric.pearson_cc(df_uncertain['uncertainty'], df_uncertain['precision'])

    if 'mode_max' in df_uncertain:
        scores['mode_max.mean'] = df_uncertain['mode_max'].mean()
        scores['mode_max.std'] = df_uncertain['mode_max'].std()
        scores['pcc-uncertain-mode_max'] = audmetric.pearson_cc(df_uncertain['uncertainty'],
                                                                df_uncertain['mode_max'])

    scores['UAR'] = audmetric.unweighted_average_recall(df['target'], df['prediction'])
    scores['Acc'] = audmetric.accuracy(df['target'], df['prediction'])

    true_df = df[df['target'] == df['prediction']]['uncertainty']
    false_df = df[df['target'] != df['prediction']]['uncertainty']
    scores['mean uncertainty correct'] = true_df.mean()
    scores['mean uncertainty incorrect'] = false_df.mean()
    # the uncertainty for incorrect samples is x times higher
    scores['uncertainty ratio'] = false_df.mean() / true_df.mean()

    df['correct'] = df.apply(lambda row: 'correct' if row.prediction == row.target else 'false', axis=1)
    df.sort_values(by=['correct'], inplace=True)
    plt.close()
    sns.ecdfplot(df, x='uncertainty', hue='correct')
    plt.legend(title='', labels=['false', 'correct'], loc='lower right')
    plt.xlabel('uncertainty')

    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{name}.png')
    plt.close()

    df_len = len(df)
    steps = 5

    plt.clf()
    _, ax = plt.subplots(1, steps, figsize=[len(classes) * steps + 1, len(classes)])

    for i in range(steps):
        df2 = df.nlargest(int(df_len / steps), 'uncertainty')
        df.drop(df2.index, inplace=True)
        scores[f'UAR {(100 / steps) * i}% - {(100 / steps) * (i + 1)}%'] = \
            audmetric.unweighted_average_recall(df2['target'], df2['prediction'])
        scores[f'Acc {(100 / steps) * i}% - {(100 / steps) * (i + 1)}%'] = \
            audmetric.accuracy(df2['target'], df2['prediction'])

        audplot.confusion_matrix(
            df2['target'],
            df2['prediction'],
            percentage=True,
            show_both=True,
            labels=classes,
            ax=ax[i],
        )
        ax[i].set_title(f'{(100 / steps) * i}% - {(100 / steps) * (i + 1)}%')

    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{name}.cm.png')
    plt.close()

    with open(f'{cfg.results_root}/{name}.yaml', 'w') as fp:
        yaml.dump(scores, fp)
    return scores


def eval_rater_agreement(name,
                         uncertainty_method,
                         classes: List,
                         agreement_column,
                         df_true: pd.DataFrame,
                         df_pred: pd.DataFrame,
                         cfg: omegaconf.DictConfig,
                         per_class: bool = False,
                         ):
    scores = {}

    df = get_uncertainty(uncertainty_method,
                         classes,
                         df_pred)

    df['agreement'] = df_true[agreement_column]
    df.dropna(inplace=True)

    scores['PCC-lower is better'] = audmetric.pearson_cc(df['agreement'], df['uncertainty'])
    if per_class:
        df['pred'] = df_pred.idxmax(axis=1)
        for c in classes:
            df_class = df.loc[df['pred'] == c]
            scores[f'PCC-{c}'] = audmetric.pearson_cc(df_class['agreement'], df_class['uncertainty'])

    scores['uncertainty.mean'] = df['uncertainty'].mean()
    scores['uncertainty.std'] = df['uncertainty'].std()
    if 'precision' in df and 'mode_max' in df:
        scores['PCC (precision)'] = audmetric.pearson_cc(df['agreement'], df['precision'])
        scores['precision.mean'] = df['precision'].mean()
        scores['precision.std'] = df['precision'].std()

        scores['PCC (mode_max)'] = audmetric.pearson_cc(df['agreement'], df['mode_max'])
        scores['mode_max.mean'] = df['mode_max'].mean()
        scores['mode_max.std'] = df['mode_max'].std()

    plt.clf()
    if 'precision' in df and 'mode_max' in df:
        _, ax = plt.subplots(3, 1)
        audplot.scatter(
            df['agreement'].astype('float'),
            df['precision'].astype('float'),
            fit=True,
            ax=ax[1],
        )
        ax[1].set_xlabel('agreement')
        ax[1].set_ylabel('precision')

        audplot.scatter(
            df['agreement'].astype('float'),
            df['mode_max'].astype('float'),
            fit=True,
            ax=ax[2],
        )
        ax[2].set_xlabel('agreement')
        ax[2].set_ylabel('mode_max')

    else:
        _, ax = plt.subplots(1, 1)

    audplot.scatter(
        df['agreement'].astype('float'),
        df['uncertainty'].astype('float'),
        fit=True,
        ax=ax[0],
    )
    ax[0].set_xlabel('agreement')
    ax[0].set_ylabel('uncertainty')

    plt.tight_layout()
    plt.savefig(f'{cfg.results_root}/{name}.cm.png')
    plt.close()

    with open(f'{cfg.results_root}/{name}.yaml', 'w') as fp:
        yaml.dump(scores, fp)
    return scores


def calibrate_uncertainty(df_dev,
                          prediction_classes,
                          predict_func,
                          test_cfg_name,
                          test_cfg,
                          cfg,
                          ):
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.dev.pkl'

    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            max_signal_dur=8.0,
            resample=True,
            verbose=True,
        )
        if test_cfg.uncertainty_method == 'mc_dropout':
            if 'mc_dropout_forward_passes' not in test_cfg:
                print('No number of MC Dropout forward passes specified, set it to 20. ')
                with open_dict(test_cfg):
                    test_cfg.mc_dropout_forward_passes = 20

            pred_dfs = []
            # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
            for i in range(test_cfg.mc_dropout_forward_passes):
                print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
                new_df = interface.process_index(df_dev.index, preserve_index=True)
                pred_dfs.append(new_df)

            df_pred = pd.concat(pred_dfs)
        else:
            df_pred = interface.process_index(df_dev.index, preserve_index=True)

        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)

    if test_cfg.uncertainty_method == 'mc_dropout':
        def calc_var(x):
            return x.var().sum()

        variance = df_pred.groupby(level=df_pred.index.names).apply(calc_var)
        df_pred = df_pred.groupby(level=df_pred.index.names).mean()
        df_pred['variance'] = variance

    df_uncertain = get_uncertainty(uncertainty_method=test_cfg.uncertainty_method,
                                   classes=prediction_classes,
                                   df_pred=df_pred,
                                   )

    if 'data_source_labeled_test' in test_cfg:
        df_uncertain['db'] = '-'.join([f'{x.name}-{x.version}-{x.table}' for x in test_cfg.data_source_labeled_dev])
    elif 'data_source_artificaial_test' in test_cfg:
        df_uncertain['db'] = test_cfg.data_source_artificaial_test
    else:
        raise NotImplementedError('No datasource selected')

    df_uncertain['uncertainty_method'] = f'{test_cfg.uncertainty_method}'
    if 'post_processing' in test_cfg:
        df_uncertain['post_processing'] = f'{test_cfg.post_processing}'
    else:
        df_uncertain['post_processing'] = 'without'

    if test_cfg.uncertainty_method == 'entropy':
        df_uncertain['conf'] = 1 - (df_uncertain['uncertainty'] / np.log(len(prediction_classes)))
    elif test_cfg.uncertainty_method == 'max_logit':
        df_uncertain['conf'] = 1 - df_uncertain['uncertainty']
    elif test_cfg.uncertainty_method == 'mc_dropout':
        raise NotImplementedError('Confidance cant be calculated for mc dropout')
    elif test_cfg.uncertainty_method == 'edl':
        df_uncertain['conf'] = 1 - df_uncertain['uncertainty']
    else:
        raise NotImplementedError('No uncertainty prediction method specified')

    col_name = list(test_cfg.important_columns_labels.keys())[0]
    df_uncertain[col_name] = df_dev[col_name]
    df_uncertain['correct'] = df_uncertain.apply(lambda row: 'correct' if row.prediction == row.emotion else 'false',
                                                 axis=1)


def test_uncertainty_calibration(df_true,
                                 prediction_classes,
                                 predict_func,
                                 test_cfg_name,
                                 test_cfg,
                                 cfg,
                                 cal=None,
                                 ):
    cache_path_pred = f'{cfg.results_root}/{test_cfg_name}.pred.pkl'

    if not os.path.exists(cache_path_pred):
        interface = audinterface.Feature(
            prediction_classes,
            process_func=predict_func,
            sampling_rate=cfg.sampling_rate,
            max_signal_dur=8.0,
            resample=True,
            verbose=True,
        )
        if test_cfg.uncertainty_method == 'mc_dropout':
            if 'mc_dropout_forward_passes' not in test_cfg:
                print('No number of MC Dropout forward passes specified, set it to 20. ')
                with open_dict(test_cfg):
                    test_cfg.mc_dropout_forward_passes = 20

            pred_dfs = []
            # for _ in audeer.progress_bar(range(20), desc='MC Dropout prediction'):
            for i in range(test_cfg.mc_dropout_forward_passes):
                print(f'MC Dropout run {i} / {test_cfg.mc_dropout_forward_passes}')
                new_df = interface.process_index(df_true.index, preserve_index=True)
                pred_dfs.append(new_df)

            df_pred = pd.concat(pred_dfs)
        else:
            df_pred = interface.process_index(df_true.index, preserve_index=True)

        df_pred.to_pickle(cache_path_pred)

    df_pred = pd.read_pickle(cache_path_pred)

    if test_cfg.uncertainty_method == 'mc_dropout':
        def calc_var(x):
            return x.var().sum()

        variance = df_pred.groupby(level=df_pred.index.names).apply(calc_var)
        df_pred = df_pred.groupby(level=df_pred.index.names).mean()
        df_pred['variance'] = variance

    if 'calibration_method' in test_cfg and test_cfg.calibration_method == 'temp_scaling':
        assert cal
        print('dfpred_in', df_pred)

        array_preds = cal.predict(df_pred.to_numpy())
        df_pred = pd.DataFrame(array_preds, index=df_pred.index, columns=df_pred.columns)
        print('dfpred', df_pred)

    scores = {}

    df_uncertain = get_uncertainty(uncertainty_method=test_cfg.uncertainty_method,
                                   classes=prediction_classes,
                                   df_pred=df_pred,
                                   )

    if 'data_source_labeled_test' in test_cfg:
        df_uncertain['db'] = '-'.join([f'{x.name}-{x.version}-{x.table}' for x in test_cfg.data_source_labeled_test])
    elif 'data_source_artificaial_test' in test_cfg:
        df_uncertain['db'] = test_cfg.data_source_artificaial_test
    else:
        raise NotImplementedError('No datasource selected')

    df_uncertain['uncertainty_method'] = f'{test_cfg.uncertainty_method}'
    if 'post_processing' in test_cfg:
        df_uncertain['post_processing'] = f'{test_cfg.post_processing}'
    else:
        df_uncertain['post_processing'] = 'without'

    if test_cfg.uncertainty_method == 'entropy':
        df_uncertain['conf'] = 1 - (df_uncertain['uncertainty'] / np.log(len(prediction_classes)))
    elif test_cfg.uncertainty_method == 'max_logit':
        df_uncertain['conf'] = 1 - df_uncertain['uncertainty']
    elif test_cfg.uncertainty_method == 'mc_dropout':
        raise NotImplementedError('Confidance cant be calculated for mc dropout')
    elif test_cfg.uncertainty_method == 'edl':
        df_uncertain['conf'] = 1 - df_uncertain['uncertainty']
    else:
        raise NotImplementedError('No uncertainty prediction method specified')

    col_name = test_cfg.important_columns_labels
    df_uncertain[col_name] = df_true[col_name]
    df_uncertain['correct'] = df_uncertain.apply(lambda row: 'correct' if row.prediction == row.emotion else 'false',
                                                 axis=1)

    fig_rel, axes_rel = plt.subplots(
        1, len(prediction_classes),
        figsize=(18, 6)
    )

    for i, label in enumerate(prediction_classes):
        # map to one-vs-rest
        mapper = {key: 0 for key in prediction_classes}
        mapper.update({label: 1})  # true label is `1`

        y_correct_binary = df_uncertain.loc[df_uncertain['emotion'] == label]['prediction'].apply(
            lambda row: mapper[row]).astype(int)
        conf = df_uncertain.loc[df_uncertain['emotion'] == label]['conf']
        # conf = result[label].values
        # conf[conf < args.min_conf] = args.min_conf
        # conf[conf > args.max_conf] = args.max_conf

        if 'calibration_method' in test_cfg and test_cfg.calibration_method == 'hist_bin':
            assert cal
            conf = cal.predict(conf)

        # bin
        bins = np.linspace(
            0,
            1,
            10 + 1
        )
        binids = np.digitize(conf, bins) - 1

        # count confs, positives and total number per bin
        bin_sums = np.bincount(binids, weights=conf, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_correct_binary, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        print('bin sum', bin_total)
        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        plot_reliability_diagram(
            bins=bins[nonzero],
            prob_pred=prob_pred,
            prob_true=prob_true,
            bin_width=bins[1] - bins[0],
            bin_total=bin_total[nonzero],
            label=label,
            ax=axes_rel[i],
        )

        ece = compute_ece(
            bin_total=bin_total[nonzero],
            acc=prob_true,
            conf=prob_pred,
        )
        axes_rel[i].text(
            0.05,
            0.75,
            f'ece={ece:.{3}}',
            transform=axes_rel[i].transAxes,
            size=10,
            bbox=dict(boxstyle="round"),
        )

    plt.savefig(f'{cfg.results_root}/{test_cfg_name}.ece.png')
    plt.close()

    return scores


def plot_reliability_diagram(
        bins,
        prob_pred,
        prob_true,
        bin_width,
        bin_total,
        label,
        ax,
):
    ax.bar(
        x=bins,
        height=prob_true,
        width=bin_width,
        color='b',
        align='edge',
        edgecolor='k',
        linewidth=1,  # edge
    )

    ax.set_xticks(bins.tolist() + [bins[-1] + bin_width])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')
    ax.set_title(f'{label}')

    rects = ax.patches

    for rect, label in zip(rects, bin_total):
        print('lll', label)
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
        )

    # draw lines
    # ax.plot([0, bins[-1] + bin_width], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot([0, 0.9 + bin_width], [0, 1], "k:", label="Perfectly calibrated")
    # line_kwargs = {'label': args.uid}
    ax.plot(prob_pred, prob_true, "r--")
    ax.legend(loc='upper left')


def compute_ece(
        bin_total,
        acc,
        conf
):
    # expected calibration error
    # ECE = sum_m [ |I_m|/N * |acc(I_m) - conf(I_m)|
    # normalize conf with min-max to [-1, 1]
    conf = (conf - conf.min()) / (conf.max() - conf.min())
    return sum(bin_total * abs(acc - conf) / sum(bin_total))
