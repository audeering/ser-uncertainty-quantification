import os
import audeer
import matplotlib.pyplot as plt

import omegaconf
import pandas as pd
import seaborn as sns

from src.data.create_data_frames import load_testing_data, load_dev_data
from src.evaluation.plots import plot_uncertainty_bin_barplot
from src.evaluation.prediction_functions import get_torch_dropout_predict_function, get_torch_predict_function
from src.evaluation.tests import test_uncertainty_dropout, test_uncertainty, test_categorical, test_dimensional, \
    test_deviaion, test_categorical_noise, test_ood_data, test_uncertainty_calibration, calibrate_uncertainty


def evaluate(model, path, cfg):
    audeer.mkdir(cfg.results_root)
    print('model', cfg.model.prediction.items())
    categorical_predictions = []
    dimensional_predictions = []

    for c, l in cfg.model.prediction.items():
        if l:
            categorical_predictions.append([c, l])
        else:
            dimensional_predictions.append(c)

    # load data
    scores = {}
    for test_cfg_name in cfg.testing.tests:
        # test_result_root = audeer.mkdir(cfg.results_root + f'/{test_cfg_name}')
        print(f'run {test_cfg_name} Test')
        test_cfg = cfg.testing.tests[test_cfg_name]
        test_cfg.cache_path = f'{cfg.testing.cache_path}/{test_cfg_name}'
        df_true = load_testing_data(test_cfg)

        if 'post_processing' in test_cfg:
            post_processing = test_cfg.post_processing
        else:
            post_processing = None

        if 'uncertainty_method' in test_cfg and test_cfg.uncertainty_method == 'mc_dropout':
            predict_func = get_torch_dropout_predict_function(model, path, cfg, post_processing=post_processing)
            test_agreement = test_uncertainty_dropout
        else:
            predict_func = get_torch_predict_function(model, path, cfg, post_processing=post_processing)
            test_agreement = test_uncertainty

        if test_cfg.type == 'categorical':
            assert len(categorical_predictions) == 1, 'test works only for one classification task'
            prediction_classes = categorical_predictions[0][1]
            scores[test_cfg_name] = test_categorical(df_true, prediction_classes, predict_func,
                                                     test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'dimensional':
            scores[test_cfg_name] = test_dimensional(df_true, dimensional_predictions, predict_func,
                                                     test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'agreement':
            assert len(categorical_predictions) == 1, 'test works only for one classification task'
            prediction_classes = categorical_predictions[0][1]
            scores[test_cfg_name] = test_agreement(df_true, prediction_classes, predict_func,
                                                   test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'correctness':
            assert len(categorical_predictions) == 1, 'test works only for one classification task'
            prediction_classes = categorical_predictions[0][1]
            scores[test_cfg_name] = test_agreement(df_true, prediction_classes, predict_func,
                                                   test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'deviation':
            scores[test_cfg_name] = test_deviaion(df_true, dimensional_predictions, predict_func,
                                                  test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'categorical_noise':
            prediction_classes = categorical_predictions[0][1]
            scores[test_cfg_name] = test_categorical_noise(df_true, prediction_classes, predict_func,
                                                           test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'ood':
            prediction_classes = categorical_predictions[0][1]
            scores[test_cfg_name] = test_ood_data(df_true, prediction_classes, predict_func,
                                                  test_cfg_name, test_cfg, cfg)
        elif test_cfg.type == 'cat_calibration_test':
            prediction_classes = categorical_predictions[0][1]
            cal = None
            if 'calibration_method' in test_cfg:
                df_dev = load_dev_data(test_cfg)
                cal = calibrate_uncertainty(df_dev, prediction_classes, predict_func,
                                            test_cfg_name, test_cfg, cfg)

            scores[test_cfg_name] = test_uncertainty_calibration(df_true, prediction_classes, predict_func,
                                                                 test_cfg_name, test_cfg, cfg, cal=cal)
        else:
            NotImplementedError(f'test: "{test_cfg.type}" is not implemented')
    plt.close()
    evaluate_combined_results(cfg=cfg)

    # clear/create results.md
    open(f'{cfg.results_root}/results.md', 'w').close()

    for key, _ in list(scores.items()):
        if '_noise' in key or 'added' in key:
            score = scores.pop(key)
            table_scores = pd.DataFrame(score)
            table_scores.sort_index(inplace=True)
            with open(f'{cfg.results_root}/results.md', 'a') as fp:
                fp.write(f'Noise test results for {key}: \n \n')
                fp.write(table_scores.round(3).to_markdown())
                fp.write('\n\n')

    for table in ['cat', 'agreement', 'level', 'dropout', 'crema_d', 'emodb', 'erik', 'msp_test1', 'correctness']:
        table_scores = {}
        for key, _ in list(scores.items()):
            if table in key:
                table_scores[key] = scores.pop(key)
        table_scores = pd.DataFrame(table_scores)
        table_scores.sort_index(inplace=True)
        with open(f'{cfg.results_root}/results.md', 'a') as fp:
            fp.write(f'Table for all the "{table}" results: \n \n')
            fp.write(table_scores.round(3).to_markdown())
            fp.write('\n\n')

        if table == 'correctness':
            plot_uncertainty_bin_barplot(table_scores, cfg)
    print(scores)
    scores = pd.DataFrame(scores)
    scores.sort_index(inplace=True)
    with open(f'{cfg.results_root}/results.md', 'a') as fp:
        fp.write(f'Table with all the remaining tests: \n\n')
        fp.write(scores.round(3).to_markdown())
    scores.to_csv(f'{cfg.results_root}/results.csv')


def short_exp_name(exp_name):
    for input_name, out in [('emodb', 'emodb'), ('musan-1.0.0-speech', 'm-speech'),
                            ('musan-1.0.0-music', 'm-music'), ('musan-1.0.0-noise', 'm-noise'),
                            ('crema-d', 'crema-d'), ('msppodcast', 'msppodcast'), ('cochlscene', 'cochlescene'),
                            ('white', 'white-noise')]:
        if input_name in exp_name:
            return out

    return exp_name


def evaluate_combined_results(cfg):
    cache_path_combined = f'{cfg.results_root}/{cfg.testing.combined_df}.pkl'
    if not os.path.exists(cache_path_combined):
        print('No combined data available')
        return
    df_combined = pd.read_pickle(cache_path_combined)
    df_combined['experiment'] = df_combined['db'] + '-' + df_combined['uncertainty_method'] + \
                                '-' + df_combined['post_processing']
    df_combined['experiment'] = df_combined['experiment'].str. \
        replace('.gold_standard', '').replace('.categories', '').replace('.test', '')

    for uq_method in ['entropy', 'mc_dropout', 'edl']:
        df_uq = df_combined[df_combined['uncertainty_method'] == uq_method]

        if not df_uq.empty:
            for post_processing in df_uq['post_processing'].unique():
                df_up_post = df_uq[df_uq['post_processing'] == post_processing]
                plt.close()
                sns.ecdfplot(df_up_post, x='uncertainty', hue='experiment')

                # Paper viz:

                # print(df_up_post)
                # if 'plot_combined' not in df_up_post:
                #     df_up_post['plot_combined'] = True
                # df_up_post['plot_combined'].fillna(True, inplace=True)
                # combined_plot_df = df_up_post.loc[df_up_post['plot_combined']]
                # print(combined_plot_df)
                # pal1 = sns.color_palette(palette='Blues_r')
                # pal2 = sns.color_palette(palette='Reds_r')
                # print("combined_plot_df", combined_plot_df['experiment'].unique())
                #
                # combined_plot_df['experiment'] = combined_plot_df['experiment'].apply(short_exp_name)
                # print("combined_plot_df", combined_plot_df['experiment'].unique())
                #
                # combined_plot_df_in = combined_plot_df[
                #     combined_plot_df['experiment'].isin(['emodb', 'm-speech', 'crema-d', 'msppodcast'])]
                # combined_plot_df_out = combined_plot_df[
                #     combined_plot_df['experiment'].isin(['m-music', 'm-noise', 'cochlescene'])]
                # combined_plot_df_white = combined_plot_df[combined_plot_df['experiment'].isin(['white-noise'])]
                # print("combined_plot_df_in", combined_plot_df_in['experiment'].unique())
                # print("combined_plot_df_out", combined_plot_df_out['experiment'].unique())
                # print("combined_plot_df_white", combined_plot_df_white['experiment'].unique())
                # plt.rcParams['figure.figsize'] = 4.5, 3.5
                # if not combined_plot_df_in.empty:
                #     sns.ecdfplot(combined_plot_df_in, x='uncertainty', hue='experiment', palette=pal1)
                # if not combined_plot_df_out.empty:
                #     sns.ecdfplot(combined_plot_df_out, x='uncertainty', hue='experiment', palette=pal2)
                # if not combined_plot_df_white.empty:
                #     sns.ecdfplot(combined_plot_df_white, x='uncertainty', hue='experiment', palette=['grey'])
                # plt.legend(title='', loc='lower right', labels=['Speech Data', 'OOD Data', 'White Noise'],
                #            labelcolor=[pal1[1], pal2[1], 'grey'], markerscale=0)
                # plt.xlabel('variance')
                # plt.xlim(0, 0.15)

                # ax = plt.gca()
                # leg = ax.get_legend()
                # if len(leg.legendHandles) >= 3:
                #     leg.legendHandles[0].set_color(pal1[1])
                #     leg.legendHandles[1].set_color(pal2[1])
                #     leg.legendHandles[2].set_color('grey')

                plt.tight_layout()
                plt.savefig(f'{cfg.results_root}/density-{uq_method}-{post_processing}.png')
                plt.close()

                if 'precision' in df_up_post and 'mode_max' in df_up_post:
                    sns.ecdfplot(df_up_post, x='precision', hue='experiment')
                    plt.tight_layout()
                    plt.savefig(f'{cfg.results_root}/density-{uq_method}-{post_processing}-precision.png')
                    plt.close()
                    sns.ecdfplot(df_up_post, x='mode_max', hue='experiment')
                    plt.tight_layout()
                    plt.savefig(f'{cfg.results_root}/density-{uq_method}-{post_processing}-mode_max.png')
                    plt.close()

                for col in ['emotion', 'scene', 'background_noise', 'vocals']:
                    if col not in df_up_post.columns:
                        continue
                    df_emo = df_up_post[~df_up_post[col].isna()]
                    if not df_emo.empty:
                        sns.ecdfplot(df_up_post, x='uncertainty', hue=col)
                        plt.savefig(f'{cfg.results_root}/density-{col}-{uq_method}-{post_processing}.png')
                        plt.close()
                        if 'precision' in df_up_post and 'mode_max' in df_up_post:
                            sns.ecdfplot(df_up_post, x='precision', hue=col)
                            plt.savefig(f'{cfg.results_root}/density-{col}-{uq_method}-{post_processing}-precision.png')
                            plt.close()
                            sns.ecdfplot(df_up_post, x='mode_max', hue=col)
                            plt.savefig(f'{cfg.results_root}/density-{col}-{uq_method}-{post_processing}-mode_max.png')
                            plt.close()

            df_combined = df_combined.drop(df_uq.index)

        if not df_combined.empty:
            sns.ecdfplot(df_combined, x='uncertainty', hue='experiment')
            # axes[1].get_legend().remove()
            plt.savefig(f'{cfg.results_root}/density-rest.png')
            plt.close()
