import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import omegaconf


def plot_uncertainty_bin_barplot(table_scores: pd.DataFrame,
                                 cfg: omegaconf.DictConfig):
    for special_vis in ['emodb', 'crema_d']:
        columns = list(table_scores.filter(regex=special_vis))
        if not columns:
            continue
        res = pd.concat([table_scores.pop(col) for col in columns], axis=1)
        if len(columns) == 2:
            res['difference'] = res[columns[0]] - res[columns[1]]

        for metric in ['UAR', 'Acc']:
            # visualizing of certainty bins per db for UAR and Accuracy
            table_vis = res[res.index.str.contains('%')]
            table_vis = table_vis[table_vis.index.str.contains(metric)]
            if table_vis.empty:
                continue
            plt.clf()
            table_vis = table_vis.T

            fig = table_vis.plot(kind='bar')
            labels = [label.replace('_uncertainty_correctness', '') for label in table_vis.index]
            fig.set_xticklabels(labels, rotation=0, ha="right")
            plt.tight_layout()
            plt.savefig(f'{cfg.results_root}/correctness.{special_vis}.{metric}.png')

    if table_scores.empty:
        return
    for metric in ['UAR', 'Acc']:
        # visualizing of certainty bins per db for UAR and Accuracy
        table_vis = table_scores[table_scores.index.str.contains('%')]
        table_vis = table_vis[table_vis.index.str.contains(metric)]
        if table_vis.empty:
            continue
        plt.clf()
        table_vis = table_vis.T
        fig = table_vis.plot(kind='bar')
        labels = [label.replace('_uncertainty_correctness', '') for label in table_vis.index]
        fig.set_xticklabels(labels, rotation=25, ha="right")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.savefig(f'{cfg.results_root}/correctness.{metric}.png')
