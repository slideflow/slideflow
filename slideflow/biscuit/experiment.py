import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from skmisc.loess import loess
from scipy import stats
from tqdm import tqdm
from statistics import mean
from os.path import join, exists

import slideflow as sf
from slideflow.util import log
from . import utils, threshold
from . import hp as biscuit_hp
from .errors import MatchError, ModelNotFoundError, ThresholdError

# -----------------------------------------------------------------------------

ALL_EXP = {
    'AA': 'full',
    'U': 800,
    'T': 700,
    'S': 600,
    'R': 500,
    'A': 400,
    'L': 350,
    'M': 300,
    'N': 250,
    'D': 200,
    'O': 176,
    'P': 150,
    'Q': 126,
    'G': 100,
    'V': 90,
    'W': 80,
    'X': 70,
    'Y': 60,
    'Z': 50,
    'ZA': 40,
    'ZB': 30,
    'ZC': 20,
    'ZD': 10
}

# -----------------------------------------------------------------------------

class Experiment:
    def __init__(
        self,
        train_project,
        eval_projects=None,
        outcome='cohort',
        outcome1='LUAD',
        outcome2='LUSC',
        outdir='results'
    ):
        """Supervises uncertainty thresholding experiments."""

        if eval_projects is None:
            eval_projects = []

        if isinstance(train_project, str):
            self.train_project = sf.Project(train_project)
        elif isinstance(train_project, sf.Project):
            self.train_project = train_project
        else:
            raise ValueError(f"Unrecognized value for train_project: {train_project}")

        self.eval_projects = []
        for ep in eval_projects:
            if isinstance(ep, str):
                self.eval_projects += [sf.Project(ep)]
            elif isinstance(ep, sf.Project):
                self.eval_projects += [ep]
            else:
                raise ValueError(f"Unrecognized value for eval_project: {eval_projects}")

        self.outcome = outcome
        self.outcome1 = outcome1
        self.outcome2 = outcome2
        self.outdir = outdir

    def add(self, path, label, out1, out2, order='f', order_col='order', gan=0):
        """Adds a sample size experiment to the given project annotations file.

        Args:
            path (str): Path to project annotations file.
            label (str): Experimental label.
            out1 (int): Number of lung adenocarcinomas (LUAD) to include in the
                experiment.
            out2 (int): Number of lung squamous cell carcinomas (LUSC) to include
                in the experiment.
            outcome (str, optional): Annotation header which indicates the outcome
                of interest. Defaults to 'cohort'.
            order (str, optional): 'f' (forward) or 'r' (reverse). Indicates which
                direction to follow when sequentially adding slides.
                Defaults to 'f'.
            order_col (str, optional): Annotation header column to use when
                sequentially adding slides. Defaults to 'order'.
            gan (int, optional): Number of GAN slides to include in experiment.
                Defaults to 0.

        Returns:
            None
        """

        assert isinstance(out1, int)
        assert isinstance(out2, int)
        assert isinstance(gan, (int, float)) and 0 <= gan < 1
        assert order in ('f', 'r')

        ann = pd.read_csv(path, dtype=str)
        print(f"Setting up exp. {label} with order {order} (sort by {order_col})")
        ann[order_col] = pd.to_numeric(ann[order_col])
        ann.sort_values(
            ['gan', self.outcome, order_col],
            ascending=[True, True, (order != 'r')],
            inplace=True
        )
        gan_out1 = round(gan * out1)
        gan_out2 = round(gan * out2)
        out1_indices = np.where((ann['site'].to_numpy() != 'GAN')
                                & (ann[self.outcome] == self.outcome1))[0]
        out2_indices = np.where((ann['site'].to_numpy() != 'GAN')
                                & (ann[self.outcome] == self.outcome2))[0]
        gan_out1_indices = np.where((ann['site'].to_numpy() == 'GAN')
                                    & (ann[self.outcome] == self.outcome1))[0]
        gan_out2_indices = np.where((ann['site'].to_numpy() == 'GAN')
                                    & (ann[self.outcome] == self.outcome2))[0]

        assert out1 <= out1_indices.shape[0]
        assert out2 <= out2_indices.shape[0]
        assert gan_out1 <= gan_out1_indices.shape[0]
        assert gan_out2 <= gan_out2_indices.shape[0]

        include = np.array(['exclude' for _ in range(len(ann))])
        include[out1_indices[:out1]] = 'include'
        include[out2_indices[:out2]] = 'include'
        include[gan_out1_indices[:gan_out1]] = 'include'
        include[gan_out2_indices[:gan_out2]] = 'include'
        ann[f'include_{label}'] = include
        ann.to_csv(path, index=False)

    @staticmethod
    def config(name_pattern, subset, ratio, **kwargs):
        """Configures a set of experiments.

        Args:
            name_pattern (str): String pattern for experiment naming.
            subset (list(str)): List of experiment ID/labels.
            ratio (float): Float 0-1. n_out1 / n_out2 (or n_out2 / n_out1)
        """

        if not isinstance(ratio, (int, float)) and ratio >= 1:
            raise ValueError("Invalid ratio; must be float >= 1")
        config = {}
        for exp in ALL_EXP:
            if exp not in subset:
                continue
            if exp == 'AA' and ratio != 1:
                raise ValueError("Cannot create full dataset exp. with ratio != 1")

            exp_name = name_pattern.format(exp)
            if ratio != 1:
                n1 = round(ALL_EXP[exp] / (1 + (1/ratio)))
                n2 = ALL_EXP[exp] - n1

                config.update({
                    exp_name:     {'out1': n1, 'out2': n2, **kwargs},
                    exp_name+'i': {'out1': n2, 'out2': n1, **kwargs}
                })
            else:
                if ALL_EXP[exp] == 'full':
                    n_out1 = 467
                    n_out2 = 474
                else:
                    n_out1 = n_out2 = int(ALL_EXP[exp] / 2)
                config.update({
                    exp_name: {'out1': n_out1, 'out2': n_out2, **kwargs},
                })
        return config

    def display(self, df, eval_dfs, hue='uq', palette='tab10', relplot_uq_compare=True,
                boxplot_uq_compare=True, ttest_uq_groups=['all', 'include'],
                prefix=''):
        """Creates plots from assmebled results, exports results to CSV.

        Args:
            df (pandas.DataFrame): Cross-validation results metrics, as generated
                by results()
            eval_dfs (dict(pandas.DataFrame)): Dict of external eval dataset names
                (keys) mapped to pandas DataFrame of result metrics (values).
            hue (str, optional): Comparison to show with different hue on plots.
                Defaults to 'uq'.
            palette (str, optional): Seaborn color palette. Defaults to 'tab10'.
            relplot_uq_compare (bool, optional): For the Relplot display, ensure
                non-UQ and UQ results are generated from the same models/preds.
            boxplot_uq_compare (bool, optional): For the boxplot display, ensure
                non-UQ and UQ results are generated from the same models/preds.
            ttest_uq_groups (list(str)): UQ groups to compare via t-test. Defaults
                to ['all', 'include'].
            prefix (str, optional): Prefix to use when saving figures.
                Defaults to empty string.

        Returns:
            None
        """

        if not len(df):
            log.error("No results to display")
            return

        # Filter out UQ results if n_slides < 100
        df = df.loc[~ ((df['n_slides'] < 100)
                    & (df['uq'].isin(['include', 'exclude'])))]

        # --- Paired t-tests ---------------------------------------------------
        if ttest_uq_groups and len(ttest_uq_groups) != 2:
            raise ValueError("Length of ttest_uq_groups must be exactly 2")
        ttest_df = df.loc[df['uq'].isin(ttest_uq_groups)].copy()
        ttest_df = ttest_df.sort_values(['id', 'fold'])

        def perform_paired_testing(level):
            print(f"Paired t-tests ({level}-level):")
            for n in sorted(ttest_df['n_slides'].unique()):
                exp_df = ttest_df[ttest_df['n_slides'] == n]
                try:
                    ttest_result = stats.ttest_rel(
                        exp_df.loc[exp_df['uq'] == ttest_uq_groups[0]][f'{level}_auc'],
                        exp_df.loc[exp_df['uq'] == ttest_uq_groups[1]][f'{level}_auc'],
                        alternative='less')
                    print(n, '\t', 'p =', ttest_result.pvalue)
                except ValueError:
                    print(n, '\t', 'p = (error)')

        perform_paired_testing('patient')
        perform_paired_testing('slide')

        # --- Cross-validation plots -------------------------------------------

        if len(df):
            # AUC (relplot)
            if relplot_uq_compare:
                rel_df = df.loc[df['uq'] != 'none']
            else:
                rel_df = df
            sns.relplot(
                x='n_slides',
                y='slide_auc',
                data=rel_df,
                hue=hue,
                marker='o',
                kind='line',
                palette=palette
            )
            plt.title('Cross-val AUC')
            ax = plt.gca()
            ax.set_ylim([0.5, 1])
            ax.grid(visible=True, which='both', axis='both', color='white')
            ax.set_facecolor('#EAEAF2')
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(100))
            plt.subplots_adjust(top=0.9)
            plt.savefig(join(self.outdir, f'{prefix}relplot.svg'))

            f, axes = plt.subplots(1, 3)
            f.set_size_inches(18, 6)

            # AUC boxplot
            if boxplot_uq_compare:
                box_df = df.loc[df['uq'] != 'none']
            else:
                box_df = df
            sns.boxplot(
                x='n_slides',
                y='slide_auc',
                hue=hue,
                data=box_df,
                ax=axes[0],
                palette=palette
            )
            axes[0].title.set_text('Cross-val AUC')
            axes[0].set_ylabel('')
            axes[0].tick_params(labelrotation=90)

            # AUC scatter - LOESS & standard error
            df = df.sort_values(by=['n_slides'])
            x = df['n_slides'].to_numpy().astype(np.float32)
            y = df['slide_auc'].to_numpy()
            lo = loess(x, y)
            try:
                lo.fit()
                pred = lo.predict(x, stderror=True)
                conf = pred.confidence()
                z = pred.values
                ll = conf.lower
                ul = conf.upper
                axes[1].plot(x, y, '+', ms=6)
                axes[1].plot(x, z)
                axes[1].fill_between(x, ll, ul, alpha=.33)
            except ValueError:
                pass

            axes[1].xaxis.set_minor_locator(plticker.MultipleLocator(20))
            axes[1].spines['bottom'].set_linewidth(0.5)
            axes[1].spines['bottom'].set_color('black')
            axes[1].tick_params(axis='x', colors='black')
            axes[1].grid(visible=True, which='both', axis='both', color='white')
            axes[1].set_facecolor('#EAEAF2')
            axes[1].set_xscale('log')
            axes[1].title.set_text('Cross-val AUC')

            # % slides included
            sns.lineplot(
                x='n_slides',
                y='patient_uq_perc',
                data=df,
                marker='o',
                ax=axes[2],
                zorder=3
            )
            axes[2].set_ylabel('')
            axes[2].title.set_text('% Patients Included with UQ (cross-val)')
            axes[2].xaxis.set_minor_locator(plticker.MultipleLocator(100))
            axes[2].tick_params(labelrotation=90)
            axes[2].grid(visible=True, which='both', axis='both', color='white', zorder=0)
            axes[2].set_facecolor('#EAEAF2')
            axes[2].set_xlim(100)
            axes[2].scatter(x=df.groupby('n_slides', as_index=False).median().n_slides.values, y=df.groupby('n_slides').median().patient_uq_perc.values, marker='x', zorder=5)

            plt.subplots_adjust(bottom=0.2)
            plt.savefig(join(self.outdir, f'{prefix}crossval.svg'))

        # --- Evaluation plots ----------------------------------------------------

        if eval_dfs:
            for eval_name, eval_df in eval_dfs.items():
                if not len(eval_df):
                    continue
                has_uq = len(eval_df.loc[eval_df['uq'].isin(['include', 'exclude'])])

                # Prepare figure
                sns.set(rc={"xtick.bottom": True, "ytick.left": True})
                f, axes = plt.subplots(1, (4 if has_uq else 3))
                f.suptitle(f'{eval_name} Evaluation Dataset')
                f.set_size_inches(16, 4)

                # AUC
                if not len(eval_df):
                    continue
                eval_df = eval_df.loc[~ ((eval_df['n_slides'] < 100)
                                        & (eval_df['uq'].isin(['include', 'exclude'])))]
                sns.lineplot(
                    x='n_slides',
                    y='patient_auc',
                    hue=hue,
                    data=eval_df,
                    marker="o",
                    ax=axes[0]
                )
                sns.scatterplot(
                    x='n_slides',
                    y='slide_auc',
                    hue=hue,
                    data=eval_df,
                    marker="x",
                    ax=axes[0]
                )
                axes[0].get_legend().remove()
                axes[0].title.set_text('AUC')

                # Accuracy
                sns.lineplot(
                    x='n_slides',
                    y='patient_acc',
                    hue=hue,
                    data=eval_df,
                    marker="o",
                    ax=axes[1]
                )
                sns.scatterplot(
                    x='n_slides',
                    y='slide_acc',
                    hue=hue,
                    data=eval_df,
                    marker="x",
                    ax=axes[1]
                )
                axes[1].get_legend().remove()
                axes[1].title.set_text('Accuracy')

                # Youden's index
                sns.lineplot(
                    x='n_slides',
                    y='patient_youden',
                    hue=hue,
                    data=eval_df,
                    marker="o",
                    ax=axes[2]
                )
                sns.scatterplot(
                    x='n_slides',
                    y='slide_youden',
                    hue=hue,
                    data=eval_df,
                    marker="x",
                    ax=axes[2]
                )
                axes[2].title.set_text("Youden's J")
                axes[2].get_legend().remove()

                # % slides included
                if has_uq:
                    sns.lineplot(
                        x='n_slides',
                        y='patient_incl',
                        data=eval_df.loc[eval_df['uq'] == 'include'],
                        marker='o'
                    )
                    sns.scatterplot(
                        x='n_slides',
                        y='slide_incl',
                        data=eval_df.loc[eval_df['uq'] == 'include'],
                        marker='x'
                    )
                    axes[3].title.set_text('% Included')
                for ax in axes:
                    ax.set_ylabel('')
                    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=100))
                    ax.tick_params(labelrotation=90)
                plt.subplots_adjust(top=0.8)
                plt.subplots_adjust(bottom=0.2)
                plt.savefig(join(self.outdir, f'{prefix}eval.svg'))

    def plot_uq_calibration(self, label, tile_uq, slide_uq, slide_pred, epoch=1):
        """Plots a graph of predictions vs. uncertainty.

        Args:
            label (str): Experiment label.
            kfold (int): Validation k-fold.
            tile_uq (float): Tile-level uncertainty threshold.
            slide_uq (float): Slide-level uncertainty threshold.
            slide_pred (float): Slide-level prediction threshold.

        Returns:
            None
        """

        val_dfs = [
            pd.read_csv(
                join(
                    utils.find_model(self.train_project, label, kfold=k, outcome=self.outcome),
                    f'tile_predictions_val_epoch{epoch}.csv'),
                dtype={'slide': str})
            for k in range(1, 4)
        ]
        for v in range(len(val_dfs)):
            utils.rename_cols(val_dfs[v], outcome=self.outcome)
        _df = val_dfs[0]
        _df = pd.concat([_df, val_dfs[1]], axis=0, join='outer', ignore_index=True)
        _df = pd.concat([_df, val_dfs[2]], axis=0, join='outer', ignore_index=True)

        # Plot tile-level uncertainty
        patients = self.train_project.dataset().patients()
        _df, _ = threshold.process_tile_predictions(_df, patients=patients)
        threshold.plot_uncertainty(
            _df,
            kind='tile',
            threshold=tile_uq,
            title=f'CV UQ Calibration: {label}'
        )
        # Plot slide-level uncertainty
        _df = _df[_df['uncertainty'] < tile_uq]
        _s_df, _ = threshold.process_group_predictions(
            _df,
            pred_thresh=slide_pred,
            level='slide'
        )
        threshold.plot_uncertainty(
            _s_df,
            kind='slide',
            threshold=slide_uq,
            title=f'CV UQ Calibration: {label}'
        )

    def results(self, exp_to_run, uq=True, eval=True, plot=False):
        """Assembles results from experiments, applies UQ thresholding,
        and returns pandas dataframes with metrics.

        Args:
            exp_to_run (list): List of experiment IDs to search for results.
            uq (bool, optional): Apply UQ thresholds. Defaults to True.
            eval (bool, optional): Calculate results of external evaluation models.
                Defaults to True.
            plot (bool, optional): Show plots. Defaults to False.

        Returns:
            pandas.DataFrame: Cross-val results,
            pandas.DataFrame: Dxternal eval results
        """

        # === Initialize projects & prepare experiments ===========================

        P = self.train_project
        eval_Ps = self.eval_projects
        df = pd.DataFrame()
        eval_dfs = {val_P.name: pd.DataFrame() for val_P in eval_Ps}
        prediction_thresholds = {}
        slide_uq_thresholds = {}
        tile_uq_thresholds = {}
        pred_uq_thresholds = {}

        # === Show results from designated epoch ==================================
        for exp in exp_to_run:
            try:
                models = utils.find_cv(P, f'EXP_{exp}', outcome=self.outcome)
            except MatchError:
                log.debug(f"Unable to find cross-val results for {exp}; skipping")
                continue
            for i, m in enumerate(models):
                try:
                    results = utils.get_model_results(m, outcome=self.outcome, epoch=1)
                except FileNotFoundError:
                    print(f"Unable to open cross-val results for {exp}; skipping")
                    continue
                m_slides = sf.util.get_slides_from_model_manifest(m, dataset=None)
                df = pd.concat([df, pd.DataFrame([{
                    'id': exp,
                    'n_slides': len(m_slides),
                    'fold': i+1,
                    'uq': 'none',
                    'patient_auc': results['pt_auc'],
                    'patient_ap': results['pt_ap'],
                    'slide_auc': results['slide_auc'],
                    'slide_ap': results['slide_ap'],
                    'tile_auc': results['tile_auc'],
                    'tile_ap': results['tile_ap'],
                }])], axis=0, join='outer', ignore_index=True)

        # === Add UQ Crossval results (non-thresholded) ===========================
        for exp in exp_to_run:
            try:
                skip = False
                models = utils.find_cv(P, f'EXP_{exp}_UQ', outcome=self.outcome)
            except MatchError:
                continue
            all_pred_thresh = []
            for i, m in enumerate(models):
                try:
                    results = utils.get_model_results(m, outcome=self.outcome, epoch=1)
                    all_pred_thresh += [results['opt_thresh']]
                    df = pd.concat([df, pd.DataFrame([{
                        'id': exp,
                        'n_slides': len(sf.util.get_slides_from_model_manifest(m, dataset=None)),
                        'fold': i+1,
                        'uq': 'all',
                        'patient_auc': results['pt_auc'],
                        'patient_ap': results['pt_ap'],
                        'slide_auc': results['slide_auc'],
                        'slide_ap': results['slide_ap'],
                        'tile_auc': results['tile_auc'],
                        'tile_ap': results['tile_ap'],
                    }])], axis=0, join='outer', ignore_index=True)
                except FileNotFoundError:
                    log.debug(f"Skipping UQ crossval (non-thresholded) results for {exp}; not found")
                    skip = True
                    break
            if not skip:
                prediction_thresholds[exp] = mean(all_pred_thresh)

        # === Get & Apply Nested UQ Threshold =====================================
        if uq:
            pb = tqdm(exp_to_run)
            for exp in pb:
                # Skip UQ for experiments with n_slides < 100
                if exp in ('V', 'W', 'X', 'Y', 'Z', 'ZA', 'ZB', 'ZC', 'ZD'):
                    continue
                pb.set_description(f"Calculating thresholds (exp {exp})...")
                try:
                    _df, thresh = self.thresholds_from_nested_cv(
                        f'EXP_{exp}_UQ', id=exp
                    )
                    df = pd.concat([df, _df], axis=0, join='outer', ignore_index=True)
                except (MatchError, FileNotFoundError, ModelNotFoundError) as e:
                    log.debug(str(e))
                    log.debug(f"Skipping UQ crossval results for {exp}; not found")
                    continue
                except ThresholdError as e:
                    log.debug(str(e))
                    log.debug(f'Skipping UQ crossval results for {exp}; could not find thresholds in cross-validation')
                    continue

                tile_uq_thresholds[exp] = thresh['tile_uq']
                slide_uq_thresholds[exp] = thresh['slide_uq']
                pred_uq_thresholds[exp] = thresh['slide_pred']
                # Show CV uncertainty calibration
                if plot and exp == 'AA':
                    print("Plotting UQ calibration for cross-validation (exp. AA)")
                    self.plot_uq_calibration(
                        label=f'EXP_{exp}_UQ',
                        **thresh
                    )
                    plt.show()

        # === Show external validation results ====================================
        if eval:
            # --- Step 7A: Show non-UQ external validation results ----------------
            for val_P in eval_Ps:
                name = val_P.name
                pb = tqdm(exp_to_run, ncols=80)
                for exp in pb:
                    pb.set_description(f'Working on {name} eval (EXP {exp})...')

                    # Read and prepare model results
                    try:
                        eval_dir = utils.find_eval(val_P, f'EXP_{exp}_FULL', outcome=self.outcome)
                        results = utils.get_eval_results(eval_dir, outcome=self.outcome)
                    except (FileNotFoundError, MatchError):
                        log.debug(f"Skipping eval for exp {exp}; eval not found")
                        continue
                    if not utils.model_exists(P, f'EXP_{exp}_FULL', outcome=self.outcome, epoch=1):
                        log.debug(f'Skipping eval for exp {exp}; trained model not found')
                        continue
                    if exp not in prediction_thresholds:
                        log.warn(f"No predictions threshold for experiment {exp}; using slide-level pred threshold of 0.5")
                        pred_thresh = 0.5
                    else:
                        pred_thresh = prediction_thresholds[exp]

                    # Patient-level and slide-level predictions & metrics
                    patient_yt, patient_yp = utils.read_group_predictions(
                        join(
                            eval_dir,
                            f'patient_predictions_{self.outcome}_eval.csv'
                        )
                    )
                    patient_metrics = utils.prediction_metrics(
                        patient_yt,
                        patient_yp,
                        threshold=pred_thresh
                    )
                    patient_metrics = {
                        f'patient_{m}': patient_metrics[m]
                        for m in patient_metrics
                    }
                    slide_yt, slide_yp = utils.read_group_predictions(
                        join(
                            eval_dir,
                            f'patient_predictions_{self.outcome}_eval.csv'
                        )
                    )
                    slide_metrics = utils.prediction_metrics(
                        slide_yt,
                        slide_yp,
                        threshold=pred_thresh
                    )
                    slide_metrics = {
                        f'slide_{m}': slide_metrics[m]
                        for m in slide_metrics
                    }
                    model = utils.find_model(P, f'EXP_{exp}_FULL', outcome=self.outcome, epoch=1)
                    n_slides = len(sf.util.get_slides_from_model_manifest(model, dataset=None))
                    eval_dfs[name] = pd.concat([eval_dfs[name], pd.DataFrame([{
                        'id': exp,
                        'n_slides': n_slides,
                        'uq': 'none',
                        'incl': 1,
                        'patient_auc': results['pt_auc'],
                        'patient_ap': results['pt_ap'],
                        'slide_auc': results['slide_auc'],
                        'slide_ap': results['slide_ap'],
                        **patient_metrics,
                        **slide_metrics,
                    }])], axis=0, join='outer', ignore_index=True)

                    # --- [end patient-level predictions] -------------------------

                    if exp not in prediction_thresholds:
                        log.debug(f"Unable to calculate eval UQ performance; no prediction thresholds found for exp {exp}")
                        continue

                    # --- Step 7B: Show UQ external validation results ------------
                    if uq:
                        if exp in tile_uq_thresholds:
                            for keep in ('high_confidence', 'low_confidence'):
                                tile_pred_df = pd.read_csv(
                                    join(
                                        eval_dir,
                                        'tile_predictions_eval.csv'
                                    ), dtype={'slide': str}
                                )
                                new_cols = {
                                    f'{self.outcome}_y_pred1': 'y_pred',
                                    f'{self.outcome}_y_true0': 'y_true',
                                    f'{self.outcome}_uncertainty1': 'uncertainty'
                                }
                                tile_pred_df.rename(columns=new_cols, inplace=True)
                                thresh_tile = tile_uq_thresholds[exp]
                                thresh_slide = slide_uq_thresholds[exp]

                                val_patients = val_P.dataset(verification=None).patients()

                                def get_metrics_by_level(level):
                                    return threshold.apply(
                                        tile_pred_df,
                                        tile_uq=thresh_tile,
                                        slide_uq=thresh_slide,
                                        tile_pred=0.5,
                                        slide_pred=pred_uq_thresholds[exp],
                                        plot=(plot and level == 'slide' and keep == 'high_confidence' and exp == 'AA'),
                                        title=f'{name}: Exp. {exp} Uncertainty',
                                        keep=keep,  # Keeps only LOW or HIGH-confidence slide predictions
                                        patients=val_patients,
                                        level=level
                                    )

                                s_results, _ = get_metrics_by_level('slide')
                                p_results, _ = get_metrics_by_level('patient')
                                if (plot and keep == 'high_confidence' and exp == 'AA'):
                                    plt.savefig(join(self.outdir, f'{name}_uncertainty_v_preds.svg'))

                                full_model = utils.find_model(P, f'EXP_{exp}_FULL', outcome=self.outcome, epoch=1)
                                n_slides = len(sf.util.get_slides_from_model_manifest(full_model, dataset=None))
                                eval_dfs[name] = pd.concat([eval_dfs[name], pd.DataFrame([{
                                    'id': exp,
                                    'n_slides': n_slides,
                                    'uq': ('include' if keep == 'high_confidence' else 'exclude'),
                                    'slide_incl': s_results['percent_incl'],
                                    'slide_auc': s_results['auc'],
                                    'slide_acc': s_results['acc'],
                                    'slide_sens': s_results['sensitivity'],
                                    'slide_spec': s_results['specificity'],
                                    'slide_youden': s_results['sensitivity'] + s_results['specificity'] - 1,
                                    'patient_incl': p_results['percent_incl'],
                                    'patient_auc': p_results['auc'],
                                    'patient_acc': p_results['acc'],
                                    'patient_sens': p_results['sensitivity'],
                                    'patient_spec': p_results['specificity'],
                                    'patient_youden': p_results['sensitivity'] + p_results['specificity'] - 1,
                                }])], axis=0, join='outer', ignore_index=True)
            for eval_name in eval_dfs:
                eval_dfs[eval_name].to_csv(
                    join(self.outdir, f'{eval_name}_results.csv'),
                    index=False
                )
        else:
            eval_dfs = None
        df.to_csv(join(self.outdir, 'crossval_results.csv'), index=False)
        return df, eval_dfs

    def run(self, exp_to_run, steps=None, hp='nature2022'):
        """Trains the designated experiments.

        Args:
            exp_to_run (dict): Dict containing experiment configuration,
                as provided by config().
            steps (list(int)): Steps to run. Defaults to all steps, 1-6.
            hp (slideflow.ModelParams, optional): Hyperparameters object.
                Defaults to hyperparameters used for publication.

        Returns:
            None
        """

        # === Initialize projects & prepare experiments ===========================
        print(sf.util.bold("Initializing experiments..."))
        P = self.train_project
        eval_Ps = self.eval_projects
        exp_annotations = join(P.root, 'experiments.csv')
        if P.annotations != exp_annotations:
            if not exists(exp_annotations):
                shutil.copy(P.annotations, exp_annotations)
            P.annotations = exp_annotations
        exp_to_add = [
            e for e in exp_to_run
            if f'include_{e}' not in pd.read_csv(exp_annotations).columns.tolist()
        ]
        for exp in exp_to_add:
            self.add(exp_annotations, label=exp, **exp_to_run[exp])

        full_epoch_exp = [e for e in exp_to_run if e in ('AA', 'A', 'D', 'G')]

        if hp == 'nature2022':
            exp_hp = biscuit_hp.nature2022()
        else:
            exp_hp = hp

        # Configure steps to run
        if steps is None:
            steps = range(7)

        # === Step 1: Initialize full-epochs experiments ==========================
        if 1 in steps:
            print(sf.util.bold("[Step 1] Running full-epoch experiments..."))
            exp_hp.epochs = [1, 3, 5, 10]
            for exp in full_epoch_exp:
                val_k = [
                    k for k in range(1, 4)
                    if not utils.model_exists(P, f'EXP_{exp}', outcome=self.outcome, kfold=k)
                ]
                if not len(val_k):
                    print(f'Skipping Step 1 for experiment {exp}; already done.')
                    continue
                elif val_k != list(range(1, 4)):
                    print(f'[Step 1] Some k-folds done; running {val_k} for {exp}')
                self.train(
                    hp=exp_hp,
                    label=f'EXP_{exp}',
                    filters={f'include_{exp}': ['include']},
                    splits=f'splits_{exp}.json',
                    val_k=val_k,
                    val_strategy='k-fold',
                    save_model=False
                )

        # === Step 2: Run the rest of the experiments at the designated epoch =====
        if 2 in steps:
            print(sf.util.bold("[Step 2] Running experiments at target epoch..."))
            exp_hp.epochs = [1]
            for exp in exp_to_run:
                if exp in full_epoch_exp:
                    continue  # Already done in Step 2
                val_k = [
                    k for k in range(1, 4)
                    if not utils.model_exists(P, f'EXP_{exp}', outcome=self.outcome, kfold=k)
                ]
                if not len(val_k):
                    print(f'Skipping Step 2 for experiment {exp}; already done.')
                    continue
                elif val_k != list(range(1, 4)):
                    print(f'[Step 2] Some k-folds done; running {val_k} for {exp}')
                self.train(
                    hp=exp_hp,
                    label=f'EXP_{exp}',
                    filters={f'include_{exp}': ['include']},
                    save_predictions=True,
                    splits=f'splits_{exp}.json',
                    val_k=val_k,
                    val_strategy='k-fold',
                    save_model=False
                )

        # === Step 3: Run experiments with UQ & save predictions ==================
        if 3 in steps:
            print(sf.util.bold("[Step 3] Running experiments with UQ..."))
            exp_hp.epochs = [1]
            exp_hp.uq = True
            for exp in exp_to_run:
                val_k = [
                    k for k in range(1, 4)
                    if not utils.model_exists(P, f'EXP_{exp}_UQ', outcome=self.outcome, kfold=k)
                ]
                if not len(val_k):
                    print(f'Skipping Step 3 for experiment {exp}; already done.')
                    continue
                elif val_k != list(range(1, 4)):
                    print(f'[Step 3] Some k-folds done; running {val_k} for {exp}')
                self.train(
                    hp=exp_hp,
                    label=f'EXP_{exp}_UQ',
                    filters={f'include_{exp}': ['include']},
                    save_predictions=True,
                    splits=f'splits_{exp}.json',
                    val_k=val_k,
                    val_strategy='k-fold',
                    save_model=False
                )

        # === Step 4: Run nested UQ cross-validation ==============================
        if 4 in steps:
            print(sf.util.bold("[Step 4] Running nested UQ experiments..."))
            exp_hp.epochs = [1]
            exp_hp.uq = True
            for exp in exp_to_run:
                total_slides = exp_to_run[exp]['out2'] + exp_to_run[exp]['out1']
                if total_slides >= 50:
                    self.train_nested_cv(
                        hp=exp_hp,
                        label=f'EXP_{exp}_UQ',
                        val_strategy='k-fold'
                    )
                else:
                    print(f"[Step 4] Skipping UQ for {exp}, need >=50 slides")

        # === Step 5: Train models across full datasets ===========================
        if 5 in steps:
            print(sf.util.bold("[Step 5] Training across full datasets..."))
            exp_hp.epochs = [1]
            exp_hp.uq = True
            for exp in exp_to_run:
                if utils.model_exists(P, f'EXP_{exp}_FULL', outcome=self.outcome):
                    print(f'Skipping Step 5 for experiment {exp}; already done.')
                else:
                    stop_batch = utils.find_cv_early_stop(P, f'EXP_{exp}', outcome=self.outcome, k=3)
                    print(f"Using detected early stop batch {stop_batch}")
                    self.train(
                        hp=exp_hp,
                        label=f'EXP_{exp}_FULL',
                        filters={f'include_{exp}': ['include']},
                        save_model=True,
                        val_strategy='none',
                        steps_per_epoch_override=stop_batch
                    )

        # === Step 6: External validation  ========================================
        if 6 in steps:
            for val_P in eval_Ps:
                print(sf.util.bold(f"[Step 6] Running eval ({val_P.name})..."))
                for exp in exp_to_run:
                    full_model = utils.find_model(P, f'EXP_{exp}_FULL', outcome=self.outcome, epoch=1)
                    if utils.eval_exists(val_P, f'EXP_{exp}_FULL', outcome=self.outcome, epoch=1):
                        print(f'Skipping eval for experiment {exp}; already done.')
                    else:
                        filters = {self.outcome: [self.outcome1, self.outcome2]}
                        val_P.evaluate(
                            full_model,
                            self.outcome,
                            filters=filters,
                            save_predictions=True,
                        )

    def thresholds_from_nested_cv(self, label, outer_k=3, inner_k=5, id=None,
                                  threshold_params=None, epoch=1,
                                  tile_filename='tile_predictions_val_epoch1.csv',
                                  y_true=None, y_pred=None, uncertainty=None):
        """Detects tile- and slide-level UQ thresholds and slide-level prediction
        thresholds from nested cross-validation."""

        if id is None:
            id = label
        patients = self.train_project.dataset(verification=None).patients()
        if threshold_params is None:
            threshold_params = {
                'tile_pred':     'detect',
                'slide_pred':    'detect',
                'plot':          False,
                'patients':      patients
            }
        all_tile_uq_thresh = []
        all_slide_uq_thresh = []
        all_slide_pred_thresh = []
        df = pd.DataFrame()
        for k in range(1, outer_k+1):

            try:
                dfs = utils.df_from_cv(
                    self.train_project,
                    f'{label}-k{k}',
                    outcome=self.outcome,
                    k=inner_k,
                    y_true=y_true,
                    y_pred=y_pred,
                    uncertainty=uncertainty)
            except ModelNotFoundError:
                log.warn(f"Could not find {label} k-fold {k}; skipping")
                continue

            val_path = join(
                utils.find_model(self.train_project, f'{label}', kfold=k, outcome=self.outcome),
                tile_filename
            )
            if not exists(val_path):
                log.warn(f"Could not find {label} k-fold {k}; skipping")
                continue
            tile_uq = threshold.from_cv(
                dfs,
                tile_uq='detect',
                slide_uq=None,
                **threshold_params
            )['tile_uq']
            thresholds = threshold.from_cv(
                dfs,
                tile_uq=tile_uq,
                slide_uq='detect',
                **threshold_params
            )
            all_tile_uq_thresh += [tile_uq]
            all_slide_uq_thresh += [thresholds['slide_uq']]
            all_slide_pred_thresh += [thresholds['slide_pred']]
            if sf.util.path_to_ext(val_path).lower() == 'csv':
                tile_pred_df = pd.read_csv(val_path, dtype={'slide': str})
            elif sf.util.path_to_ext(val_path).lower() in ('parquet', 'gzip'):
                tile_pred_df = pd.read_parquet(val_path)
            else:
                raise OSError(f"Unrecognized prediction filetype {val_path}")
            utils.rename_cols(tile_pred_df, self.outcome, y_true=y_true, y_pred=y_pred, uncertainty=uncertainty)

            def uq_auc_by_level(level):
                results, _ = threshold.apply(
                    tile_pred_df,
                    plot=False,
                    patients=patients,
                    level=level,
                    **thresholds
                )
                return results['auc'], results['percent_incl']

            pt_auc, pt_perc = uq_auc_by_level('patient')
            slide_auc, slide_perc = uq_auc_by_level('slide')
            model = utils.find_model(
                self.train_project,
                f'{label}',
                kfold=k,
                epoch=1,
                outcome=self.outcome
            )
            m_slides = sf.util.get_slides_from_model_manifest(model, dataset=None)
            df = pd.concat([df, pd.DataFrame([{
                'id': id,
                'n_slides': len(m_slides),
                'fold': k,
                'uq': 'include',
                'patient_auc': pt_auc,
                'patient_uq_perc': pt_perc,
                'slide_auc': slide_auc,
                'slide_uq_perc': slide_perc
            }])], axis=0, join='outer', ignore_index=True)

        thresholds = {
            'tile_uq': None if not all_tile_uq_thresh else mean(all_tile_uq_thresh),
            'slide_uq': None if not all_slide_uq_thresh else mean(all_slide_uq_thresh),
            'slide_pred': None if not all_slide_pred_thresh else mean(all_slide_pred_thresh),
        }
        return df, thresholds

    def train(self, hp, label, filters=None, save_predictions='csv',
              validate_on_batch=32, validation_steps=32, **kwargs):
        r"""Train outer cross-validation models.

        Args:
            hp (:class:`slideflow.ModelParams`): Hyperparameters object.
            label (str): Experimental label.
            filters (dict, optional): Dataset filters to use for
                selecting slides. See :meth:`slideflow.Dataset.filter` for
                more information. Defaults to None.
            save_predictions (bool, optional): Save validation predictions to
                model folder. Defaults to 'csv'.

        Keyword args:
            validate_on_batch (int): Frequency of validation checks during
                training, in steps. Defaults to 32.
            validation_steps (int): Number of validation steps to perform
                during each mid-training evaluation check. Defaults to 32.
            **kwargs: All remaining keyword arguments are passed to
                :meth:`slideflow.Project.train`.

        Returns:
            None
        """
        self.train_project.train(
            self.outcome,
            exp_label=label,
            filters=filters,
            params=hp,
            save_predictions=save_predictions,
            validate_on_batch=validate_on_batch,
            validation_steps=validation_steps,
            **kwargs
        )

    def train_nested_cv(self, hp, label, outer_k=3, inner_k=5, **kwargs):
        r"""Train models using nested cross-validation (outer_k=3, inner_k=5),
        skipping already-generated models.

        Args:
            hp (slideflow.ModelParams): Hyperparameters object.
            label (str): Experimental label.

        Keyword args:
            outer_k (int): Number of outer cross-folds. Defaults to 3.
            inner_k (int): Number of inner cross-folds. Defaults to 5.
            **kwargs: All remaining keyword arguments are passed to
                :meth:`slideflow.Project.train`.

        Returns:
            None
        """
        k_models = utils.find_cv(self.train_project, label, k=outer_k, outcome=self.outcome)
        for ki, k_model in enumerate(k_models):
            inner_k_to_run = [
                k for k in range(1, inner_k+1)
                if not utils.model_exists(self.train_project, f'{label}-k{ki+1}', outcome=self.outcome, kfold=k)
            ]
            if not len(inner_k_to_run):
                print(f'Skipping nested cross-val (inner k{ki+1} for experiment '
                    f'{label}; already done.')
            else:
                if inner_k_to_run != list(range(1, inner_k+1)):
                    print(f'Only running k-folds {inner_k_to_run} for nested '
                        f'cross-val k{ki+1} in experiment {label}; '
                        'some k-folds already done.')
                train_slides = sf.util.get_slides_from_model_manifest(
                    k_model, dataset='training'
                )
                self.train(
                    hp=hp,
                    label=f"{label}-k{ki+1}",
                    filters={'slide': train_slides},
                    val_k_fold=inner_k,
                    val_k=inner_k_to_run,
                    save_predictions=True,
                    save_model=False,
                    **kwargs
                )
