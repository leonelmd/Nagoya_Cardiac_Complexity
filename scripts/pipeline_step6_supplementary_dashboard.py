import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

def plot_final_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'calculations/Full_HRV_Evolution_1min.csv')
    results_dir = os.path.join(base_dir, 'figures/Supplementary')
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    # Rename Alpha1 to DFA_alpha1 to match script expectations
    if 'Alpha1' in df.columns:
        df = df.rename(columns={'Alpha1': 'DFA_alpha1'})
        
    df['Plot_Time'] = (df['Time_h'] + 12) % 24  # Center on Midnight
    df['Time_Bin'] = df['Plot_Time'].round(2)
    
    metrics = ['HR', 'SDNN', 'SD2', 'Complexity', 'Norm_Comp', 'DFA_alpha1']
    titles = ['Heart Rate (bpm)', 'SDNN (s)', 'Poincare SD2 (s)', 'Complexity Index (1-5)', 'Normalized Complexity (Index/HR)', 'Fractal DFA alpha1']
    
    colors = {'control': '#2E86AB', 'PD': '#D62828'}
    
    # 1. Dashboard Plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        if m not in df.columns:
            print(f"Warning: Metric {m} not found in columns: {df.columns}")
            continue
            
        ax = axes[i]
        # Aggregate per subject first to avoid pseudo-replication in error bars
        subj_bins = df.groupby(['Group', 'Subject', 'Time_Bin'])[m].mean().reset_index()
        summary = subj_bins.groupby(['Group', 'Time_Bin'])[m].agg(['mean', 'sem']).reset_index()
        
        for group in ['control', 'PD']:
            data = summary[summary['Group'] == group].sort_values('Time_Bin')
            # Smoothing
            y_smooth = data['mean'].rolling(window=10, center=True).mean()
            sem_smooth = data['sem'].rolling(window=10, center=True).mean()
            
            ax.plot(data['Time_Bin'], y_smooth, color=colors[group], linewidth=3, label=group.capitalize())
            ax.fill_between(data['Time_Bin'], y_smooth - sem_smooth, y_smooth + sem_smooth, color=colors[group], alpha=0.2)
            
        ax.set_title(titles[i], fontsize=16, fontweight='bold')
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xticklabels(['12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM', '9 AM', '12 PM'])
        ax.axvline(12, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.1)
        if i == 0: ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'final_24h_hrv_trends.svg'), format='svg')
    print("Dashboard saved.")

    # 2. HR Statistics & Significance Window Summary
    win_stats = []
    
    def get_window_stats(start_h, end_h, label):
        mask = (df['Time_h'] >= start_h) & (df['Time_h'] < end_h)
        df_win = df[mask]
        subj_avg = df_win.groupby(['Group', 'Subject'])[metrics].mean().reset_index()
        
        for m in metrics:
            if m not in subj_avg.columns: continue
            
            c = subj_avg[subj_avg['Group'] == 'control'][m].dropna()
            p = subj_avg[subj_avg['Group'] == 'PD'][m].dropna()
            if len(c) > 0 and len(p) > 0:
                t, pval = ttest_ind(c, p)
                drop = 100 * (p.mean() - c.mean()) / c.mean()
                win_stats.append({
                    'Window': label, 'Metric': m, 'Control_Mean': c.mean(), 'PD_Mean': p.mean(), 'P_val': pval, 'Drop_%': drop
                })

    get_window_stats(10, 16, 'Day (10am-4pm)')
    get_window_stats(0, 6, 'Night (12am-6am)')
    
    stats_df = pd.DataFrame(win_stats)
    stats_df.to_csv(os.path.join(results_dir, 'final_statistical_summary.csv'), index=False)
    print("Stats summary saved.")

if __name__ == "__main__":
    plot_final_results()
