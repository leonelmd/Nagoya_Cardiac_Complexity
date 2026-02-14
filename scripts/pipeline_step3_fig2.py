import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind

def add_stat_annotation(ax, x1, x2, y, h, text, color='black'):
    """Draws a bracket with text using plot coordinates."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=color, fontsize=12, fontweight='bold')

def get_sig_chars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def parse_time(t_str):
    try:
        if pd.isna(t_str): return np.nan
        h, m, s = map(int, str(t_str).split(':'))
        return h + m/60.0 + s/3600.0
    except:
        return np.nan

def generate_figure2():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ev_path = os.path.join(base_dir, 'calculations/Full_HRV_Evolution_1min.csv')
    metadata_path = os.path.join(base_dir, 'data/metadata/metadata.csv')
    fig_dir = os.path.join(base_dir, 'results/plots/Figure2')
    os.makedirs(fig_dir, exist_ok=True)
    
    colors = {'Control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    # Load Data
    df_ev = pd.read_csv(ev_path)
    df_meta = pd.read_csv(metadata_path)
    
    # Standardize Groups
    df_ev['Group'] = df_ev['Group'].str.lower().replace({'control': 'Control', 'pd': 'PD'})
    df_meta['Group'] = df_meta['Group'].str.lower().replace({'control': 'Control', 'pd': 'PD'})
    
    # Align Subject IDs
    if 'Subject' not in df_meta.columns:
        if 'Subject_ID' in df_meta.columns:
            df_meta = df_meta.rename(columns={'Subject_ID': 'Subject'})
            
    # Calculate Precise Clock Alignment
    # The new Julia pipeline already aligns Time_h to absolute clock time (0=Midnight)
    df_ev['Clock_Time'] = df_ev['Time_h']
    
    # Plotting Logic: Center at Midnight (12.0)
    # 0 maps to Noon, 12 maps to Midnight, 24 maps to Noon
    df_ev['Plot_Time'] = (df_ev['Clock_Time'] - 12) % 24
    
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)
    
    # --- STAGE 1: 24h Evolution (Top 2 rows) ---
    metrics = ['HR', 'Complexity', 'SDNN', 'RMSSD']
    titles = ['Heart Rate (BPM)', 'Complexity Index (MSE 1-5)', 'SDNN (s)', 'RMSSD (s)']
    
    labels = ['A', 'B', 'C', 'D']
    for idx, m in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        ax.text(-0.05, 1.1, labels[idx], transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')
        
        # Bin data for smooth plotting (coarser as requested)
        df_ev['Bin_Time'] = df_ev['Plot_Time'].round(2)
        subj_bins = df_ev.groupby(['Group', 'Subject', 'Bin_Time'])[m].mean().reset_index()
        summary = subj_bins.groupby(['Group', 'Bin_Time'])[m].agg(['mean', 'sem']).reset_index()
        
        for group in ['Control', 'PD']:
            data = summary[summary['Group'] == group].sort_values('Bin_Time')
            y_smooth = data['mean'].rolling(15, center=True, min_periods=1).mean()
            s_smooth = data['sem'].rolling(15, center=True, min_periods=1).mean()
            ax.plot(data['Bin_Time'], y_smooth, color=colors[group], linewidth=3, label=group)
            ax.fill_between(data['Bin_Time'], y_smooth - s_smooth, y_smooth + s_smooth, color=colors[group], alpha=0.15)
            
        ax.set_title(titles[idx], fontsize=18, fontweight='bold')
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['12 PM', '6 PM', '12 AM', '6 AM', '12 PM'])
        ax.axvline(12, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
        if idx == 0: ax.legend(loc='upper right', fontsize=12)

    # --- STAGE 2: Exhaustive Heatmap (Bottom Left) ---
    print("Generating Heatmap...")
    win_len = 4
    win_data = {}
    for start_h in range(24):
        end_h = start_h + win_len
        if end_h > 24:
            mask = (df_ev['Clock_Time'] >= start_h) | (df_ev['Clock_Time'] < (end_h % 24))
        else:
            mask = (df_ev['Clock_Time'] >= start_h) & (df_ev['Clock_Time'] < end_h)
        label = f"{start_h:02d}"
        win_data[label] = df_ev[mask].groupby(['Group', 'Subject'])['Complexity'].mean().reset_index()
    
    all_pairs = []
    labels_h = list(win_data.keys())
    for i in range(len(labels_h)):
        for j in range(len(labels_h)):
            if i == j: continue
            df1, df2 = win_data[labels_h[i]], win_data[labels_h[j]]
            df_pair = pd.merge(df1, df2, on=['Subject', 'Group'], suffixes=('_1', '_2'))
            df_pair['Delta'] = df_pair['Complexity_1'] - df_pair['Complexity_2']
            c = df_pair[df_pair['Group'] == 'Control']['Delta'].dropna()
            pd_vals = df_pair[df_pair['Group'] == 'PD']['Delta'].dropna()
            if len(c) > 10 and len(pd_vals) > 10:
                t, pval = ttest_ind(c, pd_vals)
                all_pairs.append({'W1': labels_h[i], 'W2': labels_h[j], 'LogP': -np.log10(pval)})
    
    df_h = pd.DataFrame(all_pairs)
    pivot = df_h.pivot(index='W1', columns='W2', values='LogP')
    
    ax_heat = fig.add_subplot(gs[2, 0])
    ax_heat.text(-0.05, 1.1, 'E', transform=ax_heat.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')
    sns.heatmap(pivot, cmap='Spectral_r', ax=ax_heat, cbar_kws={'label': '-log10(p)'})
    ax_heat.set_title(f'Exhaustive Search: Complexity Delta ({win_len}h Windows)', fontsize=18, fontweight='bold')
    ax_heat.set_xlabel('Reference Window (Start Hour)')
    ax_heat.set_ylabel('Comparison Window (Start Hour)')

    # --- STAGE 3: Best Delta Boxplot (Bottom Right) ---
    # Using 07:00-11:00 vs 13:00-17:00
    w1_start, w2_start = 7, 13
    df_w1 = win_data[f"{w1_start:02d}"]
    df_w2 = win_data[f"{w2_start:02d}"]
    df_final = pd.merge(df_w1, df_w2, on=['Subject', 'Group'], suffixes=('_Morning', '_Afternoon'))
    df_final['Delta'] = df_final['Complexity_Morning'] - df_final['Complexity_Afternoon']
    
    ax_box = fig.add_subplot(gs[2, 1])
    ax_box.text(-0.05, 1.1, 'F', transform=ax_box.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')
    
    sns.boxplot(x='Group', y='Delta', data=df_final, palette=colors, ax=ax_box, width=0.5, showfliers=False, order=['Control', 'PD'])
    sns.stripplot(x='Group', y='Delta', data=df_final, palette=colors, ax=ax_box, dodge=False, alpha=0.6, color='black', order=['Control', 'PD'])
    
    c_vals = df_final[df_final['Group'] == 'Control']['Delta']
    p_vals = df_final[df_final['Group'] == 'PD']['Delta']
    t, pval = ttest_ind(c_vals, p_vals)
    
    y_max = df_final['Delta'].max()
    y_min = df_final['Delta'].min()
    dist = y_max - y_min
    add_stat_annotation(ax_box, 0, 1, y_max + dist*0.05, dist*0.05, f'p = {pval:.4f} {get_sig_chars(pval)}')
    
    ax_box.set_title('Best Biomarker: Morning vs. Afternoon Delta', fontsize=18, fontweight='bold')
    ax_box.set_ylabel(r'$\Delta$ Complexity (07-11 vs. 13-17)', fontsize=14)
    ax_box.set_xlabel('')
    ax_box.set_ylim(y_min - dist*0.1, y_max + dist*0.25)
    ax_box.axhline(0, color='black', linestyle='--', alpha=0.3)
    sns.despine(ax=ax_box)

    plt.suptitle("Figure 2: Circadian Fractal Dynamics and Biomarker Discovery", fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = os.path.join(fig_dir, 'Figure2.png')
    plt.savefig(out_path, dpi=300)
    print(f"Figure 2 updated and saved to {out_path}")

if __name__ == "__main__":
    generate_figure2()
