import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind

def add_stat_annotation(ax, x1, x2, y, h, text, color='black'):
    """
    Draws a bracket with text using plot coordinates.
    x1, x2: x-coordinates of the two bars
    y: maximum height
    h: height of the bracket legs
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=color, fontsize=12, fontweight='bold')

def get_sig_chars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def generate_figure2():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ev_path = os.path.join(base_dir, 'calculations/Full_HRV_Evolution_1min.csv')
    day_path = os.path.join(base_dir, 'calculations/Day_Window_Stats.csv')
    night_path = os.path.join(base_dir, 'calculations/Night_Window_Stats.csv')
    mse_day_path = os.path.join(base_dir, 'calculations/MSE_curves_Day.csv')
    mse_night_path = os.path.join(base_dir, 'calculations/MSE_curves_Night.csv')
    fig_dir = os.path.join(base_dir, 'figures/Figure2')
    os.makedirs(fig_dir, exist_ok=True)
    
    colors = {'control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.2)
    
    # 1. 24h Evolution
    df_ev = pd.read_csv(ev_path)
    df_ev['Plot_Time'] = (df_ev['Time_h'] + 12) % 24
    
    metrics = ['HR', 'Complexity']
    titles = ['Heart Rate (BPM)', 'Complexity Index (MSE 1-5)']
    
    for idx, m in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        subj_bins = df_ev.groupby(['Group', 'Subject', df_ev['Plot_Time'].round(2)])[m].mean().reset_index()
        summary = subj_bins.groupby(['Group', 'Plot_Time'])[m].agg(['mean', 'sem']).reset_index()
        
        for group in ['control', 'PD']:
            data = summary[summary['Group'] == group].sort_values('Plot_Time')
            y_smooth = data['mean'].rolling(15, center=True).mean()
            s_smooth = data['sem'].rolling(15, center=True).mean()
            ax.plot(data['Plot_Time'], y_smooth, color=colors[group], linewidth=3, label=group.capitalize())
            ax.fill_between(data['Plot_Time'], y_smooth - s_smooth, y_smooth + s_smooth, color=colors[group], alpha=0.15)
            
        ax.set_title(titles[idx], fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xticklabels(['12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM', '9 AM', '12 PM'])
        ax.axvline(12, color='black', linestyle='--', alpha=0.3)
        if idx == 0: ax.legend()

    # 2. MSE Curves (Day vs Night) - AUTO SCALING
    if os.path.exists(mse_day_path) and os.path.exists(mse_night_path):
        mse_day = pd.read_csv(mse_day_path)
        mse_night = pd.read_csv(mse_night_path)
        
        for idx, (data, label) in enumerate([(mse_day, 'Day (10am-4pm)'), (mse_night, 'Night (12am-6am)')]):
            ax = fig.add_subplot(gs[1, idx])
            sns.lineplot(data=data, x='Scales', y='MSE', hue='Group', palette=colors, ax=ax, 
                         errorbar='sd', linewidth=2.5, hue_order=['control', 'PD'])
            ax.set_title(f'MSE Curves: {label}', fontsize=16, fontweight='bold')
            # Removed set_ylim(0, 2.5) to let it auto-scale for better curve appreciation
            # Optionally add a small margin
            ax.margins(x=0.01)
            ax.grid(True, alpha=0.2)

    # 3. Complexity Comparison
    df_day = pd.read_csv(day_path)
    df_night = pd.read_csv(night_path)
    df_day['Window'] = 'Day'
    df_night['Window'] = 'Night'
    df_stats = pd.concat([df_day, df_night])
    
    comp_types = ['Complexity', 'Norm_Comp']
    comp_titles = ['Complexity Index (Unnormalized)', 'Heart-Rate Normalized Complexity']
    
    for idx, ct in enumerate(comp_types):
        ax = fig.add_subplot(gs[2, idx])
        
        # Draw boxplot
        sns.boxplot(x='Window', y=ct, hue='Group', data=df_stats, palette=colors, ax=ax, 
                    width=0.6, showfliers=False, hue_order=['control', 'PD'])
        sns.stripplot(x='Window', y=ct, hue='Group', data=df_stats, palette=colors, ax=ax, 
                      dodge=True, alpha=0.4, color='black', edgecolor='white', linewidth=0.5, hue_order=['control', 'PD'])
        
        # Add P-value Brackets
        for i, window in enumerate(['Day', 'Night']):
            w_data = df_stats[df_stats['Window'] == window]
            c = w_data[w_data['Group'] == 'control'][ct].dropna()
            p = w_data[w_data['Group'] == 'PD'][ct].dropna()
            t, pval = ttest_ind(c, p)
            
            # Find y position
            y_max = w_data[ct].max()
            # If using overall max might be cleaner
            y_overall_max = df_stats[ct].max()
            h = y_overall_max * 0.05
            
            # Position for bracket
            # Boxplot places groups at i-0.2 and i+0.2 roughly
            x1, x2 = i - 0.2, i + 0.2
            
            sig_text = get_sig_chars(pval)
            # Lift over the highest point in this specific group
            y_bar = y_max + h*0.5
            
            add_stat_annotation(ax, x1, x2, y_bar, h*0.5, sig_text)

        ax.set_title(comp_titles[idx], fontsize=16, fontweight='bold')
        ax.set_ylabel('Index Value', fontsize=14)
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Figure2_Complexity_Detailed.svg'), format='svg')
    print(f"Figure 2 saved to {fig_dir}")

if __name__ == "__main__":
    generate_figure2()
