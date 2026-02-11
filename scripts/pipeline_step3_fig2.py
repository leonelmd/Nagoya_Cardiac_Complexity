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

def plot_illustration(ax):
    """
    Creates a schematic illustration of the Diurnal Amplitude metric.
    Uses a synthetic sine wave to represent a healthy circadian rhythm.
    """
    t = np.linspace(0, 24, 200)
    # Synthetic "Healthy" Curve: Peak at 14:00 (t=14), Trough at 02:00 (t=2, 26)
    # Shifted sine: -cos((t - 14)/24 * 2pi) ? No.
    # Simple sine centered at 0: cos(t). Peak at 0.
    # We want peak at 14h.
    # y = A * cos((t - 14) * 2pi / 24)
    y = 0.5 * np.cos((t - 14) * 2 * np.pi / 24) + 1.0 # Mean 1.0, Amp 0.5
    
    # Plot Curve
    ax.plot(t, y, color='#2E86AB', linewidth=3, label='Healthy Rhythm')
    
    # Define Windows
    day_mask = (t >= 10) & (t <= 16)
    night_mask = (t >= 0) & (t <= 6)
    
    # Calculate Means of Synthetic Data
    day_mean = np.mean(y[day_mask])
    night_mean = np.mean(y[night_mask]) # Note: 0-6 is correct for this 0-24 grid
    
    # Shade Regions
    ax.fill_between(t, 0, 2, where=day_mask, color='#FFD700', alpha=0.3, label='Day Window (10a-4p)')
    ax.fill_between(t, 0, 2, where=night_mask, color='#0B3D91', alpha=0.2, label='Night Window (12a-6a)')
    
    # Plot Mean Levels
    ax.hlines(day_mean, 0, 24, colors='#FFD700', linestyles='--', linewidth=2)
    ax.hlines(night_mean, 0, 24, colors='#0B3D91', linestyles='--', linewidth=2)
    
    # Annotation Arrow
    # Draw arrow at t=20 (neutral zone) or t=14?
    # Let's draw it at t=22 to be clear
    x_arrow = 22
    ax.annotate('', xy=(x_arrow, day_mean), xytext=(x_arrow, night_mean),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(x_arrow + 0.5, (day_mean + night_mean)/2, 'Diurnal\nAmplitude', 
            ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.text(13, day_mean + 0.1, 'Day Mean', color='#B8860B', fontweight='bold', ha='center')
    ax.text(3, night_mean - 0.1, 'Night Mean', color='#0B3D91', fontweight='bold', ha='center')

    ax.set_title('Metric Calculation: Diurnal Amplitude', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time of Day (h)')
    ax.set_ylim(0.4, 1.6)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['12a', '6a', '12p', '6p', '12a'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

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
    
    fig = plt.figure(figsize=(22, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.2)
    
    # 1. 24h Evolution
    df_ev = pd.read_csv(ev_path)
    df_ev['Plot_Time'] = (df_ev['Time_h'] + 12) % 24
    
    metrics = ['HR', 'Complexity', 'SDNN', 'RMSSD']
    titles = ['Heart Rate (BPM)', 'Complexity Index (MSE 1-5)', 'SDNN (s)', 'RMSSD (s)']
    
    for idx, m in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
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

    # 2. Methodology & MSE Curves (Middle Row)
    # Left: Illustration
    ax_ill = fig.add_subplot(gs[2, 0])
    plot_illustration(ax_ill)
    
    # Right: MSE Curves (Combined Day/Night)
    if os.path.exists(mse_day_path) and os.path.exists(mse_night_path):
        mse_day = pd.read_csv(mse_day_path)
        mse_night = pd.read_csv(mse_night_path)
        
        ax_mse = fig.add_subplot(gs[2, 1])
        
        # Plot Day (Solid)
        sns.lineplot(data=mse_day, x='Scales', y='MSE', hue='Group', palette=colors, ax=ax_mse, 
                     errorbar=None, linewidth=2.5, hue_order=['control', 'PD'])
        
        # Plot Night (Dashed) - We need to manually handle this to avoid hue confusion or use style
        # Easy hack: Plot manually loop
        # Or just plot Day here and maybe Night in supplement? 
        # User asked for "MSE Curves for Day and Night" in step 0. 
        # Let's try to overlay them with styles.
        
        # Overlay Night with dashed lines
        for g in ['control', 'PD']:
            dat = mse_night[mse_night['Group'] == g]
            # Aggregate mean for simple plotting if seaborn gives trouble with style+hue
            # Let's use seaborn but specify style
            pass

        # Re-doing with style
        mse_day['Period'] = 'Day'
        mse_night['Period'] = 'Night'
        mse_all = pd.concat([mse_day, mse_night])
        
        sns.lineplot(data=mse_all, x='Scales', y='MSE', hue='Group', style='Period', palette=colors, ax=ax_mse,
                     linewidth=2, hue_order=['control', 'PD'], style_order=['Day', 'Night'])
        
        ax_mse.set_title('MSE Curves: Day (Solid) vs Night (Dashed)', fontsize=16, fontweight='bold')
        ax_mse.grid(True, alpha=0.2)
        ax_mse.legend(loc='upper right', fontsize=10)

    # 3. Circadian Amplitude Analysis (Bottom Row)
    # Use sub-gridspec for 4 columns
    gs_bottom = gs[3, :].subgridspec(1, 4, wspace=0.4)
    
    df_day = pd.read_csv(day_path)
    df_night = pd.read_csv(night_path)
    df_m = pd.merge(df_day, df_night, on=['Group', 'Subject'], suffixes=('_Day', '_Night'))
    
    amp_metrics = ['HR', 'Complexity', 'SDNN', 'RMSSD']
    amp_titles = ['HR', 'Complexity', 'SDNN', 'RMSSD']
    
    for idx, m in enumerate(amp_metrics):
        # Calculate Difference
        df_m[f'{m}_Amp'] = df_m[f'{m}_Day'] - df_m[f'{m}_Night']
        
        ax = fig.add_subplot(gs_bottom[0, idx])
        
        sns.boxplot(x='Group', y=f'{m}_Amp', data=df_m, palette=colors, ax=ax, width=0.5, showfliers=False, order=['control', 'PD'])
        sns.stripplot(x='Group', y=f'{m}_Amp', data=df_m, palette=colors, ax=ax, dodge=False, alpha=0.6, color='black', edgecolor='white', linewidth=0.5, order=['control', 'PD'])
        
        # Stats
        c = df_m[df_m['Group'] == 'control'][f'{m}_Amp'].dropna()
        p_vals = df_m[df_m['Group'] == 'PD'][f'{m}_Amp'].dropna()
        t, pval = ttest_ind(c, p_vals)
        
        y_max = df_m[f'{m}_Amp'].max()
        y_min = df_m[f'{m}_Amp'].min()
        h = (y_max - y_min) * 0.1
        if h == 0: h = 1.0
        
        add_stat_annotation(ax, 0, 1, y_max + h, h*0.5, get_sig_chars(pval))
        
        ax.set_title(f'Δ {amp_titles[idx]}', fontsize=14, fontweight='bold')
        if idx == 0: ax.set_ylabel('Amplitude (Day - Night)', fontsize=12)
        else: ax.set_ylabel('')
        ax.set_xlabel('')
        ax.axhline(0, linestyle='--', color='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Figure2.svg'), format='svg')
    print(f"Figure 2 saved to {fig_dir}")

if __name__ == "__main__":
    generate_figure2()
