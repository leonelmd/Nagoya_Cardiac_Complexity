import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

def load_and_filter(file_path, start_time_str, window_min=30):
    """Load, clean artifacts, and apply low-pass filter (moving average)."""
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, names=['rel_time', 'rri'])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
        
    # Artifact removal: RRi > 2s or < 0.3s
    data = data[(data['rri'] > 0.3) & (data['rri'] < 2.0)]
    if data.empty: return None, None
    
    # Time alignment (seconds from midnight)
    start_t = datetime.strptime(start_time_str, '%H:%M:%S')
    seconds_from_midnight_start = (start_t - start_t.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    data['abs_sec'] = data['rel_time'] + seconds_from_midnight_start
    
    # Grid for interpolation
    grid_step = 30 
    t_grid = np.arange(data['abs_sec'].min(), data['abs_sec'].max(), grid_step)
    rri_interp = np.interp(t_grid, data['abs_sec'], data['rri'])
    
    window_size = int((window_min * 60) / grid_step)
    rri_smooth = pd.Series(rri_interp).rolling(window=window_size, center=True).mean()
    
    # Wrap to 24h
    t_day = t_grid % (24 * 3600)
    # Center on Midnight
    t_plot = (t_day + 12 * 3600) % (24 * 3600)
    
    return t_plot, rri_smooth

def plot_fig1_comprehensive():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data/metadata/metadata.csv')
    processed_dir = os.path.join(base_dir, 'data/processed_rri')
    results_dir = os.path.join(base_dir, 'figures/Figure1')
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return

    df = pd.read_csv(db_path)
    
    traces = {'control': [], 'PD': []}
    all_mean_rri = [] 
    
    # Common grid for averaging
    common_grid_t = np.arange(0, 24*3600, 30) # 0 to 24h in 30s steps
    matrix_data = {'control': [], 'PD': []}

    print(f"Processing signals for visualization (from {processed_dir})...")
    for idx, row in df.iterrows():
        file_path = os.path.join(processed_dir, row['Filename'])
        if not os.path.exists(file_path): continue
             
        t_plot, rri_smooth = load_and_filter(file_path, row['Start_Time'], window_min=30)
        
        if t_plot is not None:
             # Add to stats dict
             all_mean_rri.append({'Group': row['Group'], 'RRi': row['Avg_RRi']})
             
             # Store T and RRI for individual lines
             traces[row['Group']].append((t_plot, rri_smooth))

             # Prepare for Average Calculation
             idx_sort = np.argsort(t_plot)
             t_sorted = t_plot[idx_sort]
             rri_sorted = rri_smooth.iloc[idx_sort].values
             _, unique_indices = np.unique(t_sorted, return_index=True)
             t_sorted = t_sorted[unique_indices]
             rri_sorted = rri_sorted[unique_indices]
             
             rri_common = np.interp(common_grid_t, t_sorted, rri_sorted, left=np.nan, right=np.nan)
             matrix_data[row['Group']].append(rri_common)

    # Convert to matrix for averaging
    mean_curves = {}
    for g in ['control', 'PD']:
        mat = np.array(matrix_data[g])
        mean_curves[g] = np.nanmean(mat, axis=0)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    ax_ctrl = fig.add_subplot(gs[0, 0])
    ax_pd = fig.add_subplot(gs[0, 1], sharey=ax_ctrl)
    ax_dist = fig.add_subplot(gs[1, :]) 

    colors = {'control': '#2E86AB', 'PD': '#D62828'}
    
    # 1. TOP PANELS: Individual RRi curves + Average Curve
    for ax, group in zip([ax_ctrl, ax_pd], ['control', 'PD']):
        group_traces = traces[group]
        
        # Plot Individual Traces
        for (t_vals, rri_vals) in group_traces:
            idx_s = np.argsort(t_vals)
            t_s = t_vals[idx_s]
            r_s = rri_vals.iloc[idx_s]
            t_hours = t_s / 3600.0
            ax.plot(t_hours, r_s, color=colors[group], alpha=0.15, linewidth=0.8) 
            
        # Plot Average Curve
        if group in mean_curves:
            x_hours_common = common_grid_t / 3600.0
            ax.plot(x_hours_common, mean_curves[group], color=colors[group], linewidth=3.5, label=f'Group Mean', alpha=0.8)
            # Or use group color but darker/thicker? 'black' contrast is usually good for mean over cloud.
            
        ax.set_title(f'{group.capitalize()} Group', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time of Day', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.axvline(12, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 24)
        ax.set_ylim(0.4, 1.4)
        
        tick_pos = np.arange(0, 25, 3)
        tick_labels = ['12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM', '9 AM', '12 PM']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.legend(loc='upper right')

    ax_ctrl.set_ylabel('RRi (s)', fontsize=14)

    # 2. BOTTOM PANEL: Distributions of Average RRi
    if all_mean_rri:
        df_rri = pd.DataFrame(all_mean_rri)
        order = ['control', 'PD']
        
        sns.violinplot(x='Group', y='RRi', hue='Group', data=df_rri, palette=colors, 
                       inner="quart", legend=False, ax=ax_dist, order=order, hue_order=order)
        sns.swarmplot(x='Group', y='RRi', data=df_rri, color="black", alpha=0.5, ax=ax_dist, order=order)
        
        ax_dist.set_title('Average RRi Distribution', fontsize=16, fontweight='bold')
        ax_dist.set_ylabel('Mean 24h RR Interval (s)', fontsize=14)
        ax_dist.set_xlabel('Group', fontsize=14)
    else:
        ax_dist.text(0.5, 0.5, 'No Data', ha='center', va='center')

    plt.suptitle('Figure 1: 24h Heart Rate Variability Trends and Global Distribution', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'Figure1_Comprehensive.svg'), format='svg')
    plt.close('all')
    print(f"Figures saved to {results_dir}")

if __name__ == "__main__":
    plot_fig1_comprehensive()
