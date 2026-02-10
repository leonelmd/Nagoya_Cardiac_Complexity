import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

def analyze_age_complexity_day_night():
    # Paths (Relative to script location)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(base_dir, 'data/metadata/metadata.csv')
    day_path = os.path.join(base_dir, 'calculations/Day_Window_Stats.csv')
    night_path = os.path.join(base_dir, 'calculations/Night_Window_Stats.csv')
    
    output_dir = os.path.join(base_dir, 'figures/Figure4')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    if not os.path.exists(meta_path) or not os.path.exists(day_path) or not os.path.exists(night_path):
        print("Data files not found.")
        return

    df_meta = pd.read_csv(meta_path)
    df_day = pd.read_csv(day_path)
    df_night = pd.read_csv(night_path)
    
    # Merge
    # df_stats has Subject (e.g. SUB-002), df_meta has Subject_ID (e.g. SUB-001)
    df_day = pd.merge(df_day, df_meta[['Subject_ID', 'Age', 'Gender']], left_on='Subject', right_on='Subject_ID')
    df_night = pd.merge(df_night, df_meta[['Subject_ID', 'Age', 'Gender']], left_on='Subject', right_on='Subject_ID')
    
    # --- Visualization ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    colors = {'control': '#2E86AB', 'PD': '#D62828'}
    group_labels = {'control': 'Control', 'PD': 'PD'}
    
    metrics = ['Complexity', 'Norm_Comp']
    periods = [('Day (10am-4pm)', df_day), ('Night (12am-6am)', df_night)]
    
    for row_idx, (period_name, df) in enumerate(periods):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # Scatter
            sns.scatterplot(data=df, x='Age', y=metric, hue='Group', palette=colors, s=100, ax=ax, alpha=0.7)
            
            # Regression and Stats per Group
            stats_texts = []
            
            for i, group in enumerate(['control', 'PD']):
                sub = df[df['Group'] == group]
                if len(sub) < 3: continue
                
                # Linear Regression
                slope, intercept, r_val, p_val, std_err = stats.linregress(sub['Age'], sub[metric])
                
                # Plot Line
                x_vals = np.array([sub['Age'].min(), sub['Age'].max()])
                y_vals = slope * x_vals + intercept
                ax.plot(x_vals, y_vals, color=colors[group], linewidth=2, linestyle='--')
                
                # Stats Text
                sig = "*" if p_val < 0.05 else "n.s."
                txt = f"{group_labels[group]}: m={slope:.4f}, r={r_val:.2f}, p={p_val:.3f} ({sig})"
                stats_texts.append((txt, colors[group]))

            ax.set_title(f'{period_name}: Age vs {metric}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(metric)
            
            # Add stats legend/text
            for i, (txt, col) in enumerate(stats_texts):
                pos_y = 0.95 - (i * 0.05) 
                ax.text(0.95, pos_y, txt, transform=ax.transAxes, ha='right', va='top', 
                        fontsize=11, fontweight='bold', color=col, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.suptitle('Complexity vs Age: Day and Night Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'Age_Complexity_DayNight_Analysis.svg')
    fig.savefig(save_path, format='svg')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    analyze_age_complexity_day_night()
