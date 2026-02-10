import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind

def generate_figure3():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cetram_comp_path = os.path.join(base_dir, 'data/cetram_comparison/Complexity_Index_sample.csv')
    cetram_hrv_path = os.path.join(base_dir, 'data/cetram_comparison/HRV_metrics_cleaned.csv')
    japan_ev_path = os.path.join(base_dir, 'calculations/Full_HRV_Evolution_1min.csv')
    fig_dir = os.path.join(base_dir, 'figures/Figure3')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load & Process CETRAM
    df_c_comp = pd.read_csv(cetram_comp_path)
    df_c_hrv = pd.read_csv(cetram_hrv_path)
    df_c_hrv['HR'] = 60000.0 / df_c_hrv['HRV_MeanNN']
    df_c = pd.merge(df_c_comp, df_c_hrv[['Subject', 'HR']], on='Subject')
    df_c['Norm_Comp'] = df_c['Complexity'] / df_c['HR']
    df_c['Dataset'] = 'CETRAM (Chile)'
    df_c['State'] = 'Seated Rest'
    df_c['Group_Plot'] = df_c['Group'].str.upper().replace({'CONTROL': 'Control', 'PD': 'Parkinson'})

    # 2. Load & Process Japan (Nagoya)
    df_j_full = pd.read_csv(japan_ev_path)
    # Peak Daytime (8am-8pm)
    df_j_day = df_j_full[(df_j_full['Time_h'] >= 8) & (df_j_full['Time_h'] <= 20)]
    japan_peak = df_j_day.groupby(['Subject', 'Group'])['Norm_Comp'].max().reset_index()
    japan_peak['Dataset'] = 'Nagoya (Japan)'
    japan_peak['State'] = 'Daytime Peak'
    
    # Avg Daytime (10am-4pm)
    japan_avg = df_j_full[(df_j_full['Time_h'] >= 10) & (df_j_full['Time_h'] < 16)]
    japan_avg = japan_avg.groupby(['Subject', 'Group'])['Norm_Comp'].mean().reset_index()
    japan_avg['Dataset'] = 'Nagoya (Japan)'
    japan_avg['State'] = 'Day Avg (10a-4p)'

    # Combine for plot
    japan_combined = pd.concat([japan_peak, japan_avg], ignore_index=True)
    japan_combined['Group_Plot'] = japan_combined['Group'].replace({'control': 'Control', 'PD': 'Parkinson'})
    
    df_all = pd.concat([df_c[['Norm_Comp', 'Dataset', 'State', 'Group_Plot']], 
                        japan_combined[['Norm_Comp', 'Dataset', 'State', 'Group_Plot']]], ignore_index=True)

    # 3. Plotting
    plt.figure(figsize=(18, 9))
    sns.set_style("ticks")
    colors = {'Control': '#2E86AB', 'Parkinson': '#D62828'}
    
    order = ['Seated Rest', 'Daytime Peak', 'Day Avg (10a-4p)']
    x_labels = ['Chile (CETRAM)\nResting', 'Japan (Nagoya)\nPeak Active', 'Japan (Nagoya)\nAverage Day']
    
    sns.boxplot(x='State', y='Norm_Comp', hue='Group_Plot', data=df_all, order=order, 
                palette=colors, width=0.6, showfliers=False, boxprops=dict(alpha=0.3), hue_order=['Control', 'Parkinson'])
    sns.stripplot(x='State', y='Norm_Comp', hue='Group_Plot', data=df_all, order=order,
                  palette=colors, dodge=True, alpha=0.5, edgecolor='white', linewidth=0.5, hue_order=['Control', 'Parkinson'])

    # Add Stats
    for i, state in enumerate(order):
        data_s = df_all[df_all['State'] == state]
        c = data_s[data_s['Group_Plot'] == 'Control']['Norm_Comp'].dropna()
        p = data_s[data_s['Group_Plot'] == 'Parkinson']['Norm_Comp'].dropna()
        t, pval = ttest_ind(c, p)
        drop = 100 * (p.mean() - c.mean()) / c.mean()
        
        plt.text(i, df_all['Norm_Comp'].max()*1.05, f'P = {pval:.4f}', ha='center', fontsize=14, fontweight='bold')
        plt.text(i, df_all['Norm_Comp'].max()*1.15, f'Δ = {drop:.1f}%', ha='center', fontsize=14, color='#D62828', fontweight='bold')

    plt.title('Global Datasets Validation: Heart-Rate Normalized Cardiac Complexity', fontsize=22, fontweight='bold', pad=40)
    plt.ylabel('Normalized Complexity (AUC 1-5 / HR)', fontsize=16)
    plt.xlabel('Recording State / Cohort', fontsize=16)
    plt.xticks(ticks=[0, 1, 2], labels=x_labels, fontsize=14)
    plt.ylim(0, 0.05)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Clinical Group", loc='upper right', fontsize=14)
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Figure3_Multicenter_Standardization.svg'), format='svg')
    print(f"Figure 3 saved to {fig_dir}")

if __name__ == "__main__":
    generate_figure3()
