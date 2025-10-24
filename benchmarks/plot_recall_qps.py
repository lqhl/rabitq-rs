#!/usr/bin/env python3
"""Plot recall vs QPS curves for MSTG and IVF"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_recall_qps(csv_path='recall_qps_results.csv', output_path='recall_qps_plot.png'):
    # Read data
    df = pd.read_csv(csv_path)

    # Split by method
    mstg_df = df[df['method'] == 'MSTG'].sort_values('recall_at_100')
    ivf_df = df[df['method'] == 'IVF'].sort_values('recall_at_100')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curves
    ax.plot(mstg_df['recall_at_100'] * 100, mstg_df['qps'],
            marker='o', linewidth=2, markersize=8, label='MSTG', color='#2E86AB')
    ax.plot(ivf_df['recall_at_100'] * 100, ivf_df['qps'],
            marker='s', linewidth=2, markersize=8, label='IVF', color='#A23B72')

    # Labels and title
    ax.set_xlabel('Recall@100 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queries per Second (QPS)', fontsize=12, fontweight='bold')
    ax.set_title('MSTG vs IVF: Recall vs Throughput Trade-off\nGIST 1M Dataset (960D)',
                 fontsize=14, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Log scale for QPS to show differences better
    ax.set_yscale('log')

    # Annotate some key points
    # Find highest recall points for each method
    mstg_best = mstg_df.iloc[-1]
    ivf_best = ivf_df.iloc[-1]

    ax.annotate(f"MSTG best:\n{mstg_best['recall_at_100']*100:.1f}% @ {mstg_best['qps']:.0f} QPS",
                xy=(mstg_best['recall_at_100']*100, mstg_best['qps']),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#2E86AB', alpha=0.3),
                fontsize=9)

    ax.annotate(f"IVF best:\n{ivf_best['recall_at_100']*100:.1f}% @ {ivf_best['qps']:.0f} QPS",
                xy=(ivf_best['recall_at_100']*100, ivf_best['qps']),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#A23B72', alpha=0.3),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")

    # Also create a latency plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(mstg_df['recall_at_100'] * 100, mstg_df['latency_ms'],
             marker='o', linewidth=2, markersize=8, label='MSTG', color='#2E86AB')
    ax2.plot(ivf_df['recall_at_100'] * 100, ivf_df['latency_ms'],
             marker='s', linewidth=2, markersize=8, label='IVF', color='#A23B72')

    ax2.set_xlabel('Recall@100 (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('MSTG vs IVF: Recall vs Latency Trade-off\nGIST 1M Dataset (960D)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.set_yscale('log')

    plt.tight_layout()
    latency_output = output_path.replace('.png', '_latency.png')
    plt.savefig(latency_output, dpi=300, bbox_inches='tight')
    print(f"✓ Latency plot saved to {latency_output}")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nMSTG:")
    print(f"  Recall range: {mstg_df['recall_at_100'].min()*100:.1f}% - {mstg_df['recall_at_100'].max()*100:.1f}%")
    print(f"  QPS range:    {mstg_df['qps'].min():.0f} - {mstg_df['qps'].max():.0f}")
    print(f"  Latency range: {mstg_df['latency_ms'].min():.2f}ms - {mstg_df['latency_ms'].max():.2f}ms")

    print(f"\nIVF:")
    print(f"  Recall range: {ivf_df['recall_at_100'].min()*100:.1f}% - {ivf_df['recall_at_100'].max()*100:.1f}%")
    print(f"  QPS range:    {ivf_df['qps'].min():.0f} - {ivf_df['qps'].max():.0f}")
    print(f"  Latency range: {ivf_df['latency_ms'].min():.2f}ms - {ivf_df['latency_ms'].max():.2f}ms")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'recall_qps_results.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'recall_qps_plot.png'

    plot_recall_qps(csv_path, output_path)
