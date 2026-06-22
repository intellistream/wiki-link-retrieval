#!/usr/bin/env python3
"""Generate publication-quality PDFs for all 4 paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

# ── Figure 1: Motivating Example ──────────────────────────────────────
def fig1_motivation():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Query box at top center
    fig.text(0.5, 0.93, 'Query: "The director of Identity (2003)\nalso directed what 1997 film?"',
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF2CC', edgecolor='#D6B656'))

    # Common data
    docs = [
        ('Identity (film)', 'BM25: 12.4', '#DAE8FC', '#6C8EBF', False),
        ('John Cusack', 'BM25: 8.1', '#DAE8FC', '#6C8EBF', False),
        ('Mangold, James', 'BM25: 3.2', '#DAE8FC', '#6C8EBF', False),
    ]
    gold_miss = ('James Mangold', 'Gold doc (missed)', '#F8CECC', '#B85450', True)
    gold_hit  = ('James Mangold', 'Gold doc (recovered!)', '#D5E8D4', '#82B366', False)

    def draw_list(ax, items, extra=None):
        y_positions = [0.78, 0.58, 0.38, 0.18]
        for i, (title, sub, fc, ec, dashed) in enumerate(items):
            box = FancyBboxPatch((0.05, y_positions[i]-0.06), 0.9, 0.12,
                                 boxstyle="round,pad=0.02", facecolor=fc,
                                 edgecolor=ec, linewidth=1.2,
                                 linestyle='--' if dashed else '-')
            ax.add_patch(box)
            ax.text(0.5, y_positions[i]+0.02, title, ha='center', va='center',
                    fontsize=9, fontweight='bold')
            ax.text(0.5, y_positions[i]-0.03, sub, ha='center', va='center',
                    fontsize=7, color='#555555')

    # (a) BM25 only
    ax_l.set_title('(a) BM25 only', fontsize=10, fontweight='bold', pad=8)
    draw_list(ax_l, docs + [gold_miss])

    # (b) BM25 + wiki-link
    ax_r.set_title('(b) BM25 + Wiki-Link', fontsize=10, fontweight='bold', pad=8)
    draw_list(ax_r, docs + [gold_hit])

    # Wiki-link arrow on right panel
    ax_r.annotate('', xy=(0.98, 0.18), xytext=(0.98, 0.78),
                  arrowprops=dict(arrowstyle='->', color='#82B366', lw=2,
                                  connectionstyle='arc3,rad=0.3'))
    ax_r.text(0.99, 0.48, 'wiki-\nlink', fontsize=7, color='#82B366',
              ha='left', va='center', fontstyle='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('fig1_motivation.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('  fig1_motivation.pdf')


# ── Figure 2: Architecture Pipeline ───────────────────────────────────
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#DAE8FC', ec='#6C8EBF', fs=8, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        fw = 'bold' if bold else 'normal'
        lines = text.split('\n')
        for i, line in enumerate(lines):
            fsi = fs if i == 0 else fs - 1.5
            ax.text(x + w/2, y + h/2 + (len(lines)-1-2*i)*0.12, line,
                    ha='center', va='center', fontsize=fsi, fontweight=fw)

    def arrow(x1, y1, x2, y2, color='black', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    # Offline phase
    ax.text(5, 3.85, 'offline', ha='center', fontsize=8, fontstyle='italic', color='#888888')
    box(0.3, 3.2, 1.8, 0.55, 'Wiki Pages\n(.md files)')
    box(3.0, 3.2, 1.8, 0.55, 'Link Parser\nregex + resolve')
    box(5.7, 3.2, 2.2, 0.55, 'Adjacency List\nG = (V, E) bidir.', '#FFF2CC', '#D6B656')

    arrow(2.1, 3.47, 3.0, 3.47)
    arrow(4.8, 3.47, 5.7, 3.47)

    # Divider
    ax.plot([0.1, 9.9], [2.9, 2.9], '--', color='#CCCCCC', lw=1)

    # Online phase
    ax.text(3.9, 2.65, 'online', ha='center', fontsize=8, fontstyle='italic', color='#888888')
    box(0.3, 1.6, 1.8, 0.55, 'User Query', bold=True)
    box(3.0, 1.6, 1.8, 0.55, 'BM25 Retriever\ntop-k hits')
    box(5.7, 1.6, 1.8, 0.55, '1-Hop Expansion\nα · score')
    box(8.2, 1.6, 1.5, 0.55, 'Augmented\nResult Set', '#FFF2CC', '#D6B656')

    arrow(2.1, 1.87, 3.0, 1.87)
    arrow(4.8, 1.87, 5.7, 1.87)
    arrow(7.5, 1.87, 8.2, 1.87)

    # Graph feeds expansion (vertical arrow)
    ax.annotate('', xy=(6.6, 2.15), xytext=(6.8, 3.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                connectionstyle='arc3,rad=-0.15'))

    # Legend
    bm25_patch = mpatches.Patch(facecolor='#DAE8FC', edgecolor='#6C8EBF', label='Processing step')
    data_patch = mpatches.Patch(facecolor='#FFF2CC', edgecolor='#D6B656', label='Data structure')
    ax.legend(handles=[bm25_patch, data_patch], loc='lower center', ncol=2,
              fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('fig2_architecture.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('  fig2_architecture.pdf')


# ── Figure 3: Worked Example (Link Graph) ─────────────────────────────
def fig3_example():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    def node(x, y, w, h, text, fc, ec):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.06", facecolor=fc,
                              edgecolor=ec, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    def edge(x1, y1, x2, y2, label='', color='#6C8EBF', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my+0.15, label, fontsize=7, color=color,
                    fontstyle='italic', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.9))

    # Nodes
    node(1.2, 3.0, 2.0, 0.6, 'Identity (film)', '#DAE8FC', '#6C8EBF')
    node(4.2, 3.5, 2.0, 0.6, 'James Mangold', '#D5E8D4', '#82B366')
    node(4.8, 1.5, 1.7, 0.6, 'Cop Land', '#DAE8FC', '#6C8EBF')
    node(1.2, 1.0, 1.7, 0.6, 'John Cusack', '#DAE8FC', '#6C8EBF')

    # Edges
    edge(2.2, 3.1, 3.2, 3.4, 'director', '#82B366', 2.0)
    edge(4.5, 3.2, 4.7, 1.8, 'filmography', '#6C8EBF', 1.3)
    edge(1.2, 2.7, 1.2, 1.3, 'cast', '#6C8EBF', 1.3)

    # Legend
    bm25_patch = mpatches.Patch(facecolor='#DAE8FC', edgecolor='#6C8EBF', label='BM25 retrieved')
    gold_patch = mpatches.Patch(facecolor='#D5E8D4', edgecolor='#82B366', label='Gold answer')
    ax.legend(handles=[bm25_patch, gold_patch], loc='lower right', fontsize=7,
              frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig('fig3_example.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('  fig3_example.pdf')


# ── Figure 4: Recall-Precision Tradeoff ───────────────────────────────
def fig4_tradeoff():
    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    m_vals = [1, 3, 5, 10]
    ndcg = [0.3735, 0.3642, 0.3524, 0.3244]
    recall = [0.8366, 0.8922, 0.9100, 0.9210]

    # Left axis: nDCG@10
    color_ndcg = '#B85450'
    ax1.set_xlabel('Max Expansion Budget ($m$)', fontsize=9)
    ax1.set_ylabel('nDCG@10', fontsize=9, color=color_ndcg)
    ax1.set_ylim(0.30, 0.40)
    ax1.set_xticks(m_vals)
    ax1.tick_params(axis='y', labelcolor=color_ndcg, labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.grid(axis='y', alpha=0.2)

    line1 = ax1.plot(m_vals, ndcg, 'o-', color=color_ndcg, linewidth=2,
                     markersize=6, label='nDCG@10', zorder=5)
    for i, v in enumerate(ndcg):
        ax1.annotate(f'{v:.4f}', (m_vals[i], v), textcoords="offset points",
                     xytext=(5, 8), fontsize=7, color=color_ndcg)

    # Right axis: Recall@100
    ax2 = ax1.twinx()
    color_r100 = '#6C8EBF'
    ax2.set_ylabel('Recall@100', fontsize=9, color=color_r100)
    ax2.set_ylim(0.80, 0.95)
    ax2.tick_params(axis='y', labelcolor=color_r100, labelsize=8)

    line2 = ax2.plot(m_vals, recall, 's-', color=color_r100, linewidth=2,
                     markersize=6, label='Recall@100', zorder=5)
    for i, v in enumerate(recall):
        ax2.annotate(f'{v:.4f}', (m_vals[i], v), textcoords="offset points",
                     xytext=(5, -14), fontsize=7, color=color_r100)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('fig4_tradeoff.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('  fig4_tradeoff.pdf')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('Generating figures...')
    fig1_motivation()
    fig2_architecture()
    fig3_example()
    fig4_tradeoff()
    print('Done.')
