import json
import subprocess

with open('Project_15_WavCaps.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that evaluates the model (the one with "# Evaluation Logic")
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any("--- RESULTS ---" in line for line in cell['source']):
        insert_idx = i
        break

new_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "# Plot Retrieval Results (Text-to-Audio)\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "labels = ['Baseline (Random)', 'Main (Trained)']\n",
    "r1_scores = [b_r1, results['R@1']]\n",
    "r5_scores = [b_r5, results['R@5']]\n",
    "\n",
    "x = np.arange(len(labels))\n",
    "width = 0.35\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "rects1 = ax.bar(x - width/2, r1_scores, width, label='R@1')\n",
    "rects2 = ax.bar(x + width/2, r5_scores, width, label='R@5')\n",
    "\n",
    "ax.set_ylabel('Recall (%)')\n",
    "ax.set_title('Text-to-Audio Retrieval Performance')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(labels)\n",
    "ax.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
  ]
}

if insert_idx != -1:
    nb['cells'].insert(insert_idx + 1, new_cell)

with open('Project_15_WavCaps.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Added the plotting cell to the notebook.")
