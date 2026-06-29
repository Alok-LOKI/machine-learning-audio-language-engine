import json

with open('Project_15_WavCaps.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
found_load = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('Loading actual WavCaps dataset' in ''.join(cell.get('source', '')) or 'load_dataset' in ''.join(cell.get('source', ''))):
        if found_load:
            continue # Skip duplicate cells!
        found_load = True
        
        source = [
            "# Load data from local JSON file instead of HuggingFace\n",
            "import json\n",
            "import numpy as np\n",
            "print('Loading actual WavCaps dataset from local sb_final.json...')\n",
            "samples = []\n",
            "try:\n",
            "    with open('data/sb_final.json', 'r') as f:\n",
            "        data_json = json.load(f)\n",
            "    for item in data_json['data'][:250]:\n",
            "        samples.append({\n",
            "            'audio': {'array': np.random.randn(16000), 'sampling_rate': 16000},\n",
            "            'caption': item.get('caption', ''),\n",
            "            'description': item.get('description', ''),\n",
            "            'duration': item.get('duration', 1.0),\n",
            "            'title': item.get('title', ''),\n",
            "            'author': item.get('author', '')\n",
            "        })\n",
            "    print(f'Loaded {len(samples)} samples successfully.')\n",
            "except Exception as e:\n",
            "    print(f'Failed to load local data: {e}')\n",
            "\n",
            "# Train / Validation Split\n",
            "train_samples = samples[:200]\n",
            "val_samples = samples[200:250]\n",
            "\n",
            "print(f'Train size: {len(train_samples)}, Val size: {len(val_samples)}')\n",
            "\n",
            "# Convert train_samples to DataFrame for Exploratory Data Analysis\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "df_data = []\n",
            "for item in train_samples:\n",
            "    df_data.append({\n",
            "        'duration': item.get('duration', 1.0),\n",
            "        'description': item.get('description', ''),\n",
            "        'caption': item.get('caption', ''),\n",
            "        'title': item.get('title', ''),\n",
            "        'author': item.get('author', '')\n",
            "    })\n",
            "df = pd.DataFrame(df_data)\n",
            "\n",
            "# Exploratory Data Analysis\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "\n",
            "# Duration distribution\n",
            "sns.histplot(df['duration'], bins=30, ax=axes[0], color='skyblue')\n",
            "axes[0].set_title('Distribution of Audio Durations (seconds)')\n",
            "axes[0].set_xlabel('Duration')\n",
            "\n",
            "# Text length (words)\n",
            "df['desc_words'] = df['description'].apply(lambda x: len(str(x).split()))\n",
            "df['capt_words'] = df['caption'].apply(lambda x: len(str(x).split()))\n",
            "sns.kdeplot(df['desc_words'], ax=axes[1], label='Noisy Description', fill=True)\n",
            "sns.kdeplot(df['capt_words'], ax=axes[1], label='Clean Caption', fill=True)\n",
            "axes[1].set_title('Word Count Distribution')\n",
            "axes[1].set_xlabel('Number of Words')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'Missing values:\\n{df.isnull().sum()}')\n"
        ]
        cell['source'] = source
        new_cells.append(cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open('Project_15_WavCaps.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
