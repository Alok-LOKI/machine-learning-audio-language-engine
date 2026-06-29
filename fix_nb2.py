import json

with open('Project_15_WavCaps.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first code cell is where imports belong!
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Ensure it has the imports!
        source_str = ''.join(cell['source'])
        if 'AutoTokenizer' not in source_str:
            imports = [
                "# Install dependencies if necessary\n",
                "# !pip install torch torchvision torchaudio transformers datasets librosa matplotlib seaborn scikit-learn\n",
                "\n",
                "import os\n",
                "import random\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "import torchaudio\n",
                "import librosa\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from transformers import AutoTokenizer, AutoModel\n",
                "from datasets import load_dataset\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.metrics import pairwise_distances\n",
                "from tqdm import tqdm\n",
                "\n",
                "# Set random seeds for reproducibility\n",
                "def set_seed(seed=42):\n",
                "    random.seed(seed)\n",
                "    np.random.seed(seed)\n",
                "    torch.manual_seed(seed)\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.manual_seed_all(seed)\n",
                "\n",
                "set_seed(42)\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")\n\n"
            ]
            cell['source'] = imports + cell['source']
        break

with open('Project_15_WavCaps.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
