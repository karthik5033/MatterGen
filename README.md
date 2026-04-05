<p align="center">
  <img src="https://img.shields.io/badge/MatterGen-X-7c3aed?style=for-the-badge&logo=atom&logoColor=white" alt="MatterGen X" />
  <img src="https://img.shields.io/badge/Status-Research_Prototype-10b981?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/License-MIT-3b82f6?style=for-the-badge" alt="License" />
</p>

<h1 align="center">⚛️ MatterGen X</h1>
<h3 align="center">AI-Powered Material Discovery & Visualization Platform</h3>

<p align="center">
  <em>Accelerate the discovery of novel, stable crystal structures with Generative AI.<br/>
  Combining inverse design, high-fidelity property prediction, and interactive 3D visualization<br/>
  into a single research-grade platform.</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-model--training-pipeline">Model & Training</a> •
  <a href="#-screenshots">Screenshots</a>
</p>

---

## 🌟 Overview

**MatterGen X** is a full-stack AI material discovery platform that enables researchers to generate, analyze, and compare novel crystal structures through natural language prompts and configurable optimization parameters. The platform bridges the gap between cutting-edge machine learning models and practical materials science research by providing:

- **Natural Language → Crystal Generation**: Describe a material in plain English and receive AI-generated crystal candidates with predicted properties.
- **Interactive 3D Crystal Viewer**: Explore generated structures with a fully interactive, rotatable 3D molecular visualization rendered directly in the browser.
- **Comprehensive AI-Powered Reports**: Gemini-powered technical analysis including executive summaries, scientific deep dives, synthesis guides, risk profiles, and economic outlooks.
- **AI Discovery Assistant ("Aether")**: A conversational chatbot that refines your material specifications, suggesting optimal prompts and slider configurations.
- **Real-Time Property Prediction**: CGCNN (Crystal Graph Convolutional Neural Network) backed property predictions for band gap, formation energy, density, and more.

---

## ✨ Features

### 🔬 Core Discovery Engine
| Feature | Description |
|---|---|
| **Natural Language Generation** | Input a text prompt (e.g., *"High-capacity lithium cobalt oxide cathode"*) and generate crystal candidates from a 230MB+ materials dataset |
| **Optimization Sliders** | Fine-tune priorities for Density, Stability, Band Gap, Shear Modulus, Thermal Conductivity, and Refractive Index |
| **Industry Presets** | Pre-configured prompts for Energy & Sustainability, Electronics & Computing, Structural & Aerospace, Optical & Photonic, and more |
| **Multi-Candidate Ranking** | Generates and ranks multiple candidates using a weighted scoring function based on formation energy, energy above hull, and band gap |

### 🧊 Interactive 3D Crystal Viewer
- Real-time 3D molecular rendering with atom-level detail
- Color-coded atoms by element type
- Rotate, zoom, and pan controls
- CIF (Crystallographic Information File) structure generation via **pymatgen**

### 📊 Comprehensive Technical Reports (Gemini AI)
Each generated material receives a detailed, AI-generated report containing:
- **Executive Summary** — Strategic importance and market disruption potential
- **Scientific Deep Dive** — Band structure analysis, crystallographic stability, bonding mechanisms, synthesis pathways
- **Industrial Applications** — 3 detailed use-cases with performance metrics
- **Ratings Dashboard** — Commercial Viability, Sustainability Index, Manufacturing Complexity (scored 0–100)
- **Synthesis Guide** — Method, precursors, equipment, step-by-step procedure, and key challenges
- **Risk Profile** — Flammability, toxicity, handling precautions, disposal protocols
- **Economic Outlook** — Estimated cost, scalability, supply chain risks, target market sectors
- **Future Tech Lore** — Creative sci-fi flavor text for each material

### 🤖 Aether — AI Discovery Assistant
A floating chatbot powered by Gemini that understands natural language research intent and translates it to:
- Refined scientific prompts optimized for the generation engine
- Suggested optimization slider weights
- One-click "Apply Settings" to instantly load configurations

### 🗺️ Latent Space Embedding Map
- UMAP-projected 2D visualization of 64-dimensional material embeddings
- Color-coded by band gap classification (Metal / Semiconductor / Insulator)
- Interactive hover tooltips and click-to-select functionality
- Scientific scatter plot with axes, grid, and legends

### 🧪 Periodic Table Heatmap
- Full interactive periodic table showing element frequency across the dataset
- Gradient-scaled coloring based on occurrence count
- Hover-to-inspect element details

### 📈 Material Comparison
- Side-by-side comparison of multiple generated candidates
- Radar charts for multi-property visualization
- Benchmark bar charts against baseline materials

---

## 🏗️ Architecture

```
matterGen/
├── mattergen-x/
│   ├── frontend/               # Next.js 16 (App Router)
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── page.tsx              # Landing page (Hero + Features)
│   │   │   │   ├── generate/page.tsx     # Main generation dashboard
│   │   │   │   ├── material/[id]/page.tsx # Detailed material report view
│   │   │   │   ├── compare/page.tsx      # Side-by-side comparison
│   │   │   │   └── features/[slug]/      # Feature detail pages
│   │   │   ├── components/
│   │   │   │   ├── CrystalViewer.tsx           # 3D crystal structure renderer
│   │   │   │   ├── InputPanel.tsx              # Prompt + slider controls
│   │   │   │   ├── MaterialCard.tsx            # Candidate result cards
│   │   │   │   ├── ReportModal.tsx             # Technical report overlay
│   │   │   │   ├── WeightSliders.tsx           # Optimization weight controls
│   │   │   │   ├── MaterialRadarChart.tsx      # Radar property chart
│   │   │   │   ├── MaterialBenchmarkChart.tsx  # Benchmark bar chart
│   │   │   │   ├── charts/
│   │   │   │   │   └── ComparisonCharts.tsx    # Comparison page charts
│   │   │   │   ├── landing/
│   │   │   │   │   ├── header.tsx              # Navigation header
│   │   │   │   │   ├── hero-section.tsx        # Hero with CTA
│   │   │   │   │   ├── hero-visual.tsx         # Animated crystal visual
│   │   │   │   │   ├── features-4.tsx          # Feature showcase grid
│   │   │   │   │   ├── content-7.tsx           # Content section
│   │   │   │   │   ├── animated-background.tsx # Particle animation
│   │   │   │   │   └── logo.tsx                # Brand logo
│   │   │   │   ├── visualization/
│   │   │   │   │   ├── AetherAssistant.tsx     # AI chatbot widget
│   │   │   │   │   ├── EmbeddingMap.tsx        # UMAP latent space map
│   │   │   │   │   └── PeriodicTableHeatmap.tsx# Interactive periodic table
│   │   │   │   └── ui/                         # Reusable UI primitives
│   │   │   ├── lib/
│   │   │   │   └── api.ts                      # API client service
│   │   │   ├── types/                          # TypeScript type definitions
│   │   │   ├── hooks/                          # Custom React hooks
│   │   │   └── utils/                          # Utility functions
│   │   └── package.json
│   │
│   ├── backend/                # FastAPI (Python)
│   │   ├── main.py                             # App entry point (Uvicorn)
│   │   ├── app/
│   │   │   ├── api/endpoints/
│   │   │   │   ├── generation.py               # /api/generate, /api/chat/refine
│   │   │   │   └── analysis.py                 # /api/map, /api/stats/elements, /api/explain
│   │   │   ├── services/
│   │   │   │   ├── material_service.py         # Core generation & ranking engine
│   │   │   │   ├── gemini_service.py           # Gemini AI integration (reports + chat)
│   │   │   │   └── map_service.py              # UMAP embedding projection
│   │   │   ├── schemas/                        # Pydantic request/response models
│   │   │   └── static/                         # Served static files
│   │   └── requirements.txt
│   │
│   ├── training/               # ML Training Pipeline
│   │   ├── models/
│   │   │   ├── cgcnn.py                        # Crystal Graph Convolutional Neural Network
│   │   │   ├── layers.py                       # Custom GNN conv layers (CGCNNConv)
│   │   │   ├── deep_regressor.py               # Deep regression head
│   │   │   └── property_predictor.py           # Prediction wrapper
│   │   ├── datasets/
│   │   │   ├── loader.py                       # Dataset loading & batching
│   │   │   ├── graph_builder.py                # Crystal → Graph construction
│   │   │   ├── graph_dataset.py                # PyTorch Dataset for crystal graphs
│   │   │   └── featurizer.py                   # Atom & bond featurization
│   │   ├── scripts/
│   │   │   ├── train_cgcnn.py                  # CGCNN training script
│   │   │   ├── train_optimized_cgcnn.py        # Optimized training with LR scheduling
│   │   │   ├── train_ensemble.py               # Ensemble model training
│   │   │   ├── evaluate_cgcnn.py               # Model evaluation & metrics
│   │   │   ├── generate_cgcnn_embeddings.py    # Generate 64-dim crystal embeddings
│   │   │   ├── generate_data.py                # Synthetic data generation
│   │   │   ├── generate_metrics_report.py      # Performance metrics & plots
│   │   │   └── ...                             # Additional utility scripts
│   │   ├── visualization/
│   │   │   ├── plotting.py                     # Training curve plots
│   │   │   └── umap_projection.py              # UMAP projection for embeddings
│   │   ├── inference.py                        # Production inference wrapper
│   │   ├── explainability.py                   # Model explainability (feature attribution)
│   │   └── requirements.txt
│   │
│   ├── data/                   # Material datasets
│   ├── models/                 # Saved model checkpoints (.pt)
│   ├── docs/                   # Documentation
│   ├── shared/                 # Shared utilities
│   └── scripts/                # Utility scripts
│
├── material dataset/           # Raw data & environment keys
├── .gitignore
└── README.md                   # ← You are here
```

---

## 🛠️ Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| [Next.js](https://nextjs.org/) | 16.1.1 | React framework with App Router |
| [React](https://react.dev/) | 19.2.3 | UI library |
| [TypeScript](https://www.typescriptlang.org/) | 5.x | Type safety |
| [Tailwind CSS](https://tailwindcss.com/) | 4.x | Utility-first styling |
| [Framer Motion](https://www.framer.com/motion/) | 12.x | Animations & transitions |
| [Recharts](https://recharts.org/) | 3.6.0 | Data visualization (Radar, Bar, Line charts) |
| [Lucide React](https://lucide.dev/) | 0.562.0 | Icon system |
| [Radix UI](https://www.radix-ui.com/) | Latest | Accessible UI primitives |

### Backend
| Technology | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | High-performance async API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [Pydantic](https://docs.pydantic.dev/) | Request/response validation |
| [Google Generative AI (Gemini)](https://ai.google.dev/) | LLM-powered reports & chat |
| [pymatgen](https://pymatgen.org/) | Crystal structure generation & CIF export |

### Training / ML
| Technology | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [pymatgen](https://pymatgen.org/) | Crystal featurization & structure processing |
| [scikit-learn](https://scikit-learn.org/) | UMAP, metrics, preprocessing |
| [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) | Training visualization |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **npm** 9+
- **CUDA** (optional, for GPU-accelerated inference)

### 1. Clone the Repository

```bash
git clone https://github.com/karthik5033/MatterGen.git
cd MatterGen
```

### 2. Backend Setup

```bash
cd mattergen-x/backend

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server (runs on port 8002)
python main.py
```

The API will be available at `http://localhost:8002`.  
Swagger docs: `http://localhost:8002/docs`

### 3. Frontend Setup

```bash
cd mattergen-x/frontend

# Install dependencies
npm install

# Start development server (runs on port 3001)
npm run dev
```

The app will be available at `http://localhost:3001`.

### 4. Training Pipeline (Optional)

```bash
cd mattergen-x/training

# Install ML dependencies
pip install -r requirements.txt

# Generate synthetic training data
python scripts/generate_data.py

# Train the CGCNN model
python scripts/train_cgcnn.py

# Generate material embeddings for the latent space map
python scripts/generate_cgcnn_embeddings.py
```

### 5. Gemini API Key (Optional — for AI Reports & Chat)

Create a `.env.local` file in the `material dataset/` directory:

```env
GEMINI_API_KEY=your_google_generative_ai_key_here
```

> Without a Gemini API key, the platform falls back to a high-quality local simulation engine that generates plausible scientific reports and chat responses.

---

## 📡 API Reference

All endpoints are prefixed with `/api`.

### Generation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | Generate material candidates from a text prompt and weight configuration |
| `POST` | `/api/chat/refine` | Chat with Aether assistant to refine search parameters |

#### `POST /api/generate`

**Request Body:**
```json
{
  "prompt": "High-capacity lithium cobalt oxide cathode material",
  "weights": {
    "density": 0.8,
    "stability": 0.9,
    "band_gap": 0.3,
    "shear_modulus": 0.5,
    "thermal_conductivity": 0.5,
    "refractive_index": 0.3
  },
  "n_candidates": 4
}
```

**Response:** Array of `MaterialCandidate` objects with formula, predicted properties, CIF structure, embedding, and score.

### Analysis

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/map` | Retrieve UMAP-projected 2D embedding map of all materials |
| `GET` | `/api/stats/elements` | Get element frequency statistics for the periodic table heatmap |
| `POST` | `/api/explain` | Get CGCNN model explainability for a specific structure |

---

## 🧠 Model & Training Pipeline

### CGCNN — Crystal Graph Convolutional Neural Network

The core ML model is a **CGCNN** implementation that learns material property predictions directly from crystal graph representations.

**Architecture:**
```
Raw Crystal Structure (CIF/pymatgen)
    ↓
Graph Construction (atoms=nodes, bonds=edges)
    ↓
Node Featurization (Z, Group, Period, Electronegativity)
    ↓
Edge Featurization (Gaussian Distance Expansion, 41-dim)
    ↓
Linear Embedding (4 → 64-dim)
    ↓
3× CGCNNConv (Message Passing + Gate Mechanism)
    ↓
Global Mean Pooling (per-crystal vector)
    ↓
FC → BatchNorm → Softplus → FC
    ↓
Multi-Target Regression (Band Gap, Formation Energy, Density)
```

**Key Specs:**
- Node input dimension: **4** (atomic number, group, period, electronegativity)
- Edge input dimension: **41** (Gaussian distance expansion)
- Hidden dimension: **64**
- Convolution layers: **3**
- Pooling: Global Mean
- Output targets: **3** (band gap, formation energy, density)
- Weight initialization: Xavier Normal

### Training Scripts

| Script | Description |
|---|---|
| `train_cgcnn.py` | Standard training loop with validation |
| `train_optimized_cgcnn.py` | Advanced training with LR scheduling, early stopping, gradient clipping |
| `train_ensemble.py` | Ensemble model training for uncertainty quantification |
| `evaluate_cgcnn.py` | Generates comprehensive evaluation metrics and plots |
| `generate_cgcnn_embeddings.py` | Extracts 64-dim crystal-level embeddings for UMAP visualization |
| `generate_data.py` | Creates synthetic training data from Materials Project |

### Inference & Explainability

- **`inference.py`**: Production-ready `CGCNNPredictor` class for single-structure predictions
- **`explainability.py`**: `CGCNNExplainer` for feature attribution and model interpretability

---

## 📸 Screenshots

### Landing Page
> A sleek, modern landing page with animated crystal visuals, feature highlights, and clear CTAs.

### Generation Dashboard  
> The primary interface for inputting natural language prompts, adjusting optimization sliders, browsing industry presets, and viewing AI-generated crystal candidates with interactive 3D structures.

### Comprehensive Technical Report
> Detailed per-material analysis featuring executive summaries, physical property breakdowns, commercial viability scores, scientific deep dives, synthesis guides, and risk profiles.

### Latent Space Embedding Map
> UMAP-projected scatter plot of material embeddings color-coded by band gap classification (Metal / Semiconductor / Insulator).

---

## 🧩 Key Design Decisions

1. **Lazy Loading Architecture**: The 230MB+ material dataset is loaded lazily on first request using a singleton pattern, keeping cold starts fast.

2. **Graceful AI Fallback**: When Gemini API keys are unavailable or rate-limited, the platform seamlessly falls back to a high-quality rule-based simulation engine — ensuring the platform is always functional.

3. **Deterministic Generation**: Crystal structures and candidate selection use formula-seeded random number generators for reproducible results across sessions.

4. **pymatgen Integration**: Real crystallographic structures are generated using `pymatgen.core.Structure` with proper lattice parameters, fractional coordinates, and CIF export — not random data.

5. **Multi-Key Rotation**: The Gemini service supports multiple API keys with automatic rotation and fallback, maximizing availability under rate limits.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Materials Project](https://materialsproject.org/)** — Source dataset for crystal structures and properties
- **[CGCNN Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)** — Xie & Grossman (2018), *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*
- **[Google Gemini](https://ai.google.dev/)** — Powering technical report generation and the Aether discovery assistant
- **[pymatgen](https://pymatgen.org/)** — Crystal structure analysis and CIF generation

---

<p align="center">
  <strong>Built with ⚛️ by Karthik K P</strong>
</p>
