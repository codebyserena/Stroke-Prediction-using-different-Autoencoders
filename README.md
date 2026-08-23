# Stroke Prediction Using Autoencoders

Predicts stroke risk from patient demographic and clinical data using three types of autoencoders (Standard, Denoising, and Variational) to learn compressed feature representations, then feeds those representations into a classifier. The comparison is the point: each autoencoder learns a different kind of representation, and the project tests which one actually helps prediction.

## Highlights

- Compares three autoencoder architectures (AE, DAE, VAE) as feature extractors on the same clinical dataset
- Denoising Autoencoder tests robustness to noisy or incomplete patient data
- Variational Autoencoder models the latent space as a distribution rather than a point estimate, capturing uncertainty
- Combines features from all three into an ensemble, rather than picking a single "best" autoencoder upfront

## Autoencoders Used

1. **Standard Autoencoder (AE)** — compresses high-dimensional input into a lower-dimensional latent space and reconstructs it. Used for dimensionality reduction and feature extraction.
2. **Denoising Autoencoder (DAE)** — reconstructs the original data from a noisy version of the input. Used to improve robustness of the learned features.
3. **Variational Autoencoder (VAE)** — models the latent space as a distribution rather than a fixed point, capturing variability and uncertainty in the data.

## Dataset

- **Source**: the Stroke Prediction dataset, containing patient age, gender, hypertension status, heart disease, smoking habits, and related fields
- **Target variable**: whether the patient had a stroke (`1`) or not (`0`)

## Workflow

1. **Data preprocessing**: missing value handling, one-hot encoding for categorical variables, normalising numerical features
2. **Autoencoder training**: AE, DAE, and VAE each trained independently to learn compressed feature representations
3. **Feature extraction**: encoded features pulled from each trained autoencoder
4. **Ensemble**: features from all three autoencoders combined and passed to a Random Forest classifier
5. **Evaluation**: accuracy, precision, recall, and F1-score

## Installation

Requires Python 3.x, TensorFlow 2.0+, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

```
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## How to Run

Open `Stroke Prediction using AutoEncoder.ipynb` in Jupyter and run the cells in order.

```
jupyter notebook "Stroke Prediction using AutoEncoder.ipynb"
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Author

Serena Mendanha — [portfolio](https://serena-mendanha-portfolio.netlify.app) · [LinkedIn](https://www.linkedin.com/in/serena-mendanha/)
