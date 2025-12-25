# Efficient and Adaptive Gesture Classification via Grassmann and Riemannian Manifolds

This repository contains the implementation framework for the paper:

**Efficient and Adaptive Gesture Classification via Grassmann and Riemannian Manifolds**

The work addresses robust gesture recognition from surface electromyogram (sEMG) signals by leveraging Riemannian geometry and Grassmann manifolds to handle inter-subject variability and non-Euclidean data characteristics.

---


---

## Keywords

- Gesture Recognition  
- Grassmann Manifold  
- Riemannian Manifold  
- Inter-subject Variation  
- Surface Electromyogram (sEMG)

---

## Methodology Overview

1. Signal Acquisition and Preprocessing  
   - sEMG normalization  
   - Noise and artifact mitigation  
   - Sliding-window segmentation  

2. Feature Construction  
   - Covariance matrix computation  
   - SPD representation of sEMG signals  

3. Riemannian Manifold Learning  
   - Mapping SPD matrices to Riemannian space  
   - Geometry-aware distance computation  

4. Grassmann Manifold Projection  
   - Subspace modeling to reduce inter-subject variability  
   - Compact and discriminative representations  

5. Classification  
   - Manifold-aware classifiers  
   - Cross-subject evaluation strategy  

---


---

## Requirements

- Python ≥ 3.8  
- NumPy  
- SciPy  
- scikit-learn  
- PyRiemann  
- Matplotlib  

---

## Usage

1. Preprocess sEMG Signals  
python utils/preprocessing.py  

2. Extract Covariance and Grassmann Features  
python features/covariance/extract_cov.py  
python features/grassmann/compute_grassmann.py  

3. Run Cross-Subject Classification  
python experiments/cross_subject.py  

---

## Evaluation Protocol

- Cross-subject validation  
- Accuracy, Precision, Recall, F1-score  

---

## Citation

TBD

---

## Contact

Muhammad Haris  
Email: m.haris@kkkuk.edu.pk

---

## Notes

Manifold-based learning improves robustness but does not compensate for poor signal quality.
