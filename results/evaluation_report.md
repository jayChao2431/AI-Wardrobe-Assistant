# AI Wardrobe Assistant - Classification Evaluation Report

## Overall Performance

**Accuracy:** 73.47%  
**Average Confidence:** 0.738  
**Total Errors:** 98

---

## Per-Class Metrics

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Blazer** | 1.000 | 0.500 | 0.667 | 14 |
| **Blouse** | 0.737 | 1.000 | 0.848 | 14 |
| **Dress** | 0.750 | 0.643 | 0.692 | 14 |
| **Skirt** | 0.600 | 0.643 | 0.621 | 14 |
| **Tee** | 0.588 | 0.714 | 0.645 | 14 |
| **Pants** | 0.706 | 0.857 | 0.774 | 14 |
| **Shorts** | 1.000 | 0.786 | 0.880 | 14 |

---

## Sample Errors (First 10)

### 1. Shorts Classification
- **File:** `data/augmented/shorts/aug_0222.jpg`
- **True Label:** Shorts
- **Predicted:** shorts
- **Confidence:** 0.903

### 2. Blouse Classification
- **File:** `data/augmented/blouse/aug_0314.jpg`
- **True Label:** Blouse
- **Predicted:** blouse
- **Confidence:** 0.531

### 3. Dress Classification
- **File:** `datasets/dresscode_mens/train/images/s7-209396_alternate10_jpg.rf.97e464671141644841225e8ccd9bb90b.jpg`
- **True Label:** Dress
- **Predicted:** dress
- **Confidence:** 0.050

### 4. Pants Classification
- **File:** `datasets/images/50505107.jpg`
- **True Label:** Pants
- **Predicted:** pants
- **Confidence:** 0.980

### 5. Skirt Classification
- **File:** `datasets/images/46129736.jpg`
- **True Label:** Skirt
- **Predicted:** skirt
- **Confidence:** 0.873

### 6. Pants Classification
- **File:** `datasets/images/154634115.jpg`
- **True Label:** Pants
- **Predicted:** pants
- **Confidence:** 0.979

### 7. Blazer Classification
- **File:** `data/augmented/blazer/aug_0061.jpg`
- **True Label:** Blazer
- **Predicted:** blazer
- **Confidence:** 0.050

### 8. Skirt Classification
- **File:** `datasets/images/197190526.jpg`
- **True Label:** Skirt
- **Predicted:** skirt
- **Confidence:** 0.530

### 9. Pants Classification
- **File:** `datasets/images/193683028.jpg`
- **True Label:** Pants
- **Predicted:** pants
- **Confidence:** 0.950

### 10. Shorts Classification
- **File:** `datasets/images/118448257.jpg`
- **True Label:** Shorts
- **Predicted:** shorts
- **Confidence:** 0.653

---

**Note:** This evaluation report shows the model's performance across different clothing categories. The model achieves perfect precision on Blazer and Shorts categories, with Shorts showing the highest F1-score (0.880). Blouse category has perfect recall (1.000), indicating the model captures all blouse instances.
