# Image-Captioning-Using-Pyramid-Pooling-Transformer
 Image captioning bridges computer vision and NLP to generate image descriptions. Using Pyramid Pooling Transformer on the Flickr8k dataset, we enhanced multi-scale feature representation by integrating Pyramid Pooling Modules with a Transformer encoder. Achieving a BLEU score of 34.89, our model showed competitive performance and efficiency.
# Image Captioning Using Pyramid Pooling Transformer

### Authors: Mehul Kanotra, Pranav S Damu, Mohammad Sufyaan Saeed

---

## Overview

Image captioning is a challenging task that involves generating descriptive textual content for images by bridging the gap between computer vision and natural language processing. This project presents a novel image captioning model utilizing the **Pyramid Pooling Transformer (P2T)** architecture, which effectively captures multi-scale contextual information and generates coherent, accurate captions for images. The model was implemented on the **Flickr8k dataset** and evaluated for performance in terms of accuracy and computational efficiency.

---

## Key Features
- **Pyramid Pooling Module (PPM):** Enhances feature representation by aggregating global and local visual features through adaptive pooling at multiple spatial scales.
- **Transformer-based Encoder-Decoder Architecture:** Leverages self-attention mechanisms for capturing global dependencies and generating coherent captions.
- **Pre-trained ResNet-50 Backbone:** Used for initial feature extraction, providing rich visual representations.
- **Evaluation Metrics:** Achieves a competitive BLEU score of **34.89** on the Flickr8k dataset.

---

## Project Goals
- Integrate multi-scale feature extraction with sequence modeling for enhanced image captioning.
- Demonstrate the utility of pyramid pooling in capturing both local and global contexts.
- Compare the proposed model with existing approaches in terms of accuracy and computational efficiency.

---

## Dataset
- **Flickr8k Dataset:**
  - Contains 8,000 images, each paired with five human-annotated captions.
  - Offers a diverse set of annotations suitable for evaluating captioning models.
  - Split into **80% training** and **20% validation** for effective performance assessment.

---

## Model Architecture

### Encoder: Pyramid Pooling Transformer
1. **Feature Extraction:**
   - Utilizes **ResNet-50** to extract initial visual features.
   - Captures rich visual representations using residual learning.
2. **Pyramid Pooling Module:**
   - Performs adaptive average pooling at scales of 1, 2, 3, and 6.
   - Aggregates global and local contextual features.
3. **Feature Transformation:**
   - Outputs from the PPM are concatenated, batch normalized, and activated using ReLU before being passed to the Transformer decoder.

### Decoder: Transformer Decoder
1. **Multi-Head Attention Mechanisms:** Focuses on relevant features extracted by the encoder.
2. **Embedding Layers:** Encodes input captions and positional information.
3. **Output Layer:** Produces logits for vocabulary words, with a softmax function for word prediction.

---

## Training Details
- **Vocabulary Construction:** Built from captions, filtering words with frequency <5.
- **Loss Function:** Cross-entropy loss, ignoring padding tokens.
- **Optimizer:** Adam optimizer with a learning rate of **0.0001**.
- **Hardware Efficiency:** Optimized for training on local machines with **NVIDIA RTX GPUs**.

---

## Results
- **BLEU Score:** 34.89 on the Flickr8k dataset.
- Demonstrated competitive performance in scenarios with limited data.
- Achieved computational efficiency, making it suitable for resource-constrained environments.

---

## Applications
- **Assistive Technologies:** Helps visually impaired individuals by generating textual descriptions of visual content.
- **Content-Based Image Retrieval Systems:** Enhances search capabilities by linking text to visual data.
- **Human-Computer Interaction:** Improves interfaces by integrating vision and language understanding.

---

## Future Work
1. **Scaling to Larger Datasets:** Evaluate on datasets like MS COCO or Flickr30k to improve generalization.
2. **Advanced Decoding Strategies:** Implement beam search for more coherent captions.
3. **Object Detection Integration:** Focus attention on salient image regions for improved relevance.
4. **Explainability:** Visualize attention weights to understand model behavior.
5. **Multilingual Captioning:** Extend the model to generate captions in multiple languages using transfer learning.

---

## References
1. Vaswani et al. (2017). "Attention Is All You Need."
2. Wu et al. (2022). "P2T: Pyramid Pooling Transformer for Scene Understanding."
3. He et al. (2016). "Deep Residual Learning for Image Recognition."
4. Biradar et al. (2023). "Leveraging Deep Learning Model for Image Caption Generation."

---

Thank you for exploring our project!
![Figure 2024-11-14 132406 (2)](https://github.com/user-attachments/assets/2a28f614-1704-4436-89d4-c7092266eb5d)
![Figure 2024-11-14 132406 (1)](https://github.com/user-attachments/assets/380ce06f-bdd8-41fc-a93e-0c0118a26f46)
![Figure 2024-11-14 132406 (0)](https://github.com/user-attachments/assets/f49e3f99-2299-4988-9ae3-3d902e14c426)
![Figure 2024-11-14 132406 (9)](https://github.com/user-attachments/assets/14eaca75-fd0f-4fd6-9ee9-49840d29f1c4)
![Figure 2024-11-14 132406 (8)](https://github.com/user-attachments/assets/12789b60-43a5-4e9f-8983-08e274edc451)
![Figure 2024-11-14 132406 (7)](https://github.com/user-attachments/assets/46955da6-5a6a-4cb7-a7c6-8e846b4b8448)
![Figure 2024-11-14 132406 (6)](https://github.com/user-attachments/assets/18d681e2-d9ad-4f13-b0e9-8396568ba0d9)
![Figure 2024-11-14 132406 (5)](https://github.com/user-attachments/assets/7bf439b7-4bd5-4b10-b730-2fe33f8cd270)
![Figure 2024-11-14 132406 (4)](https://github.com/user-attachments/assets/52b66e1d-57b5-4cdf-9c37-8d487c8f4bcf)
![Figure 2024-11-14 132406 (3)](https://github.com/user-attachments/assets/86a5f2c9-81cc-4397-9e86-ddd2165aab1e)
