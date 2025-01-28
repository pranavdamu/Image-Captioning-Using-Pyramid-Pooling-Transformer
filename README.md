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
