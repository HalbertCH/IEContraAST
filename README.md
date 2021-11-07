# This is the official PyTorch implementation of our paper: "Artistic Style Transfer with Internal-external Learning and Contrastive Learning".  

This is the official PyTorch implementation of our paper: "Artistic Style Transfer with Internal-external Learning and Contrastive Learning".  

Although existing artistic style transfer methods have achieved significant improvement with deep neural networks, they still suffer from artifacts such as disharmonious colors and repetitive patterns. Motivated by this, we propose an internal-external style transfer method with two contrastive losses. Specifically, we utilize internal statistics of a single style image to determine the colors and texture patterns of the stylized image, and in the meantime, we leverage the external information of the large-scale style dataset to learn the human-aware style information, which makes the color distributions and texture patterns in the stylized image more reasonable and harmonious. In addition, we argue that existing style transfer methods only consider the content-to-stylization and style-to-stylization relations, neglecting the stylization-to-stylization relations. To address this issue, we introduce two contrastive losses, which pull the multiple stylization embeddings closer to each other when they share the same content or style, but push far away otherwise. We conduct extensive experiments, showing that our proposed method can not only produce visually more harmonious and satisfying artistic images, but also promote the stability and consistency of rendered video clips.

![image](https://github.com/HalbertCH/IEContraAST/blob/main/figures/pipeline.jpg)  
  
## Requirements  
We recommend the following configurations:  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1

