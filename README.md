# Artistic Style Transfer with Internal-external Learning and Contrastive Learning
This is the official PyTorch implementation of our paper: "Artistic Style Transfer with Internal-external Learning and Contrastive Learning".  

Although existing artistic style transfer methods have achieved significant improvement with deep neural networks, they still suffer from artifacts such as disharmonious colors and repetitive patterns. Motivated by this, we propose an internal-external style transfer method with two contrastive losses. Specifically, we utilize internal statistics of a single style image to determine the colors and texture patterns of the stylized image, and in the meantime, we leverage the external information of the large-scale style dataset to learn the human-aware style information, which makes the color distributions and texture patterns in the stylized image more reasonable and harmonious. In addition, we argue that existing style transfer methods only consider the content-to-stylization and style-to-stylization relations, neglecting the stylization-to-stylization relations. To address this issue, we introduce two contrastive losses, which pull the multiple stylization embeddings closer to each other when they share the same content or style, but push far away otherwise. We conduct extensive experiments, showing that our proposed method can not only produce visually more harmonious and satisfying artistic images, but also promote the stability and consistency of rendered video clips.

<div align=center>
<img src="https://github.com/HalbertCH/IEContraAST/blob/main/figures/pipeline.jpg" width="800" alt="Pipeline"/><br/>
</div>
  
## Requirements  
We recommend the following configurations:  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1

## Model Training  
- Download the content dataset: [MS-COCO](https://cocodataset.org/#download).
- Download the style dataset: [WikiArt](https://www.kaggle.com/c/painter-by-numbers).
- Download the pre-trained [VGG-19](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing) model.
- Set your available GPU ID in Line94 of the file "train.py".
- Run the following command:
```
python train.py --content_dir /data/train2014 --style_dir /data/WikiArt/train
```

## Model Testing
- Put your trained model to *./model/* folder.
- Put some sample photographs to *./input/content/* folder.
- Put some artistic style images to *./input/style/* folder.
- Run the following command:
```
python Eval.py --content input/content/1.jpg --style input/style/1.jpg
```
![image](https://github.com/HalbertCH/IEContraAST/blob/main/figures/comparison.jpg) 

![image](https://github.com/HalbertCH/IEContraAST/blob/main/figures/table.png) 
  
We provide the pre-trained model in [link](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing).  

