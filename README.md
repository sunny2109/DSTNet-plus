### 📖 DSTNet+
> <a href="https://colab.research.google.com/drive/19DdsNFeOYR8om8QCCi9WWzr_WkWTLHZd?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)]()
![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunny2109/DSTNet-plus) <br>

<!-- > [Jinshan Pan](https://jspan.github.io/), [Long Sun](https://github.com/sunny2109), [Boming Xu](https://github.com/xuboming8), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN), and [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao)
> [IMAG Lab](https://imag-njust.net/), Nanjing University of Science and Technology -->


---
This repo is a official implementation of "[Learning Efficient Deep Discriminative Spatial and Temporal Networks for Video Deblurring]()".

DSTNet+ is an extension of [DSTNet](https://github.com/xuboming8/DSTNet).

---
### Update
- **2025.03.24**: Visual results of all test sets can be download [here](https://huggingface.co/Meloo/DSTNetPlus/upload/main).
- **2025.03.14**: This paper is accepted by TPAMI.
- **2024.01.08**: This repo is created.

---
### Results
- **Model efficiency** (PSNR vs. Runtime vs. Params) 
<img width="770" src="figs/runtime.png"> 

- **Quantitative evaluations** <br>
&emsp;    &emsp;   &emsp; &emsp;    &emsp;Evaluation on **GoPro** dataset  &emsp;    &emsp;   &emsp;  &emsp;    &emsp;   &emsp;  &emsp; Evaluation on **DVD** dataset <br>
<img width="380" src="figs/table_gopro.png">  <img width="330" src="figs/table_dvd.png"> 

- Deblurred results on **GoPro** dataset
<img width="780" src="figs/gopro.png">

- Deblurred results on **DVD** dataset
<img width="780" src="figs/dvd.png">

- Deblurred results on **Real-world** blurry frames
<img width="800" src="figs/real_world.png">


## 📧 Contact
If you have any questions, please feel free to reach us out at cs.longsun@gmail.com

## 📎 Citation 

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝 
```bibtex
@article{RDG,
  title={Learning Efficient Deep Discriminative Spatial and Temporal Networks for Video Deblurring},
  author={Pan, Jinshan and Sun, Long and Xu Boming and Dong, Jiangxin and Tang, Jinhui},
  journal={CVPR},
  year={2025}
}
