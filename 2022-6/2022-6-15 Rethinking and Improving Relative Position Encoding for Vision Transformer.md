# Rethinking and Improving Relative Position Encoding for Vision Transformer
## 摘要
相对位置编码(RPE)在NLP中证明了功效，然而在CV中没有得到很好的研究，存在争议。作者分析了现有的相对位置编码方法，并提出新的相对位置编码方法iRPE，该方法考虑了self-attention中的方向性、相对距离和key和query的交互。iRPE很简单，很容易插入transformer中。不引入额外超参数的情况下，加入该方法就可以实现ImageNet和COCO分别1.5%(top-1 Acc)和1.3%(mAP)的稳定提升。

## 介绍
近年来，transformer的性能和捕获远程依赖的能力引起广泛关注，其核心在于self-attention，然而self-attention是缺陷是它不能捕捉token的顺序，因此显式的位置信息表示方法对transformer来说非常重要。

位置信息表示的两种编码方法：绝对的和相对的。绝对方法对token的绝对位置进行编码，即每个位置都有一个单独的编码向量；相对方法通过token之间的相对距离学习token的成对关系，相对方法通常通过一个查找表计算，表中包含可学习的参数。相对位置编码在NLP领域被证实有效，然而在CV中不明且有争议。例如，Dosovitskiy et al.的研究发现相对位置编码与绝对位置编码相比没有任何增益，而Srinivas et al.发现相对位置编码有明显的增益。

NLP中的相对位置编码针对的输入数据是一维序列，而CV中的输入通常是二维图像或视频，其像素是高度结构化的。从一维扩展到二维是否有效？

## 方法
### 位置编码
#### 绝对位置编码
由于transformer不包含递归和卷积，为了让模型学习到序列的顺序，就需要注入位置信息。初始的self-attention将绝对位置编码$p=(p_1,\dots,p_n)$加入token
$$
x_i=x_i+p_i,
$$
其中$p_i,x_i\in\mathbb{R}^{d_x}$。$p$有多种选择，例如用不同频率的sin和cos固定编码或用可训练参数。

#### 相对位置编码
考虑token间的成对关系，即相对位置。将输入元素$x_i,x_j$之间的相对位置编码成向量$p_{ij}^V,p_{ij}^Q,p_{ij}^{K}\in\mathbb{R}^{d_z},d_z=d_x$，之后通过self-attention即
$$
z_i=\sum_{j=1}^n\alpha_{ij}(x_jW^V+p_{ij}^V),
$$
$$
e_{ij}=\frac{(x_iW^Q+p_{ij}^Q)(x_jW^K+p_{ij}^K)^T}{\sqrt{d_z}}.
$$

### Shaw's RPE
Shaw et al.提出将输入token建模为一个有向的全连接图，两个任意位置$i,j$间的边用一个可学习的向量$p_{ij}\in\mathbb{R}^{d_z}$表示。此时，作者认为精确的相对位置信息在超过一定距离后失效，因为引入clip减少参数的数量
$$
z_i=\sum_{j=1}^{n}\alpha_{ij}(x_jW^V+p_{clip(i-j,k)}^V),
$$
$$
e_{ij}=\frac{(x_iW^Q)(x_jW_K+p_{clip(i-j,k)}^K)^T}{\sqrt{d_z}},
$$
$$
clip(x,k)=\max(-k, \min(k, x))
$$
其中$p^V=(p_{-k}^V,\dots,p_k^V),p^K=(p_{-k}^K,\dots,p_k^K)$是可学习的权重，$k$表示最大相对距离。

### RPE in Transformer-XL
Dai et al.为query引入额外的bias，并用sin表示相对位置编码
$$

e_{ij}=\frac{(x_iW^Q+u)(x_jW^K)^T+(x_iW^Q+v)(s_{i-j}W^R)^T}{\sqrt{d_z}},
$$
正弦编码向量$s$提供了相对位置的优先级，$W^R\in\mathbb{R}^{d_z\times d_z}$是一个可训练矩阵，负责将$s_{i-j}$投影到对应位置上的key向量。

### Huang's RPE
Huang et al.提出了一种同时考虑query、key和相对位置交互作用的新方法。
$$
e_{ij}=\frac{(x_iW^Q+p_{ij})(x_jW^K+p_{ij})^T-p_{ij}p_{ij}^T}{\sqrt{d_z}},
$$
其中$p_{ij}\in\mathbb{R}^{d_z}$是query和key共享的相对位置编码。

### RPE in SASA
上述方法是针对一维向量的NLP任务设计的。Radmachandran et al.提出了一种二维图像的编码方法，它将二维分为水平方向和垂直方向，每个方法通过一维编码建模
$$
e_{ij}=\frac{(x_iW^Q)(x_jW^k+concat(p_{\delta\tilde{x}}^K,p_{\delta\tilde{y}}^K))^T}{\sqrt{d_z}},
$$
其中$\delta\tilde{x}=\tilde{x}_i-\tilde{x}_j,\delta\tilde{y}=\tilde{y}_i-\tilde{y}_j$分别表示x轴和y轴上相对位置的偏移量，$p_{\delta\tilde{x}}^K,p_{\delta\tilde{y}}^K$是长为$\frac12d_z$的可学习向量，两者拼接形成长为$d_z$的相对编码。

### RPE in Axial-Deeplab
Wang et al.介绍了一种在self-attention中加入qkv的位置偏置的方法，位置灵敏度应用沿高度轴和宽度轴，当相对距离大于阈值时设为0。我们发现远程相对位置信息时有用的，如果加入分段函数，就可以进一步改进，实现更有效的长距离依赖。

### 提出的相对位置编码方法
作者设计了图像RPE(iRPE)方法来分析以前的工作。
1. 为了研究编码是否能独立于输入的token，引入两个模型：偏移和上下文
2. 提出分段函数来映射相对位置的编码，而不是传统的clip

#### 偏移和上下文模型
引入偏移模型和上下文模型来研究编码是否可以独立于输入，具体来说如下式
$$
e_{ij}=\frac{(x_iW^Q)(x_jW^K)^T+b_{ij}}{\sqrt{d_z}},
$$
对于偏置模式，$b_{ij}=r_{ij}$，其中$r_{ij}$是一个可学习的标量，表示$i,j$之间的相对位置权重。
对于上下文模式，$b_{ij}=(x_iW^Q)r_{ij}^T$，其中$r_{ij}\in\mathbb{R}^{d_z}$是一个可学习的向量，与query相互作用。上下文模式有多种变体，例如
$$
b_{ij}=(x_iW^Q)(r_{ij}^K)^T+(x_jW^K)(r_{ij}^Q)^T,
$$
其中$r_{ij}^K,r_{ij}^Q$是可学习向量。上下文还可以嵌入value，如下式
$$
z_i=\sum_{j=1}^n\alpha_{ij}(x_jW_V+r_{ij}^V),
$$
其中$r_{ij}^V\in\mathbb{R}^{d_z}$。
![[Pasted image 20220615223440.png#center|偏置模型和上下文模型]]

### 分段索引函数
引入一个多对一函数，将相对距离的集合对应一个整数，然后通过整数对$r_{ij}$进行索引，并在不同的关系位置之间共享。这种索引可以大大降低长序列的计算开销和参数。

![[Pasted image 20220615231052.png#center|分段函数和截断函数的比较]]
clip函数将相对距离大于阈值的位置被分配到相同的编码，这样不可避免会漏掉远距离相对位置的上下文信息。引入分段函数$g(x)$来索引到相应编码的相对距离，该函数基于近邻比远邻更重要的假设
$$
g(x)=\left\{\begin{align}
& |x|,&|x|\le\alpha,\\
& sign(x)\times\min(\beta, [\alpha+\frac{\ln(|x|/\alpha)}{\ln(\gamma/\alpha)}(\beta-\alpha)]),&|x|>\alpha.\\
\end{align}\right.
$$
其中$[\cdot]$表示取整，$sign(x)$对正数返回1，负数返回-1，其余返回0。$\alpha$决定分段点，$\beta$控制输出范围，$\gamma$调整曲率。

### 相对位置计算
提出两种无向映射方法：基于欧式距离的方法和量化的方法，提出两种有向映射方法：交叉方法和乘积方法。
#### 欧式距离
$$
r_{ij}=p_I(i,j),\ I(i,j)=g(\sqrt{(\tilde{x_i}-\tilde{x_j})^2+(\tilde{y_i}-\tilde{y_j})^2}),
$$
其中$p_I(i,j)$是偏置模式下可学习的标量或上下文模式下的向量。

#### 量化方法
上述方法使具有不同相对距离的两个近邻被映射到同一索引，例如相对位置(1,0)和(1,1)都被映射到索引1，于是提出了一种量化的欧氏距离度量
$$
I(i,j)=g(quant(\sqrt{(\tilde{x_i}-\tilde{x_j})^2+(\tilde{y_i}-\tilde{y_j})^2})
$$
quant将一组实数$\{0,1,1.41,2,\dots\}$映射到一组整数$\{0,1,2,3,\dots\}$，从而区分不同的相对位置。

#### 交叉
像素的位置方向对图像也很重要，提出交叉法，分别对水平和垂直方向进行编码后归纳。
$$
r_{ij}=p_{I^\tilde{x}(i,j)}^{\tilde{x}}+p_{I^\tilde{y}(i,j)}^{\tilde{y}},
$$
$$
I^{\tilde{x}}(i,j)=g(\tilde{x_i}-\tilde{x_j}),I^{\tilde{y}}(i,j)=g(\tilde{y_i}-\tilde{y_j}),
$$
其中$p_{I^\tilde{x}(i,j)}^{\tilde{x}},p_{I^\tilde{y}(i,j)}^{\tilde{y}}$是可学习的标量或向量。

#### 乘积
如果一个方向上的距离相同，交叉方法会将不同的相对位置编码到相同的嵌入。提出乘积方法
$$
r_{ij}=p_{I^\tilde{x}(i,j),I^\tilde{y}(i,j)},
$$
$I^{\tilde{x}}(i,j),I^{\tilde{y}}(i,j)$在交叉法中定义。

## 实验
比较不同方法下偏置和上下文模型的性能提升，无论在哪种方法下，上下文模型都能带来更大的提升。
![[Pasted image 20220616001552.png]]

比较clip和分段的性能差异，在图像分类任务下几乎可以忽略不计，不过作者说在目标检测任务下分段的表现更好。
![[Pasted image 20220616001848.png]]

桶数量(即查询表中的元素数量，最大距离限制* 2 + 1)与效果的关系，少于50个桶时增加桶可以提升效果，多于50个桶时增加桶就没有明显的效果。
![[Pasted image 20220616002014.png]]