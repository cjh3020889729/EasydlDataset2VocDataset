# EasydlDataset2VocDataset
> 实现EasyDL平台标注的目标检测数据到VOC格式数据的转换

## 示例

```python
from easydl2voc import generate_from_easydl_to_voc

generate_from_easydl_to_voc(easydl_dataset_dir='data/data216381/1829142_33_1683621026',
                            output_dir='voc_dataset', train_ratio=0.85)
```
- `easydl_dataset_dir`: 原始数据集目录
- `output_dir`: 输出目录
- `train_ratio`: 训练集比例

## 核心代码简要说明

```python
def generate_from_easydl_to_voc(easydl_dataset_dir, output_dir='voc_dataset', train_ratio=0.85):
    """指定easydl数据集目录, 生成指定VOC格式数据集到输出目录下
        easydl_dataset_dir --> output_dir
            |- Annotations
            |- JPEGImages
            |- label_list.txt
            |- train_list.txt
            |- eval_list.txt
    """
```
