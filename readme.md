# Fuzzy Semantics for Arbitrary-Shaped Scene Text Detection

Project code for TIP2023 paper: "[Fuzzy Semantics for Arbitrary-Shaped Scene Text Detection](https://ieeexplore.ieee.org/document/9870672)"

```
@article{fuzzytext,
  author={Wang, Fangfang and Xu, Xiaogang and Chen, Yifeng and Li, Xi},
  journal={IEEE Transactions on Image Processing}, 
  title={Fuzzy Semantics for Arbitrary-Shaped Scene Text Detection}, 
  year={2023},
  volume={32},
  number={},
  pages={1-12},
}
```

Contact:

Fangfang Wang (fangfangliana@gmail.com);


## Dependencies 

Please install python dependencies from `requirements.txt`
```bash
pip install -r requirements.txt # install dependencies
git submodule update --init # clone mmdetection
```

## Training

Use `train.sh` in `experiments` directory to train from a config file inside it.

For example:
```python
bash ./experiments/ctw/train.sh
```

## Test

Use `test.sh` in `experiments` directory to test from a config file inside it.

For example:
```python
bash ./experiments/ctw/test.sh
```
