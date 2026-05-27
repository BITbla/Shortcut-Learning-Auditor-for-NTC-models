# Shortcut-Learning-Auditor-for-NTC-models
A tool to audit the shortcut-learning for NTC models.
At present, only core code is provided in the library.

## Repository Layout

```text
  protocol_parser.py          # Protocol-field schema and byte-index mapping
  attribution.py              # YaTC/MFR field attribution
  attribution_tf.py           # TrafficFormer byte-space field attribution
  run_field_attribution.py    # YaTC attribution entry point
  run_tf_field.py             # TrafficFormer attribution entry point
  run_field_removal.py        # YaTC progressive field-removal validation
  run_field_removal_rand.py   # Random-replacement field-removal validation
  run_sanitized_baseline.py   # YaTC sanitization training/evaluation
  run_compression.py          # Post-hoc zeroing stress test
  prep_trafficformer.py       # PCAP-to-TSV preprocessing for TrafficFormer
  utils.py                    # Model and dataset loading helpers
requirements.txt
README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n protocol-field-audit python=3.10
conda activate protocol-field-audit
pip install -r requirements.txt
```

## External Model Code

This repository contains the audit logic, not full reimplementations of YaTC or
TrafficFormer. To reproduce the full experiments, place the corresponding model
code in the project root or adapt the import paths in `paca/utils.py`.

Expected external models:

- YaTC: "Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic
  Transformer with Multi-Level Flow Representation" (AAAI 2023).
- TrafficFormer: "TrafficFormer: An Efficient Pre-trained Model for Traffic
  Data" (IEEE S&P 2025).

## Dataset Access Plan

The datasets used by the paper are public or available through the original
authors' release channels. They are not redistributed here. Download each
dataset from its official source and follow the license or usage terms provided
by the dataset maintainers.

| Dataset | Used for | Official access |
| --- | --- | --- |
| USTC-TFC2016 | YaTC and TrafficFormer experiments | ScienceDB entry for USTC-TFC2016: https://www.scidb.cn/en/detail?dataSetId=923204944127127552 |
| ISCX VPN-nonVPN 2016 | YaTC and TrafficFormer experiments | Canadian Institute for Cybersecurity, VPN-nonVPN dataset: https://www.unb.ca/cic/datasets/vpn.html |
| ISCX Tor-nonTor 2016 | YaTC and TrafficFormer experiments | Canadian Institute for Cybersecurity, Tor-nonTor dataset: https://www.unb.ca/cic/datasets/tor.html |
| CSTNET-TLS1.3 | YaTC and TrafficFormer experiments | Follow the dataset instructions in the ET-BERT release: https://github.com/linwhitehat/ET-BERT |
| CICIoT2022 | YaTC extension and sanitization experiments | Canadian Institute for Cybersecurity, CIC IoT Profiling Dataset 2022: https://www.unb.ca/cic/datasets/iotdataset-2022.html |

## Basic Usage

### 1. Run YaTC Field Attribution

```bash
python -m paca.run_field_attribution \
  --dataset USTC \
  --checkpoint output/baselines/USTC/best_checkpoint.pth \
  --mode B \
  --max_samples 500 \
  --device cuda
```

The output is written under:

```text
output/attribution/field/USTC/mode_B/results.json
```

The result contains field-level margin-drop scores for the supported
counterfactual operators. The paper uses random replacement (`T_rand`) as the
primary attribution score.

### 2. Run TrafficFormer Field Attribution

```bash
python -m paca.run_tf_field \
  --dataset USTC \
  --checkpoint output/trafficformer/USTC/finetuned_model.bin \
  --max_samples 500 \
  --device cuda
```

TrafficFormer attribution uses the same protocol-field intervention surface but
passes modified bytes through the TrafficFormer tokenization path.

## Dataset Citations

Please cite the original dataset papers when using the corresponding data.

USTC-TFC2016:

```bibtex
@inproceedings{wang2017ustc,
  author    = {Wang, Wei and Zhu, Ming and Zeng, Xuewen and Ye, Xiaozhou and Sheng, Yiqiang},
  title     = {Malware Traffic Classification Using Convolutional Neural Network for Representation Learning},
  booktitle = {2017 International Conference on Information Networking (ICOIN)},
  pages     = {712--717},
  year      = {2017},
  doi       = {10.1109/ICOIN.2017.7899588}
}
```

ISCX VPN-nonVPN 2016:

```bibtex
@inproceedings{draper2016iscxvpn,
  author    = {Draper-Gil, Gerard and Lashkari, Arash Habibi and Mamun, Mohammad Saiful Islam and Ghorbani, Ali A.},
  title     = {Characterization of Encrypted and VPN Traffic Using Time-Related Features},
  booktitle = {Proceedings of the International Conference on Information Systems Security and Privacy},
  pages     = {407--414},
  year      = {2016},
  doi       = {10.5220/0005740704070414}
}
```

ISCX Tor-nonTor 2016:

```bibtex
@inproceedings{lashkari2017iscxtor,
  author    = {Lashkari, Arash Habibi and Draper-Gil, Gerard and Mamun, Mohammad Saiful Islam and Ghorbani, Ali A.},
  title     = {Characterization of Tor Traffic Using Time Based Features},
  booktitle = {Proceedings of the International Conference on Information Systems Security and Privacy},
  pages     = {253--262},
  year      = {2017},
  doi       = {10.5220/0006105602530262}
}
```

CSTNET-TLS1.3 / ET-BERT data source:

```bibtex
@inproceedings{lin2022bert,
  title     = {ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification},
  author    = {Lin, Xinjie and Xiong, Gang and Gou, Gaopeng and Li, Zhen and Shi, Junzheng and Yu, Jing},
  booktitle = {Proceedings of the ACM Web Conference 2022},
  pages     = {633--642},
  year      = {2022}
}
```

CICIoT2022:

```bibtex
@inproceedings{dadkhah2022ciciot,
  author    = {Dadkhah, Sajjad and Mahdikhani, Hassan and Kyei Danso, Priscilla and Zohourian, Alireza and Truong, Kevin Anh and Ghorbani, Ali A.},
  title     = {Towards the Development of a Realistic Multidimensional IoT Profiling Dataset},
  booktitle = {2022 19th Annual International Conference on Privacy, Security \& Trust},
  year      = {2022},
  doi       = {10.1109/PST55820.2022.9851966}
}
```

## Model Citations

YaTC:

```bibtex
@inproceedings{zhao2023yatc,
  title     = {Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation},
  author    = {Zhao, Ruijie and Zhan, Mingwei and Deng, Xianwen and Wang, Yanhao and Wang, Yijun and Gui, Guan and Xue, Zhi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  number    = {4},
  pages     = {5420--5427},
  year      = {2023}
}
```

TrafficFormer:

```bibtex
@inproceedings{zhou2025trafficformer,
  title     = {TrafficFormer: An Efficient Pre-trained Model for Traffic Data},
  author    = {Zhou, Guangmeng and Guo, Xiongwen and Liu, Zhuotao and Li, Tong and Li, Qi and Xu, Ke},
  booktitle = {2025 IEEE Symposium on Security and Privacy},
  pages     = {1844--1860},
  year      = {2025}
}
```
