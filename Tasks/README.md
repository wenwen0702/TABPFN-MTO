# Benchmark data

## Benchmark reference

The benchmark definitions correspond to the nine two-task problems described in:

Bingshui Da, Yew-Soon Ong, Liang Feng, A. K. Qin, Abhishek Gupta, Zexuan Zhu,
Chuan-Kang Ting, Ke Tang, and Xin Yao (2017).
*Evolutionary Multitasking for Single-objective Continuous Optimization:
Benchmark Problems, Performance Metric, and Baseline Results.*
arXiv:1706.03470. https://arxiv.org/abs/1706.03470

## Included files

| Problem identifier | MAT file | Task dimensions |
| --- | --- | --- |
| CIHS | `CI_H.mat` | 50 / 50 |
| CIMS | `CI_M.mat` | 50 / 50 |
| CILS | `CI_L.mat` | 50 / 50 |
| PIHS | `PI_H.mat` | 50 / 50 |
| PIMS | `PI_M.mat` | 50 / 50 |
| PILS | `PI_L.mat` | 50 / 25 |
| NIHS | `NI_H.mat` | 50 / 50 |
| NIMS | `NI_M.mat` | 50 / 50 |
| NILS | `NI_L.mat` | 50 / 50 |

These files contain benchmark transformation parameters used by
`CEC2017MTSO.py`, such as shifts and rotation matrices. They are loaded with
`scipy.io.loadmat`; MATLAB or MATLAB Engine is not needed.

## Download source

The nine MAT files were downloaded from
**MTO-Platform (MToP)**:

https://github.com/intLyc/MTO-Platform

For a fresh upstream download, open the repository and use **Code > Download ZIP**
(or clone it), locate the CEC2017 single-objective multitask benchmark parameter
files, and place the nine matching filenames listed above in this `Tasks/`
directory. Do not substitute result archives such as `MTOData.mat` for these
benchmark parameter files. The nine files are already included in this repository,
so no separate download is needed for this bundled copy.

## MToP acknowledgement and usage notes

The upstream README identifies copyright as belonging to Yanchi Li, allows
research use of MToP, and requests acknowledgement and citation in publications
that use the platform. Consult that README for the current upstream notice:

https://github.com/intLyc/MTO-Platform#copyright

Platform reference:

Yanchi Li, Wenyin Gong, Tingyu Zhang, Fei Ming, Shuijia Li, Qiong Gu,
and Yew-Soon Ong (2026). *MToP: A MATLAB Benchmarking Platform for
Evolutionary Multitasking*. ACM Transactions on Evolutionary Learning
and Optimization. https://doi.org/10.1145/3812535

Please cite the benchmark reference and acknowledge MToP as the data source.
Copyright and usage terms for these third-party files remain with their respective
owners; the project-code MIT license does not relicense them.

MToP is a MATLAB platform, but this Python implementation loads the MAT files
through SciPy and does not require MATLAB or MATLAB Engine.
