# Learning from Yesterday's Error: An Efficient Online Learning Method for Traffic Demand Prediction
## Abstract
Accurately predicting short-term traffic demand is critical for intelligent transportation systems. While deep learning models excel in this domain, their performance significantly degrades when trip distribution shifts due to external events or changes in urban dynamics. Frequent model retraining in order to adapt to these changes could imposes a prohibitive computational burden. Therefore, this paper proposed a method (**FOPRESEE** (**F**orecasting **O**nline with **Re**sidual **S**moothing and **E**nsemble **E**xperts)) to adapt the models in evolving traffic patterns with minimal computational cost. Our method is rooted in a simple yet powerful rule: adjust tomorrow's forecast for each region using yesterday's prediction error. To stabilize this adjustment, we employ exponential smoothing with a mixture of experts (MoE) strategy. Furthermore, we introduce an adaptive spatiotemporal smooth component to capture the relationship of errors across neighboring regions and time slots. Extensive experiments on seven real-world traffic demand datasets and three models demonstrate the effectiveness of our our method. Notably, it achieves superior performance with the least computational costs. Our contributions offer a practical and efficient solution for maintaining prediction accuracy in non-stationary urban environments.
## Main results
| Dataset | Model | Metrics | RMSE | MAE | RMSE | MAE | RMSE | MAE |
|---------|-------|---------|------|-----|------|-----|------|-----|
| **NTCBIKE** | **STGCN** | Ori | 11.043 | 5.142 | - | - | - | - |
| | | FSNet | 9.716 | 4.88 | - | - | - | - |
| | | Onenet | 10.593 | 5.264 | - | - | - | - |
| | | ADCSD | 9.439 | 4.843 | - | - | - | - |
| | | OGD | 9.83 | 4.942 | - | - | - | - |
| | | ELF | 10.792 | 5.745 | - | - | - | - |
| | | DSOF | **9.293** | **4.45** | - | - | - | - |
| | | Proceed | 9.638 | 4.874 | - | - | - | - |
| | | PETSA | 10.486 | 5.152 | - | - | - | - |
| | | TAFAS | 10.561 | 5.113 | - | - | - | - |
| | | Foresee | **9.148** | **4.429** | - | - | - | - |
| | **GWNET** | Ori | - | - | 10.492 | 5.168 | - | - |
| | | FSNet | - | - | **9.674** | **4.793** | - | - |
| | | ADCSD | - | - | 9.995 | 5.126 | - | - |
| | | OGD | - | - | 9.93 | 5.012 | - | - |
| | | ELF | - | - | 11.212 | 5.896 | - | - |
| | | DSOF | - | - | 11.147 | 5.064 | - | - |
| | | Proceed | - | - | 10.736 | 5.335 | - | - |
| | | PETSA | - | - | 12.248 | 5.941 | - | - |
| | | TAFAS | - | - | 12.665 | 6.032 | - | - |
| | | Foresee | - | - | **9.656** | **4.754** | - | - |
| | **Opencity** | Ori | - | - | - | - | 12.909 | 6.912 |
| | | ADCSD | - | - | - | - | 12.732 | 6.862 |
| | | ELF | - | - | - | - | **12.696** | **6.832** |
| | | TAFAS | - | - | - | - | - | - |
| | | Foresee | - | - | - | - | **12.29** | **6.475** |
| **NYCTAXI** | **STGCN** | Ori | 17.188 | 6.713 | - | - | - | - |
| | | FSNet | 11.478 | 4.667 | - | - | - | - |
| | | Onenet | 11.618 | 4.879 | - | - | - | - |
| | | ADCSD | **10.756** | **4.31** | - | - | - | - |
| | | OGD | 11.148 | 4.855 | - | - | - | - |
| | | ELF | 13.636 | 5.923 | - | - | - | - |
| | | DSOF | 11.103 | 5.121 | - | - | - | - |
| | | Proceed | 10.896 | 4.813 | - | - | - | - |
| | | PETSA | 13.719 | 4.981 | - | - | - | - |
| | | TAFAS | 13.681 | 4.914 | - | - | - | - |
| | | Foresee | **10.381** | **3.909** | - | - | - | - |
| | **GWNET** | Ori | - | - | 18.623 | 6.575 | - | - |
| | | FSNet | - | - | 11.635 | 4.563 | - | - |
| | | ADCSD | - | - | **11.138** | **4.323** | - | - |
| | | OGD | - | - | 11.041 | 4.452 | - | - |
| | | ELF | - | - | 13.697 | 5.456 | - | - |
| | | DSOF | - | - | 16.948 | 6.006 | - | - |
| | | Proceed | - | - | 11.3 | 4.463 | - | - |
| | | PETSA | - | - | 14.598 | 5.105 | - | - |
| | | TAFAS | - | - | 14.697 | 5.167 | - | - |
| | | Foresee | - | - | **10.567** | **3.933** | - | - |
| | **Opencity** | Ori | - | - | - | - | 15.964 | 5.763 |
| | | ADCSD | - | - | - | - | **15.386** | **5.563** |
| | | ELF | - | - | - | - | 15.937 | 5.667 |
| | | TAFAS | - | - | - | - | - | - |
| | | Foresee | - | - | - | - | **14.704** | **5.306** |
