---
layout: single
title:  "Water-Quality-Data-Analysis"
categories: coding
tag: [python, jupyter, machine learning]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


### 데이터 불러오기


Water Quality Data [Telangana Groundwater] <br>

데이터 출처: https://www.kaggle.com/datasets/sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data <br>

코드 참고: https://www.inflearn.com/course/파이썬-머신러닝-완벽가이드 강의



```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

data1 = pd.read_csv("ground_water_quality_2018_post.csv")
data2 = pd.read_csv("ground_water_quality_2019_post.csv")
data3 = pd.read_csv("ground_water_quality_2020_post.csv")
```


```python
data3.drop('Unnamed: 8', axis=1, inplace=True)  # 비어있는 컬럼 삭제
```


```python
# 컬럼명 맞추기
data2.columns = data1.columns
data3.columns = data1.columns
```


```python
# 데이터 합친 후 reindex
data = pd.concat([data1, data2, data3], axis=0)
data.index = range(data.shape[0])
data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sno</th>
      <th>district</th>
      <th>mandal</th>
      <th>village</th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>season</th>
      <th>pH</th>
      <th>E.C</th>
      <th>...</th>
      <th>SO4</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
      <th>Classification</th>
      <th>RSC  meq  / L</th>
      <th>Classification.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ADILABAD</td>
      <td>Adilabad</td>
      <td>Adilabad</td>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>postmonsoon 2018</td>
      <td>8.28</td>
      <td>745</td>
      <td>...</td>
      <td>46.0</td>
      <td>49.00</td>
      <td>4.00</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
      <td>C2S1</td>
      <td>-1.198684</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ADILABAD</td>
      <td>Bazarhatnur</td>
      <td>Bazarhatnur</td>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>postmonsoon 2018</td>
      <td>8.29</td>
      <td>921</td>
      <td>...</td>
      <td>68.0</td>
      <td>42.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
      <td>C3S1</td>
      <td>-3.397862</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ADILABAD</td>
      <td>Gudihatnoor</td>
      <td>Gudihatnoor</td>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>postmonsoon 2018</td>
      <td>7.69</td>
      <td>510</td>
      <td>...</td>
      <td>44.0</td>
      <td>45.00</td>
      <td>2.00</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
      <td>C2S1</td>
      <td>-0.398684</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>ADILABAD</td>
      <td>Jainath</td>
      <td>Jainath</td>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>postmonsoon 2018</td>
      <td>8.09</td>
      <td>422</td>
      <td>...</td>
      <td>35.0</td>
      <td>27.00</td>
      <td>1.00</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
      <td>C2S1</td>
      <td>0.000658</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>ADILABAD</td>
      <td>Narnoor</td>
      <td>Narnoor</td>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>postmonsoon 2018</td>
      <td>8.21</td>
      <td>2321</td>
      <td>...</td>
      <td>280.0</td>
      <td>298.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
      <td>C4S2</td>
      <td>-4.396875</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>375</td>
      <td>YADADRI</td>
      <td>S.Narayanpur</td>
      <td>S.Narayanpur</td>
      <td>17.144719</td>
      <td>78.860010</td>
      <td>9.90</td>
      <td>Post-monsoon 2020</td>
      <td>7.8</td>
      <td>2324</td>
      <td>...</td>
      <td>33.0</td>
      <td>169.30</td>
      <td>2.60</td>
      <td>160.0</td>
      <td>97.240</td>
      <td>799.835526</td>
      <td>2.602728</td>
      <td>C4S1</td>
      <td>-8.596711</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>376</td>
      <td>YADADRI</td>
      <td>Thurkapally</td>
      <td>Gandamalla</td>
      <td>17.733101</td>
      <td>78.853831</td>
      <td>5.74</td>
      <td>Post-monsoon 2020</td>
      <td>8.26</td>
      <td>2109</td>
      <td>...</td>
      <td>33.0</td>
      <td>211.30</td>
      <td>43.30</td>
      <td>48.0</td>
      <td>116.688</td>
      <td>599.802632</td>
      <td>3.751176</td>
      <td>C3S1</td>
      <td>-3.396053</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>377</td>
      <td>YADADRI</td>
      <td>Valigonda</td>
      <td>T. somaram</td>
      <td>17.399953</td>
      <td>78.952290</td>
      <td>1.72</td>
      <td>Post-monsoon 2020</td>
      <td>8.77</td>
      <td>1115</td>
      <td>...</td>
      <td>15.0</td>
      <td>60.44</td>
      <td>3.04</td>
      <td>80.0</td>
      <td>53.482</td>
      <td>419.909539</td>
      <td>1.282386</td>
      <td>C3S1</td>
      <td>-4.398191</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>378</td>
      <td>YADADRI</td>
      <td>Valigonda</td>
      <td>Vemulakonda</td>
      <td>17.347782</td>
      <td>79.143433</td>
      <td>1.65</td>
      <td>Post-monsoon 2020</td>
      <td>7.76</td>
      <td>5053</td>
      <td>...</td>
      <td>109.0</td>
      <td>465.20</td>
      <td>3.30</td>
      <td>400.0</td>
      <td>92.378</td>
      <td>1379.843750</td>
      <td>5.444988</td>
      <td>C4S1</td>
      <td>-21.996875</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>379</td>
      <td>YADADRI</td>
      <td>Y.Gutta</td>
      <td>Mallapuram</td>
      <td>17.633555</td>
      <td>78.911638</td>
      <td>4.92</td>
      <td>Post-monsoon 2020</td>
      <td>8.13</td>
      <td>2280</td>
      <td>...</td>
      <td>34.0</td>
      <td>170.70</td>
      <td>5.60</td>
      <td>152.0</td>
      <td>97.240</td>
      <td>779.835526</td>
      <td>2.657689</td>
      <td>C4S1</td>
      <td>-11.596711</td>
      <td>P.S.</td>
    </tr>
  </tbody>
</table>
<p>1106 rows × 26 columns</p>
</div>


### 데이터 전처리


 **각 컬럼별 Null값 개수 확인**



```python
data.isnull().sum()
```

<pre>
sno                   0
district              0
mandal                0
village               0
lat_gis               0
long_gis              0
gwl                  11
season                0
pH                    0
E.C                   0
TDS                   0
CO3                 160
HCO3                  0
Cl                    0
F                     0
NO3                   0
SO4                   0
Na                    0
K                     0
Ca                    0
Mg                    0
T.H                   0
SAR                   0
Classification        0
RSC  meq  / L         0
Classification.1      0
dtype: int64
</pre>
gwl (ground water level) Null값 11개, CO3 Null값 160개



```python
# gwl, CO3 각각 다른 컬럼과의 상관관계 탐색 
data.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1106 entries, 0 to 1105
Data columns (total 26 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   sno               1106 non-null   int64  
 1   district          1106 non-null   object 
 2   mandal            1106 non-null   object 
 3   village           1106 non-null   object 
 4   lat_gis           1106 non-null   float64
 5   long_gis          1106 non-null   float64
 6   gwl               1095 non-null   float64
 7   season            1106 non-null   object 
 8   pH                1106 non-null   object 
 9   E.C               1106 non-null   int64  
 10  TDS               1106 non-null   float64
 11  CO3               946 non-null    float64
 12  HCO3              1106 non-null   float64
 13  Cl                1106 non-null   int64  
 14  F                 1106 non-null   float64
 15  NO3               1106 non-null   float64
 16  SO4               1106 non-null   float64
 17  Na                1106 non-null   float64
 18  K                 1106 non-null   float64
 19  Ca                1106 non-null   float64
 20  Mg                1106 non-null   float64
 21  T.H               1106 non-null   float64
 22  SAR               1106 non-null   float64
 23  Classification    1106 non-null   object 
 24  RSC  meq  / L     1106 non-null   float64
 25  Classification.1  1106 non-null   object 
dtypes: float64(16), int64(3), object(7)
memory usage: 224.8+ KB
</pre>

```python
# 숫자형 값인데 object type으로 되어있는 pH 컬럼 type 변경
data.loc[999, 'pH'] = "8.05"
data['pH'] = data['pH'].astype('float64')
```


```python
data.corr()['gwl']
```

<pre>
sno              0.121790
lat_gis         -0.077241
long_gis        -0.228986
gwl              1.000000
pH              -0.104560
E.C             -0.032740
TDS             -0.032740
CO3             -0.028910
HCO3            -0.007824
Cl              -0.026457
F                0.068213
NO3             -0.000483
SO4             -0.133135
Na              -0.082886
K               -0.063463
Ca               0.045627
Mg              -0.021886
T.H              0.014397
SAR             -0.069024
RSC  meq  / L   -0.021590
Name: gwl, dtype: float64
</pre>
gwl은 다른 컬럼들과 상관성이 거의 없음<br>

따라서 gwl의 평균값으로 Null값을 채움



```python
data['gwl'].fillna(data['gwl'].mean(),inplace=True)
```


```python
data.isnull().sum()
```

<pre>
sno                   0
district              0
mandal                0
village               0
lat_gis               0
long_gis              0
gwl                   0
season                0
pH                    0
E.C                   0
TDS                   0
CO3                 160
HCO3                  0
Cl                    0
F                     0
NO3                   0
SO4                   0
Na                    0
K                     0
Ca                    0
Mg                    0
T.H                   0
SAR                   0
Classification        0
RSC  meq  / L         0
Classification.1      0
dtype: int64
</pre>

```python
data.corr()['CO3']
```

<pre>
sno             -0.042642
lat_gis         -0.165900
long_gis         0.046555
gwl             -0.028548
pH               0.576520
E.C             -0.076437
TDS             -0.076437
CO3              1.000000
HCO3            -0.039647
Cl              -0.110447
F                0.191556
NO3             -0.091382
SO4             -0.044441
Na               0.102308
K               -0.017172
Ca              -0.223167
Mg              -0.123764
T.H             -0.211227
SAR              0.243574
RSC  meq  / L    0.271018
Name: CO3, dtype: float64
</pre>
pH 컬럼과 양의 상관관계가 있음(0.57628413)<br>

따라서 같은 pH값을 갖는 행들의 CO3 평균값으로 Null값을 채움



```python
ind = data[data['CO3'].isnull()].index
print(ind)
```

<pre>
Int64Index([383, 384, 387, 388, 389, 390, 391, 392, 394, 395,
            ...
            688, 689, 690, 691, 692, 693, 694, 695, 696, 697],
           dtype='int64', length=160)
</pre>

```python
import math

for i in ind:
    data.loc[i, 'CO3'] = (data['CO3'][data['pH']==data.loc[i, 'pH']]).mean()
    if math.isnan(data.loc[i, 'CO3']):
        data.loc[i, 'CO3'] = (data['CO3'][(data.loc[i, 'pH'] - 0.05 <= data['pH']) & (data['pH'] <= data.loc[i, 'pH'] + 0.05)]).mean()
    if math.isnan(data.loc[i, 'CO3']):
        data.loc[i, 'CO3'] = 0
```


```python
data.isnull().sum()
```

<pre>
sno                 0
district            0
mandal              0
village             0
lat_gis             0
long_gis            0
gwl                 0
season              0
pH                  0
E.C                 0
TDS                 0
CO3                 0
HCO3                0
Cl                  0
F                   0
NO3                 0
SO4                 0
Na                  0
K                   0
Ca                  0
Mg                  0
T.H                 0
SAR                 0
Classification      0
RSC  meq  / L       0
Classification.1    0
dtype: int64
</pre>
**범주형 변수 (season 컬럼) one-hot encoding**



```python
data.head()

# 불필요한 컬럼과 target 컬럼 제외한 설계행렬
X_features = data.drop(['sno', 'district', 'mandal', 'village', 'Classification', 'RSC  meq  / L', 'Classification.1'], axis=1, inplace=False)
X_features
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>season</th>
      <th>pH</th>
      <th>E.C</th>
      <th>TDS</th>
      <th>CO3</th>
      <th>HCO3</th>
      <th>Cl</th>
      <th>F</th>
      <th>NO3</th>
      <th>SO4</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>postmonsoon 2018</td>
      <td>8.28</td>
      <td>745</td>
      <td>476.80</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>60</td>
      <td>0.44</td>
      <td>42.276818</td>
      <td>46.0</td>
      <td>49.00</td>
      <td>4.00</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>postmonsoon 2018</td>
      <td>8.29</td>
      <td>921</td>
      <td>589.44</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>80</td>
      <td>0.56</td>
      <td>100.659091</td>
      <td>68.0</td>
      <td>42.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>postmonsoon 2018</td>
      <td>7.69</td>
      <td>510</td>
      <td>326.40</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>30</td>
      <td>0.66</td>
      <td>41.471545</td>
      <td>44.0</td>
      <td>45.00</td>
      <td>2.00</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>postmonsoon 2018</td>
      <td>8.09</td>
      <td>422</td>
      <td>270.08</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>10</td>
      <td>0.58</td>
      <td>10.669864</td>
      <td>35.0</td>
      <td>27.00</td>
      <td>1.00</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>postmonsoon 2018</td>
      <td>8.21</td>
      <td>2321</td>
      <td>1485.44</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>340</td>
      <td>2.56</td>
      <td>128.843636</td>
      <td>280.0</td>
      <td>298.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>17.144719</td>
      <td>78.860010</td>
      <td>9.90</td>
      <td>Post-monsoon 2020</td>
      <td>7.80</td>
      <td>2324</td>
      <td>1487.36</td>
      <td>0.0</td>
      <td>370.0</td>
      <td>370</td>
      <td>0.58</td>
      <td>336.161100</td>
      <td>33.0</td>
      <td>169.30</td>
      <td>2.60</td>
      <td>160.0</td>
      <td>97.240</td>
      <td>799.835526</td>
      <td>2.602728</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>17.733101</td>
      <td>78.853831</td>
      <td>5.74</td>
      <td>Post-monsoon 2020</td>
      <td>8.26</td>
      <td>2109</td>
      <td>1349.76</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>260</td>
      <td>1.08</td>
      <td>332.175000</td>
      <td>33.0</td>
      <td>211.30</td>
      <td>43.30</td>
      <td>48.0</td>
      <td>116.688</td>
      <td>599.802632</td>
      <td>3.751176</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>17.399953</td>
      <td>78.952290</td>
      <td>1.72</td>
      <td>Post-monsoon 2020</td>
      <td>8.77</td>
      <td>1115</td>
      <td>713.60</td>
      <td>20.0</td>
      <td>180.0</td>
      <td>220</td>
      <td>0.34</td>
      <td>44.201420</td>
      <td>15.0</td>
      <td>60.44</td>
      <td>3.04</td>
      <td>80.0</td>
      <td>53.482</td>
      <td>419.909539</td>
      <td>1.282386</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>17.347782</td>
      <td>79.143433</td>
      <td>1.65</td>
      <td>Post-monsoon 2020</td>
      <td>7.76</td>
      <td>5053</td>
      <td>3233.92</td>
      <td>0.0</td>
      <td>280.0</td>
      <td>1360</td>
      <td>0.44</td>
      <td>76.355960</td>
      <td>109.0</td>
      <td>465.20</td>
      <td>3.30</td>
      <td>400.0</td>
      <td>92.378</td>
      <td>1379.843750</td>
      <td>5.444988</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>17.633555</td>
      <td>78.911638</td>
      <td>4.92</td>
      <td>Post-monsoon 2020</td>
      <td>8.13</td>
      <td>2280</td>
      <td>1459.20</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>340</td>
      <td>1.12</td>
      <td>506.234700</td>
      <td>34.0</td>
      <td>170.70</td>
      <td>5.60</td>
      <td>152.0</td>
      <td>97.240</td>
      <td>779.835526</td>
      <td>2.657689</td>
    </tr>
  </tbody>
</table>
<p>1106 rows × 19 columns</p>
</div>



```python
X_features = pd.get_dummies(X_features)
X_features.columns
```

<pre>
Index(['lat_gis', 'long_gis', 'gwl', 'pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl',
       'F', 'NO3 ', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR',
       'season_Post-monsoon 2020', 'season_post monsoon 2019',
       'season_postmonsoon 2018 '],
      dtype='object')
</pre>

```python
X_features
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>pH</th>
      <th>E.C</th>
      <th>TDS</th>
      <th>CO3</th>
      <th>HCO3</th>
      <th>Cl</th>
      <th>F</th>
      <th>...</th>
      <th>SO4</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
      <th>season_Post-monsoon 2020</th>
      <th>season_post monsoon 2019</th>
      <th>season_postmonsoon 2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>8.28</td>
      <td>745</td>
      <td>476.80</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>60</td>
      <td>0.44</td>
      <td>...</td>
      <td>46.0</td>
      <td>49.00</td>
      <td>4.00</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>8.29</td>
      <td>921</td>
      <td>589.44</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>80</td>
      <td>0.56</td>
      <td>...</td>
      <td>68.0</td>
      <td>42.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>7.69</td>
      <td>510</td>
      <td>326.40</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>30</td>
      <td>0.66</td>
      <td>...</td>
      <td>44.0</td>
      <td>45.00</td>
      <td>2.00</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>8.09</td>
      <td>422</td>
      <td>270.08</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>10</td>
      <td>0.58</td>
      <td>...</td>
      <td>35.0</td>
      <td>27.00</td>
      <td>1.00</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>8.21</td>
      <td>2321</td>
      <td>1485.44</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>340</td>
      <td>2.56</td>
      <td>...</td>
      <td>280.0</td>
      <td>298.00</td>
      <td>5.00</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>17.144719</td>
      <td>78.860010</td>
      <td>9.90</td>
      <td>7.80</td>
      <td>2324</td>
      <td>1487.36</td>
      <td>0.0</td>
      <td>370.0</td>
      <td>370</td>
      <td>0.58</td>
      <td>...</td>
      <td>33.0</td>
      <td>169.30</td>
      <td>2.60</td>
      <td>160.0</td>
      <td>97.240</td>
      <td>799.835526</td>
      <td>2.602728</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>17.733101</td>
      <td>78.853831</td>
      <td>5.74</td>
      <td>8.26</td>
      <td>2109</td>
      <td>1349.76</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>260</td>
      <td>1.08</td>
      <td>...</td>
      <td>33.0</td>
      <td>211.30</td>
      <td>43.30</td>
      <td>48.0</td>
      <td>116.688</td>
      <td>599.802632</td>
      <td>3.751176</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>17.399953</td>
      <td>78.952290</td>
      <td>1.72</td>
      <td>8.77</td>
      <td>1115</td>
      <td>713.60</td>
      <td>20.0</td>
      <td>180.0</td>
      <td>220</td>
      <td>0.34</td>
      <td>...</td>
      <td>15.0</td>
      <td>60.44</td>
      <td>3.04</td>
      <td>80.0</td>
      <td>53.482</td>
      <td>419.909539</td>
      <td>1.282386</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>17.347782</td>
      <td>79.143433</td>
      <td>1.65</td>
      <td>7.76</td>
      <td>5053</td>
      <td>3233.92</td>
      <td>0.0</td>
      <td>280.0</td>
      <td>1360</td>
      <td>0.44</td>
      <td>...</td>
      <td>109.0</td>
      <td>465.20</td>
      <td>3.30</td>
      <td>400.0</td>
      <td>92.378</td>
      <td>1379.843750</td>
      <td>5.444988</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>17.633555</td>
      <td>78.911638</td>
      <td>4.92</td>
      <td>8.13</td>
      <td>2280</td>
      <td>1459.20</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>340</td>
      <td>1.12</td>
      <td>...</td>
      <td>34.0</td>
      <td>170.70</td>
      <td>5.60</td>
      <td>152.0</td>
      <td>97.240</td>
      <td>779.835526</td>
      <td>2.657689</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1106 rows × 21 columns</p>
</div>


**Target 컬럼**

* Classification 컬럼: C3S1, C2S1, C4S1, C4S2, C3S2, C4S4, C3S3, C1S1 등 9개 등급으로 분류되는 수질 예측에 사용됨

* Classification.1 컬럼: RSC에 의한 분류, RSC는 알칼리성을 초과하는 탄산염과 중탄산염의 양으로 정의됨



**RSC값과 TDS값 컬럼** <br>

<br>

< RSC에 의한 지하수 분류 >

* RSC < 1.25 -- safe --

* 1.25 <= RSC <= 2.50 -- marginal --

* 2.50 < RSC -- unsuitable --



< 가축 및 가금류에 대한 지하수 이용 >

* TDS < 1000 mg/L -- Excellent -- 

* TDS b/w 1000-3000 -- very satisfactory --

* TDS b/w 3000-5000 -- Satisfactory for livestock Unfit for poultry -- 

* TDS b/w 5000-7000 -- Limited use for livestock Unfit for poultry -- 

* TDS b/w 7000-10,000 -- Very limited use -- 

* TDS > 10,000 -- Not recommended -- 



```python
y_target1 = data.iloc[:, 23]
y_target2 = data.iloc[:, 25]
print(y_target1)
print(y_target2)
```

<pre>
0       C2S1
1       C3S1
2       C2S1
3       C2S1
4       C4S2
        ... 
1101    C4S1
1102    C3S1
1103    C3S1
1104    C4S1
1105    C4S1
Name: Classification, Length: 1106, dtype: object
0       P.S.
1       P.S.
2       P.S.
3       P.S.
4       P.S.
        ... 
1101    P.S.
1102    P.S.
1103    P.S.
1104    P.S.
1105    P.S.
Name: Classification.1, Length: 1106, dtype: object
</pre>
### 수질 등급 분류 (타겟값: Classification 컬럼)


**PCA 수행을 위해 데이터 scaling**



```python
data_ohe = X_features.copy()
data_ohe['RSC'] = data.iloc[:, 24]
data_ohe['Classification1'] = data.iloc[:, 23]
data_ohe['Classification2'] = data.iloc[:, 25]
data_ohe = data_ohe[['lat_gis', 'long_gis', 'gwl', 'pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl',
       'F', 'NO3 ', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC',
       'season_Post-monsoon 2020', 'season_post monsoon 2019',
       'season_postmonsoon 2018 ', 'Classification1',
       'Classification2']]
data_ohe
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>pH</th>
      <th>E.C</th>
      <th>TDS</th>
      <th>CO3</th>
      <th>HCO3</th>
      <th>Cl</th>
      <th>F</th>
      <th>...</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
      <th>RSC</th>
      <th>season_Post-monsoon 2020</th>
      <th>season_post monsoon 2019</th>
      <th>season_postmonsoon 2018</th>
      <th>Classification1</th>
      <th>Classification2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>8.28</td>
      <td>745</td>
      <td>476.80</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>60</td>
      <td>0.44</td>
      <td>...</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
      <td>-1.198684</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>C2S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>8.29</td>
      <td>921</td>
      <td>589.44</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>80</td>
      <td>0.56</td>
      <td>...</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
      <td>-3.397862</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>C3S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>7.69</td>
      <td>510</td>
      <td>326.40</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>30</td>
      <td>0.66</td>
      <td>...</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
      <td>-0.398684</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>C2S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>8.09</td>
      <td>422</td>
      <td>270.08</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>10</td>
      <td>0.58</td>
      <td>...</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
      <td>0.000658</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>C2S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>8.21</td>
      <td>2321</td>
      <td>1485.44</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>340</td>
      <td>2.56</td>
      <td>...</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
      <td>-4.396875</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>C4S2</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>17.144719</td>
      <td>78.860010</td>
      <td>9.90</td>
      <td>7.80</td>
      <td>2324</td>
      <td>1487.36</td>
      <td>0.0</td>
      <td>370.0</td>
      <td>370</td>
      <td>0.58</td>
      <td>...</td>
      <td>160.0</td>
      <td>97.240</td>
      <td>799.835526</td>
      <td>2.602728</td>
      <td>-8.596711</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>C4S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>17.733101</td>
      <td>78.853831</td>
      <td>5.74</td>
      <td>8.26</td>
      <td>2109</td>
      <td>1349.76</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>260</td>
      <td>1.08</td>
      <td>...</td>
      <td>48.0</td>
      <td>116.688</td>
      <td>599.802632</td>
      <td>3.751176</td>
      <td>-3.396053</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>C3S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>17.399953</td>
      <td>78.952290</td>
      <td>1.72</td>
      <td>8.77</td>
      <td>1115</td>
      <td>713.60</td>
      <td>20.0</td>
      <td>180.0</td>
      <td>220</td>
      <td>0.34</td>
      <td>...</td>
      <td>80.0</td>
      <td>53.482</td>
      <td>419.909539</td>
      <td>1.282386</td>
      <td>-4.398191</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>C3S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>17.347782</td>
      <td>79.143433</td>
      <td>1.65</td>
      <td>7.76</td>
      <td>5053</td>
      <td>3233.92</td>
      <td>0.0</td>
      <td>280.0</td>
      <td>1360</td>
      <td>0.44</td>
      <td>...</td>
      <td>400.0</td>
      <td>92.378</td>
      <td>1379.843750</td>
      <td>5.444988</td>
      <td>-21.996875</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>C4S1</td>
      <td>P.S.</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>17.633555</td>
      <td>78.911638</td>
      <td>4.92</td>
      <td>8.13</td>
      <td>2280</td>
      <td>1459.20</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>340</td>
      <td>1.12</td>
      <td>...</td>
      <td>152.0</td>
      <td>97.240</td>
      <td>779.835526</td>
      <td>2.657689</td>
      <td>-11.596711</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>C4S1</td>
      <td>P.S.</td>
    </tr>
  </tbody>
</table>
<p>1106 rows × 24 columns</p>
</div>



```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(data_ohe.iloc[:, :19])
```


```python
pd.DataFrame(X_scaled)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.298941</td>
      <td>-0.325420</td>
      <td>-0.467696</td>
      <td>0.942173</td>
      <td>-0.717159</td>
      <td>-0.717159</td>
      <td>-0.367023</td>
      <td>-0.597892</td>
      <td>-0.669591</td>
      <td>-0.865088</td>
      <td>-0.336689</td>
      <td>0.106131</td>
      <td>-0.657005</td>
      <td>-0.184979</td>
      <td>-0.553204</td>
      <td>-0.310241</td>
      <td>-0.527881</td>
      <td>-0.529633</td>
      <td>0.213044</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.053290</td>
      <td>-0.537968</td>
      <td>-0.466359</td>
      <td>0.964065</td>
      <td>-0.512668</td>
      <td>-0.512668</td>
      <td>-0.367023</td>
      <td>-0.523851</td>
      <td>-0.569844</td>
      <td>-0.711785</td>
      <td>0.262650</td>
      <td>0.513042</td>
      <td>-0.718522</td>
      <td>-0.136691</td>
      <td>-0.429432</td>
      <td>0.309979</td>
      <td>-0.073181</td>
      <td>-0.657223</td>
      <td>-0.231752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.131493</td>
      <td>-0.340674</td>
      <td>-0.482393</td>
      <td>-0.349489</td>
      <td>-0.990202</td>
      <td>-0.990202</td>
      <td>-0.367023</td>
      <td>-0.745973</td>
      <td>-0.819210</td>
      <td>-0.584032</td>
      <td>-0.344956</td>
      <td>0.069139</td>
      <td>-0.692158</td>
      <td>-0.281554</td>
      <td>-0.924521</td>
      <td>-0.310241</td>
      <td>-0.755308</td>
      <td>-0.513353</td>
      <td>0.374849</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.371969</td>
      <td>-0.184468</td>
      <td>-0.379512</td>
      <td>0.526214</td>
      <td>-1.092447</td>
      <td>-1.092447</td>
      <td>-0.367023</td>
      <td>-1.042137</td>
      <td>-0.918957</td>
      <td>-0.686234</td>
      <td>-0.661158</td>
      <td>-0.097325</td>
      <td>-0.850345</td>
      <td>-0.329842</td>
      <td>-0.800748</td>
      <td>-0.806417</td>
      <td>-0.982611</td>
      <td>-0.651913</td>
      <td>0.455618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.096431</td>
      <td>0.075496</td>
      <td>-0.860514</td>
      <td>0.788925</td>
      <td>1.113969</td>
      <td>1.113969</td>
      <td>-0.367023</td>
      <td>-0.005564</td>
      <td>0.726859</td>
      <td>1.843265</td>
      <td>0.551985</td>
      <td>4.434187</td>
      <td>1.531261</td>
      <td>-0.136691</td>
      <td>-0.429432</td>
      <td>1.054243</td>
      <td>0.381487</td>
      <td>1.032406</td>
      <td>-0.433808</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>-0.661348</td>
      <td>0.084489</td>
      <td>0.174976</td>
      <td>-0.108671</td>
      <td>1.117454</td>
      <td>1.117454</td>
      <td>-0.367023</td>
      <td>0.512722</td>
      <td>0.876479</td>
      <td>-0.686234</td>
      <td>2.680257</td>
      <td>-0.134317</td>
      <td>0.400217</td>
      <td>-0.252582</td>
      <td>1.179608</td>
      <td>1.178287</td>
      <td>1.442785</td>
      <td>-0.058684</td>
      <td>-1.283249</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>0.028854</td>
      <td>0.076935</td>
      <td>-0.380848</td>
      <td>0.898387</td>
      <td>0.867649</td>
      <td>0.867649</td>
      <td>-0.367023</td>
      <td>0.956967</td>
      <td>0.327873</td>
      <td>-0.047472</td>
      <td>2.639337</td>
      <td>-0.134317</td>
      <td>0.769323</td>
      <td>1.712733</td>
      <td>-0.553204</td>
      <td>1.674463</td>
      <td>0.684568</td>
      <td>0.348162</td>
      <td>-0.231386</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>-0.361945</td>
      <td>0.197299</td>
      <td>-0.917967</td>
      <td>2.014909</td>
      <td>-0.287262</td>
      <td>-0.287262</td>
      <td>0.874291</td>
      <td>-0.894055</td>
      <td>0.128381</td>
      <td>-0.992840</td>
      <td>-0.316931</td>
      <td>-0.467244</td>
      <td>-0.556468</td>
      <td>-0.231335</td>
      <td>-0.058115</td>
      <td>0.061891</td>
      <td>0.002690</td>
      <td>-0.526424</td>
      <td>-0.434074</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>-0.423145</td>
      <td>0.430967</td>
      <td>-0.927319</td>
      <td>-0.196241</td>
      <td>4.288233</td>
      <td>4.288233</td>
      <td>-0.367023</td>
      <td>-0.153646</td>
      <td>5.813925</td>
      <td>-0.865088</td>
      <td>0.013159</td>
      <td>1.271377</td>
      <td>3.000650</td>
      <td>-0.218780</td>
      <td>4.892776</td>
      <td>1.054243</td>
      <td>3.641283</td>
      <td>0.948207</td>
      <td>-3.993509</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>-0.087918</td>
      <td>0.147603</td>
      <td>-0.490410</td>
      <td>0.613784</td>
      <td>1.066331</td>
      <td>1.066331</td>
      <td>-0.367023</td>
      <td>-0.745973</td>
      <td>0.726859</td>
      <td>0.003629</td>
      <td>4.426192</td>
      <td>-0.115821</td>
      <td>0.412521</td>
      <td>-0.107718</td>
      <td>1.055836</td>
      <td>1.178287</td>
      <td>1.366976</td>
      <td>-0.039213</td>
      <td>-1.890016</td>
    </tr>
  </tbody>
</table>
<p>1106 rows × 19 columns</p>
</div>


**PCA 변환 (범주형 변수 제외)** <br>

시각화를 위해 component 개수 2로 설정



```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)
```

<pre>
(1106, 2)
</pre>

```python
data_pca = pd.DataFrame(X_pca, columns=['component1', 'component2'])
data_pca['target'] = data_ohe['Classification1']
data_pca.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component1</th>
      <th>component2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.021253</td>
      <td>-0.627406</td>
      <td>C2S1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.121287</td>
      <td>-0.920919</td>
      <td>C3S1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.410542</td>
      <td>-0.837598</td>
      <td>C2S1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.079971</td>
      <td>-0.744420</td>
      <td>C2S1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.628757</td>
      <td>2.105698</td>
      <td>C4S2</td>
    </tr>
  </tbody>
</table>
</div>


**시각화**



```python
import matplotlib.pyplot as plt
cl = data['Classification'].unique()
length = len(data['Classification'].unique())
markers = ['.', 'v', '^', 's', 'p', '*', 'x', 'D', '<', '>', 'h', '1', '2']

for i in range(length):
    x_axis = data_pca[data_pca['target']==cl[i]]['component1']
    y_axis = data_pca[data_pca['target']==cl[i]]['component2']
    plt.scatter(x_axis, y_axis, marker=markers[i], label=cl[i])

plt.legend()
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()
```

<img src="data:image/png;base64,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"/>

원본 데이터와 PCA 변환 데이터 예측 성능 비교 (RandomForestClassifier 이용)



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(random_state=100)
scores = cross_val_score(rcf, data_ohe.iloc[:, :22], data_ohe['Classification1'],
                        scoring='accuracy', cv=3)

print('원본 데이터 CV accuracy:', scores)
print('원본 데이터 CV 평균 accuracy:', np.mean(scores))
```

<pre>
원본 데이터 CV accuracy: [0.95663957 0.95121951 0.95108696]
원본 데이터 CV 평균 accuracy: 0.952982011704175
</pre>

```python
pca_X = pd.concat([data_ohe[['season_Post-monsoon 2020', 'season_post monsoon 2019',
       'season_postmonsoon 2018 ']], data_pca[['component1', 'component2']]], axis=1)
scores_pca = cross_val_score(rcf, pca_X, data_pca['target'], scoring='accuracy', cv=3 )
print('PCA 변환 데이터 CV accuracy:',scores_pca)
print('PCA 변환 데이터 CV 평균 accuracy:', np.mean(scores_pca))
```

<pre>
PCA 변환 데이터 CV accuracy: [0.71273713 0.68563686 0.375     ]
PCA 변환 데이터 CV 평균 accuracy: 0.5911246612466124
</pre>
components의 개수가 2인 경우 예측 성능이 매우 떨어지므로 components의 개수에 따른 예측 성능 비교 (6개가 적당함)



```python
for i in range(11):
    pca = PCA(n_components=i)

    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    data_pca = pd.DataFrame(X_pca)
    data_pca

    pca_X = pd.concat([data_ohe[['season_Post-monsoon 2020', 'season_post monsoon 2019',
           'season_postmonsoon 2018 ']], data_pca], axis=1)
    scores_pca = cross_val_score(rcf, pca_X, data_ohe['Classification1'], scoring='accuracy', cv=3)
    print('Component 개수:', i)
    print('PCA 변환 데이터 CV accuracy:', scores_pca)
    print('PCA 변환 데이터 CV 평균 accuracy:', np.mean(scores_pca))
    print('')
```

<pre>
Component 개수: 0
PCA 변환 데이터 CV accuracy: [0.22493225 0.07588076 0.08967391]
PCA 변환 데이터 CV 평균 accuracy: 0.13016230705785317

Component 개수: 1
PCA 변환 데이터 CV accuracy: [0.50135501 0.66124661 0.27717391]
PCA 변환 데이터 CV 평균 accuracy: 0.4799251796865795

Component 개수: 2
PCA 변환 데이터 CV accuracy: [0.71273713 0.68563686 0.375     ]
PCA 변환 데이터 CV 평균 accuracy: 0.5911246612466124

Component 개수: 3
PCA 변환 데이터 CV accuracy: [0.8401084  0.85907859 0.875     ]
PCA 변환 데이터 CV 평균 accuracy: 0.8580623306233063

Component 개수: 4
PCA 변환 데이터 CV accuracy: [0.84281843 0.83739837 0.75543478]
PCA 변환 데이터 CV 평균 accuracy: 0.8118838615922391

Component 개수: 5
PCA 변환 데이터 CV accuracy: [0.87262873 0.8699187  0.8451087 ]
PCA 변환 데이터 CV 평균 accuracy: 0.8625520403754762

Component 개수: 6
PCA 변환 데이터 CV accuracy: [0.91056911 0.88346883 0.88858696]
PCA 변환 데이터 CV 평균 accuracy: 0.8942082989670476

Component 개수: 7
PCA 변환 데이터 CV accuracy: [0.90243902 0.87804878 0.89402174]
PCA 변환 데이터 CV 평균 accuracy: 0.8915031813361612

Component 개수: 8
PCA 변환 데이터 CV accuracy: [0.90785908 0.88346883 0.88586957]
PCA 변환 데이터 CV 평균 accuracy: 0.8923991594988414

Component 개수: 9
PCA 변환 데이터 CV accuracy: [0.899729   0.87804878 0.88315217]
PCA 변환 데이터 CV 평균 accuracy: 0.8869766505636071

Component 개수: 10
PCA 변환 데이터 CV accuracy: [0.90243902 0.87262873 0.88315217]
PCA 변환 데이터 CV 평균 accuracy: 0.88607330819685

</pre>
PCA 변환을 하는 경우 components의 개수는 6개가 적당하지만, 이는 시각화 측면에서 유리하지 않으며, <br>

성능면에서도 PCA 변환 데이터보다 원본 데이터를 이용하는 것이 우수하므로 PCA 변환을 적용하는 것이 바람직하지 않음 <br>

(원본 데이터보다 PCA 변환 데이터가 성능이 더 좋은 경우는 feature들 간의 상관관계가 높은 다중공선성 문제가 있는 경우)


**Random Forest**



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(random_state=100)
scores = cross_val_score(rfc, data_ohe.iloc[:, :22], data_ohe['Classification1'],
                        scoring='accuracy', cv=3)

print('Random Forest CV accuracy:', scores)
print('Random Forest CV 평균 accuracy:', np.mean(scores))
```

<pre>
Random Forest CV accuracy: [0.95663957 0.95121951 0.95108696]
Random Forest CV 평균 accuracy: 0.952982011704175
</pre>

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_ohe.iloc[:, :22], data_ohe['Classification1'],
                                                   test_size=0.2, random_state=100)
rfc = RandomForestClassifier(random_state=100)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Random Forest Classifier Accuracy: {0:.4f}'.format(accuracy))
```

<pre>
Random Forest Classifier Accuracy: 0.9595
</pre>

```python
import seaborn as sns

rfc.fit(data_ohe.iloc[:, :22], data_ohe['Classification1'])
ftr_importances_values = rfc.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=data_ohe.iloc[:, :22].columns)
ftr_importances = ftr_importances.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
plt.title('Feature importances')
sns.barplot(x=ftr_importances, y=ftr_importances.index)
plt.show()
```

<img src="data:image/png;base64,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9HCI98qVK2rTpo22b9+utLS0mzrXQw895LBdq1YtWSwWtW3b1l7n5uamqlWrOjx6mKVr166yWCz27dDQUN1zzz1av369JCkpKUnHjx9Xt27dVKzY/33l8PX1VceOHbV169Zsj6P9dfzXk5CQoIcfflhBQUEqXry43N3d1b17d2VmZmrfvn0ObYODg9W4cWOHurp16zqM7auvvlL16tXVsmXLHM+5YsUKBQQEqH379g7XpH79+goODrb/DtWvX18eHh569tlnNWvWLIcZNwAFAws8AAAKpVq1ajld4OF///ufJKlRo0ZOj7v6S33Xrl21du1avf7662rUqJH8/f1lsVj04IMP2h+1utWyHi/7a7xZj6I5c/r0afn4+NzwuQIDAx22PTw85O3tLU9Pz2z1qamp2Y4PDg52Wvfjjz9Kkk6dOiUp+5ikP1cmvHLlis6cOeOwiIOztjk5evSo7rvvPtWoUUP/+te/FBYWJk9PT3333Xfq169ftmsUFBSUrQ+r1erQ7rffflPFihWved7//e9/Onv2rDw8PJzuz3pEs0qVKvrmm280duxY9evXT2lpaapcubIGDhyoF198MdfjBOA6JEsAgCKlZMmSkqSFCxcqNDQ0x3YpKSlasWKFRowYoddee81en56ertOnT+f6fJ6enkpPT89W//vvv9tjudrVMzVXxzt58uQcV7UrU6ZMruO5lU6cOOG0LispyfrfrHd9rnb8+HEVK1ZMJUqUcKj/6/ivZenSpUpLS9PixYsdruXOnTtz3cdflSpVSr/++us125QsWVJBQUFatWqV0/1+fn72n++77z7dd999yszM1I4dOzR58mQNGjRIZcqUUZcuXW46TgC3B8kSAKBIiYqKkpubmw4cOHDNR74sFouMMbJarQ71H330kTIzMx3qsto4m20KCwvTrl27HOr27dunpKQkp8nSXzVt2lQBAQHas2eP+vfvf932t9PcuXM1ePBge4Jz5MgRbd68Wd27d5ck1ahRQ+XLl9ecOXP08ssv29ulpaVp0aJF9hXyrufqz9fLy8ten9Xf1dfIGKMZM2bc9Jjatm2rN954Q+vWrdP999/vtM1DDz2kefPmKTMzU02aNMlVv8WLF1eTJk1Us2ZNzZ49Wz/88APJElAAkCwBAIqUsLAwvfXWWxo+fLgOHjyoNm3aqESJEvrf//6n7777Tj4+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00+SJGOMLBbLDZ83JiZGNpvNXkJCQm7ZmAAAAOBaJEsoFHx8fFS1alXVrVtXkyZNUnp6ut58802HNkFBQercubPGjx+vxMRElStXTuPGjZMkVa9eXQcOHFBGRsYNnTc6OlopKSn2cuzYsVs2JgAAALgWyRIKpREjRmjcuHE6fvy40/0eHh6qUqWK0tLSJEldu3bV+fPn9eGHHzptn9MCD1arVf7+/g4FAAAAhQPvLKFQioyM1B133KHRo0erTZs2mjdvnrp06aLq1avLGKPly5dr5cqViouLkyQ1adJEr7zyioYMGaL//ve/evTRR1WuXDn98ssvmjp1qu69916HVfIAAABQ+JEsodAaPHiwnn76aXXu3Fne3t4aMmSIjh07JqvVqmrVqumjjz5St27d7O3fffddNWjQQLGxsZo6daquXLmiKlWqqFOnTurRo4cLRwIAAABXsJic1lIGcMNSU1Nls9k0esb98vTm3yJutZe6fu3qEAAAQAGX9X0tJSXluq9Q8M4SAAAAADhBsgQAAAAATvCcEJAH+j22hJXxAAAACjhmlgAAAADACZIlAAAAAHCCZAkAAAAAnOCdJSAPRH/xD1lZOvyWmtBxlatDAAAARQwzSwAAAADgBMkSAAAAADhBsgQAAAAATpAsId/p2bOnHnnkkWz18fHxslgsOnv2rCTJGKPp06erSZMm8vX1VUBAgBo2bKj3339ff/zxh/2406dPa9CgQQoLC5OHh4fKli2rp59+WkePHnXof8qUKapbt678/f3l7++viIgIffXVV3k5VAAAAORjJEsosLp166ZBgwapQ4cOWr9+vXbu3KnXX39dX3zxhVavXi3pz0Tp7rvv1jfffKMPP/xQv/zyiz7//HMdOHBAjRo10sGDB+39VahQQWPGjNGOHTu0Y8cO3X///erQoYN2797tqiECAADAhViuCwXS/PnzNXv2bC1dulQdOnSw14eFhenhhx9WamqqJGn48OE6fvy4fvnlFwUHB0uSKlasqK+//lrVqlVTv3797LNH7du3dzjHO++8oylTpmjr1q264447btPIAAAAkF8ws4QCafbs2apRo4ZDopTFYrHIZrPpypUrmjdvnp588kl7opTFy8tLffv21ddff63Tp09n6yMzM1Pz5s1TWlqaIiIi8mwcAAAAyL+YWUK+tGLFCvn6+jrUZWZm2n/e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c2dJUkZGhqZOnaoqVapIkvr376+33nrL3s/kyZMVHR2tRx99VJL0wQcfaOXKlbmK4dy5c5o1a5bmzJmjBx54QJIUFxencuVy/htGR48eVXBwsFq2bCl3d3dVrFhRjRs3zrF9enq60tPT7dupqam5ig0AAAD5HzNLyBOJiYlyc3NTkyZN7HVBQUGqUaOGEhMT7XXe3t72REmSypYtq5MnT0qSUlJS9L///c8hWSlevLgaNGiQqxgOHjyojIwMh+NtNptq1KiR4zGdO3fWhQsXVLlyZfXp00dLlizR5cuXc2wfExMjm81mLyEhIbmKDQAAAPkfyRLyhDEmx3qLxWLfdnd3d9hvsViyHXt1+2v1nVMMN3J8SEiIkpKSFBsbKy8vL/Xt21fNmjVzeHTwatHR0UpJSbGXY8eO5So2AAAA5H8kS8gT4eHhunz5srZt22avO3XqlPbt26datWrlqg+bzaYyZcrou+++s9dlZmYqISEhV8dXqVJF7u7uDsenpqY6LDLhjJeXlx5++GFNmjRJ8fHx2rJli3766Senba1Wq/z9/R0KAAAACgfeWUKeqFatmjp06KA+ffpo2rRp8vPz02uvvaby5curQ4cOue5nwIABiomJUdWqVVWzZk1NnjxZZ86cyTZb5Iyfn5969OihoUOHKjAwUKVLl9aIESNUrFixHI+fOXOmMjMz1aRJE3l7e+vTTz+Vl5eXQkNDcx0zAAAACgdmlpBn4uLi1KBBAz300EOKiIiQMUYrV67M9ujdtbz66qt64okn1L17d0VERMjX11dRUVHy9PTM1fETJkxQRESEHnroIbVs2VJNmzZVrVq1cjw+ICBAM2bMUNOmTVW3bl2tXbtWy5cvV1BQUK5jBgAAQOFgMbl9AQTIB65cuaJatWrpscce06hRo274+LS0NJUvX17jx49Xr169bnl8qampstlsGvbZXnl6+910P68/mvOKfQAAALh5Wd/XUlJSrvsKBY/hIV87cuSIVq9erebNmys9PV0ffPCBDh06pK5du+bq+ISEBO3du1eNGzdWSkqKfVnyG3kUEAAAAEUTyRLytWLFimnmzJl6+eWXZYxR7dq19c0336hWrVo6evSowsPDczx2z549kqRx48YpKSlJHh4eatCggTZu3KiSJUveriEAAACggOIxPBRYly9f1uHDh3PcHxYWJje32/vvATcy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"/>

**XGBoost**



```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data_ohe['Classification1'] = le.fit_transform(data_ohe['Classification1'])

X_train, X_test, y_train, y_test = train_test_split(data_ohe.iloc[:, :22], data_ohe['Classification1'],
                                                   test_size=0.2, random_state=100)
```


```python
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.01, max_depth=3)
xgb.fit(X_train, y_train, verbose=True)
pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('XGBoost Classifier Accuracy: {0:.4f}'.format(accuracy))
```

<pre>
XGBoost Classifier Accuracy: 0.9550
</pre>
### RSC값 예측 

**RSC값 예측으로 Classification.1 분류 가능**



```python
# 타겟값 분포도 확인
sns.histplot(data_ohe['RSC'], kde=True)
```

<pre>
<AxesSubplot:xlabel='RSC', ylabel='Count'>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABI9klEQVR4nO3deXxU1d0/8M+dNfuErJNAEgIGEMIiIFC0ZVHRVPFR3Kg+FluldYFqwceW2j5gX1WqrWh/qMU+Vdyw0FZFrVZkR8u+hCVsAbKRlayTyTLr+f0xmSFDEsgykztz5/N+ve7rxdx75873aGA+OfeceyQhhAARERGRQqnkLoCIiIjInxh2iIiISNEYdoiIiEjRGHaIiIhI0Rh2iIiISNEYdoiIiEjRGHaIiIhI0TRyFxAInE4nysrKEB0dDUmS5C6HiIiIukEIgcbGRqSmpkKl6rr/hmEHQFlZGdLS0uQug4iIiHqhpKQEgwYN6vI4ww6A6OhoAK7/WDExMTJXQ0RERN1hMpmQlpbm+R7vCsMO4Ll1FRMTw7BDREQUZK40BIUDlImIiEjRGHaIiIhI0Rh2iIiISNEYdoiIiEjRGHaIiIhI0Rh2iIiISNEYdoiIiEjRGHaIiIhI0Rh2iIiISNEYdoiIiEjRGHaIiIhI0Rh2iIiISNEYdoiIiEjRGHaIiIhI0TRyF0BERNQdVqsVubm5XvvGjRsHnU4nT0EUNBh2iIgoKOTm5uKVv29GckYWAKCyKB8/BzBp0iR5C6OAx7BDRERBIzkjCxkjxshdBgUZWcfs7NixA7Nnz0ZqaiokScL69eu9jkuS1On2hz/8wXPO9OnTOxyfO3duP7eEiIiIApWsYaepqQljx47Fa6+91unx8vJyr+3tt9+GJEm46667vM6bP3++13lvvvlmf5RPREREQUDW21g5OTnIycnp8rjRaPR6/emnn2LGjBkYMmSI1/6IiIgO5xIREREBQTT1vLKyEl988QUefvjhDsfWrFmDhIQEjBo1Ck8//TQaGxsvey2LxQKTyeS1ERERkTIFzQDld999F9HR0ZgzZ47X/gceeACZmZkwGo04duwYlixZgsOHD2Pjxo1dXmv58uV47rnn/F0yERERBYCgCTtvv/02HnjgAYSFhXntnz9/vufP2dnZyMrKwsSJE3Hw4EGMHz++02stWbIEixYt8rw2mUxIS0vzT+FEREQkq6AIO9988w1OnTqFdevWXfHc8ePHQ6vVIj8/v8uwo9frodfrfV0mERERBaCgGLPz1ltvYcKECRg7duwVz83Ly4PNZkNKSko/VEZERESBTtaeHbPZjDNnznheFxQUIDc3F3FxcUhPTwfgusX0j3/8Ay+//HKH9589exZr1qzB97//fSQkJOD48eNYvHgxrrnmGlx33XX91g4iIiIKXLKGnf3792PGjBme1+5xNPPmzcM777wDAFi7di2EEPjBD37Q4f06nQ6bN2/Gn/70J5jNZqSlpeHWW2/F0qVLoVar+6UNREREFNhkDTvTp0+HEOKy5/zkJz/BT37yk06PpaWlYfv27f4ojYiIiBQiKMbsEBEREfUWww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpGsMOERERKRrDDhERESkaww4REREpmqxhZ8eOHZg9ezZSU1MhSRLWr1/vdfyhhx6CJEle25QpU7zOsVgsWLhwIRISEhAZGYnbb78d58+f78dWEBERUSCTNew0NTVh7NixeO2117o855ZbbkF5ebln+/LLL72OP/XUU/jkk0+wdu1afPvttzCbzbjtttvgcDj8XT4REREFAY2cH56Tk4OcnJzLnqPX62E0Gjs91tDQgLfeegvvv/8+brzxRgDABx98gLS0NGzatAk333yzz2smIiKi4BLwY3a2bduGpKQkDBs2DPPnz0dVVZXn2IEDB2Cz2TBr1izPvtTUVGRnZ2Pnzp1dXtNiscBkMnltREREpEwBHXZycnKwZs0abNmyBS+//DL27duHmTNnwmKxAAAqKiqg0+kwYMAAr/clJyejoqKiy+suX74cBoPBs6Wlpfm1HURERCQfWW9jXcl9993n+XN2djYmTpyIjIwMfPHFF5gzZ06X7xNCQJKk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"/>


```python
X_features2 = data_ohe.drop(['RSC', 'Classification1', 'Classification2'], axis=1, inplace=False)
X_features2.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>pH</th>
      <th>E.C</th>
      <th>TDS</th>
      <th>CO3</th>
      <th>HCO3</th>
      <th>Cl</th>
      <th>F</th>
      <th>...</th>
      <th>SO4</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
      <th>season_Post-monsoon 2020</th>
      <th>season_post monsoon 2019</th>
      <th>season_postmonsoon 2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>8.28</td>
      <td>745</td>
      <td>476.80</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>60</td>
      <td>0.44</td>
      <td>...</td>
      <td>46.0</td>
      <td>49.0</td>
      <td>4.0</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>8.29</td>
      <td>921</td>
      <td>589.44</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>80</td>
      <td>0.56</td>
      <td>...</td>
      <td>68.0</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>7.69</td>
      <td>510</td>
      <td>326.40</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>30</td>
      <td>0.66</td>
      <td>...</td>
      <td>44.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>8.09</td>
      <td>422</td>
      <td>270.08</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>10</td>
      <td>0.58</td>
      <td>...</td>
      <td>35.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>8.21</td>
      <td>2321</td>
      <td>1485.44</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>340</td>
      <td>2.56</td>
      <td>...</td>
      <td>280.0</td>
      <td>298.0</td>
      <td>5.0</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



```python
X_train, X_test, y_train, y_test = train_test_split(X_features2, data_ohe['RSC'],
                                                   test_size=0.2, random_state=100)
```

**Regularized Linear Model - Ridge Regression**



```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)

rmse = mean_squared_error(y_test, pred, squared=False)

print("RMSE: {:.3f}".format(rmse))
```

<pre>
RMSE: 0.078
</pre>

```python
# alpha값에 따른 RMSE
alphas = [0, 0.1, 1, 10, 50, 100, 200]
coeff_df = pd.DataFrame()

for i in alphas:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)

    print("alpha = {0}일 때, RMSE: {1:.5f}".format(i, rmse))
    
    coeff = pd.Series(data=ridge.coef_, index=X_train.columns)
    colname = 'alpha:' + str(i)
    coeff_df[colname] = coeff
    
coeff_df
```

<pre>
alpha = 0일 때, RMSE: 0.19638
alpha = 0.1일 때, RMSE: 0.19614
alpha = 1일 때, RMSE: 0.19541
alpha = 10일 때, RMSE: 0.19265
alpha = 50일 때, RMSE: 0.18914
alpha = 100일 때, RMSE: 0.18857
alpha = 200일 때, RMSE: 0.18882
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha:0</th>
      <th>alpha:0.1</th>
      <th>alpha:1</th>
      <th>alpha:10</th>
      <th>alpha:50</th>
      <th>alpha:100</th>
      <th>alpha:200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lat_gis</th>
      <td>-2.204223e-02</td>
      <td>-0.022069</td>
      <td>-0.022067</td>
      <td>-0.021854</td>
      <td>-0.020579</td>
      <td>-0.019042</td>
      <td>-0.016546</td>
    </tr>
    <tr>
      <th>long_gis</th>
      <td>-1.836327e-02</td>
      <td>-0.018266</td>
      <td>-0.018339</td>
      <td>-0.018608</td>
      <td>-0.018498</td>
      <td>-0.017591</td>
      <td>-0.015592</td>
    </tr>
    <tr>
      <th>gwl</th>
      <td>-3.195652e-03</td>
      <td>-0.003183</td>
      <td>-0.003155</td>
      <td>-0.003091</td>
      <td>-0.002894</td>
      <td>-0.002693</td>
      <td>-0.002403</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-4.832663e-02</td>
      <td>-0.048215</td>
      <td>-0.047976</td>
      <td>-0.043822</td>
      <td>-0.031466</td>
      <td>-0.023284</td>
      <td>-0.015255</td>
    </tr>
    <tr>
      <th>E.C</th>
      <td>7.915002e-04</td>
      <td>0.000792</td>
      <td>0.000792</td>
      <td>0.000794</td>
      <td>0.000799</td>
      <td>0.000804</td>
      <td>0.000812</td>
    </tr>
    <tr>
      <th>CO3</th>
      <td>3.770137e-03</td>
      <td>0.003473</td>
      <td>0.002347</td>
      <td>0.001394</td>
      <td>0.001097</td>
      <td>0.000985</td>
      <td>0.000902</td>
    </tr>
    <tr>
      <th>HCO3</th>
      <td>4.786286e-03</td>
      <td>0.004491</td>
      <td>0.003369</td>
      <td>0.002471</td>
      <td>0.002325</td>
      <td>0.002300</td>
      <td>0.002282</td>
    </tr>
    <tr>
      <th>Cl</th>
      <td>1.815244e-03</td>
      <td>0.001819</td>
      <td>0.001817</td>
      <td>0.001807</td>
      <td>0.001778</td>
      <td>0.001753</td>
      <td>0.001718</td>
    </tr>
    <tr>
      <th>F</th>
      <td>3.085268e-02</td>
      <td>0.030748</td>
      <td>0.030739</td>
      <td>0.030227</td>
      <td>0.027973</td>
      <td>0.025578</td>
      <td>0.021863</td>
    </tr>
    <tr>
      <th>NO3</th>
      <td>1.492468e-03</td>
      <td>0.001495</td>
      <td>0.001495</td>
      <td>0.001489</td>
      <td>0.001471</td>
      <td>0.001456</td>
      <td>0.001435</td>
    </tr>
    <tr>
      <th>SO4</th>
      <td>2.296756e-03</td>
      <td>0.002302</td>
      <td>0.002300</td>
      <td>0.002295</td>
      <td>0.002278</td>
      <td>0.002262</td>
      <td>0.002238</td>
    </tr>
    <tr>
      <th>Na</th>
      <td>-4.986169e-03</td>
      <td>-0.004996</td>
      <td>-0.004993</td>
      <td>-0.004982</td>
      <td>-0.004954</td>
      <td>-0.004929</td>
      <td>-0.004890</td>
    </tr>
    <tr>
      <th>K</th>
      <td>-2.875159e-03</td>
      <td>-0.002882</td>
      <td>-0.002881</td>
      <td>-0.002877</td>
      <td>-0.002867</td>
      <td>-0.002865</td>
      <td>-0.002868</td>
    </tr>
    <tr>
      <th>Ca</th>
      <td>-6.095186e-02</td>
      <td>-0.060656</td>
      <td>-0.059517</td>
      <td>-0.049981</td>
      <td>-0.029292</td>
      <td>-0.019419</td>
      <td>-0.011726</td>
    </tr>
    <tr>
      <th>Mg</th>
      <td>-9.973387e-02</td>
      <td>-0.099246</td>
      <td>-0.097368</td>
      <td>-0.081696</td>
      <td>-0.047711</td>
      <td>-0.031506</td>
      <td>-0.018887</td>
    </tr>
    <tr>
      <th>T.H</th>
      <td>2.006669e-02</td>
      <td>0.020243</td>
      <td>0.020908</td>
      <td>0.017996</td>
      <td>0.009880</td>
      <td>0.005964</td>
      <td>0.002911</td>
    </tr>
    <tr>
      <th>SAR</th>
      <td>1.733916e-02</td>
      <td>0.017338</td>
      <td>0.017309</td>
      <td>0.017052</td>
      <td>0.016339</td>
      <td>0.015779</td>
      <td>0.014913</td>
    </tr>
    <tr>
      <th>RSC</th>
      <td>-1.251003e-01</td>
      <td>-0.110162</td>
      <td>-0.054127</td>
      <td>-0.009314</td>
      <td>-0.002186</td>
      <td>-0.001141</td>
      <td>-0.000570</td>
    </tr>
    <tr>
      <th>season_Post-monsoon 2020</th>
      <td>6.093422e+11</td>
      <td>-0.027065</td>
      <td>-0.026997</td>
      <td>-0.026210</td>
      <td>-0.023088</td>
      <td>-0.020047</td>
      <td>-0.015858</td>
    </tr>
    <tr>
      <th>season_post monsoon 2019</th>
      <td>6.093422e+11</td>
      <td>0.008846</td>
      <td>0.009040</td>
      <td>0.009000</td>
      <td>0.008163</td>
      <td>0.007247</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>season_postmonsoon 2018</th>
      <td>6.093422e+11</td>
      <td>0.018219</td>
      <td>0.017957</td>
      <td>0.017210</td>
      <td>0.014925</td>
      <td>0.012800</td>
      <td>0.009986</td>
    </tr>
  </tbody>
</table>
</div>


**회귀 트리**



```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

rf = RandomForestRegressor(n_estimators=500, random_state=0)
gb = GradientBoostingRegressor(n_estimators=500, random_state=0)
xgb = XGBRegressor(n_estimators=500)
lgb = LGBMRegressor(n_estimators=500)

models = [rf, gb, xgb, lgb]

for model in models: 
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)

    print("model: {}, RMSE: {:.3f}".format(model.__class__.__name__, rmse))
```

<pre>
model: RandomForestRegressor, RMSE: 0.706
model: GradientBoostingRegressor, RMSE: 0.462
model: XGBRegressor, RMSE: 0.546
model: LGBMRegressor, RMSE: 1.116
</pre>

```python
# Feature importances

gb = GradientBoostingRegressor(n_estimators=500, random_state=0)
gb.fit(X_features2, data_ohe['RSC'])

feature_series = pd.Series(data=gb.feature_importances_, index=X_features2.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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UdXD5JkqnQULFiA4OBgvvfQSOnfujJCQEAQEBMDOzg7An0s68u8uP//88wCAZs2aQafToWXLlnB0dCy32ImIiKhiU8nDi0GJKqE7d+6gbt26mD9/PoYNG1aq1zIYDNDpdNDr9UzCiYiILERx3r/5gUGqdBITE3H69Gm0bdsWer0eH330EQCgV69e5RwZERERWTomz1QpzZs3D6mpqbC1tUVQUBD27t2LmjVrlndYREREZOGYPFOl07JlS/z666/lHQYRERFVQvzAIBERERGRmZg8ExERERGZickzEREREZGZmDwTEREREZmJyTNZLJVK9cgSHh5eoJ29vT18fHwQHh5e6IcKly5diubNm8Pe3h5OTk5o2bIl5syZU8YjIyIiooqKu22QxcrKylK+Xr9+PT788EOkpqYqdRqNRvk6NjYW3bp1w71795CWloZly5ahXbt2WLFiBYYMGQIA+PLLLzFhwgQsXLgQHTp0gNFoxPHjx5GcnFx2gyIiIqIKjckzWSxXV1fla51OB5VKZVL3MCcnJ+VYgwYN0LVrVwwdOhRjxoxBz549UaNGDXz33Xd49dVXTZ5C2Lhx49IdBBEREVkULtugKusf//gHbt26hR07dgD4Mxk/ePAgLly4YHYfRqMRBoPBpBAREVHlxeSZqix/f38AwPnz5wEAkZGRcHJyQoMGDeDn54fw8HBs2LABeXl5RfYRFRUFnU6nFA8Pj7IInYiIiMoJk2eqskQEwJ8fKAQANzc3HDhwACdOnMC4ceOQk5ODoUOHolu3bkUm0BEREdDr9UrJzMwss/iJiIio7HHNM1VZKSkpAAAvLy+T+iZNmqBJkyYYPXo09u3bh+effx579uxBx44dC/ShVquhVqvLJF4iIiIqf7zzTFXWp59+CkdHR3Tu3LnINoGBgQCAO3fulFVYREREVIHxzjNVCTdv3sSVK1dgNBqRlpaGpUuXYuvWrfj3v/8NJycnAMDbb78Nd3d3vPDCC6hXrx6ysrLw8ccfo1atWggODi7fARAREVGFwOSZqoTXX38dAGBnZ4e6deviueeew+HDh9GqVSulTefOnbFixQosWbIE169fR82aNREcHIydO3fCxcWlvEInIiKiCkQl+Z+aIqKnZjAYoNPpoNfr4ejoWN7hEBERkRmK8/7NNc9ERERERGZi8kxEREREZCYmz0REREREZmLy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"/>

Gradient Boosting Regressor Model보다 Linear Regression Model이 더 적절


### TDS값 예측



```python
# 타겟값 분포도 확인
sns.histplot(data_ohe['TDS'], kde=True)
```

<pre>
<AxesSubplot:xlabel='TDS', ylabel='Count'>
</pre>
<img src="data:image/png;base64,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PXq0TFURERGRUoR9yw4ALFiwAPfeey+GDx+OUaNGYdWqVTh9+jQeffRRuUsjIiIimUVE2PnFL36ByspKPP/88ygtLUVubi4++ugj9OjRI2i/U6/XY/HixS26w6gl3quO4f3qGN6vjuH96hjeryun5HslCXG5+VpERERE4Svsx+wQERERtYdhh4iIiCIaww4RERFFNIYdIiIiimgMO53wyiuvIDs7GzExMRg2bBi++OILuUsKum3btmHatGnIzMyEJEn44IMP/M4LIZCXl4fMzEwYDAZMmDABRUVFftfYbDbMmTMHqampiIuLw+23344zZ874XVNVVYV7770XJpMJJpMJ9957L6qrq4P86QIrPz8fI0aMgNFoRFpaGqZPn46jR4/6XcP7ddHKlStx7bXX+hYiGzVqFD7++GPfed6r9uXn50OSJMyfP993jPfsory8PEiS5Pcwm82+87xXLZ09exb33HMPUlJSEBsbiyFDhqCwsNB3Pizv2VVuTRV11q1bJ7RarXj99dfF4cOHxbx580RcXJw4deqU3KUF1UcffSQWLVok3nvvPQFArF+/3u/873//e2E0GsV7770nDh48KH7xi1+IjIwMYbVafdc8+uijomvXrqKgoEDs3btX3HjjjWLw4MHC6XT6rrnllltEbm6u2LFjh9ixY4fIzc0VU6dODdXHDIibb75ZrF69Whw6dEjs379f3HbbbaJ79+6irq7Odw3v10UbNmwQ//73v8XRo0fF0aNHxa9+9Suh1WrFoUOHhBC8V+35+uuvRc+ePcW1114r5s2b5zvOe3bR4sWLxcCBA0VpaanvUV5e7jvPe+XvwoULokePHuL+++8XX331lSguLhabNm0SJ06c8F0TjveMYaeDrr/+evHoo4/6HevXr5949tlnZaoo9H4YdtxutzCbzeL3v/+971hTU5MwmUzi1VdfFUIIUV1dLbRarVi3bp3vmrNnzwqVSiU++eQTIYQQhw8fFgDErl27fNfs3LlTABDffvttkD9V8JSXlwsAYuvWrUII3q8rkZSUJP7617/yXrWjtrZW5OTkiIKCAjF+/Hhf2OE987d48WIxePDgVs/xXrX0zDPPiLFjx7Z5PlzvGbuxOsBut6OwsBBTpkzxOz5lyhTs2LFDpqrkV1xcDIvF4ndf9Ho9xo8f77svhYWFcDgcftdkZmYiNzfXd83O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"/>


```python
# 로그 변환 후 타겟값 분포도 확인
log_y = np.log1p(data_ohe['TDS'])
sns.histplot(log_y, kde=True)
```

<pre>
<AxesSubplot:xlabel='TDS', ylabel='Count'>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABazUlEQVR4nO3deXhU5cH+8e/MJJnsC9lDEggQtgRkFcEFKIIbWuur1ipWa+trq1XR1u1nq9S28KqtYrXa2rdVrKK+VbF2cQEVXBDZ90BYQgZCtsm+zWSZ8/sjkBpJIAlJzszk/lzXXMrMmfE+HpLcOec5z2MxDMNARERExE9ZzQ4gIiIi0pdUdkRERMSvqeyIiIiIX1PZEREREb+msiMiIiJ+TWVHRERE/JrKjoiIiPi1ALMDeAOPx8PRo0eJiIjAYrGYHUdERES6wDAMampqSElJwWrt/PyNyg5w9OhR0tLSzI4hIiIiPXD48GFSU1M7fV1lB4iIiABa/2dFRkaanEZERES6orq6mrS0tLaf451R2YG2S1eRkZEqOyIiIj7mVENQNEBZRERE/JrKjoiIiPg1U8vOJ598wqWXXkpKSgoWi4W333677bWmpibuu+8+xo0bR1hYGCkpKXz3u9/l6NGj7T7D7XZz++23ExcXR1hYGJdddhlHjhzp5z0RERERb2Vq2amrq+OMM87gmWeeOeG1+vp6Nm/ezM9//nM2b97MW2+9RW5uLpdddlm77RYuXMiKFSt47bXX+Oyzz6itrWX+/Pm0tLT0126IiIiIF7MYhmGYHQJaBxetWLGCyy+/vNNtNmzYwJlnnkl+fj7p6elUVVURHx/PX//6V7797W8D/7mN/N///jcXXHBBl/7b1dXVREVFUVVVpQHKIiIiPqKrP799asxOVVUVFouF6OhoADZt2kRTUxPz5s1r2yYlJYXs7GzWrl3b6ee43W6qq6vbPURERMQ/+UzZcblc3H///Vx77bVt7a2oqIigoCBiYmLabZuYmEhRUVGnn7VkyRKioqLaHppQUERExH/5RNlpamrimmuuwePx8Oyzz55ye8MwTnrP/QMPPEBVVVXb4/Dhw70ZV0RERLyI15edpqYmrr76avLy8li5cmW7a3JJSUk0NjZSUVHR7j0lJSUkJiZ2+pl2u71tAkFNJCgiIuLfvLrsHC86+/btY9WqVcTGxrZ7ffLkyQQGBrJy5cq25woLC9m5cyczZszo77giIiLihUxdLqK2tpb9+/e3/TkvL4+tW7cyaNAgUlJSuPLKK9m8eTP//Oc/aWlpaRuHM2jQIIKCgoiKiuL73/8+P/nJT4iNjWXQoEH89Kc/Zdy4cZx//vlm7ZaIiIh4EVNvPV+9ejWzZ88+4fkbbriBRYsWkZGR0eH7Pv74Y2bNmgW0Dly+5557WL58OQ0NDcyZM4dnn322W4OOdeu5iIiI7+nqz2+vmWfHTCo7IiIivscv59kRERER6S5Tx+yIiDgcDpxO52l9RlxcHOnp6b2USET8jcqOiJjG4XAweswYGurrT+tzQkJD2ZOTo8IjIh1S2RER0zidThrq67nuvsdJTB/eo88odhzglUfvwel0quyISIdUdkTEdInpw0nNzDI7hoj4KQ1QFhEREb+msiMiIiJ+TWVHRERE/JrKjoiIiPg1lR0RERHxayo7IiIi4tdUdkRERMSvqeyIiIiIX1PZEREREb+msiMiIiJ+TWVHRERE/JrKjoiIiPg1lR0RERHxayo7IiIi4tdUdkRERMSvBZgdQESku1o8BvlldZTWuDnitBE3/6e8sbuGmvBSzh0Rh9VqMTuiiHgRlR0R8RnNLR52Ha1mk6OCGlfzsWdthGXNYvnOWpbvXM/opAgWnj+SC7ISsVhUekREl7FExEeU1zXy+sbDrM4tpcbVTEigjTHJEWRHN1Ox+kVmDgkhwh7AnqIafvjyJu54bSuuphazY4uIF1DZERGvt7eohlfXO3DWNhISaGP2qHhuOnso88YmMSrSQ/WXb3DntGg+vW82t80eToDVwj+2HeX6P39JRV2j2fFFxGQqOyLi1XYUVPHeriKaPQapMSFcNy2d8anRBNhO/PYVHRrEPReMZtlNZxIRHMCGQxUs+POX1Dc2d/DJIjJQqOyIiNfadriSj/aUAHBGahTfmjiYMPuphxqePSKON380g9iwIHYdreaev23HMIy+jisiXkplR0S80p6ialbnlgIwOT2GmSPjsXZjwPHIxAj+cP1kAm0W/rWjkKc/2t9XUUXEy6nsiIjXKahoYNXu1jM6E9OjOXtEbI/urJo6dBC//GY2AE+uymVTfkWv5hQR36CyIyJepbK+kX9uP0qLYTAiPpxzR8Sd1i3k15yZzhWTBmMY8MBb22ls9vRiWhHxBSo7IuI1mls8/GtHIa5mD4mR9l6bK+fnl4wlNiyI3OJanv/kQC8kFRFforIjIl7j472lbbeXzx+f0uEdVz0RExbEQ5eOBeB3H+0nz1nXK58rIr5BMyiLiFfYfbSa3YXVWICLspMI78JdV1+Vk5Nz0tdTDYMJiUFsLW7kZ6+v4yfTY9pec7vd2O32nsRuExcXR3p6+ml9hoj0DZUdETFdbTOszm0dkHzWsFjSBoV2+b3V5a13bC1YsOCU2wbGDyXlpmf4/LCLvz1yJU0lecdesQCnd2t6SGgoe3JyVHhEvJDKjoiYy2JlU1kATS0Gg6NDmDI05tTv+YqG2moALrnlQUaNn3zK7dc7Wzhcb2PSj57i7IRmctav4d1lT3X5/R0pdhzglUfvwel0quyIeCGVHRExVeTUy3G6rQTaLMwdm9ituXS+KjZlCKmZWafcLmxwI39dl0+Ry4o1fhiDkg506/0i4ns0QFlETJNf2UT0udcDcN7IeKJCAvv8vxkTGkRWciQA6/PK+/y/JyLmU9kREVO4m1t4an0lloBAkkM8bQWkP0wZOggLkF9eTx2nNzBZRLyfyo6ImOKpVfs4VNlMS30VkwY198p8Ol0VFRLI8PhwAI4yqN/+uyJiDpUdEel3m/LL+cOa1rEyZe8/Q7Ct/zNMGhINQAmRWMOi+z+AiPQblR0R6Vd17mbu/r9teAyYNSSEhtwvTMmRHBVCclQwBlYiJs03JYOI9A+VHRHpV4v/nUN+WT0pUcF8f2L/jdPpyMT0aAAiJlyE5/Sm2RERL6ayIyL95uO9JbzypQOA31x1BmFB5n4LGh4XThBN2EKjcLYEm5pFRPqOyo6I9IuKukbue2M7AN87eygzRsSZnAisVguJVAFQ2BxmchoR6SsqOyLSL37+952U1LgZHh/GfReONjtOm0QqMQwPlR47lfWNZscRkT6gsiMife7vWwv45/ZCbFYLT357AsGBJtx+1YlgmnDlbQFg19Fqk9OISF9Q2RGRPlVU5eLnb+8E4PZvjGB8arS5gTpQu+19AHYXVtOikcoifkdrY4lIjzkcDpxOZ6evG4bBLz8pp9rVzPCYQGZE17B58+a213Nycvoj5inV7/+SQFqobwRHeT0ZcRq/I+JPVHZEpEccDgejx4yhob6+023CJ15M7Lxb8TS5+eSxHzLt/iMdbldbW9tXMbvG00JCQAMFzeHsKapW2RHxMyo7ItIjTqeThvp6rrvvcRLTh5/wenUTfFQUSIsBExNsXPXL352wTc76Nby77ClcLld/RD6pBFtr2TlYWkdjs4egAF3lF/EXpn41f/LJJ1x66aWkpKRgsVh4++23271uGAaLFi0iJSWFkJAQZs2axa5du9pt43a7uf3224mLiyMsLIzLLruMI0c6/u1RRHpfYvpwUjOz2j2Sho1hc004LYaF9EGhzJw4+oRtUjOzGJSUanb8NhHWJqJDAmn2GBwoNflMk4j0KlPLTl1dHWeccQbPPPNMh68/9thjPPHEEzzzzDNs2LCBpKQk5s6dS01NTds2CxcuZMWKFbz22mt89tln1NbWMn/+fFpaWvprN0Tkaz7d56SstpGQQBvzxib26yKfPWWxwOikCAD2FNWcYmsR8SWmXsa66KKLuOiiizp8zTAMli5dyoMPPsgVV1wBwLJly0hMTGT58uXccsstVFVV8ec//5m//vWvnH/++QC8/PLLpKWlsWrVKi644IJ+2xcRabW/pJbtBa0T9V2QlUiY3Xeulo9KimBdXjmHy+upczf7VHYR6ZzXXpTOy8ujqKiIefPmtT1nt9uZOXMma9euBWDTpk00NTW12yYlJYXs7Oy2bTridruprq5u9xCR01ftamJVTjEAk9NjGBLrWwN9o0ODSIoMxgD2lehSloi/8NqyU1RUBEBiYmK75xMTE9teKyoqIigoiJiYmE636ciSJUuIiopqe6SlpfVyepGBx+MxeH9nEe5mD4mRdqYPjzU7Uo+MTAwHYF+JLmWJ+AuvLTvHff1av2EYp7z+f6ptHnjgAaqqqtoehw8f7pWsIgPZl4fKOVrlIshm5aLsZGxW7x+n05ERCa1l52ilizp3s8lpRKQ3eG3ZSUpKAjjhDE1JSUnb2Z6kpCQaGxupqKjodJuO2O12IiMj2z1EpOeOVNSzIa8cgG+MTiAqJNDkRD0XERxIUmTrCuj7dSlLxC94bdnJyMggKSmJlStXtj3X2NjImjVrmDFjBgCTJ08mMDCw3TaFhYXs3LmzbRsR6VvuFnh/VzEGMDY5klHH7mjyZZnHzu6o7Ij4B1NvNaitrWX//v1tf87Ly2Pr1q0MGjSI9PR0Fi5cyOLFi8nMzCQzM5PFixcTGhrKtddeC0BUVBTf//73+clPfkJsbCyDBg3ipz/9KePGjWu7O0tE+tam8gBq3c3EhAYya1S82XF6xYiEcD7d76SgskF3ZYn4AVO/gjdu3Mjs2bPb/nz33XcDcMMNN/Diiy9y77330tDQwK233kpFRQXTpk3jgw8+ICLiP785PvnkkwQEBHD11VfT0NDAnDlzePHFF7HZvGdVZRF/FTH5UgobrNgsFi7KTibQ5rUni7slMiSQhAg7JTVuDpTWeuXipSLSdaaWnVmzZmEYna8wbLFYWLRoEYsWLep0m+DgYJ5++mmefvrpPkgoIp05WNFEzKybADgnM474CLvJiXpXZkI4JTVu9qvsiPg8//g1TET6VZ27mSfWVWAJCCQ5xMMZqVFmR+p1w4+N2ymoaMDdrBnZRXyZyo6IdNvD7+ziaE0LzTVOJg9q9onlILorJjSImNBAPAYccna+sruIeD+VHRHplr9vLeCNTUewWsD5j99g9+PhccPiW8/uHHTqriwRX6ayIyJdll9Wx4MrdgJw5Zhw3Id3mpyobw2La13u4pCznhZP5+MLRcS7qeyISJc0t3hY+PpWat3NTB0aw1Vjw82O1OeSooIJCbTR2OKhoLLB7Dgi0kMqOyLSJU9/tJ8tjkoiggN48tsTfHY5iO6wWiwMi289u3OwVJeyRHyVyo6InNKm/HKe/mgfAL+6PJvUmFCTE/Wf45eyDpTWnXSqDBHxXpoWVGSAcjgcOJ3OU25X3+ThJx848Rgwc0gIaZ5iNm8uJicnpx9Smi99UCg2q4VadzNldY3EhfvXfEIiA4HKjsgA5HA4GD1mDA31p76lOvbiuwgfN4fmqmL+uvB2Xmps/57aWv++vBNgs5IaE0J+WT35ZfUqOyI+SGVHZAByOp001Ndz3X2Pk5g+vNPtjtRZ+bIsADCYkzmIuCdfbnstZ/0a3l32FC6Xqx8SmysjNoz8snoOOeuYPCTG7Dgi0k0qOyIDWGL6cFIzszp8rc7dzD/X5QMezhway4Thse1eL3Yc6IeE3mFoXBjklnK0qnU2ZXuAH08uJOKHNEBZRDq0OrcUd7OH+Ag7Z2YMMjuOqaJCAok+Npvy4XLdgi7ia1R2ROQE+0tq2V9Si9UCc8ckDojbzE9laOyxCQbL6kxOIiLdpbIjIu24mlr4eG8JAJOHxPjdauY9NTS29Xb7Q2W6BV3E16jsiEg7n+wrpb6xhZjQQM4cOrAvX33V4JgQAqwW6twtOGsbzY4jIt2gsiMibfLL6sgprAFg7thEAmz6FnFcgNVK2qD/nN0REd+h72QiAkBjs4cP97RevpqQFk1yVIjJibxP26Usp8qOiC9R2RERAL7MK6PG1UxkcADTh8We+g0D0PFByoXVLlxNLSanEZGuUtkREcrrGtl6uBKAWaMSCArQt4aORIYEMigsCMMAR/mpZ58WEe+g72giA5xhGKzJLcVjQEZcGBnHFr6Ujn31riwR8Q0qOyID3EFnHY7yemwWC+dlxpkdx+sdv5SVX1avW9BFfITKjsgA1uKBT3JLAZg0JJro0CCTE3m/lOgQAm0W6htbKKlxmx1HRLpAZUdkAMutsVLtaibcHsBUzanTJTarhfRjt6Dnl2ncjogvUNkRGaBskfHsrW5d0PKcEXEEak6dLhsy6PilLI3bEfEF+u4mMkDFzL6JFsPC4OgQRiaGmx3Hpww5Nki5sNqFu1m3oIt4O5UdkQFob1kjYaPPBQxmjozHYtFCn90ReWwVdEOroIv4BJUdkQHGMAxe3t66JMSQMI8W+uyhIcfH7ZTrUpaIt1PZERlgVueWsqu0EaO5kbFRugTTU0OO3YLuKKtHd6CLeDeVHZEBxOMxePTdPQBUb/oHoQEmB/JhqTEh2CwWql3N1DabnUZETkZlR2QA+fu2AvYU1RAaaKF63d/MjuPTAm1WkqODASh26VupiDfTV6jIAOFubuG3H+QC8K3R4XhctSYn8n3H78oqbtC3UhFvpq9QkQHilXUOjlQ0kBBhZ36m1r/qDcfn2yl1W8Cma4Ii3kplR2QAqHE18czH+wFYeP5I7AG61bw3xIUHERpko8WwEDx4rNlxRKQTKjsiA8BfPjtEeV0jw+LCuHpKqtlx/IbFYmm7BT04Y5LJaUSkMyo7In6u2tXEnz87CMDCuSMJ0LIQvSr92LidEJUdEa+l73oifm7Z54eodjUzIiGcS8Ylmx3H77QuCmoQlDiM8gbNWyTijVR2RPxYjauJ//0sD4DbvzECm1VjdXpbaFAA0UGtswpuK3abnEZEOqKyI+LHXvoin6qGJobFhzF/fIrZcfxWYnBr2dlapLIj4o1UdkT8VK27mT992jpWR2d1+lZisAeAbcWNeDxaO0LE26jsiPipl744RGV9ExlxYVyqszp9KtZu4HHXU+32sPNoldlxRORrNAuWiI9yOBw4nc4OX2to8vCHj0sBmD8sgO3btrZ7PScnp6/jDShWC7jytxE6cjqf5JYyPjXa7Egi8hUqOyI+yOFwMHrMGBrq6zt8PfLMK4iZfRNN5Uf56ZWXgeHpcLvaWi0Z0Vsa8jYfKztOfvyNTLPjiMhXqOyI+CCn00lDfT3X3fc4ienD273W4oF3jwbi9sBZIxIY+swbJ7w/Z/0a3l32FC6Xq78i+z1X3mYANjsqqHE1EREcaHIiETlOZUfEhyWmDyc1M6vdc9uOVOL2lBIZHMD0MzoemFzsONBfEQeM5qpiksNtFNa2sPZAGRdkJZkdSUSO0QBlET/i8Rhszq8AYFJ6jO7A6mcTkuwAfJJbanISEfkqlR0RP7KvpJZqVzMhgTbGpkSaHWfAaSs7+0oxDN2CLuItVHZE/IRhGGzMLwfgjNQoArUGVr/Ljg8i0GbhcHkDh8o6HjwuIv1P3w1F/ISjvB5nbSMBVgvj06LNjjMghQRamTJkEKBLWSLeRGVHxE9sPDZWJ3twFCGBNpPTDFznjYwHVHZEvIlXl53m5mZ+9rOfkZGRQUhICMOGDeORRx7B4/nPnCGGYbBo0SJSUlIICQlh1qxZ7Nq1y8TUIv2vqNrFkYoGrBaYmB5tdpwB7byRcQB8cbCMxuaO5zcSkf7l1WXn0Ucf5Q9/+APPPPMMOTk5PPbYYzz++OM8/fTTbds89thjPPHEEzzzzDNs2LCBpKQk5s6dS01NjYnJRfrXpkOtZ3VGJUYQqfldTDUmKZK4cDv1jS1tY6hExFxeXXa++OILvvnNb3LJJZcwdOhQrrzySubNm8fGjRuB1rM6S5cu5cEHH+SKK64gOzubZcuWUV9fz/Lly01OL9I/Kuob2V/aOhPy5CExJqcRq9XCeZmtZ3c+ye14OQ8R6V9eXXbOOeccPvzwQ3JzcwHYtm0bn332GRdffDEAeXl5FBUVMW/evLb32O12Zs6cydq1azv9XLfbTXV1dbuHiK/a4qgEICMujNhwu7lhBNC4HRFv49UzKN93331UVVUxevRobDYbLS0t/PrXv+Y73/kOAEVFRQAkJia2e19iYiL5+fmdfu6SJUv4xS9+0XfBRfqJuwVyClvL+iSN1fEa5xw7s7O7sJrSGjfxESqhImby6jM7r7/+Oi+//DLLly9n8+bNLFu2jN/85jcsW7as3XYWS/tZYg3DOOG5r3rggQeoqqpqexw+fLhP8ov0tbxaK80eg/hwO4OjQ8yOI8fEhdvJHtw6qeOn+3R2R8RsXn1m55577uH+++/nmmuuAWDcuHHk5+ezZMkSbrjhBpKSWteeKSoqIjk5ue19JSUlJ5zt+Sq73Y7drt+0xMdZbRyobb3FfGJ69EkLvvS/8zLj2VlQzSe5pVwxKdXsOCIDmlef2amvr8dqbR/RZrO13XqekZFBUlISK1eubHu9sbGRNWvWMGPGjH7NKtLfQkedg6vFQmiQjczEcLPjyNccH7fz6T4nHo+WjhAxk1ef2bn00kv59a9/TXp6OllZWWzZsoUnnniCm266CWi9fLVw4UIWL15MZmYmmZmZLF68mNDQUK699lqT04v0HcMwiJz6TQDGp0YRYPXq31sGpEnpMYQF2Sira2R3YTXZg6PMjiQyYHl12Xn66af5+c9/zq233kpJSQkpKSnccsstPPTQQ23b3HvvvTQ0NHDrrbdSUVHBtGnT+OCDD4iIiDAxuUjf2lPWhD15JFYMxumHqFcKCrAyfXgcq3KKWZNbqrIjYiKv/nUwIiKCpUuXkp+fT0NDAwcOHOBXv/oVQUFBbdtYLBYWLVpEYWEhLpeLNWvWkJ2dbWJqkb73z9w6ANLDPIQGefXvLAPazJHH59vRIGURM3l12RGREx0ur+fLAhcAIyK0HIE3Oz5uZ1N+BbXuZpPTiAxcKjsiPualLw7hMaDh0BaigjTw1ZsNiQ1jSGwozR6DLw6UmR1HZMBS2RHxIXXuZl7b0DovVPWGv5ucRrrivEzNpixiNpUdER/y9tYCalzNJIfbcB3cZHYc6YK2pSM0uaCIaVR2RHyEYRj89YvWZVAuHBEG6BKWL5g+PJYAq4X8snryy+rMjiMyIKnsiPiIDYcq2FNUQ3CgldlDtTSErwi3B7StRq9LWSLmUNkR8RF/Xdd6VufyCYMJD9KXri85filrTa7T5CQiA5O+Y4r4gJIaF+/tLATg+ulDTE4j3TXzWNn54oCTxmZNFyDS31R2RHzAa+sP09RiMCk9mqwUzcTra8YmRxIbFkRdYwubHRVmxxEZcFR2RLxcc4uH5V86APju9KHmhpEesVotnJup2ZRFzKKyI+LlVuUUU1TtIjYsiIvGJZkdR3pIt6CLmEdlR8TLvXTsdvNvT03DHmAzOY301LnHJhfcWVBNSY3L5DQiA4vKjogX219Sw9oDZVgtcN1ZGpjsy+Ij7JyR2jre6uM9JSanERlYVHZEvNgrx8bqzBmTyOBoza3j684fkwjAyt0qOyL9SWVHxEu5mlp4a3MBANdNSzc5jfSGOcfKzmf7S2lobDE5jcjAobIj4qXe3VlIVUMTg6ND2sZ7iG8bkxzB4OgQXE0ePt+vCQZF+ovKjoiXenV96+rm356ahs1qMTmN9AaLxcL5YxIA+HBPsclpRAYOlR0RL3SgtJb1eeVYLXDVlFSz40gvOn9s66WsVTkleDxazFWkP6jsiHih1ze0ntWZPSqB5CgNTPYn0zJiCbcHUFrjZntBldlxRAYElR0RL+NubuGNTUcAuOZMDUz2N0EB1ra1slbt1qUskf6gsiPiZVbuLqa8rpHESDuzR2lgsj86f2zruJ1VOSo7Iv1BZUfEy7x2bGDy1VPSCLDpS9QfzR6VgM1qYU9RDYfL682OI+L39J1UxIs4yur5bL8Ti6W17Ih/ig4NYsqQGAA+1NkdkT6nsiPiRV7f2Dpj8jkj4kgbFGpyGulLc79yV5aI9C2VHREv0dzi4W8bWwcmX6uByX7v+GzK6w6WUe1qMjmNiH8LMDuAyEDkcDhwOtvPoLvxqIuSGjeRdiuDXEfZvLmw0/fn5OT0dUTpYxlxYQyPD+NAaR2f5JYyf3yK2ZFE/JbKjkg/czgcjB4zhob69gNT4y5/gLBRZ3Pk0zc565E/d+mzamtr+yKi9JPzxyZyYM1B3t9VrLIj0odUdkT6mdPppKG+nuvue5zE9OEAuFvgXwWBGMAVl80n6spLTvoZOevX8O6yp3C5XP2QWPrKhVlJ/HHNQT7KKcbV1EJwoM3sSCJ+SWVHxCSJ6cNJzcwCYOvhSgxKSYiwk5WVecr3FjsO9HU86YHuXl40DIO4UBvO+hZeeO9LLjkjlfR0jdcS6W0qOyJeYPfRagDGJkeanER6orq8FIAFCxZ0+70xs79P5Jnf4qHnV/CTj55lT06OCo9IL1PZETFZaY2b0lo3NouFkUkRZseRHmiobS2rl9zyIKPGT+7We8vcFlYXQ2TWuTjffQqn06myI9LLelR2hg0bxoYNG4iNjW33fGVlJZMmTeLgwYO9Ek5kINhd2PqDMiM+jBCN2fBpsSlD2i5NdtVgw2Bj5SFq3RCSMbGPkokMbD2aZ+fQoUO0tLSc8Lzb7aagoOC0Q4kMFC0eg71FNYAuYQ1UFouFEQnhAISOOsfkNCL+qVtndt555522f3///feJiopq+3NLSwsffvghQ4cO7bVwIv4uz1lHQ1MLoUE2hmjG5AErMyGcrYcrCc2cRlOLYXYcEb/TrbJz+eWXA62/idxwww3tXgsMDGTo0KH89re/7bVwIv4u59glrDFJkVitFpPTiFmSo4IJthm47GFsK3YzzexAIn6mW2XH4/EAkJGRwYYNG4iLi+uTUCIDgasF8srqABiTrIHJA5nFYmFwiIcDtTa+OOLiv80OJOJnejRmJy8vT0VH5DQ56qwYBiRG2okNt5sdR0yWGtr6y+T6AheNzR6T04j4lx7fev7hhx/y4YcfUlJS0nbG57i//OUvpx1MxN856lp/19DAZAGItRs015RRFxHL5/udzB6dYHYkEb/RozM7v/jFL5g3bx4ffvghTqeTioqKdg8RObnAuCFUNVmxWmBkoi5hCVgsUJ/7BQD/3tH5IrAi0n09OrPzhz/8gRdffJHrr7++t/OIDAhhWbOB1pWvtR6SHFe/9zMiJ8/ng93F/LrZQ1BAj34fFZGv6dFXUmNjIzNmzOjtLCIDgscwCMuaBcAozZgsX+E+spvoYCtVDU18kltqdhwRv9GjsvODH/yA5cuX93YWkQFhV0kjARFxBFoMMmLDzI4j3sTwcG56CAArtmqCVpHe0qPLWC6Xi+eff55Vq1Yxfvx4AgMD273+xBNP9Eo4EX+0Jr8BgMGhHgJsukwh7Z03JIR/5Naxancx1a4mIoMDT/0mETmpHpWd7du3M2HCBAB27tzZ7jWLRROjiXSmobGFL464ABgSptuL5UTDogPITAhnX0kt7+0o4uqpaWZHEvF5PSo7H3/8cW/nEBkQVuUU09Bs0FxVTGxajNlxxAtZLBYunziYx9/fy4otBSo7Ir1A59BF+tGKLa3jMOp2fYxOgkpnvjkhBYB1eWUcrWwwOY2I7+vRmZ3Zs2ef9HLVRx991ONAIv6qrNbNmmN32NTuWg1cYWoe8V6pMaFMyxjEl3nl/H3rUX40a7jZkUR8Wo/O7EyYMIEzzjij7TF27FgaGxvZvHkz48aN6+2MIn7hn9sLafEYDI8JpLn8iNlxxMt9a+JgAFZsOYJhaCV0kdPRozM7Tz75ZIfPL1q0iNra2tMKJOKv3jp2CWvmkBB07lNO5aJxyTz0zi5yi2vZXVhNVkqU2ZFEfFavjtlZsGCB1sUS6cDB0lq2Ha7EZrVwTnqw2XHEB0SFBHL+mNb1sd7eojl3RE5Hr5adL774guDg3v1GXlBQwIIFC4iNjSU0NJQJEyawadOmttcNw2DRokWkpKQQEhLCrFmz2LVrV69mEDldx39YnZsZR3SwloeQrrl8QuulrL9vPUqLR5eyRHqqR5exrrii/cBKwzAoLCxk48aN/PznP++VYAAVFRWcffbZzJ49m3fffZeEhAQOHDhAdHR02zaPPfYYTzzxBC+++CIjR47kV7/6FXPnzmXv3r1ERGgqfjGfYRhts+F+a+Jg8BSbnEh8xaxRCUSHBlJS42btASfnZsabHUnEJ/Wo7ERFtb92bLVaGTVqFI888gjz5s3rlWAAjz76KGlpabzwwgttzw0dOrTt3w3DYOnSpTz44INtBWzZsmUkJiayfPlybrnllg4/1+1243a72/5cXV3da5lFvm6zo4LD5Q2EBdmYNzaJnJ0qO9I1QQFW5o9P5uV1Dt7aXKCyI9JDPSo7Xy0ffemdd97hggsu4KqrrmLNmjUMHjyYW2+9lZtvvhmAvLw8ioqK2hUsu93OzJkzWbt2badlZ8mSJfziF7/ol30QeWtz61mdC7KTCAnSJSzpnv+alMrL6xy8u7OQRZdlERWi5SNEuuu0xuxs2rSJl19+mVdeeYUtW7b0VqY2Bw8e5LnnniMzM5P333+fH/7wh9xxxx289NJLABQVFQGQmJjY7n2JiYltr3XkgQceoKqqqu1x+PDhXs8uAtDY7OGf2wsBuGJiqslpxBdNSItmVGIEriYP72w7anYcEZ/UozM7JSUlXHPNNaxevZro6GgMw6CqqorZs2fz2muvER/fO6daPR4PU6ZMYfHixQBMnDiRXbt28dxzz/Hd7363bbuvT3BoGMZJJz202+3Y7fZeyShyMh/vLaGqoYnESDvTh8eaHUd8kMVi4dtT03jkn7t5fYOD688aYnYkEZ/TozM7t99+O9XV1ezatYvy8nIqKirYuXMn1dXV3HHHHb0WLjk5mbFjx7Z7bsyYMTgcDgCSkpIATjiLU1JScsLZHhEz/P3YwOTLzkjBZtX6ENIz35o4mCCblZ0F1ewsqDI7jojP6VHZee+993juuecYM2ZM23Njx47l97//Pe+++26vhTv77LPZu3dvu+dyc3MZMqT1N5uMjAySkpJYuXJl2+uNjY2sWbOGGTNm9FoOkZ6odjWxKqcEgMuPzYYr0hMxYUHMy2r9Be7/Nuqyu0h39ajseDweAgNPHCQXGBiIx+M57VDH3XXXXaxbt47Fixezf/9+li9fzvPPP89tt90GtJ7eXbhwIYsXL2bFihXs3LmTG2+8kdDQUK699tpeyyHSE+/tKKKx2UNmQjhjkyPNjiM+7pqp6UDrYrKuphaT04j4lh6VnW984xvceeedHD36n8FyBQUF3HXXXcyZM6fXwk2dOpUVK1bw6quvkp2dzS9/+UuWLl3Kdddd17bNvffey8KFC7n11luZMmUKBQUFfPDBB5pjR0z39rFLWJdPHHzSMWQiXTFjeCypMSHUuJp5d2eh2XFEfEqPys4zzzxDTU0NQ4cOZfjw4YwYMYKMjAxqamp4+umnezXg/Pnz2bFjBy6Xi5ycnLbbzo+zWCwsWrSIwsJCXC4Xa9asITs7u1cziHRXUZWLLw6WAa3jdUROl9Vq4eopaQC8vkGXskS6o0d3Y6WlpbF582ZWrlzJnj17MAyDsWPHcv755/d2PhGf9M62AgwDpg6NIW1QqNlxxE9cOTmVpatyWXewnDxnHRlxYWZHEvEJ3Tqz89FHHzF27Ni2GYfnzp3L7bffzh133MHUqVPJysri008/7ZOgIr7k7S2tl3g1MFl6U0p0COeNbJ3aQwOVRbquW2Vn6dKl3HzzzURGnjjYMioqiltuuYUnnnii18KJ+KLc4hp2F1YTaLNwybhks+OIn7lmauulrDc2HaG5pfduCBHxZ90qO9u2bePCCy/s9PV58+a1W5FcZCA6vsL5zJEJRIcGmZxG/M03RicSFx5EaY2bVTlaZ02kK7pVdoqLizu85fy4gIAASktLTzuUiK/yeAz+vrX1Eta3dAlL+kBQgJWrjg1Ufnmdw+Q0Ir6hW2Vn8ODB7Nixo9PXt2/fTnKyTtvLwLUxv4KCygbC7QHMGZNgdhzxU9eemY7FAp/td3KwtNbsOCJer1t3Y1188cU89NBDXHTRRQQHB7d7raGhgYcffpj58+f3akARb+NwOHA6nR2+9r8bW6fyn5ocyO4d2zrcJicnp8+yie/r6t+PSUl2NhW6efIfG/nehNZxlHFxcaSnp/dlPBGf1K2y87Of/Yy33nqLkSNH8uMf/5hRo0ZhsVjIycnh97//PS0tLTz44IN9lVXEdA6Hg9FjxtBQX3/ii9YAUn/8V2whEbz2P3fzYn7HZee42lr9Ri7/UV3eOgRgwYIFXdo+eNgUEq9axNvbivj9LRdiNLsJCQ1lT06OCo/I13Sr7CQmJrJ27Vp+9KMf8cADD2AYBtA6sd8FF1zAs88+qwU4xa85nU4a6uu57r7HSUwf3u61o/UWvnAGEmwz+NE9D9PZpMk569fw7rKncLlc/ZBYfEVDbeuUHpfc8iCjxk8+5faGAe8dNagPieBbv36dkPJ9vPLoPTidTpUdka/p9qSCQ4YM4d///jcVFRXs378fwzDIzMwkJiamL/KJeKXE9OGkZma1e277jkKgljGDY0jLjO/0vcWOA32cTnxZbMqQE/5udWZiUDmfHyjjSHMY53ytfIvIf/RoBmWAmJgYpk6d2ptZRHyWu7mFg846AEYnaV026R9jUyJZd7Cc4mo3FSFaf02kMz1aG0tE2ttfUkuLx2BQaBDx4Xaz48gAERoUwIjEcAAO1urbuUhn9NUh0gv2FtUAMCopQiucS78aPzgKgMP1Vqx2rZUl0hGVHZHTVOtu5nBFA9BadkT6U3JUMHHhQbQYFsLGaTFmkY6o7IicptxjZ3WSo4KJCul8hnGRvmCxWBg/OBqAiImX4Dl2l6yI/IfKjshp2lPcWnY0MFnMMiopgkCLQeCgFLYUuc2OI+J1VHZETkNZrZvSGjdWC2QmquyIOYICrAwNb10B/d/7OpjwUmSAU9kROQ17j53VGRIbRkigzeQ0MpANi2jBMDxsKXJzQOtlibSjsiPSQ4ZhtN2FpUtYYrbwAGjYvwGAl9YeMjeMiJdR2RHpocIqF9WuZoJsVjLidMuvmK9m0zsAvLHpCDWuJpPTiHgPlR2RHsopal3LaHhCGIE2fSmJ+Vz520iNDKCusYU3Nh0xO46I19B3aJEeaDEgt7h1XMSYpEiT04j8x8UjQgFYtvYQHo9uQxcBlR2RHjlab6Wx2UNEcACpMSFmxxFpM3NICBHBARwqq2dNbqnZcUS8gsqOSA/k17V+6YzW8hDiZUICrVw9JQ2AFzVQWQRQ2RHpNltYDMWu1oIzJlmXsMT7fHf6ECwWWJNbqtvQRVDZEem2sKxZgIXkqGBiQoPMjiNygiGxYcwZnQDoNnQRUNkR6RbDMAjLngPorI54txtmDAV0G7oIqOyIdMvBymaC4odixWBkQrjZcUQ6dc6IOEYkhOs2dBFUdkS6ZfWh1nWHUkI92LU8hHgxi8XSdnZHt6HLQKeyI9JFjc0ePnW4ABgS5jE5jcipXTFxsG5DF0FlR6TLPt5bQrXbQ3NtOQnB+i1ZvF+YPaDtNvS/fJ5nchoR86jsiHTRm8fGPdTt+hirptYRH3HjjKFYLfDpPif7imvMjiNiCpUdkS4oq3Xz8d4SAOp2fmRyGpGuSxsUytyxiQC8oNvQZYAKMDuAiC94c/MRmloMRgwKJN+Zb3YckU7l5OSc8Nw5CU28vwve2OjggiQ3EfbOf8+Ni4sjPT29LyOK9DuVHZFTMAyD1zYcBmBuRigfmpxHpCPV5a0DkBcsWNDh60k3LIWkEVx216NUf/lGp58TEhrKnpwcFR7xKyo7Iqew4VAFB0vrCA2ycU56sNlxRDrUUFsNwCW3PMio8ZNPeD2/1srGckiZcwM3XX9th+POih0HeOXRe3A6nSo74ldUdkRO4bX1DgAuHZ9CSGCzyWlETi42ZQipmVknPJ/k8bDrs0M0NLXQEJnOqKQIE9KJmEMDlEVOoqqhiX/tKATgmjPTTE4j0nMBVivjU6MA2Hq40twwIv1MZUfkJP6+tQB3s4dRiRFMSIs2O47IaRk3OAqbxUJRtYvCqgaz44j0G5UdkU4YhsGr61sHJl9zZhoWiybXEd8WZg9gZFLrmm5bHZXmhhHpRyo7Ip3YUVBFTmE1QQFWvjVxsNlxRHrFxLQYAPaV1mo1dBkwVHZEOnH8rM5F2UlEhwaZnEakd8RH2EmNDsEwYPuRKrPjiPQL3Y0lA47D4cDpdJ50m4YmDys2t86YPCnaxebNm4GOJ2wT8TUT0qM5UtnAjoIqzswYRKBNv/eKf1PZkQHF4XAweswYGurrT7pd2Li5xF18J03lR7nxovknvF5bW9tXEUX6XEZcGFEhgVQ1NLGnsIZxx+7SEvFXKjsyoDidThrq67nuvsdJTB/e4TaGAR8VBVDZBBOHJXDN799qey1n/RreXfYULpervyKL9DqrxcIZqVF8ss/JlsMVZA+O1AB88WsqOzIgJaYP73DiNYDCqgYqDx/BZrUwY9xIQoJsba8VOw70V0SRPjU2JZJ1B8upqG/CUV7PkNgwsyOJ9BldqBX5muMTro1MDG9XdET8iT3AxtiUSAC2aJJB8XMqOyJfUeduZn9J63icCanR5oYR6WPHJ8rML6unvK7R3DAifUhlR+QrdhRU4TEgOSqYhEgt+in+LSokkOHxrZevtISE+DOfKjtLlizBYrGwcOHCtucMw2DRokWkpKQQEhLCrFmz2LVrl3khxWe1eAx2FLTOO3KGzurIAHH87E5OYTWNLeZmEekrPlN2NmzYwPPPP8/48ePbPf/YY4/xxBNP8Mwzz7BhwwaSkpKYO3cuNTU1JiUVX7W/pJb6xhZCg2yMSAg3O45IvxgcHUJceBDNHoO8Op/5kSDSLT7xN7u2tpbrrruOP/3pT8TExLQ9bxgGS5cu5cEHH+SKK64gOzubZcuWUV9fz/Llyzv9PLfbTXV1dbuHyLYjlcCxxRKtug1XBgaLxdK2hMSBGhtYNShf/I9PlJ3bbruNSy65hPPPP7/d83l5eRQVFTFv3ry25+x2OzNnzmTt2rWdft6SJUuIiopqe6SlpfVZdvENJdUuCqtcWC2tZUdkIBmZGE5IoI2GFguhI6ebHUek13l92XnttdfYtGkTS5YsOeG1oqIiABITE9s9n5iY2PZaRx544AGqqqraHocPH+7d0OJzth47q5OZEEGYXdNPycASYLMy/tgsyhFTvmlyGpHe59Xf1Q8fPsydd97JBx98QHBw53fGfH3mT8MwTjobqN1ux26391pO8W31jc3kFrfebn5Gms7qyMA0bnAUG/LKCB48htyyRiaZHUikF3n1mZ1NmzZRUlLC5MmTCQgIICAggDVr1vC73/2OgICAtjM6Xz+LU1JScsLZHpHObD1cSYvHIDHSTpJuN5cBKsweQFqYB4B/7aszOY1I7/LqsjNnzhx27NjB1q1b2x5TpkzhuuuuY+vWrQwbNoykpCRWrlzZ9p7GxkbWrFnDjBkzTEwuvqKx2cP2I623m08ZMkjrA8mANiKiteysPeyisKrB5DQivcerL2NFRESQnZ3d7rmwsDBiY2Pbnl+4cCGLFy8mMzOTzMxMFi9eTGhoKNdee60ZkcXH7DxahbvZQ0zofyZXExmoooMMXI4dBKeP469f5HPvhaPNjiTSK7y67HTFvffeS0NDA7feeisVFRVMmzaNDz74gIiICLOjiZdr8RhscVQCMGlIjM7qiADVG/9OcPo4lq93cPs3MrU+nPgFnys7q1evbvdni8XCokWLWLRokSl5xHftKaqm1t1MmN3G6CSVYxGAhv3rSQyzUVzXxIotBVw7Ld3sSCKnzavH7Ij0FcOATfkVAExMiyHAqi8FEQAMDxdntl7SfeHzPAzDMDmQyOnTd3gZkI42WKiob8IeYCV7cKTZcUS8ypyMEMLtAewrqeWz/U6z44icNpUdGZByq1vHIYxPjcIeoDEJIl8VGmjlqimpAPzlszyT04icPp8bsyNyuuxp2ZQ3WrFZLVrdXKQDOTk5TE3L5EXg472l/GP1egZHdv3HRVxcHOnpGusj3kNlRwacqOlXAzA2OVJLQ4h8RXV5KQALFiwAIP6KnxGaeRY3Ln6B8pV/6PLnhISGsicnR4VHvIa+08uAklPaSEjGJCwYTBkSY3YcEa/SUFsNwCW3PMio8ZMpcVn4tASiJ1/CtZfNI6gLAx+KHQd45dF7cDqdKjviNVR2ZEB5bVcNAEPDPUSGBJqcRsQ7xaYMITUzi8GGQc56B87aRiqCU5isXxDER2mAsgwYXxwoY0dJI0ZLE6MjW8yOI+L1LBYLE9KigdY15Dwe3YYuvkllRwYEwzB4cmUuALXb3idU5zRFumRUYgQhgTZq3c0cKK01O45Ij6jsyICwem8p6w+VE2iFqnV/MzuOiM8IsFkZlxoFwJbDleaGEekhlR3xex6PwaPv7QHg4swwWmrKTE4k4lvGD47CaoHCKhdF1S6z44h0m8qO+L2/bytgT1ENEcEBXDE63Ow4Ij4nzB7AqMTW9eO2OCpMTiPSfSo74tfczS389oPWsTo/nDmcCLv+yov0xIT0aAD2ldRS42oyN4xIN+k7v/i1ZWsPcaSigYQIOzednWF2HBGflRARTGpMCIbRemeWiC9R2RG/5ax18/SH+wH46QWjCAnSGlgip2NSeus8OzsLqmls9picRqTrVHbEbz25MpcadzNZKZFcOSnV7DgiPm9obCgxoYE0tnjYdbTK7DgiXaayI35pb1ENr653APDQ/LFYrRaTE4n4PovFwsRjZ3c0yaD4EpUd8TuGYfDQ33fiMeDCrCSmDYs1O5KI3xiT1DrJYLVLkwyK71DZEb/zzrajfJlXTnCglQcvGWN2HBG/8tVJBjc7KjEMnd0R76eyI36lxtXEr/6VA8CPZ48gbVCoyYlE/M/4wVHYrBaKql0UVmmSQfF+KjviV55YmUtpjZuhsaHcfN4ws+OI+KUwewCjk1onGdysSQbFB6jsiN/YlF/Bi2sPAfCLb2ZjD9Ct5iJ9ZeKx1dAPlNZRWd9obhiRU1DZEb/gbm7hvje3YxhwxaTBzBwZb3YkEb8WG25nSGzrZWJNMijeTmVH/MLTH+5nf0ktceF2Hpo/1uw4IgPC8UkGdxdW42pqMTmNSOcCzA4g0h0OhwOn09nuuYMVTTy7uvW5m8aHcHDPzk7fn5OT06f5RAaStJgQ4sKDcNY2sqOgiqlDB5kdSaRDKjviMxwOB6PHjKGhvv4/T1ptJH/3CYISh1O35zNue/R/uvRZtbWaH0TkdFksFialx/DB7mK2HalsO9Mj4m1UdsRnOJ1OGurrue6+x0lMHw7Anioru6oCCLIaXDLnTILnvXXSz8hZv4Z3lz2Fy6XbZUV6w8jECD7f76TO3UJucQ0RZgcS6YDKjvicxPThpGZm4ax1s+fwYcBg1ugkRiRHnvK9xY4DfR9QZACxWS2ckRbN2gNlbHZUcJ5O7ogX0gBl8UnNLR7e21lEi2EwNDa0bc4PEel/4wZHEWC14KxtpNStdejE+6jsiE/6bL+TsrpGQoNszB2biMWib7AiZgkOtDE2pfXM6r5qzW8l3kdlR3xOYYOFbUeqAJg7NpHQIF2NFTHb8UkGi1xWAmJTzQ0j8jUqO+JTrGHRbCprLTcT0qIZGhtmciIRAYgODWJ4fOvXY+TUb5mcRqQ9lR3xGR7DIO7iu3B7LMSFB3H28FizI4nIVxy/9Tw86xuUN2iSQfEeKjviM/6VW0fIsMlYLQYXZiURYNNfXxFvkhIdQqzdgyUgkH/m1pkdR6SNflqIT9iUX8FL22sAGB/dQmy43eREItKRUZGtZ3TeP1BPVUOTyWlEWqnsiNcrq3Xz4+WbaTGgLucThoV7zI4kIp1ICjZoLM2nodng5XX5ZscRAVR2xMu1eAwWvr6VwioXgyNslL33NLrLXMR7WSxQ/eUbALzw+SEtECpeQWVHvNrvPtzHp/uchATa+OmMGIzGBrMjicgp1OV8QlyoDWetmzc2HTE7jojKjnivNbml/O6jfQD8+lvZDIkKNDmRiHSJp4Vvjmy9Df35Tw7S3KJLz2IulR3xSgWVDSx8bQuGAddOS+eKSZqkTMSXzBkWQkxoII7yet7dWWR2HBngVHbE6zQ2e7jtlc1U1DcxbnAUD80fa3YkEemm4AArN87IAOC51QcwDMPkRDKQqeyI11n87xy2Hq4kMjiAZ6+bRHCg1toR8UXfnT6EkEAbuwurWb231Ow4MoCp7IhX+ce2o7y49hAAT357AmmDQs0NJCI9FhMWxIKz0gF46sN9OrsjplHZEa+xv6SW+9/cDsCPZg1nzphEkxOJyOm6+bxh2AOsbD1cyWf7nWbHkQFKZUe8Qp27mR+9vIm6xhamD4vlJ3NHmh1JRHpBQkQw1047dnZnlc7uiDlUdsR0hmHw4Iod7CupJSHCzu++M1HrXon4kR/OHE5QgJWN+RV8caDM7DgyAOknipju5S8dvL31KDarhWeunUR8hNa9EvEniZHBfGdqGtA6dkekv6nsiKm2Ha7kl//YDcB9F47izIxBJicSkb7ww1nDCbJZ+TKvnHUHdXZH+pfKjpimoq6RW1/ZTGOLhwuyErn53GFmRxKRPpIcFcLVU1snB/2dzu5IP/PqsrNkyRKmTp1KREQECQkJXH755ezdu7fdNoZhsGjRIlJSUggJCWHWrFns2rXLpMTSVR6PwV3/t5WCygaGxIby+FVnYNEKnyJ+7UezRhBos7D2QBkbDpWbHUcGEK8uO2vWrOG2225j3bp1rFy5kubmZubNm0ddXV3bNo899hhPPPEEzzzzDBs2bCApKYm5c+dSU1NjYnI5ld9/vJ/Ve0uxB1h57rrJRAZr3SsRfzc4OoQrJ7eO3Vm6KtfkNDKQeHXZee+997jxxhvJysrijDPO4IUXXsDhcLBp0yag9azO0qVLefDBB7niiivIzs5m2bJl1NfXs3z5cpPTS2fW7nfyxLFvdL+8PJuxKZEmJxKR/nLb7NaxO5/vL+PTfZpVWfpHgNkBuqOqqgqAQYNaB7Hm5eVRVFTEvHnz2rax2+3MnDmTtWvXcsstt3T4OW63G7fb3fbn6urqPkwtxzkcDvYfKebuD5wYBszJCGGEtZTNm7v2DS8nJ6ePE4pIbznZ1+vcYcH8a189D7+1mcfOj8P6tUvYcXFxpKen93VEGUB8puwYhsHdd9/NOeecQ3Z2NgBFRa0r6SYmtp9pNzExkfz8/E4/a8mSJfziF7/ou7ByAofDwegxY4m45B5Chk2h0ZnPC7+9m780u0/95q+pra3tg4Qi0huqy1t/eVmwYEGn21hDIhl8y/9ysCKUmQvuon7Pp+1eDwkNZU9OjgqP9BqfKTs//vGP2b59O5999tkJr319YKthGCcd7PrAAw9w9913t/25urqatLS03gsrJ3A6nQRmzyNk2BSsFoOLs1OIeurVbn1Gzvo1vLvsKVwuVx+lFJHT1VDbeqb8klseZNT4yZ1ul1NlZXcVDP2ve5mXfBfWY9+yix0HeOXRe3A6nSo70mt8ouzcfvvtvPPOO3zyySekpqa2PZ+UlAS0nuFJTk5ue76kpOSEsz1fZbfbsds1cV1/2lfWSPR5NwAwa1QiWYOjuv0ZxY4DvR1LRPpIbMoQUjOzOn09odlD3tpD1DW1UBk6mPGp0f0XTgYcrx6gbBgGP/7xj3nrrbf46KOPyMjIaPd6RkYGSUlJrFy5su25xsZG1qxZw4wZM/o7rnSi2tXEE+sqsdgCGBzaQrYGJIsMeEEBVqYdm0T0y7xyGps9JicSf+bVZee2227j5ZdfZvny5URERFBUVERRURENDQ1A6+WrhQsXsnjxYlasWMHOnTu58cYbCQ0N5dprrzU5vUBrYf1/b+2guK6F5soiJg9q0Xw6IgJA9uAookICqW9sYevhSrPjiB/z6rLz3HPPUVVVxaxZs0hOTm57vP76623b3HvvvSxcuJBbb72VKVOmUFBQwAcffEBERISJyeW4FVsK+Of2QmwWKH3nMQK9+m+ciPQnm9XC9GGxAGzKr6ChscXkROKvvHrMjmEYp9zGYrGwaNEiFi1a1PeBpFsOl9fz0N9bZ7P+dlY4Swo1iZiItDcyMZxNDjulNW6+OFjGSJvZicQf6fds6RMtHoO7/28rte5mpgyJ4Vujw82OJCJeyGKxcF5mHAA7C6qobNRlbul9Xn1mR7yLw+HA6XR2ads3c2rZcKiGkAAL388KIHfvnj5OJyK+KjUmlMyEcPaV1LKtQqd2pPep7EiXtE4KOIaG+vpTbhuUNIKkBb/BYgvg8Nu/5eJff9T2miYEFJGOnJMZR56zDqfbSujoc82OI35GZUe6xOl00lBfz3X3PU5i+vBOt2v2wIdFgdQ2Wxgc2sIVP/oxFsuPNSGgiJxUZHAgU4bEsC6vnJhv/IC6Rt2KLr1HZUe6JTF9+EknCvtoTwm1zVWE2wOYP2UYwYGtp6Q1IaCInMrkITHsPFxGbUQsy3fWcO5ZZicSf6EBytJr8px17ChoXax17tjEtqIjItIVATYrEwc1A/De/nrNvSO9RmVHekV9YzMrdxcDMDE9mvRBoSYnEhFflBBsULvjQwzg/je3a2Zl6RUqO3LaDMNgVU4JDU0txIYHMePYJGEiIj1R8fGfibRb2VNUw9Mf7TM7jvgBlR05bTsKqshz1mGzWrgwK4kAm/5aiUjPeRqq+e9JrWvoPbv6ANt0OUtOk34qyWkpr2vk032tc++cPTyWuHCtJi8ip29GWgiXnpFCi8fgJ3/bpqUk5LSo7EiPtXgM3t9VRLPHIH1QKBPSos2OJCJ+5JHLsoiPsLO/pJZH/rnb7Djiw1R2pMfWHSyjpMZNcKCVuWMTtZq5iPSqmLAgnrx6AhYLvLrewT+2HTU7kvgolR3pkSMV9WzMrwBgzuhEwu2asklEet85mXHcNmsEAA+8tYM8Z53JicQXqexIt7maWnh/V+tt5lkpkYxI0CKfItJ3Fp6fydShMdS6m7n5pY3UuJrMjiQ+RmVHusUw4OO9JdS6m4kKCeS8zHizI4mInwuwWfn9tZNIigxmf0ktC1/bSovHMDuW+BCVHemWw/VWcotrsVjgwqwkggL0V0hE+l5CZDB/vH4y9gArH+4p4X/ezTE7kvgQ/aSSLrNFJrClvHUJiGkZg0iKCjY5kYgMJGekRfPYleMB+NOnefxxjdbck65R2ZEuaWoxiP/m/TQbFpKjgpk6ZJDZkURkAPrmhMH8v4tHA7Dk3T3834bDJicSX6CyI13y4rZq7CkjCbQaXJiVhNWq28xFxBz/fd5wbjlvGAD3vbWd19Y7TE4k3k5lR07pnW1HeXd/PQBTY5uJDAk0OZGIDHT3XzSa688agmHA/W/tYNnaQ2ZHEi+msiMntb+khvvf3A5A1drXSA7RHRAiYj6LxcIj38ziB+dkAPDwO7t49L09eHSXlnRAZUc6Vd/YzI9e3kx9YwvZCUFUfrbc7EgiIm0sFgsPXjKGO+dkAvDc6gPctnwz9Y3NJicTb6OyIx0yDIMHV+xkX0ktCRF27poWDYbH7FgiIu1YLBbumjuSJ64+gyCblXd3FnHZM5+zp6ja7GjiRVR2pEPPf3KQFVsKsFktPP2dicSE2MyOJCLSqSsmpbL85mkkRrYuHPrNZz7nxc/zdFlLAJUd6cCq3cX8z3t7AHho/limDYs1OZGIyKlNGTqIf99xLrNGxeNu9rDoH7u58g9ryS2uMTuamEyrN0o7OwuquPO1LRgGLDgrne9OH2J2JBGRLosNt/OXG6byzHtbeW5tIZsdlVy49BMuGB7Kt7MiiLR3/Xf8uLg40tPT+zCt9BeVHWnjKKvnxhc2UNfYwtkjYnn40iwsFs2nIyK+5ciRw9x/1Tk02kIZdP4thI6czrv76/nXrlJqtvyb6o1/x1NXecrPCQkNZU9OjgqPH1DZEQBKa9xc/5cvcda6GZMcyXMLJhNo01VOEfE9TqeThvp6rrvvFySmD6fE1cSOChuVhBF11lXEnHUlQ8I9jIxoIbyTacOKHQd45dF7cDqdKjt+QGVHKKt1s+B/vyS/rJ7UmBCWfW8qkcGaOFBEfFti+nBSM7NIBSYaBnnOOjbmV1BY5SKv1sahWhvD4sPISoliSGwoVp3J9lsqOwNceV0j1/3vl+wtriEhws5fvz+NhEgt8Cki/sVisTAsPpyMuDCOVrrYmF/OobJ6DpTWcaC0jnB7AGNTIslKidQve35IZWcAK6528d0/r2dvcQ3xEXZe/e+zyIgLMzuWiEifsVgsDI4JYXDMYMpq3ew8Ws2ewmpq3c2szytnfV45aYNCSLJYsQTYzY4rvURlZ4A6WFrL9X9eT0FlAwkRdpbffBbD48PNjiUiAkBOTk6fvzc23M7MkfGcPTyWA6V17DxaxZGKBg6XN3CYAFJ//BLPbazkh7EVTEqP1g0bPkxlZwBad7CMH728iYr6JjLiwnjppjNJGxRqdiwREarLSwFYsGDBaX9WbW1tl7YLsFkZlRTBqKQIqhqayCmsZoejjHp7GCsPNrDyubUMiw/jysmpXDExlaQoXer3NSo7A4hhGLy8Lp9f/GM3zR6D8alR/OXGqcSF61StiHiHhtrWZR4uueVBRo2f3KPPyFm/hneXPYXL5er2e6NCAjlrWCyDm4t49rGHueb+pXx5tJGDpXU89t5efvP+Xs7JjOeqyanMHZtIcKBml/cFKjsDRFVDEw+u2ME/txcCcPmEFP7nv8brC1VEvFJsyhBSM7N69N5ix4HT/u9bLOB27OCOadGMzBrPv7cX8samI6w/VM4nuaV8kltKZHAA/zU5lRumD2Woxjt6NZWdAWDtASf3/G07BZUNBFgt3HvhKG4+d5iuP4uIdEG4PYCrp6Zx9dQ0DjnreGvzEd7cXEBBZQMvfH6IF9ceYvaoBG6YMZRzR8Rhtep7q7dR2fFjlfWN/PpfOfxt0xEAhsSG8tQ1E5mQFm1uMBERHzU0Loy7541i4fkj+WRfKcvWHuLjvaV8tKeEj/aUMCw+jBtnDOWKSamE2/Uj1lvoSPgAh8OB0+ns8vZNLQbvHajnb7trqG1sXfH38qxB/PKqKURo/ggRkdNmtVqYNSqBWaMSyHPW8dIXh/jbxiMcLK3job/v4vH39vKdaenMGxqE1VXV4/+O1ufqHSo7Xs7hcDB6zBga6utPvbE1gPDs2USedTWBMckANJYeovz93/N8RT53z8whQl80IiK9KiMujIcvzeIn80bx5qYjLFt7iIPOOp7/5CB//LiZupxPqF7/Fk2lh7r92Vqfq3eo7Hi5/6zx8jiJ6cM73MbdAofqrByssVHf0nqt2G41yIpuYUhaCqXxN2mNFxGRPhZuD+CGGUO5/qwhrM4t4bf/2s6uUgjP/gbh2d8gMdhDZmQLCXaDrgyZ1PpcvUdlx0ccX+PlOMMwKKxysaOgin3FtbQYrZerQoNsTE6PYVxqVNtCnhorJyLSMz2d3DAauDaphJt+8wum/ug3FNTbKHZZKXZZiQ+3M2lINJkJEdj0DbpfqOz4mPK6RvYV15BbUkt5XWPb8wkRdsanRjEqMYIArVYuInJaenNyw6zgKs6fMIktjgp2Ha2mtNbN+7uKWXugjAlp0WSnRBEUoO/bfUllx8sZhkFg/FByqqys/jKfstr/FByb1cKoxAjGpUaRpMU7RUR6TW9PbhgVEsisUQlMGxbLjiNVbD1cSY2rmU/3Ofkyr5xxg6OYkBatO7j6iP6veqE6dzNrD5Tx0Z4SPthRQspNz7C7CqARqwXSB4WSmRjB8Lgw7N2YFLA/1poREfEnvT25YUigjTMzBjEpPZo9RTVsdlRQUd/EpvwKtjgqGJ0UyaT0aGI1s32vUtnxAq6mFrY4KvnigJO1B8rYeriSZo/R9rqnyUVKRBDZGUkMjw/v9qzHZqw1IyIinQuwWckeHEVWSiR5zjo25VdwtMrF7sJqdhdWMzQ2lDSbxvP0FpUdE7iaWthRUMX6vHLWHnCy8VAF7mZPu21SY0L4xugE0gOq+e/LZ3PVU6+RmhLVo/+e2WvNiIhIxywWC8PiwxkWH05hVQOb8yvZX1rLobJ6DhFI0nefZNXBejLHNmmetNOgstPHHA4H+w4Xs6eskT3OJvY4G8mrbOJr3YboYCvjEoIYl2AnOyGIxDAbFksjOTl5GM2NHX94N5m91oyIiHQuOSqES8aHUFnfyBZHJbuOVmJPzuTZjVX8ZdsqLsxK4srJaUwfHqu7uLpJZacPPfLmBv60age2qMQTXmuuLcddkIMrfxuu/O3klx9h20k+S5ePREQGhujQIGaPTiCdUv76yiuMu+xmjlQ38/bWo7y99SgJEXbmjElk3thEpg+P1YLOXaCy04dKKuuOFR2DqECDWPvxh4dQWziWsVOBqSf9DF0+EhEZmOw2qP7yTZ76/QPYEobzxqbDvLP1KCU1bl5d7+DV9Q5CAm2cNzKO6cNimTJ0EKOTNP1IR/ym7Dz77LM8/vjjFBYWkpWVxdKlSzn33HNNzTRnWAh//tXd3PyTn5MxSpePRESk+ywWCxPSopmQFs3P549l3cFyVu0uZlVOMYVVLt7fVcz7u4oBCAuyMTE9hglp0YxICGd4fDjDE8IIDTq9H/fdXaPx68xe48svys7rr7/OwoULefbZZzn77LP54x//yEUXXcTu3btN/Z87JCoQ16EtBKpki4hIL7AH2Jg5Mp6ZI+N55JtZ7Dpazcd7StiQX8GW/Apq3M18tt/JZ/vbF5OUqGCSo0OID7cTH2EnIcJOTFgQoUE2QgJthBz7p81q4fi9wIbROtdbcUkJ119/PY1NzWC1YbEGgM2G5fi/W1v/HasNi+0/f/7qawH2YP66+G4unDq6//+n4Sdl54knnuD73/8+P/jBDwBYunQp77//Ps899xxLliwxOZ2IiEjvs1gsZA+OIntw6526LR6D3OIaNuZXsPtoNQdKajlQWktZXSNHq1wcrer5cIiY//rFaefdesjJhScfudFnfL7sNDY2smnTJu6///52z8+bN4+1a9d2+B63243b7W77c1VVFQDV1dW9mu34oOIj+3bhbujCquUdOH4Zq+hQLgfCQn32M7whQ298hjdk6I3P8IYM3vIZ3pChNz7DGzL0xmd4QwZv+YzSI3kAbNq0qVs3qQwBhsQBcQB26pqDKKhqptLdQlWDh0q3hypXC7WNHtwtRuujufWfx5ZZbFuo1AI0NTVRWFhI5KA4ggKDsFha11y08J9/Hn9YLQbWY+8//npDXTX7Nn1K/LQbe/3n7PHPMwzj5BsaPq6goMAAjM8//7zd87/+9a+NkSNHdviehx9+2AD00EMPPfTQQw8/eBw+fPikXcHnz+wcZ7G0n3PAMIwTnjvugQce4O677277s8fjoby8nNjY2E7f05+qq6tJS0vj8OHDREZGmh2nT2lf/ZP21T8NlH0dKPsJvr+vhmFQU1NDSkrKSbfz+bITFxeHzWajqKio3fMlJSUkJp44vw2A3W7Hbm+/7kh0dHRfReyxyMhIn/zL1xPaV/+kffVPA2VfB8p+gm/va1RU1Cm38fn7hIKCgpg8eTIrV65s9/zKlSuZMWOGSalERETEW/j8mR2Au+++m+uvv54pU6Ywffp0nn/+eRwOBz/84Q/NjiYiIiIm84uy8+1vf5uysjIeeeQRCgsLyc7O5t///jdDhgwxO1qP2O12Hn744RMutfkj7at/0r76p4GyrwNlP2Hg7KvFME51v5aIiIiI7/L5MTsiIiIiJ6OyIyIiIn5NZUdERET8msqOiIiI+DWVnX62aNEiLBZLu0dSUtJJ37NmzRomT55McHAww4YN4w9/+EM/pT093d3X1atXn7C9xWJhz549/Zi65woKCliwYAGxsbGEhoYyYcIENm3adNL3+Oqx7e6++uqxHTp0aIe5b7vttk7f46vHtLv76qvHtLm5mZ/97GdkZGQQEhLCsGHDeOSRR/B4PCd9ny8e157sq68e11Pxi1vPfU1WVharVq1q+7PNZut027y8PC6++GJuvvlmXn75ZT7//HNuvfVW4uPj+a//+q/+iHtaurOvx+3du7fdTJ7x8fF9kq03VVRUcPbZZzN79mzeffddEhISOHDgwEln5vbVY9uTfT3O147thg0baGlpafvzzp07mTt3LldddVWH2/vqMYXu7+txvnZMH330Uf7whz+wbNkysrKy2LhxI9/73veIiorizjvv7PA9vnpce7Kvx/nacT2lXlmNU7rs4YcfNs4444wub3/vvfcao0ePbvfcLbfcYpx11lm9nKz3dXdfP/74YwMwKioq+ixTX7nvvvuMc845p1vv8dVj25N99eVj+1V33nmnMXz4cMPj8XT4uq8e046cal999Zhecsklxk033dTuuSuuuMJYsGBBp+/x1ePak3311eN6KrqMZYJ9+/aRkpJCRkYG11xzDQcPHux02y+++IJ58+a1e+6CCy5g48aNNDU19XXU09adfT1u4sSJJCcnM2fOHD7++ON+SHn63nnnHaZMmcJVV11FQkICEydO5E9/+tNJ3+Orx7Yn+3qcLx7b4xobG3n55Ze56aabOl0w2FeP6dd1ZV+P87Vjes455/Dhhx+Sm5sLwLZt2/jss8+4+OKLO32Prx7Xnuzrcb52XE9FZaefTZs2jZdeeon333+fP/3pTxQVFTFjxgzKyso63L6oqOiEBU0TExNpbm7G6XT2R+Qe6+6+Jicn8/zzz/Pmm2/y1ltvMWrUKObMmcMnn3zSz8m77+DBgzz33HNkZmby/vvv88Mf/pA77riDl156qdP3+Oqx7cm++vKxPe7tt9+msrKSG2+8sdNtfPWYfl1X9tVXj+l9993Hd77zHUaPHk1gYCATJ05k4cKFfOc73+n0Pb56XHuyr756XE/J7FNLA11tba2RmJho/Pa3v+3w9czMTGPx4sXtnvvss88MwCgsLOyPiL3mVPvakfnz5xuXXnppH6bqHYGBgcb06dPbPXf77bef9DS3rx7bnuxrR3zl2B43b948Y/78+SfdxleP6dd1ZV874gvH9NVXXzVSU1ONV1991di+fbvx0ksvGYMGDTJefPHFTt/jq8e1J/vaEV84rqeiMzsmCwsLY9y4cezbt6/D15OSkigqKmr3XElJCQEBAcTGxvZHxF5zqn3tyFlnndWt7c2SnJzM2LFj2z03ZswYHA5Hp+/x1WPbk33tiK8cW4D8/HxWrVrFD37wg5Nu56vH9Ku6uq8d8YVjes8993D//fdzzTXXMG7cOK6//nruuusulixZ0ul7fPW49mRfO+ILx/VUVHZM5na7ycnJITk5ucPXp0+fzsqVK9s998EHHzBlyhQCAwP7I2KvOdW+dmTLli3d2t4sZ599Nnv37m33XG5u7kkXo/XVY9uTfe2IrxxbgBdeeIGEhAQuueSSk27nq8f0q7q6rx3xhWNaX1+P1dr+R5/NZjvp7di+elx7sq8d8YXjekpmn1oaaH7yk58Yq1evNg4ePGisW7fOmD9/vhEREWEcOnTIMAzDuP/++43rr7++bfuDBw8aoaGhxl133WXs3r3b+POf/2wEBgYab7zxhlm70GXd3dcnn3zSWLFihZGbm2vs3LnTuP/++w3AePPNN83ahS5bv369ERAQYPz617829u3bZ7zyyitGaGio8fLLL7dt4y/Htif76svHtqWlxUhPTzfuu+++E17zl2N6XHf21VeP6Q033GAMHjzY+Oc//2nk5eUZb731lhEXF2fce++9bdv4y3Htyb766nE9FZWdfvbtb3/bSE5ONgIDA42UlBTjiiuuMHbt2tX2+g033GDMnDmz3XtWr15tTJw40QgKCjKGDh1qPPfcc/2cume6u6+PPvqoMXz4cCM4ONiIiYkxzjnnHONf//qXCcl75h//+IeRnZ1t2O12Y/To0cbzzz/f7nV/Orbd3VdfPrbvv/++ARh79+494TV/OqaG0b199dVjWl1dbdx5551Genq6ERwcbAwbNsx48MEHDbfb3baNvxzXnuyrrx7XU7EYhmGYeWZJREREpC9pzI6IiIj4NZUdERER8WsqOyIiIuLXVHZERETEr6nsiIiIiF9T2RERERG/prIjIiIifk1lR0RERPyayo6IiIj4NZUdEfEZFovlpI8bb7zxhO3CwsLIzMzkxhtvZNOmTSd85h//+EfOOOMMwsLCiI6OZuLEiTz66KP9vGci0pcCzA4gItJVhYWFbf/++uuv89BDD7VbgT0kJKTt31944QUuvPBCXC4Xubm5PP/880ybNo2//OUvfPe73wXgz3/+M3fffTe/+93vmDlzJm63m+3bt7N79+7+2ykR6XNaG0tEfNKLL77IwoULqaysPOE1i8XCihUruPzyy9s9f8MNN7BixQry8/OJiYnh8ssvJyYmhhdeeKF/QouIKXQZS0QGjLvuuouamhpWrlwJQFJSEuvWrSM/P9/kZCLSl1R2RGTAGD16NACHDh0C4OGHHyY6OpqhQ4cyatQobrzxRv7v//4Pj8djYkoR6W0qOyIyYBy/am+xWABITk7miy++YMeOHdxxxx00NTVxww03cOGFF6rwiPgRlR0RGTBycnIAyMjIaPd8dnY2t912G6+88gorV65k5cqVrFmzxoyIItIHVHZEZMBYunQpkZGRnH/++Z1uM3bsWADq6ur6K5aI9DHdei4ifqmyspKioiLcbje5ubn88Y9/5O233+all14iOjoagB/96EekpKTwjW98g9TUVAoLC/nVr35FfHw806dPN3cHRKTXqOyIiF/63ve+B0BwcDCDBw/mnHPOYf369UyaNKltm/PPP5+//OUvPPfcc5SVlREXF8f06dP58MMPiY2NNSu6iPQyzbMjIiIifk1jdkRERMSvqeyIiIiIX1PZEREREb+msiMiIiJ+TWVHRERE/JrKjoiIiPg1lR0RERHxayo7IiIi4tdUdkRERMSvqeyIiIiIX1PZEREREb/2/wGmAruGR4kiawAAAABJRU5ErkJggg=="/>


```python
X_features3 = data_ohe.drop(['TDS', 'Classification1', 'Classification2'], axis=1, inplace=False)
X_features3.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_gis</th>
      <th>long_gis</th>
      <th>gwl</th>
      <th>pH</th>
      <th>E.C</th>
      <th>CO3</th>
      <th>HCO3</th>
      <th>Cl</th>
      <th>F</th>
      <th>NO3</th>
      <th>...</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>T.H</th>
      <th>SAR</th>
      <th>RSC</th>
      <th>season_Post-monsoon 2020</th>
      <th>season_post monsoon 2019</th>
      <th>season_postmonsoon 2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.668300</td>
      <td>78.524700</td>
      <td>5.09</td>
      <td>8.28</td>
      <td>745</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>60</td>
      <td>0.44</td>
      <td>42.276818</td>
      <td>...</td>
      <td>49.0</td>
      <td>4.0</td>
      <td>48.0</td>
      <td>38.896</td>
      <td>279.934211</td>
      <td>1.273328</td>
      <td>-1.198684</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.458888</td>
      <td>78.350833</td>
      <td>5.10</td>
      <td>8.29</td>
      <td>921</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>80</td>
      <td>0.56</td>
      <td>100.659091</td>
      <td>...</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>56.0</td>
      <td>63.206</td>
      <td>399.893092</td>
      <td>0.913166</td>
      <td>-3.397862</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.525555</td>
      <td>78.512222</td>
      <td>4.98</td>
      <td>7.69</td>
      <td>510</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>30</td>
      <td>0.66</td>
      <td>41.471545</td>
      <td>...</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>38.896</td>
      <td>219.934211</td>
      <td>1.319284</td>
      <td>-0.398684</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.730555</td>
      <td>78.640000</td>
      <td>5.75</td>
      <td>8.09</td>
      <td>422</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>10</td>
      <td>0.58</td>
      <td>10.669864</td>
      <td>...</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>19.448</td>
      <td>159.967105</td>
      <td>0.928155</td>
      <td>0.000658</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.495665</td>
      <td>78.852654</td>
      <td>2.15</td>
      <td>8.21</td>
      <td>2321</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>340</td>
      <td>2.56</td>
      <td>128.843636</td>
      <td>...</td>
      <td>298.0</td>
      <td>5.0</td>
      <td>56.0</td>
      <td>92.378</td>
      <td>519.843750</td>
      <td>5.682664</td>
      <td>-4.396875</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



```python
X_train, X_test, y_train, y_test = train_test_split(X_features3, log_y,
                                                   test_size=0.2, random_state=100)
```

**Regularized Linear Model - Ridge Regression**



```python
# alpha값에 따른 RMSE
alphas = [0, 0.1, 1, 10, 50, 100, 200]
coeff_df2 = pd.DataFrame()

for i in alphas:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)

    print("alpha = {0}일 때, 로그 변환된 RMSE: {1:.20f}".format(i, rmse))
    
    coeff = pd.Series(data=ridge.coef_, index=X_train.columns)
    colname = 'alpha:' + str(i)
    coeff_df2[colname] = coeff



coeff_df2
```

<pre>
alpha = 0일 때, 로그 변환된 RMSE: 0.19638300293687338538
alpha = 0.1일 때, 로그 변환된 RMSE: 0.19614127466498579300
alpha = 1일 때, 로그 변환된 RMSE: 0.19541120347840121307
alpha = 10일 때, 로그 변환된 RMSE: 0.19265383184059700628
alpha = 50일 때, 로그 변환된 RMSE: 0.18914203138394836134
alpha = 100일 때, 로그 변환된 RMSE: 0.18856546447742386641
alpha = 200일 때, 로그 변환된 RMSE: 0.18882328831798447788
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha:0</th>
      <th>alpha:0.1</th>
      <th>alpha:1</th>
      <th>alpha:10</th>
      <th>alpha:50</th>
      <th>alpha:100</th>
      <th>alpha:200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lat_gis</th>
      <td>-2.204223e-02</td>
      <td>-0.022069</td>
      <td>-0.022067</td>
      <td>-0.021854</td>
      <td>-0.020579</td>
      <td>-0.019042</td>
      <td>-0.016546</td>
    </tr>
    <tr>
      <th>long_gis</th>
      <td>-1.836327e-02</td>
      <td>-0.018266</td>
      <td>-0.018339</td>
      <td>-0.018608</td>
      <td>-0.018498</td>
      <td>-0.017591</td>
      <td>-0.015592</td>
    </tr>
    <tr>
      <th>gwl</th>
      <td>-3.195652e-03</td>
      <td>-0.003183</td>
      <td>-0.003155</td>
      <td>-0.003091</td>
      <td>-0.002894</td>
      <td>-0.002693</td>
      <td>-0.002403</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-4.832663e-02</td>
      <td>-0.048215</td>
      <td>-0.047976</td>
      <td>-0.043822</td>
      <td>-0.031466</td>
      <td>-0.023284</td>
      <td>-0.015255</td>
    </tr>
    <tr>
      <th>E.C</th>
      <td>7.915002e-04</td>
      <td>0.000792</td>
      <td>0.000792</td>
      <td>0.000794</td>
      <td>0.000799</td>
      <td>0.000804</td>
      <td>0.000812</td>
    </tr>
    <tr>
      <th>CO3</th>
      <td>3.770137e-03</td>
      <td>0.003473</td>
      <td>0.002347</td>
      <td>0.001394</td>
      <td>0.001097</td>
      <td>0.000985</td>
      <td>0.000902</td>
    </tr>
    <tr>
      <th>HCO3</th>
      <td>4.786286e-03</td>
      <td>0.004491</td>
      <td>0.003369</td>
      <td>0.002471</td>
      <td>0.002325</td>
      <td>0.002300</td>
      <td>0.002282</td>
    </tr>
    <tr>
      <th>Cl</th>
      <td>1.815244e-03</td>
      <td>0.001819</td>
      <td>0.001817</td>
      <td>0.001807</td>
      <td>0.001778</td>
      <td>0.001753</td>
      <td>0.001718</td>
    </tr>
    <tr>
      <th>F</th>
      <td>3.085268e-02</td>
      <td>0.030748</td>
      <td>0.030739</td>
      <td>0.030227</td>
      <td>0.027973</td>
      <td>0.025578</td>
      <td>0.021863</td>
    </tr>
    <tr>
      <th>NO3</th>
      <td>1.492468e-03</td>
      <td>0.001495</td>
      <td>0.001495</td>
      <td>0.001489</td>
      <td>0.001471</td>
      <td>0.001456</td>
      <td>0.001435</td>
    </tr>
    <tr>
      <th>SO4</th>
      <td>2.296756e-03</td>
      <td>0.002302</td>
      <td>0.002300</td>
      <td>0.002295</td>
      <td>0.002278</td>
      <td>0.002262</td>
      <td>0.002238</td>
    </tr>
    <tr>
      <th>Na</th>
      <td>-4.986169e-03</td>
      <td>-0.004996</td>
      <td>-0.004993</td>
      <td>-0.004982</td>
      <td>-0.004954</td>
      <td>-0.004929</td>
      <td>-0.004890</td>
    </tr>
    <tr>
      <th>K</th>
      <td>-2.875159e-03</td>
      <td>-0.002882</td>
      <td>-0.002881</td>
      <td>-0.002877</td>
      <td>-0.002867</td>
      <td>-0.002865</td>
      <td>-0.002868</td>
    </tr>
    <tr>
      <th>Ca</th>
      <td>-6.095186e-02</td>
      <td>-0.060656</td>
      <td>-0.059517</td>
      <td>-0.049981</td>
      <td>-0.029292</td>
      <td>-0.019419</td>
      <td>-0.011726</td>
    </tr>
    <tr>
      <th>Mg</th>
      <td>-9.973387e-02</td>
      <td>-0.099246</td>
      <td>-0.097368</td>
      <td>-0.081696</td>
      <td>-0.047711</td>
      <td>-0.031506</td>
      <td>-0.018887</td>
    </tr>
    <tr>
      <th>T.H</th>
      <td>2.006669e-02</td>
      <td>0.020243</td>
      <td>0.020908</td>
      <td>0.017996</td>
      <td>0.009880</td>
      <td>0.005964</td>
      <td>0.002911</td>
    </tr>
    <tr>
      <th>SAR</th>
      <td>1.733916e-02</td>
      <td>0.017338</td>
      <td>0.017309</td>
      <td>0.017052</td>
      <td>0.016339</td>
      <td>0.015779</td>
      <td>0.014913</td>
    </tr>
    <tr>
      <th>RSC</th>
      <td>-1.251003e-01</td>
      <td>-0.110162</td>
      <td>-0.054127</td>
      <td>-0.009314</td>
      <td>-0.002186</td>
      <td>-0.001141</td>
      <td>-0.000570</td>
    </tr>
    <tr>
      <th>season_Post-monsoon 2020</th>
      <td>6.093422e+11</td>
      <td>-0.027065</td>
      <td>-0.026997</td>
      <td>-0.026210</td>
      <td>-0.023088</td>
      <td>-0.020047</td>
      <td>-0.015858</td>
    </tr>
    <tr>
      <th>season_post monsoon 2019</th>
      <td>6.093422e+11</td>
      <td>0.008846</td>
      <td>0.009040</td>
      <td>0.009000</td>
      <td>0.008163</td>
      <td>0.007247</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>season_postmonsoon 2018</th>
      <td>6.093422e+11</td>
      <td>0.018219</td>
      <td>0.017957</td>
      <td>0.017210</td>
      <td>0.014925</td>
      <td>0.012800</td>
      <td>0.009986</td>
    </tr>
  </tbody>
</table>
</div>


**회귀 트리**



```python
rf = RandomForestRegressor(n_estimators=500, random_state=0)
gb = GradientBoostingRegressor(n_estimators=500, random_state=0)
xgb = XGBRegressor(n_estimators=500)
lgb = LGBMRegressor(n_estimators=500)

models = [rf, gb, xgb, lgb]

for model in models: 
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)

    print("model: {}, 로그 변환된 RMSE: {:.5f}".format(model.__class__.__name__, rmse))
```

<pre>
model: RandomForestRegressor, 로그 변환된 RMSE: 0.00550
model: GradientBoostingRegressor, 로그 변환된 RMSE: 0.00778
model: XGBRegressor, 로그 변환된 RMSE: 0.01422
model: LGBMRegressor, 로그 변환된 RMSE: 0.03262
</pre>

```python
# Feature importances

rf = RandomForestRegressor(n_estimators=500, random_state=0)
rf.fit(X_features3, log_y)

feature_series = pd.Series(data=gb.feature_importances_, index=X_features3.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAs8AAAGdCAYAAADt3J7WAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABgxklEQVR4nO3dfVzN9/8/8MdJdUqnTspFRSW6UIjJxdJGm4uYYZ9hjJk2bGaYD2OLbTEjDG2sGTbx2VxtwjAzhgy5VnNRKoSszIhzXB6p5++PfXv/nFWcTBenHvfb7XW7dV7v1/v1fr5ene08e3ud11slIgIiIiIiInooi/IOgIiIiIjIXDB5JiIiIiIyEZNnIiIiIiITMXkmIiIiIjIRk2ciIiIiIhMxeSYiIiIiMhGTZyIiIiIiEzF5JiIiIiIykWV5B0BUmeTn5yMrKwv29vZQqVTlHQ4RERGZQERw/fp1uLm5wcLiwfeWmTwTPUZZWVlwd3cv7zCIiIjoEWRmZqJevXoPbMPkmegxsre3BwAkTf0c9ja25RwNERFR5VJzSN9S6Vev18Pd3V35HH8QJs9Ej1HBUg17G1vY2zJ5JiIiepwcHBxKtX9TllzyC4NERERERCZi8kyVUnh4OFQqVaHSpUuXYs/R6/WYOHEiGjVqBBsbG7i4uKBjx45Ys2YNRKQMoyciIqKKiss2qNLq0qULYmNjjerUanWRba9du4annnoKOp0On3zyCVq1agVLS0vs3LkT48ePx7PPPgtHR8cyiJqIiIgqMibPVGmp1Wq4uLiY1HbChAk4e/Ys0tLS4ObmptT7+vri5Zdfho2NTWmFSURERGaEyTNVefn5+Vi5ciUGDBhglDgX0Gg0xZ5rMBhgMBiU13q9vlRiJCIiooqBa56p0tq4cSM0Go1RmTJlSqF2ly9fxtWrV9GoUaMSXyMqKgparVYp3OOZiIiocuOdZ6q0nnnmGcyfP9+ozsnJqVC7gi8DPsoTASMiIjBmzBjldcE+kURERFQ5MXmmSsvOzg7e3t4PbVerVi3UqFEDKSkpJb6GWq0u9kuIREREVPlw2QZVeRYWFujbty+WLVuGrKysQsdv3ryJe/fulUNkREREVNEweaZKy2Aw4OLFi0bl8uXLAIBXX30VERERSttp06bB3d0dbdq0wf/+9z8kJycjPT0dixcvRvPmzXHjxo3yGgYRERFVIFy2QZXW5s2b4erqalTn5+eHkydP4vz587Cw+P9/O9aoUQP79u3D9OnT8cknn+DcuXOoUaMGmjZtik8//RRarbaswyciIqIKSCV8dBrRY6PX66HVanF69kLY29qWdzhERESVSq23XimVfgs+v3U6HRwcHB7Ylss2iIiIiIhMxGUbRKWg5pC+D/3LlYiIiMwP7zwTEREREZmIyTMRERERkYmYPBMRERERmYjJMxERERGRiZg8Ez2ESqXCunXryjsMIiIiqgCYPFOVd/HiRYwcORINGjSAWq2Gu7s7unfvjm3btpV3aERERFTBcKs6qtLOnj2LkJAQODo6YubMmQgMDERubi5++eUXvP322zh58mR5h0hEREQVCJNnqtKGDx8OlUqFAwcOwM7OTqlv3LgxXn/99XKMjIiIiCoiJs9UZeXk5GDz5s2YOnWqUeJcwNHR8aF9GAwGGAwG5bVer3+cIRIREVEFwzXPVGWdOnUKIoJGjRo9ch9RUVHQarVKcXd3f4wREhERUUXD5JmqLBEB8PduGo8qIiICOp1OKZmZmY8rPCIiIqqAmDxTleXj4wOVSoWUlJRH7kOtVsPBwcGoEBERUeXF5JmqLCcnJ4SFhSEmJgY3b94sdPzatWtlHxQRERFVaEyeqUr78ssvkZeXh9atWyMuLg7p6elISUnB3LlzERwcXN7hERERUQXD3TaoSvPy8sKRI0cwdepUjB07FtnZ2ahVqxaCgoIwf/788g6PiIiIKhiVFHxrioj+Nb1eD61WC51Ox/XPREREZqIkn99ctkFEREREZCImz0REREREJmLyTERERERkIibPREREREQmYvJMRERERGQiJs9ERERERCZi8kyVWnh4OFQqFYYNG1bo2PDhw6FSqRAeHl72gREREZFZYvJMlZ67uztWrlyJ27dvK3V37tzBihUr4OHhUY6RERERkblh8kyVXosWLeDh4YE1a9YodWvWrIG7uzueeOIJpe769esYMGAA7Ozs4OrqiujoaISGhmL06NHlEDURERFVREyeqUp47bXXEBsbq7xevHgxXn/9daM2Y8aMwZ49e7B+/Xps3boVu3btwpEjR8o6VCIiIqrAmDxTlTBw4EDs3r0bZ8+exblz57Bnzx688soryvHr169j6dKlmDVrFjp06IAmTZogNjYWeXl5D+zXYDBAr9cbFSIiIqq8LMs7AKKyULNmTXTr1g1Lly6FiKBbt26oWbOmcvzMmTPIzc1F69atlTqtVgs/P78H9hsVFYXJkyeXWtxERERUsfDOM1UZr7/+OpYsWYKlS5cWWrIhIgAAlUpVZH1xIiIioNPplJKZmfl4gyYiIqIKhckzVRldunTB3bt3cffuXYSFhRkda9iwIaysrHDgwAGlTq/XIz09/YF9qtVqODg4GBUiIiKqvLhsg6qMatWqISUlRfn5fvb29hg0aBDGjRsHJycn1K5dG5GRkbCwsCh0N5qIiIiqLt55pirlQXeH58yZg+DgYDz//PPo2LEjQkJC4O/vDxsbmzKOkoiIiCoqlTxsUSdRFXXz5k3UrVsXs2fPxuDBg006R6/XQ6vVQqfTcQkHERGRmSjJ5zeXbRD9n8TERJw8eRKtW7eGTqfDxx9/DADo2bNnOUdGREREFQWTZ6L7zJo1C6mpqbC2tkZQUBB27dpltKUdERERVW1Mnon+zxNPPIHDhw+XdxhERERUgfELg0REREREJmLyTERERERkIibPREREREQmYvJMRERERGQiJs9UKalUqgeW8PBwo/bx8fFQqVS4du1aob7q16+Pzz77rEziJiIiooqNu21QpZSdna38vGrVKnz00UdITU1V6mxtbcsjLCIiIjJzTJ6pUnJxcVF+1mq1UKlURnVEREREj4LJM9G/YDAYYDAYlNd6vb4coyEiIqLSxuSZ6D716tUrVHfr1q1i20dFRWHy5MmlGRIRERFVIEyeie6za9cu2NvbG9WFhoYW2z4iIgJjxoxRXuv1eri7u5dWeERERFTOmDwT3cfLywuOjo5GdZaWxf9nolaroVarSzkqIiIiqii4VR0RERERkYmYPFOV9MUXX6BDhw7lHQYRERGZGSbPVCVdvnwZp0+fLu8wiIiIyMyoRETKOwiiykKv10Or1UKn08HBwaG8wyEiIiITlOTzm3eeiYiIiIhMxOSZiIiIiMhETJ6JiIiIiEzE5JmIiIiIyERMnomIiIiITMTkmYiIiIjIREyeiYiIiIhMxOSZzF54eDhUKhVUKhUsLS3h4eGBt956C1evXlXaJCYm4vnnn0ft2rVhY2OD+vXro2/fvrh8+bJRX3FxcQgNDYVWq4VGo0FgYCA+/vhj5OTklPWwiIiIqAJi8kyVQpcuXZCdnY2zZ8/i66+/xoYNGzB8+HAAwKVLl9CxY0fUrFkTv/zyC1JSUrB48WK4urri1q1bSh8TJ05E37590apVK/z88884fvw4Zs+ejd9//x3ffvtteQ2NiIiIKhDL8g6A6HFQq9VwcXEBANSrVw99+/bFkiVLAAAJCQnQ6/X4+uuvYWn591vey8sLzz77rHL+gQMHMG3aNHz22Wd45513lPr69eujU6dOuHbtWpmNhYiIiCou3nmmSufMmTPYvHkzrKysAAAuLi64d+8e1q5di+KeRr9s2TJoNBrlbvU/OTo6FllvMBig1+uNChEREVVeTJ6pUti4cSM0Gg1sbW3RsGFDJCcn47333gMAPPnkk5gwYQL69++PmjVromvXrvj000/x559/Kuenp6ejQYMGSsJtqqioKGi1WqW4u7s/1nERERFRxaKS4m7FEZmJ8PBw/PHHH5g/fz5u3bqFr7/+Gmlpadi4caOyTAMArly5gu3bt2Pfvn1Yt24dcnJy8Ntvv6Fp06bo2rUrsrOzkZSUVKJrGwwGGAwG5bVer4e7uzt0Oh0cHBwe1xCJiIioFOn1emi1WpM+v3nnmSoFOzs7eHt7IzAwEHPnzoXBYMDkyZON2jg7O6NPnz6YPXs2UlJS4ObmhlmzZgEAfH19cfr0aeTm5pboumq1Gg4ODkaFiIiIKi8mz1QpRUZGYtasWcjKyiryuLW1NRo2bIibN28CAPr3748bN27gyy+/LLI9vzBIREREAHfboEoqNDQUjRs3xrRp09ClSxesXLkS/fr1g6+vL0QEGzZswKZNmxAbGwsAaNOmDcaPH4+xY8fijz/+wH/+8x+4ubnh1KlT+Oqrr/DUU08Z7cJBREREVROTZ6q0xowZg9deew19+vRB9erVMXbsWGRmZkKtVsPHxwdff/01Bg4cqLSfMWMGgoKCEBMTg6+++gr5+flo2LAhevfujUGDBpXjSIiIiKii4BcGiR6jknzhgIiIiCoGfmGQiIiIiKgUMHkmIiIiIjIRk2ciIiIiIhMxeSYiIiIiMhGTZyIiIiIiEzF5JiIiIiIyEZNnqlIuXryIkSNHokGDBlCr1XB3d0f37t2xbdu28g6NiIiIzAAfkkJVxtmzZxESEgJHR0fMnDkTgYGByM3NxS+//IK3334bJ0+eLO8QiYiIqIJj8kxVxvDhw6FSqXDgwAHY2dkp9Y0bN8brr78OAJgzZw5iY2Nx5swZODk5oXv37pg5cyY0Gk15hU1EREQVCJdtUJWQk5ODzZs34+233zZKnAs4OjoCACwsLDB37lwcP34cS5cuxfbt2zF+/Phi+zUYDNDr9UaFiIiIKi8mz1QlnDp1CiKCRo0aPbDd6NGj8cwzz8DLywvPPvsspkyZgu+//77Y9lFRUdBqtUpxd3d/3KETERFRBcLkmaoEEQEAqFSqB7bbsWMHOnXqhLp168Le3h6vvvoqrly5gps3bxbZPiIiAjqdTimZmZmPPXYiIiKqOJg8U5Xg4+MDlUqFlJSUYtucO3cOzz33HJo0aYK4uDgcPnwYMTExAIDc3Nwiz1Gr1XBwcDAqREREVHkxeaYqwcnJCWFhYYiJiSnyLvK1a9dw6NAh3Lt3D7Nnz8aTTz4JX19fZGVllUO0REREVFExeaYq48svv0ReXh5at26NuLg4pKenIyUlBXPnzkVwcDAaNmyIe/fuYd68eThz5gy+/fZbfPXVV+UdNhEREVUgTJ6pyvDy8sKRI0fwzDPPYOzYsWjSpAk6deqEbdu2Yf78+WjevDnmzJmDGTNmoEmTJli2bBmioqLKO2wiIiKqQFRS8E0qIvrX9Ho9tFotdDod1z8TERGZiZJ8fvPOMxERERGRiZg8ExERERGZiMkzEREREZGJmDwTEREREZmIyTMRERERkYmYPBMRERERmYjJMxERERGRiZg8U5UQHh4OlUqF6dOnG9WvW7cOKpWqnKIiIiIic8PkmaoMGxsbzJgxA1evXi3vUIiIiMhMMXmmKqNjx45wcXEp9pHbV65cwcsvv4x69eqhevXqaNq0KVasWFHGURIREVFFxuSZqoxq1aph2rRpmDdvHi5cuFDo+J07dxAUFISNGzfi+PHjeOONNzBw4EDs37+/2D4NBgP0er1RISIiosqLyTNVKf/5z3/QvHlzREZGFjpWt25dvPvuu2jevDkaNGiAkSNHIiwsDD/88EOx/UVFRUGr1SrF3d29NMMnIiKicsbkmaqcGTNmYOnSpUhOTjaqz8vLw9SpUxEYGAhnZ2doNBps2bIF58+fL7aviIgI6HQ6pWRmZpZ2+ERERFSOmDxTldOuXTuEhYVhwoQJRvWzZ89GdHQ0xo8fj+3btyMpKQlhYWG4e/dusX2p1Wo4ODgYFSIiIqq8LMs7AKLyEBUVhSeeeAK+vr5K3a5du9CzZ0+88sorAID8/Hykp6fD39+/vMIkIiKiCoZ3nqlKCgwMxIABAzBv3jylztvbG1u3bkVCQgJSUlLw5ptv4uLFi+UYJREREVU0TJ6pypoyZQpERHn94YcfokWLFggLC0NoaChcXFzwwgsvlF+AREREVOGo5P7sgYj+Fb1eD61WC51Ox/XPREREZqIkn9+880xEREREZCImz0REREREJmLyTERERERkIibPREREREQmYvJMRERERGQiJs9EDzBp0iQ0b968vMMgIiKiCoLJMxERERGRiZg8ExERERGZiMkzVTrXr1/HgAEDYGdnB1dXV0RHRyM0NBSjR4/GvHnz0LRpU6XtunXroFKpEBMTo9SFhYUhIiKiPEInIiKiCo7JM1U6Y8aMwZ49e7B+/Xps3boVu3btwpEjRwAAoaGhOHHiBC5fvgwA2LlzJ2rWrImdO3cCAO7du4eEhAS0b9++3OInIiKiiovJM1Uq169fx9KlSzFr1ix06NABTZo0QWxsLPLy8gAATZo0gbOzs5Isx8fHY+zYscrrgwcP4s6dO3jqqadMup7BYIBerzcqREREVHkxeaZK5cyZM8jNzUXr1q2VOq1WCz8/PwCASqVCu3btEB8fj2vXruHEiRMYNmwY8vLykJKSgvj4eLRo0QIajcak60VFRUGr1SrF3d29VMZFREREFQOTZ6pURATA30lyUfXA30s34uPjsWvXLjRr1gyOjo5o164ddu7cifj4eISGhpp8vYiICOh0OqVkZmY+lnEQERFRxcTkmSqVhg0bwsrKCgcOHFDq9Ho90tPTldcF655Xr16tJMrt27fHr7/+WuL1zmq1Gg4ODkaFiIiIKi8mz1Sp2NvbY9CgQRg3bhx27NiBEydO4PXXX4eFhYVyN7pg3fOyZcuU5Dk0NBTr1q3D7du3TV7vTERERFUPk2eqdObMmYPg4GA8//zz6NixI0JCQuDv7w8bGxsAfy/pKLi7/PTTTwMAAgMDodVq8cQTT/DuMRERERVLJfcvBiWqhG7evIm6deti9uzZGDx4cKleS6/XQ6vVQqfTMQknIiIyEyX5/LYso5iIykxiYiJOnjyJ1q1bQ6fT4eOPPwYA9OzZs5wjIyIiInPH5JkqpVmzZiE1NRXW1tYICgrCrl27ULNmzfIOi4iIiMwck2eqdJ544gkcPny4vMMgIiKiSohfGCQiIiIiMhGTZyIiIiIiEzF5JiIiIiIyEZNnIiIiIiITMXkms3fp0iW8+eab8PDwgFqthouLC8LCwrB3716jdgkJCahWrRq6dOlSqI+zZ89CpVIpRavV4sknn8SGDRvKahhERERkBpg8k9nr1asXfv/9dyxduhRpaWlYv349QkNDkZOTY9Ru8eLFGDlyJHbv3o3z588X2devv/6K7Oxs7N+/H61bt0avXr1w/PjxshgGERERmQE+YZDM2rVr11CjRg3Ex8crj9wuys2bN+Hq6oqDBw8iMjISAQEB+Oijj5TjZ8+ehZeXFxITE9G8eXMAwPXr1+Hg4IC5c+di5MiRJsXDJwwSERGZn5J8fvPOM5k1jUYDjUaDdevWwWAwFNtu1apV8PPzg5+fH1555RXExsbiQX835ubmYtGiRQAAKyurYtsZDAbo9XqjQkRERJUXk2cya5aWlliyZAmWLl0KR0dHhISEYMKECTh69KhRu2+++QavvPIKAKBLly64ceMGtm3bVqi/tm3bQqPRwMbGBmPHjkX9+vXx0ksvFXv9qKgoaLVapbi7uz/eARIREVGFwuSZzF6vXr2QlZWF9evXIywsDPHx8WjRogWWLFkCAEhNTcWBAwfQr18/AH8n3H379sXixYsL9bVq1SokJiZi/fr18Pb2xtdffw0nJ6dirx0REQGdTqeUzMzMUhkjERERVQxc80yV0pAhQ7B161acO3cO48ePx6effopq1aopx0UEVlZWyM7ORo0aNYpc87xz50706tULycnJqF27tknX5ZpnIiIi88M1z1TlBQQE4ObNm7h37x7+97//Yfbs2UhKSlLK77//Dk9PTyxbtqzYPtq3b48mTZpg6tSpZRg5ERERVWRMnsmsXblyBc8++yy+++47HD16FBkZGfjhhx8wc+ZM9OzZExs3bsTVq1cxePBgNGnSxKj07t0b33zzzQP7Hzt2LBYsWIA//vijjEZEREREFRmTZzJrGo0Gbdq0QXR0NNq1a4cmTZrgww8/xNChQ/HFF1/gm2++QceOHaHVagud26tXLyQlJeHIkSPF9v/888+jfv36vPtMREREALjmmeix4ppnIiIi88M1z0REREREpYDJMxERERGRiZg8ExERERGZiMkzEREREZGJmDwTEREREZmIyTMRERERkYmYPBMRERERmYjJM5m9S5cu4c0334SHhwfUajVcXFwQFhaG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"/>

Linear regression model보다 Random Forest Regressor Model이 더 적절

