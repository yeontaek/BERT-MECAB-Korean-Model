# BERT with MECAB for Korean text

기존 Sentencepiece로 구축된 [한국어 BERT Model](https://github.com/yeontaek/BERT-Korean-Model)은 한국어 조사가 제대로 분리되지 않는 문제가 있었습니다. 이로 인해 KorQuAD Task에서는 EM이 과도하게 떨어지는 문제가 발생하였습니다. 예를 들어 <code>"담수와"</code> 라는 토큰은 <code>"담수" + "##와"</code>로 구분되어야 하지만 아래와 같이 <code>"담" + "##수와"</code>로 구분되어 학습하였습니다.

```
담수와 염수가 급작스럽게 섞일 경우 대부분의 수생생물이 폐사하는 원인은?

['담', '##수와', '염', '##수가', '급', '##작', '##스럽게', '섞', '##일', '경우', '대부분의',
'수', '##생', '##생물', '##이', '폐', '##사', '##하는', '원인은', '?']

```

이러한 문제를 해결하고자 본 repository에서는 Mecab tokenizer를 이용해 사전을 구축하여 학습을 진행하였고 성능 결과를 공유합니다. MECAB으로 구축한 사전 파일을 첨부하였으니, 한국어 BERT Model 학습에 도움이 되었으면 합니다.


<br>


# 사전 구축 

한국어 위키데이터 350만 문장을 사용하였고 각 문장의 한 어절씩 <code>mecab.morphs</code>을 수행하였습니다. 또한 <code>wordpiece_tokenizer</code>을 그대로 사용하기 위해서 tokenizer 된 2번째 토큰부터는 "##"을 추가하였습니다.

```python
    import mecab
    mecab = mecab.MeCab()
    morph = []
    sentence = "담수와 염수가 급작스럽게 섞일 경우 대부분의 수생생물이 폐사하는 원인은?"
    for st in sentence.split(" "):
        count = 0
        for token in mecab.morphs(st):
            tk = token
            if count > 0:
                tk = "##" + tk
                morph.append(tk)
            else:
                morph.append(tk)
                count += 1
                
     print(morph)
                
['담수', '##와', '염수', '##가', '급작', '##스럽', '##게', '섞일', '경우', '대부분', '##의',
'수생', '##생물', '##이', '폐사', '##하', '##는', '원인', '##은', '##?']
```
<br>

Sentencepiece와 다르게 subword로 구분되지 않아 tokenizer 후 출현 빈도 기준으로 최대 128,000개의 단어로 사전을 구성하였고, BERT에서 필요한 5개의 token([PAD], [UNK], [CLS], [SEP], [MASK])을 추가하여 총 128,005개의 단어로 구성했습니다.  

mecab 설치와 관련된 자세한 사항은 [링크](https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/) 확인하시길 바랍니다. 

<br>


## BERT Pre-training
**한국어 위키데이터(2019.01 dump file, 약 350만 문장)** 을 사용하여 학습을 진행하였으며, 모델의 하이퍼파라미터는 논문과 동일하게 사용하였습니다. 오리지널 논문과 동일하게 구축하고자 n-gram masking은 적용하지 않았습니다. 학습 방법은 논문에 서술된 것처럼 128 length 90%, 512 length 10%씩 학습을 진행하여, 총 100만 step을 진행했습니다. 
<br>
<br>
* 학습 파라미터(seq_length=128)
```python
learning_rate = 1e-4
train_batch_size = 256 
max_seq_length = 128
masked_lm_prob = 0.15
max_predictions_per_seq = 20
num_train_steps = 900000
```   

* 학습 파라미터(seq_length=512)
```python
learning_rate = 1e-4
train_batch_size = 256 
max_seq_length = 512
masked_lm_prob = 0.15
max_predictions_per_seq = 77
num_train_steps = 100000
```   
<br>


## Step별 성능 비교
Base Model 기준으로 총 120만 step을 학습을 진행하였고 기존 Sentencepiece로 구축한 Model과 성능 비교 결과는 아래와 같습니다. 측정 기준은 step별 KorQuAD Task의 F1,EM으로 측정하였습니다.
<br>

* Base Model(12-layer, 768-hidden, 12-heads)<br>

| Step | seq_length | Sentencepiece F1 | Sentencepiece EM | Mecab F1 | Mecab EM |
|:-------:|:-------:|:-------:| :-------:| :-------:| :-------:|
| 40만 | 128 | 73.98% | 44.49% | 77.41% | 62.12% |
| 60만 | 128 | 77.15% | 46.89% | 78.63% | 63.17% |
| 90만 | 128| 78.64% | 48.33% | 80.98% | 65.25% |
| 100만 | 512 | 87.8% | 59.0% | **91.40%** | **79.47%** |
| 120만 | 512 | 88.19% | 59.87% | 00% | 00% |


<br>


## 성능 평가 
성능 비교를 위해 BERT-Multilingual Model과 실험을 진행하였으며, [Google BERT github](https://github.com/google-research/bert)의 SQUAD Task 기본 하이퍼파라미터를 사용하였습니다. KorQuAD 성능 결과는 아래와 같습니다.

| Model | F1 | EM |
|:---:|:---:| :---:|
| BERT-Base, Multilingual Cased | 89.9% | 70.29% |
| **BERT with SentencePiece(our model)** | 87.8% | 59.09% |
| **BERT with Mecab(our model)** | 91.40% | 79.47% |


