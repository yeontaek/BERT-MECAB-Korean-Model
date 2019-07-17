# BERT with MECAB for Korean text

기존 Sentencepiece로 구축된 [한국어 BERT Model](https://github.com/yeontaek/BERT-Korean-Model)은 한국어 조사가 제대로 분리되지 않는 문제가 있었습니다. 이로 인해 KorQuAD Task에서는 EM이 과도하게 떨어지는 문제가 발생하였습니다. 예를 들어 <code>"노아의"</code> 라는 토큰은 <code>"노아" + "##의"</code>로 구분되어야 하지만 아래와 같이 <code>"노" + "##아의"</code>로 구분되어 학습되었습니다. 

```
노아의 방주에서 가장 처음 밖으로 내보낸 동물은?

=> ['노', '##아의', '방', '##주는', '총', '몇', '##층으로', '되어', '있었', '##는가', '?']

```

이러한 문제를 해결하고자 본 repository에서는 Mecab tokenizer를 이용해 사전을 구축하여 학습을 진행하였고 성능 결과를 공유합니다. MECAB으로 구축한 사전 파일을 첨부하였으니, 한국어 BERT Model 학습에 도움이 되길 바랍니다. 



# 사전 구축 

한국어 위키데이터 350만 문장을 사용하였고 각 문장의 한 어절씩 <code>mecab.morphs</code>을 수행하였습니다. 

Sentencepiece와 다르게 subword로 구분되지 않아 출현 빈도 기준으로 최대 128,000개의 단어로 사전을 구성하였고, BERT에서 필요한 5개의 token([PAD], [PAD], [PAD], [PAD], [PAD])을 추가하여 총 128005개의 단어로 구성했습니다.  

```python
    import mecab
    from tqdm import tqdm
    mecab = mecab.MeCab()

    RAW_DATA_FPATH = "../corpus_data/ko-wiki_20190604.txt"
    with open(RAW_DATA_FPATH, 'r', encoding='utf-8') as f:
        sentence = f.readlines()

    dict = {}
    for st in tqdm(sentence):
        if st != "\n":
            for sen in st.split(" "):
                count = 0
                for token in mecab.morphs(sen):
                    tk = token
                    if count > 0:
                        tk = "##" + tk
                    if tk in dict:
                        value = dict.get(tk)
                        dict[tk] = value + 1
                        count += 1
                    else:
                        dict[tk] = 1
                        count += 1

```  
mecab 설치과 관련된 자세한 사항은 [링크](https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/) 확인하시길 바랍니다. 


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

## Step별 성능 비교
Base Model 기준으로 총 100만 step을 학습을 진행하였고 기존 Sentencepiece로 구축한 Model과 성능 비교 결과는 아래와 같습니다. 측정 기준은 step별 KorQuAD Task의 F1,EM으로 측정하였습니다. 
<br>

* Base Model(12-layer, 768-hidden, 12-heads)<br>

| Step | seq_length | Sentencepiece F1 | Sentencepiece EM | Mecab F1 | Mecab EM |
|:---:|:---:|:---:| :---:| :---:| :---:|
| 40만 | 128 | 73.98% | 44.49% | 77.41% | 62.12% |
| 60만 | 128 | 77.15% | 46.89% | 00.00% | 00.00% |
| 90만 | 128| 78.64% | 48.33% | 00.00% | 00.00% |
| 100만 | 512 | 87.8% | 59.0% | 00.00% | 00.00% |





