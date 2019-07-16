# BERT with MECAB for Korean text

기존 Sentencepiece로 구축된 [한국어 BERT Model](https://github.com/yeontaek/BERT-Korean-Model)은 한국어 조사가 제대로 분리되지 않는 문제가 있었습니다. 이로 인해 KorQuAD Task에서 EM이 과도하게 떨어지는 문제가 발생하였습니다. 예를 들어 "노아의 방주에서" 라는 토큰은 "노아" + "##의" + "방주"+ "##에서"로 구분되어야 하지만 아래와 같이 " 

'''
노아의 방주에서 가장 처음 밖으로 내보낸 동물은?

['노', '##아의', '방', '##주는', '총', '몇', '##층으로', '되어', '있었', '##는가', '?']

'''

이러한 문제를 해결하고자 본 repository에서는 Mecab tokenizer를 이용해 사전을 구축하는 과정과 성능 결과를 공유합니다. MECAB으로 구축한 사전 파일을 첨부하였으니, 한국어 BERT Model 학습에 도움이 되길 바랍니다. 



# 사전 구축 





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
