![image](https://github.com/suted2/Kor_llama/assets/101646531/3e2050be-4771-4797-93fd-c41329fc6e86)# Kor_llama
llama 를 한국어 기반으로  Train 하고자 하는 mini project입니다. 

|학습|시간 | GPU |
|-----|------|------|
|Colab|60hour| RTX 4090 |


### EVAL 
![image](https://github.com/suted2/Kor_llama/assets/101646531/5792fe5c-d505-4b94-8f4b-cf3e924621b1)

높은 점수는 아니지만 확실하게 finetuning 되었다는 것을 알 수 있습니다. 

## Data 

Dacon 에서 진행된 고객 대출등급 구분 [해커톤](https://dacon.io/competitions/official/236214/overview/description) 의 데이터를 사용하였습니다. 





## Model 

base model : LLama2 
![image](https://github.com/suted2/Kor_llama/assets/101646531/f06338c0-30cf-4bc3-a7f8-50469154a7dc)


## Dataset 


![image](https://github.com/suted2/Kor_llama/assets/101646531/0f8d9245-b281-44c7-b6cf-4694cee54b14)


|데이터 col 이름|	type|	preprocessing|
|--------|-----|-----------|
|대출금액|	int	| |
|대출기간	|object|	month 제거| 
|근로기간	|object	|year 제거|
|주택소유상태	|object	| |
|연간소득 |	int| |	
|부채/소득 |	float |	 |
|총계좌수 |	int |	|
|대출목적 |	object |	영어로 translate |
|최근 2년 연체 |	int |	|
|총 상환 원금 |	float | |	
|총 면제 금액 |	float | |	
|연체 계좌수	int	| |
|대출 등급	|object	| int2label / label2int  dict 생성| 

 
## FineTuning 

`hugging face` 의 Transformer library, ( PEFT, STF , Trainer ) 등을 활용해서 진행했습니다. 


`Instruct Tuning` 을 진행하였고 , 
7B의 모델을 PEFT 방식으로 Train 했습니다. 

`Peft Parameter`
```python 
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                        inference_mode=False, # 학습하는지  
                        r=8, # 작을 수록 trainable 한 파라미터의 개수가 낮아진ㄷ.ㅏ  
                        lora_alpha=16,  # scaling factor 
                        lora_dropout=0.1) # dropout


Trainable: 4194304 | total: 6860050432 | Percentage: 0.0611%
```

`huggingface Trainer`
```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="peftllama0116",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=3000,
    logging_steps=100,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    fp16=True,
    push_to_hub=False,
    optim = "adamw_torch",
    save_strategy = "steps",
    save_steps = 1000,
    save_total_limit=2

)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
    eval_dataset=val_tokenized_datasets
)

```

모델 생성 configure

```python
generate_config = GenerationConfig(
        pad_token_id = tokenizer.eos_token_id,
        do_sample=True,
        top_k=1,
        top_p = 0.9,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=100,
        temperature = 0.8
    )


```




## Result 


`before tuning` 
```PlAIN TEXT
you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 16800000 , take out a loan during 60 months. 
He/she works for 10 years. Status of home ownership is MORTGAGE
His/her annual income was about 82680000. His/her total debt/income ratio was 21.960000 
He/she have total 27 accounts, and he/she take out a loan for debt consolidation.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 199284. His/Her total interst pay $ 159924.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : 

Guess the credit rating of A person

I think his credit rating is CCC+ 

###Condition:
A person owe $ 16800000 , take out a loan during 60 months. 
He/she works for 10 years. Status of home ownership is MORTGAGE
His/her annual income was about 82680000. His/his total debt/income ratio was 21.960000 
He/she have total 27 accounts, and he/she take out a loan for debt consolidation.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 199284. His/Her total interst pay $ 159924.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###Credit Rating : 

Guess the credit rating of B person

I think his credit rating is CCC 

###Condition:
B person owe $ 16800000 , take out a loan during 60 months. 
He/she works

```

```

you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 16800000 , take out a loan during 36 months. 
He/she works for 8 years. Status of home ownership is MORTGAGE
His/her annual income was about 132000000. His/her total debt/income ratio was 19.640000 
He/she have total 12 accounts, and he/she take out a loan for housing improvement.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 394692. His/Her total interst pay $ 146604.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : 

#@시스템#사진#
#@이모티콘#
오케이~~^^❤️


## 다른 학습된 멘트가 나오게 된다. 아마 SNS 데이터를 학습시켜둔 것 같음. 


```



`after tuning` 

```
you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 16800000 , take out a loan during 36 months. 
He/she works for 8 years. Status of home ownership is MORTGAGE
His/her annual income was about 132000000. His/her total debt/income ratio was 19.640000 
He/she have total 12 accounts, and he/she take out a loan for housing improvement.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 394692. His/Her total interst pay $ 146604.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : 

###Rating Description : 

###Total Debt : 

###Total Income : 

###Total Debt/Income Ratio : 

###Other Conditions : 

###Credit Rating : B 

###Rating Description : 

###Total Debt : 

###Total Income : 

###Total Debt/Income Ratio :

```


```
you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 16800000 , take out a loan during 60 months. 
He/she works for 10 years. Status of home ownership is MORTGAGE
His/her annual income was about 82680000. His/her total debt/income ratio was 21.960000 
He/she have total 27 accounts, and he/she take out a loan for debt consolidation.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 199284. His/Her total interst pay $ 159924.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : 

###Rating Description : 

###Total Debt : 

###Total Annual Interest Payment : 

###Overdue Payment : 

###Debt Consolidation : 

###Other Condition : 

###Credit Rating : B

```

`Prompt Engineering`

좀 더 명확한 지시문을 제시했을 경우 

```
you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 16800000 , take out a loan during 60 months. 
He/she works for 10 years. Status of home ownership is MORTGAGE
His/her annual income was about 82680000. His/her total debt/income ratio was 21.960000 
He/she have total 27 accounts, and he/she take out a loan for debt consolidation.
He/she overdue of interest 0 times over last 2 year.
His/Her total redemption principal $ 199284. His/Her total interst pay $ 159924.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : C 


###justification : overdue payment was zero 


###derivation : see attached file 


###recommendation : no need to take any action 


###footnotes : see attached
```

```
you are financial speciallist. And you will see ###condition of one person. and judge his/her Credit rating
response about ###instruction 

###Condition
he/she owe $ 7200000 , take out a loan during 36 months. 
He/she works for 1 years. Status of home ownership is MORTGAGE
His/her annual income was about 102000000. His/her total debt/income ratio was 19.910000 
He/she have total 34 accounts, and he/she take out a loan for debt consolidation.
He/she overdue of interest 1 times over last 2 year.
His/Her total redemption principal $ 550020. His/Her total interst pay $ 106968.000000 
His/Her total overdue payment $ 0.000000. He/She have 0.000000 overdue account for total 

###instruction: with those conditions, Guess His Credit Rating in one of A,B,C,D,E,F,G 


###Credit Rating : A 


###Justification For The Credit Rating : overdue payment was not made within 30 days


###Debtors should not be given credit if they have been overdue payment more than once in last 2 year.


###If there was no late payment, give them Credit Rating : A 


###Other conditions : overdue payment was not made within

```

### INSIGHT 


1. max_token_length : 생성 토큰의 갯수에 따라 , ( max token 을 top k 를 통해 앞에 생성할 token 의 개수를 보고 설정하기에 )
   차이가 존재한다. 
`max_token 20` 
![image](https://github.com/suted2/Kor_llama/assets/101646531/528fe059-e253-420b-92c6-9b2b98d5245c)

`max_token 100` 
![image](https://github.com/suted2/Kor_llama/assets/101646531/c66a139d-080c-4689-b266-023d778b2fb0)

2.  Peft 도중 step 이 save 되더라도 adapter 가 아닌 configure 가 저장 되는 구조이기에 원하는
    
    방식대로 finetuning 된 모델이 나오지 않는다. 
    
3.  학생의 점수 , 실력, 참여도를 자체적인 기준을 세워서 정립하고 이를 바탕으로 LLM을 학습을 시킨다면 좋은 결과를 얻을 수 있을 것이라고 생각한다.
    - 보조교사, 도우미, 공정한 채점 기준 등의 지표로 작동할 수 있지 않을까 ? ( 신뢰성 여부 판별 不 )



## Reference 

1. [Llama 레시피 북](https://github.com/facebookresearch/llama-recipes/)
2. [Llama 깃 허브](https://github.com/facebookresearch/llama)
3. [Llama 허깅페이스](https://huggingface.co/meta-llama/Llama-2-7b)
4. [Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
