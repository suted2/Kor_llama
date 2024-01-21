# Kor_llama
llama 를 한국어 기반으로  Train 하고자 하는 mini project입니다. 

|학습|시간 | GPU |
|-----|------|------|
|Colab|60hour| RTX 4090 |

## Data 

Dacon 에서 진행된 고객 대출등급 구분 [해커톤](https://dacon.io/competitions/official/236214/overview/description) 의 데이터를 사용하였습니다. 





## Model 

base model : LLama2 


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



## Result 


`before tuning` 



`after tuning` 



## Reference 

1. [Llama 레시피 북](https://github.com/facebookresearch/llama-recipes/)
2. [Llama 깃 허브](https://github.com/facebookresearch/llama)
3. [Llama 허깅페이스](https://huggingface.co/meta-llama/Llama-2-7b)
4. [Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
