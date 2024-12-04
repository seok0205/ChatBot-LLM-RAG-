# CH 3

## LLM, RAG 활용한 Chat Bot 구현

1. 주제 및 결과 코드
    [ChatBot](./ChatBot.ipynb)  
    - RAG의 정보 검색으로 LLM의 응답에 정확한 답변 생성 기능을 추가한 챗봇.

2. 주요 기능
    [ChatBot(save result)](./ChatBot(save%20result%20fuction).py)  
    - LLM 모델에 RAG 기능(정보 검색)을 추가해 더욱 정확, 상세한 답변 출력.
    - 외부 저장된 프롬프트를 불러와 결과값을 외부 파일에 텍스트 파일로 저장.

    - [저장된 프롬프트](./Prompts)
    - [저장된 결과물](./result)

3. 주요 기술

    - RAG chains 구성.

    1. retriever가 질문에 대한 유사한 정보 검색해서 context에 저장

    2. DebugPassThrough()는 사용자가 입력한 질문을 체크(디버깅)

    3. 또 다시 DebugPassThrough()를 통해 위 1, 2 과정 잘 출력이 되고 있는지 흐름 확인

    4. ContextToText()는 검색된 context정보를 하나의 문자열로 합쳐서 모델에 전달할 텍스트 구성

    5. contextual_prompt는 모델이 이해할 수 있는 형태로 프롬프트 생성

    6. 모델에 프롬프트 전달 후 답변 생성
