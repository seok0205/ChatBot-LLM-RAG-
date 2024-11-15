import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import json

# seok.env파일에서 환경변수 로드.
load_dotenv(dotenv_path='seok.env')

# api key 변수에 환경변수 파일의 api key 저장.
api_key = os.getenv("API_KEY")

# Chat 모델 초기화.
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# PDF 파일 로드.
loader = PyPDFLoader("pdf/한권으로 끝내는 주식과 세금.pdf")

# 파일 로드.
docs = loader.load()

# 차례의 내용들이 큰 의미가 없다고 판단해 차례 페이지는 과감히 제외.
filtered_docs = [doc for i, doc in enumerate(docs) if not (3 <= i <= 13)]

# 텍스트 청킹.
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, # 200자 단위로 분할
    chunk_overlap=20,   # 문맥 유지용 20자 중복
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(filtered_docs)

# 임베딩 생성 및 Vector store 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)  # FAISS로 벡터 스토어 구축
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})  # 유사한 정보 검색 시스템(retriever)으로 사용

# DebugPassThrough class 정의(RAG chain의 디버깅을 위함. 입력된 값 그대로 출력 후 반환.)
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
    
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가(확장성 및 유연성).
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# JSON 형식의 프롬프트 파일 정의. 시스템 메시지 및 유저 질문.
prompt_files = ["Prompts/prompt1.json", "Prompts/prompt2.json", "Prompts/prompt3.json"]

# 결과 텍스트 파일 저장할 위치 생성.
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# 각 프롬프트 파일에 RAG 체인 설정.
for prompt_file in prompt_files:
    # 각 프롬프트 파일 읽기
    with open(prompt_file, 'r', encoding='utf=8') as file:  # 읽기 모드로 프롬프트 파일 염.
        prompt_text = json.load(file)   # 딕셔너리 형식으로 읽음

    # 프롬프트 파일의 내용을 사용, 프롬프트 템플릿 정의
    contextual_prompt = ChatPromptTemplate.from_messages([
        (prompt_text["System"]),
        (prompt_text["Question"])
    ])

    # 중간 단계 데이터 흐름 확인하는 DebugPassThrough 추가
    rag_chain_debug = {
        "context": retriever,   # retriever 유사 질문 검색기
        "question": DebugPassThrough()  # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
    } | DebugPassThrough() | ContextToText() | contextual_prompt | model
    
    '''
    위에서 만든 기능들 연결해서 RAG 체인 구성.
    1. retriever가 질문에 대한 유사한 정보 검색해서 context에 저장함.
    2. DebugPassThrough()는 사용자가 입력한 질문을 체크.(디버깅)
    3. 또 다시 DebugPassThrough()를 통해 위 1, 2 과정 잘 출력이 되고 있는지 흐름 확인.
    4. ContextToText()는 검색된 context정보를 하나의 문자열로 합쳐서 모델에 전달할 텍스트 구성.
    5. contextual_prompt는 모델이 이해할 수 있는 형태로 프롬프트 생성.
    6. 모델에 프롬프트 전달 후 답변 생성.
    '''

    # 질문
    query = prompt_text["Question"]

    # 결과 파일 생성.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # 결과 파일 생성 시간.
    result_filename = f"{os.path.splitext(os.path.basename(prompt_file))[0]}_{timestamp}.txt"   # 결과 파일 이름 구조.
    result_path = os.path.join(result_dir, result_filename) # 결과 파일 경로.

    with open(result_path, 'a', encoding='utf=8') as file:  # 파일 만들고 열어서 모델 출력 값 입력.
        result = rag_chain_debug.invoke(query)  # 질문을 RAG 체인에 전달 후 결과 생성.
        file.write(f"Question: {query}\n")
        file.write(f"Answer: {result}\n\n")