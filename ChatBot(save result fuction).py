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

# 환경변수 파일 로드
load_dotenv(dotenv_path='seok.env')

# api key 변수에 환경변수 파일의 api key 저장
api_key = os.getenv("API_KEY")

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("pdf/한권으로 끝내는 주식과 세금.pdf")

# 페이지 별 문서 로드
docs = loader.load()

# 차례의 내용들이 큰 의미가 없다고 판단해 차례 페이지는 과감히 제외.
filtered_docs = [doc for i, doc in enumerate(docs) if not (3 <= i <= 13)]

# 텍스트 청킹
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(filtered_docs)

# 임베딩 및 Vector store 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# DebugPassThrough class 정의
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
    
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

prompt_files = ["Prompts/prompt1.json", "Prompts/prompt2.json", "Prompts/prompt3.json"]

result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# RAG chains에서 각 단계에 DebugPassThrough 추가
for prompt_file in prompt_files:
    # 각 프롬프트 파일 읽기
    with open(prompt_file, 'r', encoding='utf=8') as file:
        prompt_text = json.load(file)

    # 프롬프트 템플릿 정의
    contextual_prompt = ChatPromptTemplate.from_messages([
        (prompt_text["System"]),
        (prompt_text["Question"])
    ])

    # RAG 체인에서 각 단계마다 DebugPassThrough 추가
    rag_chain_debug = {
        "context": retriever,   # 컨텍스트를 가져오는 retriever
        "question": DebugPassThrough()  # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
    } | DebugPassThrough() | ContextToText() | contextual_prompt | model

    query = prompt_text["Question"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{os.path.splitext(os.path.basename(prompt_file))[0]}_{timestamp}.txt"
    result_path = os.path.join(result_dir, result_filename)

    with open(result_path, 'a', encoding='utf=8') as file:
        result = rag_chain_debug.invoke(query)
        file.write(f"Question: {query}\n")
        file.write(f"Answer: {result}\n\n")