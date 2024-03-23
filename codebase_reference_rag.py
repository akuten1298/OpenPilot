from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from milvus import default_server
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings('ignore')

# from dotenv import load_dotenv
import os

class CodeBaseReference:
    def returnReference(description):
        # load_dotenv()
        # print(OPEN_AI_API_KEY)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")

        # default_server.start()

        with open("./data/data.txt") as f:
            file_text = f.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 512, chunk_overlap = 64,
        )

        file_texts = []
        texts = text_splitter.split_text(file_text)
        # print(texts[0])
        for i, chunked_text in enumerate(texts):
            file_texts.append(Document(
                page_content=chunked_text,
                metadata={"doc_title": "title", "chunk_num": i}
            ))

        embeddings = OpenAIEmbeddings()

        vector_store = Milvus.from_documents(
            file_texts,
            embedding=embeddings,
            connection_args={"host": "localhost", "port": 19530},
            collection_name="github_reference"
        )

        # print(file_texts[0])
        retriever = vector_store.as_retriever()
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # default_server.stop()
        # default_server.cleanup()
        
        return chain.invoke(f"{description}. GIVE ME ONLY THE GITHUB REPO LINK AND NOTHING ELSE.")
        
# print(CodeBaseReference.returnReference("I am making a graphics library. Can you give me a repo link thats similar to my project?"))