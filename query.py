import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        # convert the user query to embeddings for matching with vectorstore data
        return self.model.encode(text, convert_to_numpy=True).tolist()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", "Your_API")

prompt = PromptTemplate(template="""Answer the question based only on the context. \n\nContext: {context} \n\nQuestion: {question}  """, input_variables=[
                        "context", "question"])


def rag_query(question, path):
    client = Groq(api_key=GROQ_API_KEY)

    MODEL = "llama-3.1-8b-instant"

    embedding_model = SentenceTransformersEmbeddings()

    vectorstore = Chroma(persist_directory=path,
                         embedding_function=embedding_model)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message.content
