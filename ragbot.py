from datasets import load_dataset
from langchain.vectorstores import Pinecone
import pinecone
# from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import ContextCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import (ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)


dataset = load_dataset(
    "jamescalam/llama-2-arxiv-papers-chunked",
    split="train"
)

# print(dataset)

embeddings = HuggingFaceEmbeddings()
# initialize pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),,
    environment='gcp-starter',
)
index_name="ragbot2"
index = pinecone.Index(index_name)

from tqdm.auto import tqdm  # for progress bar

data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['chunk'] for _, x in batch.iterrows()]
    # embed text
    embeds = embeddings.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

# print(index.describe_index_stats())

from langchain.retrievers import ContextualCompressionRetriever

vectorstore = Pinecone(index, embeddings, 'text')
# initialize base retriever
retriever = vectorstore.as_retriever()

def load_query_gen_prompt():
    return """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.
        Don't generate the search query if the user is conversing generally or engaging in small talk. In which case just return the original question.
    Chat History:
    {chat_history}
    Question:
    {question}

    Search query:
    """

def load_system_prompt():
    return """
       System: You are an intelligent and friendly AI assistant. 
      The AI is talkative and provides lots of specific details from its context. 
      ALWAYS return a "SOURCES" part in your answer, except for small-talk conversations. 
      Answer with the facts listed in the list of sources below. If there isn't enough information below, say you don't know.
      If asking a clarifying question to the user would help, ask the question. 

      User: Hi AI, how are you today?

      Assistant: I'm great thank you. How can I help you?

      User: I'd like to understand how BERT works.

      Assistant:
      
      Sources:
      
Chat History: {chat_history}
"""


query_gen_prompt = load_query_gen_prompt()
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(query_gen_prompt)
system_msg = load_system_prompt()
llm = ChatOpenAI(temperature=0.3, verbose=False, model='gpt-3.5-turbo', openai_api_key = os.environ.get("OPENAI_API_KEY"))

memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', max_token_limit=500)
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
messages = [SystemMessagePromptTemplate.from_template(system_msg), HumanMessagePromptTemplate.from_template("{summaries}, {question}")]
prompt = ChatPromptTemplate.from_messages(messages)

answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=False,prompt=prompt)

chain = ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator,
            combine_docs_chain=answer_chain,
            verbose=False,
            memory=memory,
            rephrase_question=False
)

query = "explain llama 2 in 100 words"
result = chain({"question": query})
print(result['chat_history'][-1].content)
"""
>>>LLAMA 2 is a language model developed by Facebook AI Research. It is trained using a pretraining and fine-tuning process. 
In the pretraining phase, the model is trained on a large corpus of text data from the internet to learn general language patterns.
During fine-tuning, the model is trained on specific tasks, such as question answering or sentiment analysis, using a smaller dataset specific to the task. 
The development of LLAMA 2 includes safety measures such as safety-specific data annotation, red-teaming, iterative evaluations, and applying basic safety techniques. 
The aim is to encourage responsible AI innovation and collaboration within the AI community to improve the safety of the models.

Source: http://arxiv.org/pdf/2307.09288
"""

query = "what safety measures were used in the development of llama 2? answer in less than 100 words"
result = chain({"question": query})
print(result['chat_history'][-1].content)
"""
The safety measures used in the development of Llama 2 include safety-specific data annotation and tuning, red-teaming, 
iterative evaluations, and applying basic safety techniques at the user input and model output layers. 
The developers also provide code examples to help replicate their safe generations and share a Responsible Use Guide for guidelines on safe development and deployment. 
The aim is to encourage responsible AI innovation and collaboration within the AI community to improve the safety of the models.

Source: http://arxiv.org/pdf/2307.09288
"""
