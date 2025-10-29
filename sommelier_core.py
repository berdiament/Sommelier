"""
Sommelier AI Assistant
Author: Bernardo
Description: Interactive sommelier assistant with LangChain, LangSmith tracing, and Chroma vector store.
"""
import os
try:
    import streamlit as st
    if "LANGSMITH_API_KEY" in st.secrets:
        os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
        os.environ["LANGCHAIN_TRACING"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "sommelier-bot")
        os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

from typing import List, Literal, Optional
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()

# --- 1) Configure LangSmith tracing BEFORE LangChain imports ---
def setup_env():
    os.environ["LANGSMITH_PROJECT"]  = "sommelier-bot"
#print("LangSmith project =>", os.environ["LANGSMITH_PROJECT"])

setup_env()

# --- 2) Now safe to import LangChain ecosystem ---
from dataclasses import dataclass
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough, RunnableLambda, RunnableBranch,
    RunnableParallel, RunnableMap, Runnable
    )
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import ensure_config
from pydantic import BaseModel, Field

#-----3) Chain classes -----
@dataclass
class BuildArtifacts:
    chain: Runnable
    base_chain: Runnable
    classifier: Runnable
    wine_chain: Runnable
    food_chain: Runnable
    retriever: any
    context: Optional[str]
    embeddings: OpenAIEmbeddings
    chat_model: ChatXAI
    memory_factory: callable

# --- 5) Define schemas ---
class WineSuggestion(BaseModel):
    wine: str = Field(..., description="Types of wine (grape varietals) that best matches the dish")
    reason: str = Field(..., description="Explanation of why this pairing suggestion works")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level between 0 and 1")

class WinePairing(BaseModel):
    """Pairing dish -> wines"""
    dish: str = Field(..., description="The dish that is being paired with the wine")
    suggestions: List[WineSuggestion] = Field(..., description="List of wine suggestions for the dish")

class DishSuggestion(BaseModel):
    dish: str = Field(..., description="Dish that fits the wine")
    desc: str = Field(..., description="Short description of the dish, including sides for each of the dishes, \
                      such as preparation methods and flavor profiles")
    reason: str = Field(...,description="Explanation of the rationale behind the dish and wine pairing")
    confidence: float = Field(..., ge=0, le=1)

class FoodPairing(BaseModel):
    """Pairing wine -> dishes"""
    wine: str = Field(...,description="Wine being prompted")
    suggestions: List[DishSuggestion] = Field(...,description="List of suggestions for the given wine")


class QueryType(BaseModel):
    """Define given constraint to define pairing type"""
    task: Literal["wines_for_dish","dishes_for_wine"] = Field(...,description="Decide if food or wine is a given constraint and\
                                                              then return the appropriate pairings based on the selected task")
    item: str = Field(...,description="Normalized target (dish or wine) extracted from the query made by user")
    restrictions: str = Field(...,description="All filters or restrictions that should be used to select the requested wine\
                              or dish (e.g.: vegetarian, lactose free, red, white or rosé wine)")

#-----4) Build the chain------
def build_chain() -> BuildArtifacts:

    # --- 6) Chat model setup ---
    chat = ChatXAI(model = 'grok-4-fast-non-reasoning',
                temperature=.2,
                max_tokens=2000)
    
    embeddings_model = OpenAIEmbeddings()
    # --- 3) Memory management ---
    mem_store = {}
    def get_by_session_id(session_id):
        if session_id not in mem_store:
            mem_store[session_id] = InMemoryChatMessageHistory()
        return mem_store[session_id]

     # --- 4) Vector store loader ---
    def load_vector_store(store_location):
        vector_store = Chroma(
            embedding_function = embeddings_model,
            persist_directory = store_location
        )
        retriever = vector_store.as_retriever(search_type = "mmr",search_kwargs={"k": 5,"fetch_k": 15})
        return retriever
    
    memory_context = {}
    def merge_with_last_context(new_query, session_id):
        """Preenche campos vazios do classificador com o último contexto armazenado."""
        last = memory_context.get(session_id)
        if not last:
            memory_context[session_id] = new_query
            return new_query
        # herdamos os campos se vierem vazios
        if not new_query.item:
            new_query.item = last.item
        if not new_query.task:
            new_query.task = last.task
        if last.restrictions and new_query.restrictions:
            # acumula restrições
            new_query.restrictions = f"{last.restrictions}; {new_query.restrictions}"
        elif not new_query.restrictions:
            new_query.restrictions = last.restrictions

        memory_context[session_id] = new_query
        return new_query


    # --- 7) Build subchains ---
    clf_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Classifique a solicitação do usuário conforme as instruções abaixo:\n\n"
        "1. Se o usuário pedir algo como 'qual vinho combina com <prato>?', defina `task = 'wine_for_dish'` e `item = <prato>`.\n"
        "2. Se o usuário pedir algo como 'quais pratos combinam com <vinho>?', defina `task = 'dishes_for_wine'` e `item = <vinho>`.\n"
        "3. Utilize o histórico da conversa (se disponível) para interpretar o contexto.\n"
        "4. Se o usuário apenas refinar um pedido anterior (por exemplo, adicionar restrições como 'sem carne' ou 'leve'), "
        "mantenha o mesmo `task` e `item`, mas atualize o campo `restrictions`.\n"
        "5. Se for um novo pedido, faça a classificação do zero.\n\n"
        "Além disso, identifique se o usuário mencionou uma categoria específica, como 'entrada', 'salada', 'prato principal', "
        "'sobremesa', 'vinho tinto', 'vinho branco', 'vinho rosé', 'vinho de sobremesa' ou 'espumante', "
        "ou ainda uma uva específica, como 'Sauvignon Blanc', 'Cabernet Sauvignon', 'Syrah', 'Merlot', 'Malbec', etc.\n"
        "Se houver alguma dessas menções, inclua essa informação no campo `restrictions`.\n\n"
        "Retorne **somente** os campos definidos no schema (`task`, `item`, `restrictions`)."),
        
        ("placeholder", "{memoria}"),
        ("human", "{pergunta}")
    ])

    classifier = (clf_prompt | chat.with_structured_output(QueryType) | 
                  RunnableLambda(lambda o, cfg=None: merge_with_last_context(
                      o, ensure_config(cfg).get("configurable",{}).get("session_id","default"))
    ))
    classifier = classifier.with_config({"run_name": "Classifier"})

    # ---------- 2) Branches that return structured models ----------
    wine_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "Você é um sommelier experiente. Considere fortemente as informações abaixo quando existirem:\n\n{context}\n\n"
        "Responda em português brasileiro."),
        ("placeholder", "{memoria}"),
        ("user", 
        "Prato: {dish}\n"
        "Sugira até três vinhos que harmonizem com este prato. "
        "Respeite as seguintes restrições: {restrictions}. "
        "Retorne APENAS os campos no schema.")
    ])
    wine_chain = wine_prompt | chat.with_structured_output(WinePairing)

    food_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "Você é um sommelier experiente. Considere fortemente as informações abaixo quando existirem:\n\n{context}\n\n"
        "Quando o usuário pedir sobremesas, trate 'dish' como 'sobremesa'. Da mesma forma,"
        "se o usuário pedir saladas, trate 'dish como 'salada'"
        "Responda em português brasileiro."),
        ("placeholder", "{memoria}"),
        ("user", 
        "Vinho: {wine}\n"
        "Sugira até três harmonizações (entradas, pratos principais ou sobremesas, conforme o pedido). "
        "Respeite as seguintes restrições: {restrictions}. "
        "Inclua uma breve descrição. Retorne APENAS os campos no schema.")
    ])
    food_chain = food_prompt | chat.with_structured_output(FoodPairing)

    # Small helpers to adapt classifier output into branch inputs
    to_wine_input = RunnableLambda(lambda o: {"dish": o["item"],"restrictions": o["restrictions"]})
    to_food_input = RunnableLambda(lambda o: {"wine": o["item"],"restrictions": o["restrictions"]})


    # --- 8) Retrieval and context injection ---
    def retrieve_context(inputs):
        vs_retr = load_vector_store(store_location=Path('chroma_vector_store'))
        docs = vs_retr.invoke(inputs["pergunta"])
        return {"context": "\n\n".join([d.page_content for d in docs])}
    
    # Router: choose branch by QueryType.task
    context_injection = (
        RunnablePassthrough.assign(context = RunnableLambda(retrieve_context))
    )
    router = RunnableBranch(
        (lambda o: o["task"] == "wine_for_dish", to_wine_input | RunnablePassthrough.assign(context=lambda x: x.get("context", "")) 
         | wine_chain),
        (lambda o: o["task"] == "dishes_for_wine", to_food_input | RunnablePassthrough.assign(context=lambda x: x.get("context", "")) 
         | food_chain),
        # Fallback: default to wine_for_dish if uncertain
        to_wine_input | RunnablePassthrough.assign(context=lambda x: x.get("context", "")) | wine_chain
    )
    router = router.with_config({"run_name": "Router"})

    # ---------- 4) Full chain ----------
    def split_output_for_memory(output):
        """Provide both structured and string versions for safe memory logging."""
        return {"structured": output, "output": AIMessage(str(output))}

    def unwrap_structured(result):
        """Extract only the structured part for the final user output."""
        return result["structured"]

    # --- 9) Combine full chain ---
    pairing_chain = RunnableParallel(
        context = context_injection, classifier = classifier
    ) | RunnableLambda(lambda x: {**x["classifier"].model_dump(), "context": x["context"]}) | router
    pairing_chain = pairing_chain.with_config({"run_name": "PairingChain"})

    pairing_chain_mem = (
        RunnableWithMessageHistory(
        pairing_chain | RunnableLambda(split_output_for_memory),
        get_by_session_id,
        input_messages_key='pergunta',
        history_messages_key='memoria'
    ) | RunnableLambda(unwrap_structured))
    pairing_chain_mem = pairing_chain_mem.with_config({"run_name": "PairingChain_Mem"})

    return BuildArtifacts(
        chain = pairing_chain_mem,
        base_chain = router,
        classifier = classifier,
        wine_chain = wine_chain,
        food_chain = food_chain,
        retriever = retrieve_context,
        context = context_injection,
        embeddings = embeddings_model,
        chat_model = chat,
        memory_factory = mem_store
        )

config = {'configurable':{'session_id':'user_a'}}
terms_use = {'wine':'Vinho','dish':'Prato','confidence':'Assertividade','reason':'Explicação','desc':'Descrição'}

def main():
# --- 10) CLI loop ---
    art = build_chain()
    print("🍷 Sommelier AI pronto! Digite 'sair' para encerrar.\n")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() in ['sair','quit','exit']:
            print("\n\nSommelier: Até logo!\n")
            sys.exit(0)
        else:
            print(f"Você - {pergunta}")
        try:
            res = art.chain.with_config({
                "run_name": f"Sommelier - {pergunta[:20]}"
            }).invoke({"pergunta":pergunta},config=config)
            res = res.model_dump()
            k, v = next(iter(res.items()))
            print(f"{terms_use.get(k,k)}: {v}\n\nSugestões:\n")
            
            for sugg in res['suggestions']:
                for k, v in sugg.items():
                    print(terms_use.get(k,k),":",v)
                print('\n')
        except Exception as e:
            print(f"\n[Erro]: {e}\n")

def get_chain(session_id: str = "default"):
    art = build_chain()
    config = {"configurable":{"session_id":session_id}}
    return art,config

if __name__ == "__main__":
    main()
