import streamlit as st
from sommelier_core import get_chain

# --- Configuração da página ---
st.set_page_config(page_title="🍷 Sommelier AI 🍽️", page_icon="🍷🍽️", layout="wide")

# --- Sidebar fixa ---
with st.sidebar:
    st.title("🍷 Sommelier AI 🍽️")
    st.caption("Seu sommelier pessoal com LangChain + Grok-4")

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"user_{id(st)}"

    if "art" not in st.session_state:
        art, config = get_chain(st.session_state.session_id)
        st.session_state.art = art
        st.session_state.config = config
        st.session_state.messages = []

    st.divider()

    # --- Controles laterais ---
    if st.button("🧹 Novo chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⛔ Encerrar sessão"):
        st.session_state.clear()
        st.success("Sessão encerrada. Você pode fechar a aba.")
        st.stop()

# --- Área principal do chat ---
st.markdown(
    """
    <h1 style='
    margin-top:-15px;
    margin-bottom:10px;
    font-size:2.2em
    '>
    ## 💬 Conversa com o Sommelier</h1>
    """,
    unsafe_allow_html=True
)

# Renderiza histórico com st.chat_message
chat_box = st.container(height=600)  # altura ajustável, cria scroll automático
with chat_box:
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"].lower().startswith("sommelier") or msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta sobre harmonização de vinhos...")

# Processa a entrada
if user_input:
    cmd = user_input.strip().lower()
    if cmd in {"sair", "exit", "quit"}:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "👋 Até logo! Sessão encerrada."})
        st.session_state.clear()
        st.stop()
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            with st.spinner("🍷 Sommelier está pensando..."):
                res = st.session_state.art.chain.with_config({
                    "run_name": f"Sommelier - {user_input[:20]}"
                }).invoke({"pergunta": user_input}, config=st.session_state.config)
                data = res.model_dump()

            main_item = data.get("dish", data.get("wine", "Sugestão"))
            if 'dish' in list(data.keys()):
                icon_1 = "🍽️"
                icon_2 = "🍷"
            elif 'wine' in list(data.keys()):
                icon_1 = "🍷"
                icon_2 = "🍽️"
            else:
                icon_1 = icon_2 = "⁉️"
            answer = f"### {icon_1} {main_item}\n\n"
            for s in data.get("suggestions", []):
                name = s.get('wine', s.get('dish', ''))
                reason = s.get('reason', '')
                desc = s.get('desc', '')
                desc_text = f"📖 _{desc}_\n" if desc else ""
                conf = s.get('confidence', None)
                conf_text = f"🧭 **Assertividade:** {conf:.0%}\n" if conf is not None else ""
                answer += f"{icon_2} **{name}**  \n{desc_text}  \nExplicação: {reason}  \n{conf_text}  \n---  \n"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

