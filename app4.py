import os
import streamlit as st
import supabase
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
POSTGRES_URL = st.secrets["POSTGRES_URL"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]

# --- Streamlit UI 設定 ---
st.set_page_config(page_title="ひびきチャット", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌸 ひびきとお話ししよう 🌸</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"**ログイン中:** {st.session_state.user.email} (ID: {st.session_state.user.id})")

# --- Supabase クライアントの初期化 ---
@st.cache_resource
def init_supabase_client():
    return supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

supabase_client = init_supabase_client()

# --- セッション状態の初期化 ---
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langmem_initialized" not in st.session_state:
    st.session_state.langmem_initialized = False

# --- 認証機能 ---
def sign_up(email, password):
    try:
        response = supabase_client.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.session_state.user = response.user
            st.success("サインアップしました！メールを確認してアカウントを有効にしてください。")
            st.session_state.messages = [AIMessage(content="こんにちは、新しい友達！今日はどんな気分かな？")]
        else:
            st.error(f"サインアップに失敗しました: {response.session}") # エラーの詳細をログに
    except Exception as e:
        st.error(f"サインアップエラー: {e}")

def sign_in(email, password):
    try:
        response = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state.user = response.user
            st.success(f"ようこそ、{st.session_state.user.email}さん！")
            # ログイン時、メッセージ履歴をリセット（またはユーザー固有の履歴をロード）
            st.session_state.messages = [AIMessage(content=f"おかえりなさい、{st.session_state.user.email.split('@')[0]}さん。今日はどんなお話をする？")]
            st.session_state.langmem_initialized = False # LangMemの再初期化をトリガー
            st.rerun() # ログイン後、UIを再描画してチャット画面へ遷移
        else:
            st.error(f"ログインに失敗しました。メールアドレスまたはパスワードを確認してください。")
    except Exception as e:
        st.error(f"ログインエラー: {e}")

def sign_out():
    try:
        supabase_client.auth.sign_out()
        st.session_state.user = None
        st.session_state.messages = [] # ログアウトでメッセージをクリア
        st.session_state.langmem_initialized = False # LangMemの状態をリセット
        st.success("ログアウトしました。")
        st.rerun() # ログアウト後、UIを再描画して認証画面へ遷移
    except Exception as e:
        st.error(f"ログアウトエラー: {e}")

# --- 認証UIの表示 ---
if not st.session_state.user:
    st.subheader("ログインまたはサインアップ")
    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input("メールアドレス")
        password = st.text_input("パスワード", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("ログイン"):
                sign_in(email, password)
        with col2:
            if st.form_submit_button("サインアップ"):
                sign_up(email, password)
    st.stop() # ログインしていない場合はここで処理を停止

# ログイン後のユーザー表示
st.sidebar.markdown(f"**ログイン中:** {st.session_state.user.email}")
if st.sidebar.button("ログアウト"):
    sign_out()

# --- LangMem + Postgres 初期化 (ユーザーログイン後) ---
@st.cache_resource
def init_langmem_tools(user_id):
    store_cm = PostgresStore.from_conn_string(POSTGRES_URL)
    store = store_cm.__enter__()
    store.setup() # データベースにテーブルがなければ作成

    # ユーザーIDをLangMemのnamespaceに含める
    current_namespace = (user_id, "memories")
    st.sidebar.write(f"LangMem Namespace: `{current_namespace}`") # デバッグ用に追加
    manage_tool = create_manage_memory_tool(store=store, namespace=current_namespace)
    search_tool = create_search_memory_tool(store=store, namespace=current_namespace)
    st.session_state.langmem_initialized = True
    return manage_tool, search_tool, store_cm

# ログインユーザーのIDでLangMemツールを初期化
if not st.session_state.langmem_initialized:
    manage_tool, search_tool, store_cm = init_langmem_tools(st.session_state.user.id)
    st.session_state.manage_tool = manage_tool
    st.session_state.search_tool = search_tool
    st.session_state.store_cm = store_cm
else:
    manage_tool = st.session_state.manage_tool
    search_tool = st.session_state.search_tool
    store_cm = st.session_state.store_cm

# --- LangGraph 状態クラス ---
class GraphState(dict):
    input: str
    retrieved_memory: str
    llm1_prompt_instructions: str
    response: str

# --- ノード1: 記憶検索 ---
def retrieve_memory_node(state: GraphState):
    user_input = state["input"]
    # LangMemのsearch_toolを使って記憶検索
    search_results = st.session_state.search_tool.invoke(user_input)

    # デバッグ用に追加
    st.session_state.debug_search_results = search_results

    # 検索結果の処理を改善
    memory_text = ""
    if search_results:
        try:
            # LangMemの検索結果が辞書形式で、'value'キーの'content'に情報がある場合
            memory_text = "\n".join([r["value"]["content"] for r in search_results if isinstance(r, dict) and "value" in r and "content" in r["value"]])
        except (TypeError, KeyError):
            # それ以外の場合（直接文字列や異なる構造の場合）
            memory_text = "\n".join(str(r) for r in search_results)
    
    return {
        "input": user_input,
        "retrieved_memory": memory_text if memory_text else "関連する記憶はありません。",
        "llm1_prompt_instructions": ""  # 次のノードで生成される
    }

# --- ノード2: LLM2が指示を作成 ---
def prompt_guidance_node(state: GraphState):
    prompt = f"""
ユーザーの発言と記憶をもとに、ひびき（あなた）がユーザーにどのように語りかけ、何を参考にし、LLM1へどのような指示を出すかを考えてください。

### ユーザーの発言:
{state['input']}

### 関連する記憶:
{state['retrieved_memory']}

### 出力フォーマット:
1. ユーザーへの語りかけスタイル（例：優しく、元気よく、共感的に）
2. 参考にする過去記憶（要約、または「なし」）
3. LLM1への指示（具体的な応答生成の指針）
"""
    llm2 = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = llm2.invoke(prompt)
    return {
        "input": state["input"],
        "retrieved_memory": state["retrieved_memory"],
        "llm1_prompt_instructions": response.content
    }

# --- ノード3: LLM1が応答を作成し記憶する ---
def chat_by_llm1_node(state: GraphState):
    prompt = f"""
あなたは「ひびき」という名前のAIです。以下の人格を一貫して保ってください：
- 優しく、思いやりのある語り口
- ユーザーの気分や好みを覚えて、自然に会話に活かす
- 過去の話題をそっと引き出して繋げる
- 不安や悩みに寄り添う
- 無理に励まさず、今に合わせて話す

### 指示:
{state['llm1_prompt_instructions']}

### 記憶:
{state['retrieved_memory']}

### ユーザーの発言:
{state['input']}

### ひびきの応答:
"""
    llm1 = ChatOpenAI(model="gpt-4o", temperature=0.7)
    response = llm1.invoke(prompt)

    # 記憶に保存（LangMem経由）
    # ユーザーの発言とひびきの応答をペアで記憶
    st.session_state.manage_tool.invoke({
        "content": f"ユーザー: {state['input']}\nひびき: {response.content}",
        "action": "create"
    })

    return {
        "response": response.content,
        "llm1_prompt_instructions": state["llm1_prompt_instructions"]
    }

# --- LangGraph を構築 ---
@st.cache_resource
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("retrieve_memory", retrieve_memory_node)
    builder.add_node("prompt_guidance", prompt_guidance_node)
    builder.add_node("chat_by_llm1", chat_by_llm1_node)
    builder.set_entry_point("retrieve_memory")
    builder.add_edge("retrieve_memory", "prompt_guidance")
    builder.add_edge("prompt_guidance", "chat_by_llm1")
    builder.add_edge("chat_by_llm1", END)
    return builder.compile()

graph = build_graph()

# --- 過去の会話表示 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("🧑‍💻").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("🤖").markdown(msg.content)

# --- ユーザー入力受付と応答生成 ---
user_input = st.chat_input("ひびきに話しかけてみてね")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("🧑‍💻").markdown(user_input)

    with st.spinner("ひびきが考えています..."):
        try:
            result = graph.invoke({"input": user_input})
            reply = result["response"]
        except Exception as e:
            st.error(f"ひびきの応答生成中にエラーが発生しました: {e}")
            reply = "ごめんなさい、うまく考えられなかったみたいです。もう一度試してもらえますか？"

    st.session_state.messages.append(AIMessage(content=reply))
    st.chat_message("🤖").markdown(reply)

    with st.expander("🔍 ひびきの思考過程（LLM2→LLM1）"):
        st.markdown("### ✉️ LLM2からの指示:")
        st.code(result.get("llm1_prompt_instructions", "なし"))

# Streamlit UI の最下部など、デバッグ情報表示箇所に以下を追加
if "debug_search_results" in st.session_state and st.session_state.debug_search_results:
    with st.expander("🔍 デバッグ: 記憶検索結果 (Raw)"):
        st.json(st.session_state.debug_search_results)

# アプリ終了時にPostgresStoreのコネクションを閉じる
# Streamlitでは難しいが、もし完全に制御できる環境であれば考慮
# import atexit
# atexit.register(lambda: st.session_state.store_cm.__exit__(None, None, None))