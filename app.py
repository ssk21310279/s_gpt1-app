import torch
import streamlit as st
import sentencepiece as spm
import numpy as np
from model import MiniGPTDecoder
# --- UI 初期化 ---
st.set_page_config(page_title="GPT-1 対話デモ", page_icon="🧠")
MAX_ATTEMPTS = 3  # 最大試行回数

def check_password():
    """パスワード認証 + ロック付き + 認証後は rerun でフォーム非表示"""

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0
    if "locked" not in st.session_state:
        st.session_state.locked = False

    # 認証済みならフォームを表示しない
    if st.session_state.authenticated:
        return True

    # ロック状態ならフォーム表示せず中断
    if st.session_state.locked:
        st.error("🚫 ログイン試行が制限されました。ページをリロードしてください。")
        return False

    # パスワードフォーム表示
    with st.form("login_form"):
        st.write("🔒 パスワードを入力してください")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")

        if submitted:
            if password == st.secrets["password"]:
                st.session_state.authenticated = True
                st.session_state.attempts = 0
                st.success("✅ 認証成功！")
                st.rerun()  # 🔁 ここが `st.experimental_rerun()` → `st.rerun()` に変更された箇所
            else:
                st.session_state.attempts += 1
                remaining = MAX_ATTEMPTS - st.session_state.attempts
                if remaining <= 0:
                    st.session_state.locked = True
                    st.error("🚫 試行回数を超えました。ページをリロードしてください。")
                else:
                    st.error(f"❌ パスワードが違います。残り：{remaining} 回")

    return False

# ✅ 未認証ならアプリを停止
if not check_password():
    st.stop()









# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- トークナイザー読み込み ---
sp = spm.SentencePieceProcessor()
sp.load("bpe_tokenizer.model")

pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()

# --- モデル構築・読み込み ---
vocab_size = sp.get_piece_size()
model = MiniGPTDecoder(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# --- top-k / top-p フィルタリング ---
def top_k_top_p_filtering(logits, top_k=0, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    if top_p < 1.0:
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        sorted_logits[mask] = -float('Inf')

    if top_k > 0:
        sorted_logits[..., top_k:] = -float('Inf')

    filtered_logits = torch.full_like(logits, -float('Inf'))
    filtered_logits.scatter_(1, sorted_indices, sorted_logits)
    return filtered_logits

# --- 応答生成関数 ---
def generate_response(history, max_len=50, top_k=0, top_p=0.9, temperature=1.0):
    input_ids = [bos_id] + sp.encode(history + "AI:")
    input_tensor = torch.tensor([input_ids], device=device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature
            filtered = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    response_ids = input_tensor[0].tolist()[len(input_ids):]
    return sp.decode(response_ids)


st.title("🧠 GPT-1 対話デモ")

if "history" not in st.session_state:
    st.session_state.history = ""

# --- スライダー設定 ---
st.sidebar.header("生成パラメータ")
temperature = st.sidebar.slider("温度（temperature）", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
max_len = st.sidebar.slider("最大生成長（トークン数）", min_value=10, max_value=100, value=50, step=5)

# --- 入力フォーム（エンターで送信） ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("あなた：", "")
    submitted = st.form_submit_button("送信")

if submitted and user_input.strip():
    st.session_state.history += f"User: {user_input}\n"
    response = generate_response(
        st.session_state.history,
        max_len=max_len,
        temperature=temperature,
        top_k=0,
        top_p=0.9
    )
    st.session_state.history += f"AI: {response}\n"

# --- 履歴表示 ---
st.text_area("会話履歴", st.session_state.history, height=300)

# --- リセットボタン ---
if st.button("履歴リセット"):
    st.session_state.history = ""
