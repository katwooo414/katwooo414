"""
話力鍛錬所 (Power Speaking Dojo)
タイプA（言葉が出てこないパターン）向けMVP

OpenAI APIを使用した音声トレーニングアプリ
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from src.openai_client import SpeechTrainingAI

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="話力鍛錬所",
    page_icon="🗣️",
    layout="wide"
)

# セッション状態の初期化
if "ai_client" not in st.session_state:
    try:
        st.session_state.ai_client = SpeechTrainingAI()
    except ValueError as e:
        st.error("⚠️ OpenAI APIキーが設定されていません。.envファイルを確認してください。")
        st.stop()

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None

if "feedback" not in st.session_state:
    st.session_state.feedback = None


def save_audio_file(audio_bytes: bytes, filename: str = "recorded_audio.wav") -> str:
    """
    録音された音声データをファイルに保存

    Args:
        audio_bytes: 音声データ
        filename: 保存するファイル名

    Returns:
        保存されたファイルのパス
    """
    audio_dir = Path("audio_files")
    audio_dir.mkdir(exist_ok=True)

    file_path = audio_dir / filename
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return str(file_path)


def display_feedback(feedback: dict, training_type: str):
    """
    フィードバックを表示

    Args:
        feedback: フィードバックの辞書
        training_type: トレーニングタイプ
    """
    st.subheader("📊 評価結果")

    # スコア表示
    score = feedback.get("score", 0)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("スコア", f"{score}/100")

    # プログレスバー
    if score >= 80:
        st.success(f"素晴らしい！ スコア: {score}/100")
    elif score >= 60:
        st.info(f"良い調子です！ スコア: {score}/100")
    else:
        st.warning(f"もう少し頑張りましょう！ スコア: {score}/100")

    st.progress(score / 100)

    # フィードバック詳細
    st.markdown("### 💬 総合フィードバック")
    st.write(feedback.get("feedback", ""))

    # 良かった点と改善点
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ 良かった点")
        good_points = feedback.get("good_points", [])
        if good_points:
            for point in good_points:
                st.write(f"- {point}")
        else:
            st.write("（該当なし）")

    with col2:
        st.markdown("### 📈 改善点")
        improvements = feedback.get("improvements", [])
        if improvements:
            for improvement in improvements:
                st.write(f"- {improvement}")
        else:
            st.write("（該当なし）")

    # トレーニング別の詳細情報
    if training_type == "parrot":
        st.markdown("### 📝 オウム返し分析")
        col1, col2 = st.columns(2)
        with col1:
            has_parrot = feedback.get("has_parrot", False)
            st.write(f"質問の復唱: {'✅ できている' if has_parrot else '❌ できていない'}")
        with col2:
            has_answer = feedback.get("has_answer", False)
            st.write(f"回答の提示: {'✅ できている' if has_answer else '❌ できていない'}")

    elif training_type == "commentary":
        st.markdown("### 📝 実況分析")
        col1, col2 = st.columns(2)
        with col1:
            sentence_count = feedback.get("sentence_count", 0)
            st.write(f"文の数: {sentence_count}")
        with col2:
            avg_length = feedback.get("avg_sentence_length", 0)
            st.write(f"平均文字数: {avg_length:.1f}")


def parrot_training():
    """オウム返し訓練（F-202）"""
    st.header("🦜 オウム返し訓練")

    st.markdown("""
    ### 訓練の目的
    質問に対し、まず**その質問内容を復唱（オウム返し）**し、次に話す内容を考える時間（約2秒）を意図的に稼ぐ練習です。

    ### やり方
    1. 下の「新しい質問を生成」ボタンを押して質問を取得
    2. 質問を読んで理解する
    3. 録音ボタンを押して、**質問を復唱してから回答**する
    4. AIがあなたの回答を分析してフィードバックします

    ### ポイント
    - 「〜ですか？」という質問なら、「〜について、ですね」と復唱
    - 復唱することで、2秒程度の考える時間を確保できる
    - 復唱の後に、自分の考えを述べる
    """)

    # 質問生成ボタン
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎲 新しい質問を生成", use_container_width=True):
            with st.spinner("質問を生成中..."):
                try:
                    st.session_state.current_question = st.session_state.ai_client.generate_question("parrot")
                    st.session_state.transcribed_text = None
                    st.session_state.feedback = None
                    st.rerun()
                except Exception as e:
                    st.error(f"質問生成エラー: {str(e)}")

    # 質問表示
    if st.session_state.current_question:
        st.info(f"**質問:** {st.session_state.current_question}")

        # 音声録音
        st.markdown("### 🎤 音声を録音")
        st.write("マイクボタンをクリックして録音を開始してください。もう一度クリックすると録音が停止します。")

        audio_bytes = audio_recorder(
            text="録音",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            # 音声認識と分析ボタン
            if st.button("📊 音声を分析", use_container_width=True):
                with st.spinner("音声を認識中..."):
                    try:
                        # 音声ファイルを保存
                        audio_path = save_audio_file(audio_bytes)

                        # 音声をテキストに変換
                        transcribed_text = st.session_state.ai_client.transcribe_audio(audio_path)
                        st.session_state.transcribed_text = transcribed_text

                        # 認識結果を表示
                        st.success("✅ 音声認識完了")
                        st.markdown("### 📝 認識されたテキスト")
                        st.write(transcribed_text)

                        # フィードバック生成
                        with st.spinner("フィードバックを生成中..."):
                            feedback = st.session_state.ai_client.analyze_parrot_response(
                                st.session_state.current_question,
                                transcribed_text
                            )
                            st.session_state.feedback = feedback

                            # フィードバック表示
                            display_feedback(feedback, "parrot")

                        # 音声ファイルを削除
                        os.remove(audio_path)

                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")

    else:
        st.warning("「新しい質問を生成」ボタンを押して、トレーニングを開始してください。")


def commentary_training():
    """実況トレーニング（F-203）"""
    st.header("📢 実況トレーニング")

    st.markdown("""
    ### 訓練の目的
    提示された物体を、**短い言葉で区切りながら（句点を意識して）連ねていく**発話練習です。
    長い言葉を使わず、簡潔に区切って話すことが重要です。

    ### やり方
    1. 下の「新しいトピックを生成」ボタンを押してトピックを取得
    2. そのトピック（物や場所）を観察する
    3. 録音ボタンを押して、**短い文で区切りながら実況**する
    4. AIがあなたの発話を分析してフィードバックします

    ### ポイント
    - 一文を短くする（目安：10〜15文字程度）
    - 「〜で、〜で、」のような接続詞で繋げない
    - 「これは〜です。」「色は〜です。」のように、句点で区切る
    - 抽象的な表現ではなく、具体的に描写する
    """)

    # トピック生成ボタン
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎲 新しいトピックを生成", use_container_width=True):
            with st.spinner("トピックを生成中..."):
                try:
                    st.session_state.current_question = st.session_state.ai_client.generate_question("commentary")
                    st.session_state.transcribed_text = None
                    st.session_state.feedback = None
                    st.rerun()
                except Exception as e:
                    st.error(f"トピック生成エラー: {str(e)}")

    # トピック表示
    if st.session_state.current_question:
        st.info(f"**実況対象:** {st.session_state.current_question}")

        # 音声録音
        st.markdown("### 🎤 音声を録音")
        st.write("マイクボタンをクリックして録音を開始してください。もう一度クリックすると録音が停止します。")

        audio_bytes = audio_recorder(
            text="録音",
            recording_color="#e74c3c",
            neutral_color="#27ae60",
            icon_name="microphone",
            icon_size="3x"
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            # 音声認識と分析ボタン
            if st.button("📊 音声を分析", use_container_width=True):
                with st.spinner("音声を認識中..."):
                    try:
                        # 音声ファイルを保存
                        audio_path = save_audio_file(audio_bytes)

                        # 音声をテキストに変換
                        transcribed_text = st.session_state.ai_client.transcribe_audio(audio_path)
                        st.session_state.transcribed_text = transcribed_text

                        # 認識結果を表示
                        st.success("✅ 音声認識完了")
                        st.markdown("### 📝 認識されたテキスト")
                        st.write(transcribed_text)

                        # フィードバック生成
                        with st.spinner("フィードバックを生成中..."):
                            feedback = st.session_state.ai_client.analyze_live_commentary(
                                st.session_state.current_question,
                                transcribed_text
                            )
                            st.session_state.feedback = feedback

                            # フィードバック表示
                            display_feedback(feedback, "commentary")

                        # 音声ファイルを削除
                        os.remove(audio_path)

                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")

    else:
        st.warning("「新しいトピックを生成」ボタンを押して、トレーニングを開始してください。")


def main():
    """メイン関数"""
    # タイトル
    st.title("🗣️ 話力鍛錬所")
    st.markdown("### タイプA（言葉が出てこないパターン）向けトレーニング")

    # サイドバー
    with st.sidebar:
        st.header("📋 メニュー")

        training_mode = st.radio(
            "トレーニングを選択",
            ["オウム返し訓練", "実況トレーニング"],
            help="練習したいトレーニングを選んでください"
        )

        st.markdown("---")

        st.markdown("""
        ### 💡 アプリについて

        このアプリは、**話し方の課題（言葉が出てこない）** を改善するためのトレーニングツールです。

        **主な機能:**
        - 🦜 オウム返し訓練
        - 📢 実況トレーニング
        - 🎤 音声認識
        - 📊 AIフィードバック

        毎日少しずつ練習することで、
        話す力が向上します！
        """)

        st.markdown("---")
        st.caption("Powered by OpenAI API")

    # トレーニングモードに応じた表示
    if training_mode == "オウム返し訓練":
        parrot_training()
    else:
        commentary_training()


if __name__ == "__main__":
    main()
