"""
OpenAI API クライアント
音声認識とフィードバック生成を担当
"""

import os
from typing import Optional
from openai import OpenAI


class SpeechTrainingAI:
    """話力鍛錬所のためのOpenAI APIクライアント"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初期化

        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI APIキーが設定されていません")

        self.client = OpenAI(api_key=self.api_key)

    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        音声ファイルをテキストに変換（Whisper API使用）

        Args:
            audio_file_path: 音声ファイルのパス

        Returns:
            認識されたテキスト
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ja"  # 日本語を指定
                )
            return transcript.text
        except Exception as e:
            raise Exception(f"音声認識エラー: {str(e)}")

    def analyze_parrot_response(self, question: str, user_response: str) -> dict:
        """
        オウム返し訓練の分析（F-202）
        質問を復唱できているか、その後に回答できているかを評価

        Args:
            question: 提示された質問
            user_response: ユーザーの回答

        Returns:
            フィードバックの辞書
        """
        prompt = f"""
あなたは話し方トレーニングのコーチです。
以下の「オウム返し訓練」の評価を行ってください。

【訓練の目的】
質問に対し、まずその質問内容を復唱（オウム返し）し、次に話す内容を考える時間（約2秒）を意図的に稼ぐ練習です。

【提示された質問】
{question}

【ユーザーの回答】
{user_response}

【評価項目】
1. 質問の復唱: 質問内容を適切にオウム返しできているか
2. 回答の有無: 復唱の後に、自分なりの回答を述べているか
3. 構成: 「復唱→回答」の流れができているか

以下のJSON形式で評価してください：
{{
    "has_parrot": true/false,
    "has_answer": true/false,
    "score": 0-100,
    "feedback": "具体的なフィードバック（日本語、200文字以内）",
    "good_points": ["良かった点1", "良かった点2"],
    "improvements": ["改善点1", "改善点2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは話し方トレーニングの専門コーチです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            raise Exception(f"分析エラー: {str(e)}")

    def analyze_live_commentary(self, topic: str, user_speech: str) -> dict:
        """
        実況トレーニングの分析（F-203）
        短い言葉で区切りながら話せているかを評価

        Args:
            topic: 実況対象のトピック
            user_speech: ユーザーの発話内容

        Returns:
            フィードバックの辞書
        """
        prompt = f"""
あなたは話し方トレーニングのコーチです。
以下の「実況トレーニング」の評価を行ってください。

【訓練の目的】
提示された物体を、短い言葉で区切りながら（句点を意識して）連ねていく発話練習です。
長い言葉を使わず、簡潔に区切って話すことが重要です。

【実況対象】
{topic}

【ユーザーの発話】
{user_speech}

【評価項目】
1. 句点の意識: 文章を「丸（句点）」で言い切れているか
2. 簡潔さ: 一文が短く、わかりやすいか
3. 接続詞の使用: 接続詞で繋げすぎていないか
4. 具体性: 抽象的ではなく、具体的に表現できているか

以下のJSON形式で評価してください：
{{
    "sentence_count": 文の数,
    "avg_sentence_length": 平均文字数,
    "score": 0-100,
    "feedback": "具体的なフィードバック（日本語、200文字以内）",
    "good_points": ["良かった点1", "良かった点2"],
    "improvements": ["改善点1", "改善点2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは話し方トレーニングの専門コーチです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            raise Exception(f"分析エラー: {str(e)}")

    def analyze_self_questioning(self, action: str, user_response: str) -> dict:
        """
        自問自答ダイアログの分析（F-201）
        日常の行動に対し、「なぜ？」を深掘りして理由を言語化できているかを評価

        Args:
            action: 提示された日常の行動
            user_response: ユーザーの回答

        Returns:
            フィードバックの辞書
        """
        prompt = f"""
あなたは話し方トレーニングのコーチです。
以下の「自問自答ダイアログ」の評価を行ってください。

【訓練の目的】
提示された日常の行動に対し、「なぜ？」を問いかけ、深掘りした理由を言語化させる練習です。
思考の引き出しを増やすことが目的です。

【提示された行動】
{action}

【ユーザーの回答】
{user_response}

【評価項目】
1. 深掘り度: 表面的ではなく、深い理由まで言語化できているか
2. 具体性: 抽象的ではなく、具体的に説明できているか
3. 思考の広がり: 複数の視点や理由を述べているか
4. 言語化力: 自分の考えを言葉にできているか

以下のJSON形式で評価してください：
{{
    "depth_level": 1-5（深掘りの深さ）,
    "is_specific": true/false,
    "score": 0-100,
    "feedback": "具体的なフィードバック（日本語、200文字以内）",
    "good_points": ["良かった点1", "良かった点2"],
    "improvements": ["改善点1", "改善点2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは話し方トレーニングの専門コーチです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            raise Exception(f"分析エラー: {str(e)}")

    def analyze_conclusion_first(self, question: str, user_response: str) -> dict:
        """
        結論ファースト訓練の分析（F-102）
        質問に対し、まず結論を強く言い切れているかを評価

        Args:
            question: 提示された質問
            user_response: ユーザーの回答

        Returns:
            フィードバックの辞書
        """
        prompt = f"""
あなたは話し方トレーニングのコーチです。
以下の「結論ファースト訓練」の評価を行ってください。

【訓練の目的】
質問に対し、まず結論を強く言い切る発話練習です。
冒頭に「パンチワード」または「エッセンスワード」があることが重要です。

【提示された質問】
{question}

【ユーザーの回答】
{user_response}

【評価項目】
1. 結論ファースト: 冒頭に結論が来ているか
2. 言い切り: 断定的に、強く言い切れているか
3. パンチワード: インパクトのある言葉で始まっているか
4. 明確性: 何を言いたいのかが明確か

以下のJSON形式で評価してください：
{{
    "has_conclusion_first": true/false,
    "has_punch_word": true/false,
    "assertiveness": 1-5（言い切りの強さ）,
    "score": 0-100,
    "feedback": "具体的なフィードバック（日本語、200文字以内）",
    "good_points": ["良かった点1", "良かった点2"],
    "improvements": ["改善点1", "改善点2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは話し方トレーニングの専門コーチです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            raise Exception(f"分析エラー: {str(e)}")

    def generate_question(self, training_type: str) -> str:
        """
        トレーニング用の質問を生成

        Args:
            training_type: トレーニングタイプ（"parrot", "commentary", "self_questioning", "conclusion_first"）

        Returns:
            生成された質問またはトピック
        """
        if training_type == "parrot":
            prompt = """
話し方トレーニングの「オウム返し訓練」用の質問を1つ生成してください。
日常的で答えやすい質問にしてください。

例：
- 今日の朝ごはんは何を食べましたか？
- 最近ハマっていることは何ですか？
- 好きな季節とその理由を教えてください

質問文のみを出力してください。
"""
        elif training_type == "commentary":
            prompt = """
話し方トレーニングの「実況トレーニング」用のトピックを1つ生成してください。
身の回りにある具体的な物や場所にしてください。

例：
- 自分のスマートフォン
- 今いる部屋
- 自分の手

トピック名のみを出力してください。
"""
        elif training_type == "self_questioning":
            prompt = """
話し方トレーニングの「自問自答ダイアログ」用の日常行動を1つ生成してください。
ユーザーが「なぜそれをするのか」を深掘りできるような行動にしてください。

例：
- 毎朝コーヒーを飲む
- スマホを頻繁にチェックする
- 週末に散歩をする

「〜をする」という形式で、行動のみを出力してください。
"""
        else:  # conclusion_first
            prompt = """
話し方トレーニングの「結論ファースト訓練」用の質問を1つ生成してください。
意見や考えを求める質問にしてください。

例：
- あなたにとって仕事で最も大切なことは何ですか？
- 今年の目標を一言で表すと？
- この会議で最も重要な論点は何だと思いますか？

質問文のみを出力してください。
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは話し方トレーニングの質問生成AIです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"質問生成エラー: {str(e)}")
