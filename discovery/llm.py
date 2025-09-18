import os
import openai
import google.generativeai as genai
from typing import Literal, Optional, List, Dict, Union
from langchain.memory import ConversationBufferMemory
import json
import traceback
import uuid # Gemini の tool call ID 生成に必要

class LLMClient:
    """
    A client class to interact with different Large Language Models (LLMs)
    like OpenAI and Gemini, with conversation memory and tool/function calling support.
    """

    def __init__(self, discovery, openai_api_key: Optional[str] = None, google_api_key: Optional[str] = None):
        """
        Initializes the LLMClient and the conversation memory.

        Args:
            discovery: The Discovery object.
            openai_api_key: OpenAI API key. Defaults to OS environment variable 'OPENAI_API_KEY'.
            google_api_key: Google API key. Defaults to OS environment variable 'GOOGLE_API_KEY'.
        """
        self.discovery = discovery
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        if self._openai_api_key:
            openai.api_key = self._openai_api_key
        if self._google_api_key:
            genai.configure(api_key=self._google_api_key)

        # 会話メモリを初期化
        self.memory = ConversationBufferMemory(return_messages=True)

    def _call_openai(self, messages: List[Dict], model: str, tools: Optional[List[Dict]]) -> Dict[str, Union[str, List[Dict], None]]:
        """ OpenAI API を呼び出す内部メソッド """
        if not self._openai_api_key:
            raise ValueError("OpenAI API key is not configured.")
        try:
            client = openai.OpenAI(api_key=self._openai_api_key)
            completion_args = {
                "model": model,
                "messages": messages,
            }
            if tools:
                completion_args["tools"] = tools
                completion_args["tool_choice"] = "auto"

            response = client.chat.completions.create(**completion_args)

            message = response.choices[0].message
            response_content = message.content
            response_tool_calls = None
            if message.tool_calls:
                response_tool_calls = [
                    {
                        'id': tc.id,
                        'type': tc.type,
                        'function': {'name': tc.function.name, 'arguments': tc.function.arguments}
                     } for tc in message.tool_calls
                ]
            return {'content': response_content, 'tool_calls': response_tool_calls}
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise # エラーを再発生させて get_response で処理できるようにする

    def _call_gemini(self, full_prompt: str, model: str, tools: Optional[List[Dict]], thinking_budget: Optional[int]) -> Dict[str, Union[str, List[Dict], None]]:
        """ Gemini API を呼び出す内部メソッド """
        if not self._google_api_key:
            raise ValueError("Google API key is not configured.")
        try:
            gen_model = genai.GenerativeModel(model)

            generation_config = None
            if thinking_budget is not None:
                if not isinstance(thinking_budget, int) or thinking_budget < 0:
                     raise ValueError("thinking_budget must be a non-negative integer.")
                from google.generativeai import types
                generation_config = types.GenerationConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
                )

            gemini_response = gen_model.generate_content(
                full_prompt,
                generation_config=generation_config,
                tools=tools
            )

            response_content = None
            response_tool_calls = None
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_parts = []
                tool_call_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_call_id = f"call_{uuid.uuid4()}"
                        tool_call_parts.append({
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': fc.name,
                                'arguments': json.dumps(dict(fc.args)) if fc.args else "{}" # JSON文字列に変換
                            }
                        })
                if text_parts:
                    response_content = "\n".join(text_parts)
                if tool_call_parts:
                    response_tool_calls = tool_call_parts

            return {'content': response_content, 'tool_calls': response_tool_calls}
        except Exception as e:
            print(f"Error calling Google Generative AI API: {e}")
            raise

    def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        service: Literal["openai", "gemini"],
        model: str,
        thinking_budget: Optional[int] = None,
        save_memory: bool = False,
        use_memory: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Union[str, List[Dict], None]]:
        """
        指定されたLLMサービスとモデルからレスポンスを取得します。会話メモリ機能とツール/Function Callingをサポートします。
        内部でサービス固有の呼び出しメソッド (_call_openai, _call_gemini) を使用します。

        Args:
            system_prompt: モデルの動作を制御するシステムプロンプト。
            user_prompt: ユーザーのクエリまたは指示。
            service: 使用するLLMサービス ('openai' または 'gemini')。
            model: 使用する具体的なモデル名 (例: 'gpt-4', 'gemini-pro')。
            thinking_budget: Geminiの思考プロセスのためのオプショナルなトークン予算。
                             'gemini' サービスの場合のみ適用。
                             デフォルトはNone (モデルのデフォルト動作)。
                             0を設定すると思考を無効化。
            save_memory: Trueの場合、テキスト応答をメモリに保存 (ツール呼び出し時は保存されない)。
            use_memory: Trueの場合、過去の会話履歴をプロンプトに含める。
            tools: LLM に提供するツール/関数の定義リスト (OpenAI/Gemini 形式)。

        Returns:
            以下のキーを持つ辞書:
            - 'content': モデルが生成したテキストレスポンス (ツール呼び出し時はNoneの場合あり)。
            - 'tool_calls': LLM が要求したツール呼び出しのリスト (ツール呼び出しがない場合はNone)。
                          OpenAI形式 ({'id': str, 'type': 'function', 'function': {'name': str, 'arguments': str}}) に統一。

        Raises:
            ValueError: サービスが未対応、APIキーが未設定、
                      またはthinking_budgetが無効な場合。
            Exception: API呼び出し中のエラー。
        """
        history_messages: List[Dict[str, str]] = []
        history_text: str = ""

        if use_memory:
            # メモリから過去の会話履歴を取得
            # ConversationBufferMemory(return_messages=True) の場合、 .chat_memory.messages に BaseMessage のリストが入る
            # これを OpenAI/Gemini で使える形式に変換する
            loaded_memory = self.memory.load_memory_variables({})
            # loaded_memory['history'] は BaseMessage のリスト
            base_messages = loaded_memory.get('history', [])

            # OpenAI 形式のメッセージリストを作成
            for msg in base_messages:
                if hasattr(msg, 'content'): # HumanMessage, AIMessage など
                   role = "user" if msg.type == "human" else "assistant"
                   history_messages.append({"role": role, "content": msg.content})

            # Gemini 形式のテキスト履歴を作成 (単純な連結)
            history_text = "\n".join([f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in base_messages])


        response_data = None
        try:
            if service == "openai":
                # OpenAI用のメッセージリストを作成
                messages = [{"role": "system", "content": system_prompt}]
                if use_memory:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": user_prompt})
                # OpenAI呼び出し
                response_data = self._call_openai(messages=messages, model=model, tools=tools)

            elif service == "gemini":
                # Gemini用のプロンプトテキストを作成
                prompt_parts = [system_prompt]
                if use_memory and history_text:
                    prompt_parts.append("\n\n--- Conversation History ---" + history_text)
                prompt_parts.append("\n\n--- Current Prompt ---" + user_prompt)
                full_prompt = "\n".join(prompt_parts)
                # Gemini呼び出し
                response_data = self._call_gemini(full_prompt=full_prompt, model=model, tools=tools, thinking_budget=thinking_budget)

            else:
                raise ValueError(f"Unsupported service: {service}. Choose 'openai' or 'gemini'.")

            # メモリへの保存 (テキスト応答があり、ツール呼び出しがない場合のみ)
            response_content = response_data.get('content')
            response_tool_calls = response_data.get('tool_calls')
            if save_memory and response_content and not response_tool_calls:
                self.memory.save_context({"input": user_prompt}, {"output": response_content})

            return response_data

        except Exception as e:
             # API呼び出し中のエラーをキャッチした場合など
             print(f"Error during get_response for service '{service}': {e}")
             # エラーを示す情報を返すか、Noneを返すか、あるいは再発生させるかなど検討
             # ここでは空の応答を返す例
             return {'content': None, 'tool_calls': None}

    # メモリをクリアするメソッドを追加しても良い (任意)
    def clear_memory(self):
        """会話メモリをクリアします。"""
        self.memory.clear()

    # メモリの内容を参照するメソッドを追加
    def get_memory_string(self) -> str:
        """現在の会話メモリの内容を整形された文字列として返します。"""
        loaded_memory = self.memory.load_memory_variables({})
        base_messages = loaded_memory.get('history', [])
        if not base_messages:
            return "メモリは空です。"

        history_string = "--- Conversation History ---\n"
        for msg in base_messages:
            if hasattr(msg, 'content'):
                role = "User" if msg.type == "human" else "AI"
                history_string += f"{role}: {msg.content}\n"
        return history_string.strip()

    async def handle_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        LLMから要求されたツール呼び出しを処理し、結果をtoolロールメッセージのリストで返す
        """
        tool_results = []
        if not self._google_api_key:
            print("エラー: Google API keyが初期化されていません。ツールを実行できません。")
            # エラーを示す tool result を返すことも可能
            for call in tool_calls:
                 tool_results.append({
                     "role": "tool",
                     "tool_call_id": call['id'],
                     "content": f"Error: Google API key not initialized. Cannot execute tool {call['function']['name']}."
                 })
            return tool_results

        for call in tool_calls:
            function_name = call['function']['name']
            function_args_str = call['function']['arguments']
            tool_call_id = call['id']
            result_content = "" # ツール実行結果

            print(f"\n--- Handling Tool Call ---")
            print(f"ID: {tool_call_id}")
            print(f"Function: {function_name}")
            print(f"Arguments: {function_args_str}")

            try:
                # 引数をJSONとしてパース
                args = json.loads(function_args_str)

                if function_name == "get_skill_full_code":
                    skill_name = args.get("skill_name")
                    if skill_name:
                        # get_skill_code は docstring を除くため注意。含む場合は別途実装が必要
                        # get_skill_code は非同期なので await する
                        code = await self.discovery.get_skill_code([skill_name])
                        if code and code.get(skill_name, {}).get('success'):
                            result_content = f"Source code for skill '{skill_name}':\n```python\n{code[skill_name]['code']}\n```"
                        else:
                            error_message = code.get(skill_name, {}).get('message', 'It might not exist or is inaccessible.')
                            result_content = f"Error: Could not retrieve source code for skill '{skill_name}'. Reason: {error_message}"
                    else:
                        result_content = "Error: Missing required argument 'skill_name' for get_skill_full_code."

                # --- 他のツールの処理をここに追加 ---
                # elif function_name == "other_tool":
                #    arg1 = args.get("arg1")
                #    result = await self.some_other_async_skill(arg1) # 例
                #    result_content = f"Result of other_tool: {result}"
                # ---------------------------------

                else:
                    result_content = f"Error: Unknown tool function '{function_name}'."

            except json.JSONDecodeError:
                result_content = f"Error: Invalid JSON arguments provided for tool '{function_name}': {function_args_str}"
            except Exception as e:
                # get_skill_code や他のツール実行中の予期せぬエラー
                result_content = f"Error executing tool '{function_name}': {e}"
                print(f"Error details: {traceback.format_exc()}") # 詳細ログ

            print(f"Result Content: {result_content[:200]}...") # 長すぎる場合は省略して表示
            print("--------------------------\n")

            # LLMに返す tool ロールのメッセージを作成
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            })

        return tool_results

    async def run_interactive_loop(self):
        """LLM と対話的にやり取りし、ツール呼び出しを処理するループ (デモ用)"""
        if not self._google_api_key:
            print("エラー: Google API keyが初期化されていません。ツールを実行できません。")
            return

        # get_skill_full_code ツールの定義
        tools_definition = [
            {
                "type": "function",
                "function": {
                    "name": "get_skill_full_code",
                    "description": "指定されたMinecraftボットのスキル関数の完全なソースコードを取得します。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "ソースコードを取得したいスキル関数の名前 (例: 'move_to_position', 'get_inventory_counts')。"
                            }
                        },
                        "required": ["skill_name"]
                    }
                }
            }
            # 他のツールがあればここに追加
        ]

        # 会話履歴 (システムプロンプトで初期化)
        messages = [
            {"role": "system", "content": "あなたはMinecraftボットを操作するアシスタントです。必要に応じて提供されたツールを使って情報を取得したり操作を実行したりできます。"}
        ]
        self.clear_memory() # 新しい対話セッション

        print("\n--- Interactive LLM Loop (Type 'quit' to exit) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            messages.append({"role": "user", "content": user_input})

            try:
                # LLM に応答を要求 (ツール定義を渡す)
                response_data = self.get_response(
                    system_prompt="", # 履歴に含まれるため空で良い
                    user_prompt="",   # 履歴に含まれるため空で良い
                    service="gemini", # または "openai" (Tool Calling対応モデルを選択)
                    model="gemini-pro",  # Tool Calling に適したモデルを選択
                    tools=tools_definition,
                    use_memory=False, # use_memory=Trueにして履歴管理をLLMClientに任せても良いが、ここでは手動管理
                    # messages 引数を直接渡せるように get_response を修正する方がより良い設計かも
                    # 現状は use_memory=True 相当の動作を messages 配列で模倣する
                )

                # ---- get_response が messages を直接受け取れない場合の代替実装 ----
                # 現在の実装に合わせて、最新のユーザープロンプトと履歴を使う
                current_user_prompt = messages[-1]["content"]
                # use_memory=True にして、手動の message append を減らす形も検討
                # self.memory.chat_memory.add_user_message(current_user_prompt) # メモリに手動追加する場合

                response_data = self.get_response(
                     system_prompt=messages[0]["content"], # システムプロンプトは常に渡す
                     user_prompt=current_user_prompt, # 最新のユーザープロンプト
                     service="gemini",
                     model="gemini-pro",
                     tools=tools_definition,
                     use_memory=True, # LLMClient のメモリを使う
                     save_memory=False # ループ内で手動管理するか、ここでTrueにして任せるか
                 )
                # ------------------------------------------------------------

                ai_response_content = response_data.get('content')
                tool_calls = response_data.get('tool_calls')

                # AI の思考プロセス（ツール呼び出しまたはテキスト応答）を履歴に追加
                # OpenAIの場合、messageオブジェクト全体を追加するのが一般的
                # ここでは簡略化して content / tool_calls を持つ辞書を追加
                response_message_for_history = {"role": "assistant"}
                if tool_calls:
                    response_message_for_history["tool_calls"] = tool_calls
                if ai_response_content:
                     response_message_for_history["content"] = ai_response_content
                messages.append(response_message_for_history)


                if tool_calls:
                    print("AI: (Requesting tool use...)")
                    # ツール呼び出しを処理
                    tool_results_messages = await self.handle_tool_calls(tool_calls)
                    # ツール実行結果を履歴に追加
                    messages.extend(tool_results_messages)

                    # ツール実行結果を添えて再度LLMに問い合わせ
                    # ---- 再度 get_response ----
                    # 最新の履歴 (tool results を含む) を使う
                    # この部分も get_response が messages を直接受け取る方が綺麗
                    latest_tool_result_content = tool_results_messages[0]["content"] # 簡略化のため最初の結果のみ

                    response_data_after_tool = self.get_response(
                        system_prompt=messages[0]["content"],
                        user_prompt=latest_tool_result_content, # tool結果をプロンプトとして渡すのは微妙かも
                        service="gemini",
                        model="gemini-pro",
                        tools=tools_definition,
                        use_memory=True, # 継続してメモリを使う
                        save_memory=True # 最終的なAI応答を保存
                    )
                    # -------------------------

                    final_content = response_data_after_tool.get('content')
                    if final_content:
                        print(f"AI: {final_content}")
                        messages.append({"role": "assistant", "content": final_content})
                        # LLMClientのメモリにも最終応答を保存する場合
                        # self.memory.save_context({"input": tool_results_messages[-1]["content"]}, {"output": final_content})
                    else:
                        print("AI: (Tool execution completed, but no further text response)")
                        # 応答がない場合も履歴には残す (必要に応じて)
                        # messages.append({"role": "assistant", "content": None})


                elif ai_response_content:
                    # ツール呼び出しがなく、テキスト応答があった場合
                    print(f"AI: {ai_response_content}")
                    # 応答は既に追加済みだが、save_memory=True で LLMClient 側で保存する場合
                    if self.memory: # LLMClient にメモリがあるか確認
                       self.memory.save_context({"input": user_input}, {"output": ai_response_content})


            except Exception as e:
                print(f"An error occurred during interaction: {e}")
                traceback.print_exc()
                # エラーが発生した場合、ループを継続するかどうかは要検討

        # ループ終了後のクリーンアップ
        if self._google_api_key:
            genai.disconnect()
            print("ボットをサーバーから切断しました。")


    # run メソッドを修正して、インタラクティブループを呼び出すようにする (既存の処理はコメントアウト)
    async def run(self):
        """メイン実行関数 - インタラクティブループを開始"""
        await self.run_interactive_loop()

        # --- 既存のrunの内容 (コメントアウト) ---
        # # サーバー接続確認とボット召喚
        # server_active = await self.check_server_and_join()
        # if not server_active:
        #     print("サーバーに接続できないため、終了します")
        #     return
        # # ... (既存のステータス表示、スキルリスト表示など) ...
        # # 終了時の処理（tryの外で実行）
        # if self.discovery:
        #     self.discovery.disconnect_bot()
        #     print("ボットをサーバーから切断しました")
        # --- コメントアウト終了 ---


# Example Usage (Optional - illustrating tool calling)
if __name__ == '__main__':
    # Make sure to set OPENAI_API_KEY and GOOGLE_API_KEY environment variables
    # and install langchain (`pip install langchain`)
    try:
        client = LLMClient()
        client.clear_memory() # Start with fresh memory for the example

        print("--- Tool Calling Example (OpenAI) ---")
        # ダミーのツール定義
        tools_definition = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                }
            }
        ]

        user_prompt_tool = "What's the weather like in Boston?"
        print(f"User: {user_prompt_tool}")
        tool_response = client.get_response(
            system_prompt="You are a helpful assistant that can use tools.",
            user_prompt=user_prompt_tool,
            service="openai",
            model="gpt-3.5-turbo", # Or a model that supports tool calling well
            tools=tools_definition
            # save/use_memory は False のまま
        )

        print(f"AI Response: {tool_response}")

        # ツール呼び出しがあった場合の処理例 (実際にはツールを実行し結果を返す)
        if tool_response.get('tool_calls'):
            print("\nLLM requested tool calls:")
            for call in tool_response['tool_calls']:
                print(f"  ID: {call['id']}, Function: {call['function']['name']}, Arguments: {call['function']['arguments']}")
            # ここで実際に get_current_weather(location="Boston, MA") を呼び出し、
            # その結果を次の get_response 呼び出し時に tool ロールで渡す必要がある

        # メモリの内容を確認
        print("\n--- Current Memory ---")
        print(client.get_memory_string())

        # メモリをクリア
        client.clear_memory()
        print("\n--- Memory After Clearing ---")
        print(client.get_memory_string())


    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure LangChain is installed ('pip install langchain') and API keys are set.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
