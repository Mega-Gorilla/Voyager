from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from discovery import Discovery
from skill.skills import Skills
from contextlib import asynccontextmanager
import math
from javascript import require # Vec3 を使う可能性のため (skills.pyの依存関係)
import inspect # メソッドとdocstring取得のため
import io # 標準出力/エラー出力キャプチャのため
import contextlib # redirect_stdout/stderrのため
import traceback # トレースバック取得のため
import textwrap # インデント調整のため
import os # osモジュールをインポート
from dotenv import load_dotenv # python-dotenvからload_dotenvをインポート
import ast # ast モジュールをインポート

# Discoveryインスタンスの初期化
discovery = Discovery()
skills = None
current_goal: Optional[str] = None # ★ 追加: 現在のゴールを格納する変数

# lifespanコンテキストマネージャを定義
@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリケーション起動時の処理
    global skills
    await discovery.check_server_and_join()
    skills = discovery.skills
    # サーバー接続確認（非同期で実行）
    asyncio.create_task(check_server_connection())
    
    yield
    
    # アプリケーション終了時の処理
    # 必要に応じてボットの切断処理などを実装
    if discovery:
        discovery.disconnect_bot()

# FastAPIインスタンスを作成
app = FastAPI(
    title="MineCraft Bot API",
    description="Minecraft Bot API は、MineFlayer を使用したマインクラフトボットの制御APIです。",
    version="1.0.0",
    lifespan=lifespan
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# サーバー接続確認
async def check_server_connection():
    server_active = await discovery.check_server_active(timeout=15)
    if not server_active:
        raise HTTPException(status_code=503, detail="マインクラフトサーバーに接続できません。ポートを確認してください")

# ルートエンドポイント
@app.get("/")
async def root():
    return {"message": "MineCraft Bot API is running..."}

# サーバー情報の取得
@app.get("/server/info", tags=["server"])
async def get_server_info():
    return discovery.get_server_info()

# ボット接続状態の確認 -> ボット周辺ブロックの領域分類に変更
@app.get("/bot/status", tags=["bot"], summary="ボットの状態と周辺情報（バイオーム、時間、体力、空腹度、エンティティ、インベントリ、ブロック分類）を取得")
async def get_bot_status():
    # --- ボットの基本情報を取得 ---
    try:
        bot_entity = discovery.bot.entity
        bot_pos_raw = bot_entity.position # Y座標はエンティティ基準
        bot_health = discovery.bot.health
        bot_food = discovery.bot.food
        bot_time = discovery.bot.time.timeOfDay

        # ボットがいるブロックとバイオームを取得
        center_block = discovery.bot.blockAt(bot_pos_raw)
        bottom_block = discovery.bot.blockAt(bot_pos_raw.offset(0, -1, 0))
        bot_pos = center_block.position.offset(0, 1, 0)
        bot_biome_id = discovery.bot.world.getBiome(bot_pos)
        bot_biome_name = discovery.mcdata.biomes[str(bot_biome_id)]['name']
        bot_x = bot_pos.x
        bot_z = bot_pos.z

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ボットの基本情報の取得に失敗しました: {e}")

    # --- 周囲のブロックを取得 & 分類 ---
    try:
        blocks = await skills.get_surrounding_blocks(
            position=bot_pos, # スキルの引数名に合わせる
            x_distance=3,
            y_distance=2,
            z_distance=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"周辺ブロック情報の取得に失敗しました: {e}")

    # ブロック名をグループごとに一時的に格納
    temp_grouped_block_names = {"group1": [], "group2": [], "group3": [], "group4": [], "group0": []}

    if blocks:
        for block in blocks:
            block_pos_dict = block.get('position')
            block_name = block.get('name')
            if not isinstance(block_pos_dict, dict) or block_name is None:
                continue

            block_x = block_pos_dict.get('x')
            block_z = block_pos_dict.get('z')
            if not isinstance(block_x, (int, float)) or not isinstance(block_z, (int, float)):
                continue

            dx = block_x - bot_x
            dz = block_z - bot_z

            if math.fabs(dx) < 1e-6 and math.fabs(dz) < 1e-6:
                 temp_grouped_block_names["group0"].append(block_name)
            elif dz > 1e-6 and math.fabs(dx) <= dz + 1e-6:
                temp_grouped_block_names["group1"].append(block_name)
            elif dx > 1e-6 and math.fabs(dz) <= dx + 1e-6:
                temp_grouped_block_names["group2"].append(block_name)
            elif dz < -1e-6 and math.fabs(dx) <= math.fabs(dz) + 1e-6:
                temp_grouped_block_names["group3"].append(block_name)
            elif dx < -1e-6 and math.fabs(dz) <= math.fabs(dx) + 1e-6:
                temp_grouped_block_names["group4"].append(block_name)

    # ブロック分類結果（重複除去とソート）
    classified_blocks = {
        "front_blocks": sorted(list(set(temp_grouped_block_names["group1"]))),
        "right_blocks": sorted(list(set(temp_grouped_block_names["group2"]))),
        "back_blocks": sorted(list(set(temp_grouped_block_names["group3"]))),
        "left_blocks": sorted(list(set(temp_grouped_block_names["group4"]))),
        "center_blocks": sorted(list(set(temp_grouped_block_names["group0"])))
    }

    # --- 近くのエンティティ情報を取得 ---
    nearby_entities_info = []
    try:
        # _get_nearby_entities は同期メソッドの可能性あり
        nearby_entities_raw = skills._get_nearby_entities(max_distance=16) # 範囲は適宜調整
        if nearby_entities_raw:
            for entity in nearby_entities_raw:
                # 有効なエンティティ情報のみ抽出
                if hasattr(entity, 'name') and hasattr(entity, 'position') and entity.position:
                    nearby_entities_info.append({
                        "name": entity.name,
                        "position": {
                            "x": round(entity.position.x, 1), # 小数点以下第一位で四捨五入
                            "y": round(entity.position.y, 1), # 小数点以下第一位で四捨五入
                            "z": round(entity.position.z, 1)  # 小数点以下第一位で四捨五入
                        }
                    })
    except Exception as e:
        print(f"Warning: Failed to get nearby entities ({e})")
        # エラーが発生しても処理は続行

    # --- インベントリ情報を取得 ---
    inventory_info = {}
    try:
        # get_inventory_counts は同期メソッド
        inventory_info = skills.get_inventory_counts()
    except Exception as e:
        print(f"Warning: Failed to get inventory counts ({e})")
        # エラーが発生しても処理は続行

    # --- 最終的なレスポンスを作成 ---
    final_result = {
        "biome": bot_biome_name,
        "time_of_day": bot_time,
        "health": bot_health,
        "hunger": bot_food,
        "bot_position": f"x = {center_block.position.x}, y = {center_block.position.y}, z = {center_block.position.z}",
        "nearby_entities": nearby_entities_info,
        "inventory": inventory_info,
        **classified_blocks # ブロック分類結果を展開して結合
    }

    return final_result

# Skillsクラスの関数リストを取得するエンドポイント
@app.get("/skills/list", tags=["skills"], summary="Skillsクラスで利用可能な関数（メソッド）の名前、説明、非同期フラグのリストを取得")
async def get_skills_list():
    global skills # lifespanで初期化されたskillsインスタンスを使用
    if skills is None:
        raise HTTPException(status_code=503, detail="Skillsが初期化されていません")

    skill_list = []
    # inspect.getmembersでskillsオブジェクトのメソッドを取得
    for name, method in inspect.getmembers(skills, inspect.ismethod):
        # アンダースコアで始まらない公開メソッドのみを対象とする
        if not name.startswith('_'):
            # docstringを取得し、整形
            docstring = inspect.cleandoc(method.__doc__) if method.__doc__ else "説明がありません。"
            # メソッドが非同期関数かどうかをチェック
            is_async = inspect.iscoroutinefunction(method)

            skill_list.append({
                "name": name,
                "description": docstring,
                "is_async": is_async # 非同期フラグを追加
            })

    # 名前順にソートして返す
    return sorted(skill_list, key=lambda x: x['name'])

# 新しいエンドポイント: 特定のスキル関数のソースコードを取得
@app.get("/skills/code/{skill_name}", tags=["skills"], summary="指定されたスキル関数のソースコードを取得 (docstring除外)")
async def get_skill_code(skill_name: str = Path(..., title="取得したいスキル関数の名前")):
    global skills
    if skills is None:
        raise HTTPException(status_code=503, detail="Skillsが初期化されていません")

    # skill_nameに対応するメソッドを取得
    try:
        method = getattr(skills, skill_name)
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"スキル関数 '{skill_name}' が見つかりません")

    # メソッドが呼び出し可能で、アンダースコアで始まらないことを確認
    if not callable(method) or skill_name.startswith('_'):
        raise HTTPException(status_code=404, detail=f"スキル関数 '{skill_name}' が見つかりません、またはアクセスできません")

    # --- Docstringを除去するTransformer ---
    class DocstringRemover(ast.NodeTransformer):
        def _remove_docstring(self, node):
            if not node.body:
                return
            # 関数/クラス定義内の最初の式がdocstringであるか確認
            if isinstance(node.body[0], ast.Expr):
                if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    # Docstring (Python 3.8+)
                    node.body.pop(0)
                elif isinstance(node.body[0].value, ast.Str):
                    # Docstring (Python 3.7 or earlier)
                    node.body.pop(0)

        def visit_FunctionDef(self, node):
            self._remove_docstring(node)
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            self._remove_docstring(node)
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node): # クラス定義のdocstringも除去する場合
            self._remove_docstring(node)
            self.generic_visit(node)
            return node

    # メソッドのソースコードを取得し、docstringを除去
    try:
        source_code = inspect.getsource(method)
        # ソースコードのインデントを除去 (ASTパース前にdedentが必要)
        dedented_source_code = textwrap.dedent(source_code)

        # ASTにパース
        tree = ast.parse(dedented_source_code)

        # Docstringを削除するTransformerを適用
        transformer = DocstringRemover()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree) # Location情報を修正

        # ASTをソースコード文字列に戻す (Python 3.9+)
        # ast.unparse はインデントを再構築する
        code_without_docstring = ast.unparse(new_tree)

        return {"skill_name": skill_name, "source_code": code_without_docstring}

    except (TypeError, OSError) as e:
        # ソースコードが取得できない場合
        raise HTTPException(status_code=500, detail=f"スキル関数 '{skill_name}' のソースコードを取得できませんでした: {e}")
    except SyntaxError as e:
        # AST パース失敗時のエラーハンドリング
        raise HTTPException(status_code=500, detail=f"スキル関数 '{skill_name}' のソースコードの解析に失敗しました: {e}")
    except AttributeError as e:
        # ast.unparse がない場合のエラー (Python 3.9未満)
        if "'module' object has no attribute 'unparse'" in str(e):
             raise HTTPException(status_code=501, detail="この機能にはPython 3.9以上が必要です (ast.unparse)。")
        else:
             raise HTTPException(status_code=500, detail=f"予期せぬエラーが発生しました: {e}")

# --- Pydantic モデル定義 ---
class CodeExecutionRequest(BaseModel):
    code: str

class TeleportRequest(BaseModel):
    position_x: float
    position_y: float
    position_z: float

# --- Pythonコード実行エンドポイント ---
@app.post("/execute/python_code", tags=["execute"], summary="Pythonコード文字列を実行します（非同期対応）【セキュリティ注意】")
async def execute_python_code(request: CodeExecutionRequest):
    global skills, discovery, bot
    # botインスタンスを discovery から取得してグローバルスコープに設定 (関数内でアクセス可能にするため)
    # lifespanで初期化されていることを前提とする
    if discovery and hasattr(discovery, 'bot'):
        bot = discovery.bot
    else:
        # discovery または bot が未初期化の場合のエラーハンドリング
        raise HTTPException(status_code=503, detail="Bot is not initialized")

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    # 動的に生成する非同期ラッパー関数の名前
    dynamic_async_func_name = "__dynamic_exec_async_code__"

    # LLMが生成したコードを適切にインデントする
    # textwrap.indent を使って各行の先頭にスペースを追加
    indented_user_code = textwrap.indent(request.code, '    ') # 4スペースでインデント

    # 非同期ラッパー関数のコード文字列を作成
    # skills, discovery, bot, asyncio を関数内で利用可能にする
    # グローバル変数としてアクセスする想定
    wrapper_code = f"""
import asyncio # ラッパー関数内で asyncio を利用可能にする

async def {dynamic_async_func_name}():
    # グローバル変数 skills, discovery, bot を参照
    global skills, discovery, bot
    # --- User Code Start ---
{indented_user_code}
    # --- User Code End ---

"""
    # exec のための実行コンテキスト (グローバルスコープ)
    # ラッパー関数が参照できるように、現在のグローバル変数を含める
    exec_globals = globals().copy()
    # 必要に応じて追加の変数を渡すことも可能
    # exec_globals.update({"some_other_var": some_value})

    try:
        # ラッパー関数を定義する
        # exec_globals を渡すことで、ラッパー関数が skills 等のグローバル変数を参照できる
        exec(wrapper_code, exec_globals)

        # 定義された非同期関数オブジェクトを取得
        async_func_to_run = exec_globals.get(dynamic_async_func_name)

        if async_func_to_run and inspect.iscoroutinefunction(async_func_to_run):
            # 標準出力と標準エラーをキャプチャしながら、動的に定義した非同期関数を実行
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                await async_func_to_run()
        else:
            # 関数が正しく定義されなかった場合のエラー
            # 同期コードとして実行するフォールバックも考えられるが、ここではエラーとする
            raise RuntimeError(f"Failed to define the internal async wrapper function '{dynamic_async_func_name}'.")

        # 実行結果を取得
        output = output_buffer.getvalue()
        error_output = error_buffer.getvalue()

        return {
            "success": True,
            "output": output,
            "error_output": error_output
        }

    except Exception as e:
        # exec または await 中のエラーをキャプチャ
        error_message = str(e)
        tb_str = traceback.format_exc()
        # エラー発生前のエラー出力も取得しておく
        error_output_before_exception = error_buffer.getvalue()

        return {
            "success": False,
            "error": error_message,
            "traceback": tb_str,
            # エラー発生前の標準エラー出力も返す
            "error_output": error_output_before_exception
        }

# --- テレポートエンドポイント (/bot/teleport) ---
@app.post("/bot/teleport", tags=["bot"])
async def teleport_bot(request: TeleportRequest):
    discovery.bot.chat(f"/tp bot {request.position_x} {request.position_y} {request.position_z}")
    return {"message": "ボットを転送しました"}

# --- ゴール設定/取得エンドポイント ---
class GoalRequest(BaseModel):
    goal: str

# ボットの現在の目標（ゴール）を設定します
@app.post("/bot/goal", tags=["bot"], summary="ボットの現在の目標（ゴール）を設定します")
async def set_bot_goal(request: GoalRequest):
    global current_goal
    current_goal = request.goal
    discovery.bot.chat(f"目標が 「{current_goal}」 に設定されました")
    return {"message": f"目標を設定しました: {current_goal}"}

# ボットに現在設定されている目標（ゴール）を取得します
@app.get("/bot/goal", tags=["bot"], summary="ボットに現在設定されている目標（ゴール）を取得します")
async def get_bot_goal():
    global current_goal
    if current_goal is None:
        # raise HTTPException(status_code=404, detail="目標はまだ設定されていません")
        # 404ではなく、Noneを返す仕様に変更（他のAPIに合わせる）
        return {"goal": None}
    return {"goal": current_goal}

# サーバー起動用コード
if __name__ == "__main__":
    load_dotenv() # .envファイルから環境変数を読み込む
    port = int(os.getenv("PORT", 8000)) # 環境変数 PORT を読み込む、なければデフォルト8000
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port, reload=True) 