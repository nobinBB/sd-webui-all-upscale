import os
import time
import threading
import gradio as gr
from PIL import Image
from modules import script_callbacks, shared, images, devices
from modules.shared import opts

# グローバル変数
stop_processing = False

def get_available_upscalers():
    """利用可能なアップスケーラーのリストを取得"""
    upscaler_names = []
    
    if hasattr(shared, 'sd_upscalers'):
        for upscaler in shared.sd_upscalers:
            if upscaler.name and upscaler.name != "None":
                upscaler_names.append(upscaler.name)
    
    upscaler_names = sorted(list(set(upscaler_names)))
    
    if not upscaler_names:
        upscaler_names = ["ESRGAN_4x", "LDSR", "SwinIR_4x", "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"]
    
    return upscaler_names

def get_upscaler_instance(upscaler_name):
    """指定された名前のアップスケーラーインスタンスを取得"""
    if hasattr(shared, 'sd_upscalers'):
        for upscaler in shared.sd_upscalers:
            if upscaler.name == upscaler_name:
                return upscaler
    return None

def estimate_processing_time(upscaler_name, total_files, avg_size):
    """処理時間の概算"""
    # アップスケーラーごとの基準時間（秒/メガピクセル）
    time_per_megapixel = {
        "ESRGAN_4x": 2.0,
        "R-ESRGAN 4x+": 2.5,
        "R-ESRGAN 4x+ Anime6B": 2.5,
        "LDSR": 8.0,
        "SwinIR_4x": 3.5,
    }
    
    base_time = time_per_megapixel.get(upscaler_name, 3.0)
    
    # GPUの有無で調整
    try:
        import torch
        if not torch.cuda.is_available():
            base_time *= 5
    except:
        base_time *= 5
    
    # 平均的な画像サイズ（メガピクセル）
    megapixels = (avg_size[0] * avg_size[1]) / 1_000_000
    
    # 合計時間の計算
    total_seconds = total_files * megapixels * base_time
    
    return total_seconds

def format_time(seconds):
    """秒を読みやすい形式に変換"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}分{int(seconds % 60)}秒"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}時間{minutes}分"

def count_images_and_estimate_size(folder_path):
    """画像ファイルの総数と平均サイズを取得"""
    total_files = 0
    total_width = 0
    total_height = 0
    sample_count = 0
    max_samples = 10
    
    # 親フォルダ直下の画像をチェック
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            total_files += 1
            if sample_count < max_samples:
                try:
                    with Image.open(filepath) as img:
                        total_width += img.width
                        total_height += img.height
                        sample_count += 1
                except:
                    pass
    
    # サブフォルダ内の画像をチェック
    for subdir_name in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir_name)
        if os.path.isdir(subdir_path) and not subdir_name.startswith("upscaled_"):
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    total_files += 1
                    if sample_count < max_samples:
                        try:
                            with Image.open(filepath) as img:
                                total_width += img.width
                                total_height += img.height
                                sample_count += 1
                        except:
                            pass
    
    # 平均サイズの計算
    if sample_count > 0:
        avg_width = total_width / sample_count
        avg_height = total_height / sample_count
    else:
        avg_width, avg_height = 1024, 1024
    
    return total_files, (avg_width, avg_height)

def process_batch_upscale(folder_path, upscaler_name, scale_factor, tile_size, progress=gr.Progress()):
    """バッチアップスケール処理"""
    global stop_processing
    stop_processing = False
    
    logs = []
    
    # 入力検証
    if not folder_path or not os.path.isdir(folder_path):
        return f"エラー: フォルダが見つかりません: {folder_path}"
    
    if not upscaler_name:
        return "エラー: アップスケーラーを選択してください。"
    
    # アップスケーラーの取得
    selected_upscaler = get_upscaler_instance(upscaler_name)
    if not selected_upscaler:
        return f"エラー: アップスケーラー '{upscaler_name}' が見つかりません。"
    
    # 画像数と推定時間の計算
    total_files, avg_size = count_images_and_estimate_size(folder_path)
    if total_files == 0:
        return "処理対象の画像が見つかりません。"
    
    estimated_time = estimate_processing_time(upscaler_name, total_files, avg_size)
    
    logs.append(f"使用アップスケーラー: {upscaler_name}")
    logs.append(f"倍率: {scale_factor}x")
    logs.append(f"タイルサイズ: {tile_size if tile_size > 0 else '自動'}")
    logs.append(f"親フォルダ: {folder_path}")
    logs.append(f"処理対象ファイル数: {total_files}")
    logs.append(f"推定処理時間: {format_time(estimated_time)}")
    logs.append("-" * 50)
    
    processed_count = 0
    error_count = 0
    start_time = time.time()
    
    try:
        # デバイスの設定
        devices.torch_gc()
        
        # 処理対象のファイルリストを作成
        files_to_process = []
        
        # 親フォルダ直下の画像
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                files_to_process.append((".", filename, folder_path))
        
        # サブフォルダ内の画像
        for subdir_name in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir_name)
            if os.path.isdir(subdir_path) and not subdir_name.startswith("upscaled_"):
                for filename in os.listdir(subdir_path):
                    filepath = os.path.join(subdir_path, filename)
                    if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        files_to_process.append((subdir_name, filename, subdir_path))
        
        # 各ファイルを処理
        for idx, (subdir_name, filename, base_path) in enumerate(files_to_process):
            if stop_processing:
                logs.append("\n処理が停止されました。")
                break
            
            # 進捗更新
            current_progress = (idx + 1) / total_files
            elapsed_time = time.time() - start_time
            if idx > 0:
                avg_time_per_file = elapsed_time / (idx + 1)
                remaining_time = avg_time_per_file * (total_files - idx - 1)
                progress_text = f"進捗: {idx + 1}/{total_files} ({current_progress*100:.1f}%) - 残り時間: {format_time(remaining_time)}"
            else:
                progress_text = f"進捗: {idx + 1}/{total_files} ({current_progress*100:.1f}%)"
            
            progress(current_progress, progress_text)
            
            # 出力ディレクトリの設定
            if subdir_name == ".":
                output_dir = os.path.join(folder_path, f"upscaled_{upscaler_name}_{scale_factor}x")
                folder_display_name = "親フォルダ"
            else:
                output_dir = os.path.join(base_path, f"upscaled_{scale_factor}x")
                folder_display_name = subdir_name
            
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)
            
            input_path = os.path.join(base_path, filename)
            
            try:
                # 画像を読み込み
                img = Image.open(input_path)
                
                # RGBに変換（必要な場合）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 元のサイズを記録
                original_size = img.size
                
                # Extrasと同じ処理方法を使用
                if hasattr(selected_upscaler, 'do_upscale'):
                    # より効率的なdo_upscaleメソッドがある場合
                    upscaled_img = selected_upscaler.do_upscale(img, selected_upscaler.data_path)
                else:
                    # 標準的なupscaleメソッド
                    # タイルサイズの設定
                    if tile_size > 0 and hasattr(selected_upscaler.scaler, 'tile_size'):
                        selected_upscaler.scaler.tile_size = tile_size
                        selected_upscaler.scaler.tile_pad = 10
                    
                    upscaled_img = selected_upscaler.scaler.upscale(img, scale_factor, selected_upscaler.data_path)
                
                # 出力ファイル名
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}_{scale_factor}x.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # 画像を保存
                upscaled_img.save(output_path, "PNG")
                
                new_size = upscaled_img.size
                logs.append(f"✓ [{folder_display_name}] {filename} → {output_filename} ({original_size} → {new_size})")
                processed_count += 1
                
                # メモリ解放
                del img
                del upscaled_img
                
            except Exception as e:
                logs.append(f"✗ [{folder_display_name}] {filename}: エラー - {str(e)}")
                error_count += 1
            
            # 定期的にメモリ解放
            if processed_count % 5 == 0:
                devices.torch_gc()
        
    except Exception as e:
        logs.append(f"\n処理中にエラーが発生しました: {str(e)}")
    
    # 最終的なメモリ解放
    devices.torch_gc()
    
    # 処理完了
    total_time = time.time() - start_time
    logs.append("\n" + "-" * 50)
    logs.append(f"処理完了: 成功 {processed_count} / エラー {error_count}")
    logs.append(f"実際の処理時間: {format_time(total_time)}")
    
    return "\n".join(logs)

def stop_processing_func():
    """処理停止"""
    global stop_processing
    stop_processing = True
    return "停止リクエストを送信しました..."

def create_ui():
    """UI作成"""
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("# Batch Upscale 親フォルダーの全てをアップスケール")
        gr.Markdown("Stable Diffusion WebUIの内蔵アップスケーラーを使用してバッチ処理を行います。")
        
        with gr.Row():
            with gr.Column(scale=1):
                # アップスケーラー選択
                upscalers = get_available_upscalers()
                upscaler_dropdown = gr.Dropdown(
                    choices=upscalers,
                    label="アップスケーラー",
                    value=upscalers[0] if upscalers else None,
                    interactive=True
                )
                
                # 親フォルダパス
                folder_input = gr.Textbox(
                    label="親フォルダパス",
                    placeholder="例: C:/Users/username/Pictures",
                    lines=1
                )
                
                # 倍率選択（デフォルト2倍）
                scale_slider = gr.Slider(
                    minimum=2,
                    maximum=4,
                    step=0.5,
                    value=2,  # デフォルト2倍
                    label="アップスケール倍率"
                )
                
                # タイルサイズ設定（メモリ効率化）
                tile_size_slider = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    step=64,
                    value=192,
                    label="タイルサイズ (0=自動, 大きいほど高速だがメモリ使用量増加)"
                )
                
                # ボタン
                with gr.Row():
                    run_button = gr.Button("アップスケール実行", variant="primary")
                    stop_button = gr.Button("停止", variant="stop")
                    refresh_button = gr.Button("リスト更新")
            
            with gr.Column(scale=2):
                # ログ出力
                log_output = gr.Textbox(
                    label="処理ログ",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
        
        # 使用方法
        with gr.Accordion("使用方法", open=False):
            gr.Markdown("""
            ### 特徴
            - 追加のライブラリインストール不要
            - 処理時間の概算表示
            - 処理中の停止機能
            - リアルタイムの進捗表示
            
            ### 推奨設定
            - **メモリ効率重視**: タイルサイズ192-256
            - **最高速度（VRAMに余裕がある場合）**: タイルサイズ0（自動）
            
            ### パフォーマンスのヒント
            - **タイルサイズ**: 大きいほど高速だがVRAM使用量が増加
            - **GPUメモリ不足の場合**: タイルサイズを128-192に設定
            - **CPUモードの場合**: タイルサイズを64-128に設定
            
            このツールはExtrasと同じエンジンを使用していますが、
            バッチ処理に最適化されており、メモリ効率も改善されています。
            
            ### フォルダ構造
            ```
            親フォルダ/
            ├── image1.jpg              # 親フォルダ直下の画像も処理
            ├── upscaled_ESRGAN_2x/     # 親フォルダ用の出力
            │   └── image1_2x.png
            ├── サブフォルダ1/
            │   ├── photo1.jpg
            │   └── upscaled_2x/        # サブフォルダ用の出力
            │       └── photo1_2x.png
            └── サブフォルダ2/
                └── ...
            ```
            """)
        
        # イベントハンドラ
        run_button.click(
            fn=process_batch_upscale,
            inputs=[folder_input, upscaler_dropdown, scale_slider, tile_size_slider],
            outputs=log_output,
            show_progress=True
        )
        
        stop_button.click(
            fn=stop_processing_func,
            outputs=log_output
        )
        
        def refresh_upscalers():
            upscalers = get_available_upscalers()
            return gr.Dropdown.update(choices=upscalers, value=upscalers[0] if upscalers else None)
        
        refresh_button.click(
            fn=refresh_upscalers,
            outputs=upscaler_dropdown
        )
    
    return ui

# タブとして登録
def on_ui_tabs():
    return [(create_ui(), "Batch Upscale", "batch_upscale")]

# 登録実行
try:
    script_callbacks.on_ui_tabs(on_ui_tabs)
    print("Batch Upscale (Built-in): タブ登録完了")
except Exception as e:
    print(f"Batch Upscale (Built-in): タブ登録エラー - {e}")
    import traceback
    traceback.print_exc()