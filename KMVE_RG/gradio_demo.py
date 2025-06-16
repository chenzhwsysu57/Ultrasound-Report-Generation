
# import gradio as gr
# from PIL import Image
# import os
# import datetime
# from inference_api import UltrasoundReportModel

# model = UltrasoundReportModel()

# SAVE_DIR = "/home/chenzhw/ultrasound_report_gen/US-Report-Gen/gradio_files"
# IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, "images")
# REPORT_SAVE_DIR = os.path.join(SAVE_DIR, "reports")
# os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
# os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

# def files_to_pil_images(file_list):
#     images = []
#     for f in file_list:
#         img = Image.open(f).convert("RGB")
#         images.append(img)
#     return images

# def add_images(new_files, state):
#     if new_files:
#         state.extend(new_files)
#     return state, [file.name for file in state]

# def generate_report_gradio(image_list):
#     if len(image_list) != 2:
#         return "请上传两张图像（左、右或纵横对）"
#     report = model.infer(image_list)
#     return report

# def save_report_and_images(state, report_text):
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     image_paths = []

#     try:
#         for idx, file in enumerate(state):
#             img = Image.open(file).convert("RGB")
#             image_name = f"{timestamp}_{idx+1}.jpg"
#             image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
#             img.save(image_path)
#             image_paths.append(image_path)
#     except Exception as e:
#         return f"保存图片失败: {e}"

#     try:
#         report_filename = f"{timestamp}_report.txt"
#         report_path = os.path.join(REPORT_SAVE_DIR, report_filename)
#         with open(report_path, "w", encoding="utf-8") as f:
#             f.write(report_text)
#     except Exception as e:
#         return f"保存报告失败: {e}"

#     return f"✅ 保存成功：\n图片 {len(image_paths)} 张\n报告文件名：{report_filename}"

# # ✅ 清除所有状态
# def clear_all():
#     return [], "", None, ""  # 清空：state, 文件名, 图像预览, 报告文本

# with gr.Blocks() as demo:
#     gr.Markdown("## 🩺 Ultrasound 报告生成")

#     state = gr.State([])

#     with gr.Row():
#         image_input = gr.File(file_types=["image"], file_count="multiple", label="上传两张超声图像")
#         image_names = gr.Textbox(label="当前文件", interactive=False)

#     image_preview = gr.Gallery(label="图片预览")

#     with gr.Row():
#         submit_btn = gr.Button("生成报告")
#         save_btn = gr.Button("保存报告")
#         clear_btn = gr.Button("清除输入")  # ✅ 新增按钮

#     output_box = gr.Textbox(label="生成报告（可编辑）", lines=10, interactive=True)
#     save_status = gr.Textbox(label="保存状态", interactive=False)

#     image_input.change(
#         fn=add_images,
#         inputs=[image_input, state],
#         outputs=[state, image_names]
#     )

#     def update_preview(state):
#         return files_to_pil_images(state)

#     state.change(
#         fn=update_preview,
#         inputs=state,
#         outputs=image_preview
#     )

#     submit_btn.click(
#         fn=generate_report_gradio,
#         inputs=state,
#         outputs=output_box
#     )

#     save_btn.click(
#         fn=save_report_and_images,
#         inputs=[state, output_box],
#         outputs=save_status
#     )

#     # ✅ 清除按钮点击行为
#     clear_btn.click(
#         fn=clear_all,
#         inputs=[],
#         outputs=[state, image_names, image_preview, output_box]
#     )

# demo.launch()

import gradio as gr
from PIL import Image
import os
import datetime
from inference_api import UltrasoundReportModel
from functools import partial

model = UltrasoundReportModel()

SAVE_DIR = "/home/chenzhw/ultrasound_report_gen/US-Report-Gen/gradio_files"
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, "images")
REPORT_SAVE_DIR = os.path.join(SAVE_DIR, "reports")
EMBEDDING_FILE = "/home/chenzhw/ultrasound_report_gen/USData/embedding_index.json"

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

def files_to_pil_images(file_list):
    images = []
    for f in file_list:
        img = Image.open(f).convert("RGB")
        images.append(img)
    return images

def add_images(new_files, state):
    if new_files:
        state.extend(new_files)
    return state, [file.name for file in state]

def generate_report_gradio(image_list):
    if len(image_list) != 2:
        return "请上传两张图像（左、右或纵横对）"
    report = model.infer(image_list)
    return report

def save_report_and_images(state, report_text):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_paths = []

    try:
        for idx, file in enumerate(state):
            img = Image.open(file).convert("RGB")
            image_name = f"{timestamp}_{idx+1}.jpg"
            image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            img.save(image_path)
            image_paths.append(image_path)
    except Exception as e:
        return f"保存图片失败: {e}"

    try:
        report_filename = f"{timestamp}_report.txt"
        report_path = os.path.join(REPORT_SAVE_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    except Exception as e:
        return f"保存报告失败: {e}"

    return f"✅ 保存成功：\n图片 {len(image_paths)} 张\n报告文件名：{report_filename}"

# ✅ 查询相似图像：2张图分别返回Top2
def query_similar_images(state, model):
    print(model)
    if len(state) != 2:
        return [], "❌ 请先上传两张图像"

    all_images = []
    info_lines = []

    for idx, file in enumerate(state):
        try:
            sims = model.find_topk_similar(file, EMBEDDING_FILE, topk=2)
            for rank, (path, score) in enumerate(sims):
                try:
                    img = Image.open(path).convert("RGB")
                    all_images.append(img)
                    info_lines.append(f"[图像{idx+1} Top{rank+1}] {score:.4f} @ {path}")
                except Exception as e:
                    info_lines.append(f"[图像{idx+1} Top{rank+1}] {score:.4f} @ {path} [读取失败: {e}]")
        except Exception as e:
            info_lines.append(f"图像{idx+1} 查询失败: {e}")

    return all_images, "\n".join(info_lines)

# ✅ 清除所有状态
def clear_all():
    return [], "", None, "", [], ""

with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Ultrasound 报告生成")

    state = gr.State([])

    with gr.Row():
        image_input = gr.File(file_types=["image"], file_count="multiple", label="上传两张超声图像")
        image_names = gr.Textbox(label="当前文件", interactive=False)

    image_preview = gr.Gallery(label="图片预览", height=200)

    with gr.Row():
        submit_btn = gr.Button("生成报告")
        save_btn = gr.Button("保存报告")
        clear_btn = gr.Button("清除输入")
        query_btn = gr.Button("查询相似图像")

    output_box = gr.Textbox(label="生成报告（可编辑）", lines=10, interactive=True)
    save_status = gr.Textbox(label="保存状态", interactive=False)

    with gr.Row():
        sim_gallery = gr.Gallery(label="相似图像（每图Top2）", height=200)
        sim_info = gr.Textbox(label="相似图像得分及路径", lines=8, interactive=False)

    image_input.change(
        fn=add_images,
        inputs=[image_input, state],
        outputs=[state, image_names]
    )

    def update_preview(state):
        return files_to_pil_images(state)

    state.change(
        fn=update_preview,
        inputs=state,
        outputs=image_preview
    )

    submit_btn.click(
        fn=generate_report_gradio,
        inputs=state,
        outputs=output_box
    )

    save_btn.click(
        fn=save_report_and_images,
        inputs=[state, output_box],
        outputs=save_status
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[state, image_names, image_preview, output_box, sim_gallery, sim_info]
    )

    query_btn.click(
        fn=partial(query_similar_images, model=model),
        inputs=state,
        outputs=[sim_gallery, sim_info],
        
    )

demo.launch()
