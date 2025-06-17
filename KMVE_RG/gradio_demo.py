
import gradio as gr
from PIL import Image
import os
import requests

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

def postprocess_report(report):
    report = report.replace(' ', '')
    replace_list = ['_SMM_', '_SCM_', '_Loc_', '_LocR_', '_2DS_', '_3DS_', '_r_']
    for item in replace_list:
        report = report.replace(item, '____')
    return report

def generate_report_gradio(image_list):
    if len(image_list) != 2:
        return "请上传两张图像（左、右或纵横对）"
    report = model.infer(image_list)
    
    report = postprocess_report(report)
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
        return f"归档图片失败: {e}"

    try:
        report_filename = f"{timestamp}_report.txt"
        report_path = os.path.join(REPORT_SAVE_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    except Exception as e:
        return f"归档报告失败: {e}"

    return f"✅ 归档成功：\n图片 {len(image_paths)} 张\n报告文件名：{report_filename}"
import os
import json
def find_report_from_imgpath(path):
    # 读取 all.json
    all_report_path = '/home/chenzhw/ultrasound_report_gen/USData/all.json'
    with open(all_report_path, 'r', encoding='utf-8-sig') as f:
        all_data = json.load(f)

    # 提取文件名，比如 128002_2.jpeg
    filename = os.path.basename(path)

    # 提取 UID，例如从 128002_2.jpeg 中提取 128002
    uid = filename.split('_')[0]

    # 遍历所有样本，查找包含该文件名的项
    for split in ['train', 'val', 'test']:
        if split in all_data:
            for sample in all_data[split]:
                if filename in sample.get('image_path', []):
                    return sample.get('finding', None)

    # 如果未找到
    return None

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
                    info_lines.append(f"图{idx+1}相似度 {score:.4f} \n{postprocess_report(find_report_from_imgpath(path))}\n")
                except Exception as e:
                    info_lines.append(f"[图像{idx+1} Top{rank+1}] {score:.4f} @ {path} [读取失败: {e}]")
        except Exception as e:
            info_lines.append(f"图像{idx+1} 查询失败: {e}")

    return all_images, "\n".join(info_lines)

def query_similar_texts(report_text):
    try:
        response = requests.post(
            "http://localhost:8000/search",
            headers={"Content-Type": "application/json"},
            json={"query": report_text, "top_k": 3}
        )
        if response.status_code != 200:
            return [], f"❌ 查询失败，状态码: {response.status_code}"

        data = response.json()
        results = data.get("results", [])

        images = []
        report_lines = []

        for item in results:
            score = item.get("score", 0)
            if score >= 0.9999:  # 排除相似度为1的（几乎一致）
                continue

            uid = str(item.get("uid"))
            finding = item.get("finding", "")
            score_str = f"{score:.4f}"

            # 通过 UID 查找图像路径
            image_found = False
            all_report_path = '/home/chenzhw/ultrasound_report_gen/USData/all.json'
            all_report_root = '/home/chenzhw/ultrasound_report_gen/USData/all_report'
            with open(all_report_path, 'r', encoding='utf-8-sig') as f:
                all_data = json.load(f)

            for split in ['train', 'val', 'test']:
                for sample in all_data.get(split, []):
                    if str(sample.get("uid")) == uid:
                        for img_path in sample.get("image_path", []):
                            try:
                                img = Image.open(os.path.join(all_report_root,img_path)).convert("RGB")
                                images.append(img)
                            except Exception as e:
                                report_lines.append(f"[图像加载失败]: {img_path} -> {e}")
                        report_lines.append(f"UID: {uid} | 相似度: {score_str}\n{postprocess_report(finding)}\n")
                        image_found = True
                        break
                if image_found:
                    break

        return images, "\n".join(report_lines) if report_lines else "未找到匹配图像"
    except Exception as e:
        return [], f"❌ 查询出错: {e}"


# ✅ 清除所有状态
def clear_all():
    return [], "", None, "", [], ""

with gr.Blocks() as demo:
    gr.Markdown("## 超声报告生成系统")

    state = gr.State([])

    with gr.Row():
        image_input = gr.File(file_types=["image"], file_count="multiple", label="上传两张超声图像")
        image_names = gr.Textbox(label="当前文件", interactive=False)

    image_preview = gr.Gallery(label="图片预览")

    with gr.Row():
        submit_btn = gr.Button("生成报告")
        query_btn = gr.Button("查询相似图像")
        similar_text_btn = gr.Button("查询相似报告")
        save_btn = gr.Button("归档报告")
        clear_btn = gr.Button("清除输入")
        

    output_box = gr.Textbox(label="生成报告（可编辑）", lines=10, interactive=True)
    save_status = gr.Textbox(label="归档状态", interactive=False)

    with gr.Row():
        sim_gallery = gr.Gallery(label="相似图像（每图查两张）")
        sim_info = gr.Textbox(label="相似图像及报告", lines=10, interactive=False)
    with gr.Row():
        similar_text_gallery = gr.Gallery(label="相似报告图像")
        similar_text_texts = gr.Textbox(label="相似报告", lines=10, interactive=False)

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

    similar_text_btn.click(
        fn=query_similar_texts,
        inputs=output_box,
        outputs=[similar_text_gallery, similar_text_texts]
    )

demo.launch()
