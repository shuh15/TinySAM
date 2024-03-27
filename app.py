# Code credit: [EfficientSAM Demo](https://huggingface.co/spaces/yunyangx/EfficientSAM).

import copy
import os  # noqa

import gradio as gr
import numpy as np
import torch
from PIL import ImageDraw
from torchvision.transforms import ToTensor

from utils.tools import format_results, point_prompt
from utils.tools_gradio import fast_process
from tinysam import sam_model_registry, SamPredictor
from huggingface_hub import snapshot_download
import wget


#snapshot_download("merve/tinysam", local_dir="tinysam")
URL = "https://github.com/xinghaochen/TinySAM/releases/download/1.0/tinysam.pth"
response = wget.download(URL, "tinysam.pth")

model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./tinysam.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

# Description
title = "<center><strong><font size='8'>TinySAM<font></strong> <a href='https://github.com/xinghaochen/TinySAM'><font size='6'>[GitHub]</font></a> </center>"

description_e = """This is a demo of TinySAM Model](https://github.com/xinghaochen/TinySAM).
              """

description_p = """# Interactive Instance Segmentation
                - Point-prompt instruction
                <ol>
                <li> Click on the left image (point input), visualizing the point on the right image </li>
                <li> Click the button of Segment with Point Prompt </li>
                </ol>
                - Box-prompt instruction
                <ol>
                <li> Click on the left image (one point input), visualizing the point on the right image </li>
                <li> Click on the left image (another point input), visualizing the point and the box on the right image</li>
                <li> Click the button of Segment with Box Prompt </li>
                </ol>
                - Github [link](https://github.com/xinghaochen/TinySAM)
              """

# examples
examples = [
    ["fig/1.jpg"],
    ["fig/2.jpg"],
    ["fig/3.jpg"],
    ["fig/4.jpeg"],
    ["fig/5.jpg"],
    ["fig/6.jpeg"]
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_with_boxs(
    image,
    seg_image,
    global_points,
    global_point_label,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    if len(global_points) < 2:
        return seg_image, global_points, global_point_label
    print("Original Image : ", image.size)

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    print("Scaled Image : ", image.size)
    print("Scale : ", scale)

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_points = scaled_points[:2]
    scaled_point_label = np.array(global_point_label)[:2]

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, global_points, global_point_label

    nd_image = np.array(image)
    img_tensor = ToTensor()(nd_image)

    #coord_np = np.array(session_state['coord_list'])
    #label_np = np.array(session_state['label_list'])
    print(scaled_points, scaled_point_label)
    predictor.set_image(np.array(image))
    input_box = scaled_points.reshape([4])
    print('box', input_box)
    masks, scores, logits = predictor.predict(
        point_coords=None, #scaled_points,
        point_labels=None, #scaled_point_label,
        box=input_box[None, :]
    )
    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')
    annotations = np.expand_dims(masks[scores.argmax()], axis=0)
    
    print(annotations)
    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        use_retina=use_retina,
        bbox = scaled_points.reshape([4]),
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, global_points, global_point_label


def segment_with_points(
    image,
    global_points,
    global_point_label,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    print("Original Image : ", image.size)

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    print("Scaled Image : ", image.size)
    print("Scale : ", scale)

    if global_points is None:
        return image, global_points, global_point_label
    if len(global_points) < 1:
        return image, global_points, global_point_label
    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, global_points, global_point_label

    nd_image = np.array(image)
    img_tensor = ToTensor()(nd_image)
    print(img_tensor.shape)    
    
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=global_point_label,
    )
    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')
    annotations = np.expand_dims(masks[scores.argmax()], axis=0)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        points = scaled_points,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, global_points, global_point_label


def get_points_with_draw(image, cond_image, global_points, global_point_label, evt: gr.SelectData):
    print("Starting functioning")
    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 8, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    if image is not None:
        draw = ImageDraw.Draw(image)

        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
        )

    return image, global_points, global_point_label

def get_points_with_draw_(image, cond_image, global_points, global_point_label, evt: gr.SelectData):

    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    if len(global_points) > 2:
        return image, global_points, global_point_label
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 8, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    if image is not None:
        draw = ImageDraw.Draw(image)

        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
        )

    if len(global_points) == 2:
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 < y2:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        elif x1 < x2 and y1 >= y2:
            draw.rectangle([x1, y2, x2, y1], outline="red", width=5)
            global_points[0][0] = x1
            global_points[0][1] = y2
            global_points[1][0] = x2
            global_points[1][1] = y1
        elif x1 >= x2 and y1 < y2:
            draw.rectangle([x2, y1, x1, y2], outline="red", width=5)
            global_points[0][0] = x2
            global_points[0][1] = y1
            global_points[1][0] = x1
            global_points[1][1] = y2
        elif x1 >= x2 and y1 >= y2:
            draw.rectangle([x2, y2, x1, y1], outline="red", width=5)
            global_points[0][0] = x2
            global_points[0][1] = y2
            global_points[1][0] = x1
            global_points[1][1] = y1

    return image, global_points, global_point_label


cond_img_p = gr.Image(label="Input with Point", value=default_example[0], type="pil")
cond_img_b = gr.Image(label="Input with Box", value=default_example[0], type="pil")

segm_img_p = gr.Image(
    label="Segmented Image with Point-Prompt", interactive=False, type="pil"
)
segm_img_b = gr.Image(
    label="Segmented Image with Box-Prompt", interactive=False, type="pil"
)

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

with gr.Blocks(css=css, title="TinySAM") as demo:
    global_points = gr.State([])
    global_point_label = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        # ###
        with gr.Row():
            with gr.Column():

                with gr.Column():
                    segment_btn_p = gr.Button(
                        "Segment with Point Prompt", variant="primary"
                    )
                    clear_btn_p = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    examples_per_page=6,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    with gr.Tab("Box mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_b.render()

            with gr.Column(scale=1):
                segm_img_b.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():

                with gr.Column():
                    segment_btn_b = gr.Button(
                        "Segment with Box Prompt", variant="primary"
                    )
                    clear_btn_b = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_b],

                    examples_per_page=6,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.select(get_points_with_draw, inputs = [segm_img_p, cond_img_p, global_points, global_point_label], outputs = [segm_img_p, global_points, global_point_label])

    cond_img_b.select(get_points_with_draw_, [segm_img_b, cond_img_b, global_points, global_point_label], [segm_img_b, global_points, global_point_label])

    segment_btn_p.click(
        segment_with_points, inputs=[cond_img_p, global_points, global_point_label], outputs=[segm_img_p, global_points, global_point_label]
    )

    segment_btn_b.click(
        segment_with_boxs, inputs=[cond_img_b, segm_img_b, global_points, global_point_label], outputs=[segm_img_b,global_points, global_point_label]
    )

    def clear():
        return None, None, [], []

    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p, global_points, global_point_label])
    clear_btn_b.click(clear, outputs=[cond_img_b, segm_img_b, global_points, global_point_label])

demo.queue()
demo.launch()
