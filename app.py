import gradio as gr
import numpy as np
import torch
import sys
import os
from tinysam import sam_model_registry, SamPredictor


model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weight/tinysam.pth")

predictor = SamPredictor(sam)

examples = [
    [os.path.join(os.path.dirname(__file__), "fig/picture3.jpg")],
    ["fig/picture2.jpg"],
    ["fig/picture1.jpg"],
]

default_example = examples[0]
# Description
title = "<center><strong><font size='8'>TinySAM<font></strong> <a href='https://github.com/xinghaochen/TinySAM'><font size='6'>[GitHub]</font></a> </center>"
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
css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

def infer(img):
    if img is None:
        gr.Error("Please upload an image and select a point.")
    if img["background"] is None:
        gr.Error("Please upload an image and select a point.")
    # background (original image) layers[0] ( point prompt) composite (total image)
    image = img["background"].convert("RGB")
    point_prompt = img["layers"][0]
    total_image = img["composite"]
    predictor.set_image(np.array(image))
    print("point_prompt : ", point_prompt)

    # get point prompt
    img_arr = np.array(point_prompt)
    if not np.any(img_arr):
        gr.Error("Please select a point on top of the image.")
    else:
        nonzero_indices = np.nonzero(img_arr)
        img_arr = np.array(point_prompt)
        nonzero_indices = np.nonzero(img_arr)
        center_x = int(np.mean(nonzero_indices[1]))
        center_y = int(np.mean(nonzero_indices[0]))
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
        )
        result_label = [(masks[scores.argmax(), :, :], "mask")]
        return image, result_label


with gr.Blocks(css=css, title="TinySAM") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            im = gr.ImageEditor(
                type="pil",
            )
        output = gr.AnnotatedImage()

    im.change(infer, inputs=im, outputs=output)

demo.launch(debug=True)
