import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)

    ### FILL: Implement MLS or RBF based image warping
    h, w = image.shape[:2]
    p = np.array(source_pts).astype(np.float32)
    q = np.array(target_pts).astype(np.float32)

    for x in range(w):
        for y in range(h):
            # Step1: Calculate dynamic weights
            v = np.array([x, y], dtype=np.float32)
            diff = q - v
            dist2 = np.sum(diff ** 2, axis=1)
            wi = 1.0 / ((dist2)** alpha + eps)
            # Step2: Calculate weighted centroids
            w_sum = np.sum(wi)
            p_star = np.sum(wi[:, None] * p, axis=0) / w_sum
            q_star = np.sum(wi[:, None] * q, axis=0) / w_sum
            # Step3: Switch to relative coordinates
            p_hat = p - p_star
            q_hat = q - q_star
            # Step4: Solve optimal transformation matrix M
            A = np.zeros((2, 2), dtype=np.float32)
            B = np.zeros((2, 2), dtype=np.float32)
            for i in range(len(p)):
                A += wi[i] * np.outer(q_hat[i], q_hat[i])
                B += wi[i] * np.outer(p_hat[i], q_hat[i])
            M = np.linalg.pinv(A) @ B
            # Step5: Determine final pixel position
            vs = (v - q_star) @ M + p_star
            if 0 <= vs[0] < w and 0 <= vs[1] < h:
                warped_image[y, x] = image[int(vs[1]), int(vs[0])]
            else:
                warped_image[y, x] = (255, 255, 255)
    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
