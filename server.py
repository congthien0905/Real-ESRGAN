import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import io
from PIL import Image


def xu_ly_nhieu(img):
    """Hàm xử lý ảnh bằng Real-ESRGAN."""

    model_name = 'RealESRGAN_x2plus.pth'
    denoise_strength = 0.5
    outscale = 0.95
    face_enhance = False  # Có thể thay đổi tùy theo yêu cầu
    ext = 'auto'

    # determine models according to model names
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 1
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    img = np.array(img)  # Chuyển đổi ảnh PIL thành numpy array để xử lý
    if img.shape[2] == 4:  # RGBA
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        raise RuntimeError('If you encounter CUDA out of memory, try to set --tile with a smaller number.')

    return Image.fromarray(output)


app = Flask(__name__)
CORS(app)

@app.route('/xu-ly-nhieu', methods=['POST'])
def xu_ly_anh():
    try:
        # Lấy file ảnh từ request (phải gửi dưới dạng form-data)
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        image_file = request.files['image']

        # Lấy tham số format từ yêu cầu (client có thể gửi "jpeg" hoặc "png")
        output_format = request.form.get('format', 'jpeg').lower()  # Mặc định là 'jpeg'

        # Mở file ảnh bằng PIL
        img = Image.open(image_file)

        # Cải thiện ảnh
        improved_img = xu_ly_nhieu(img)

        # Lưu ảnh đã cải thiện vào bộ đệm
        buffered = io.BytesIO()

        # Xác định định dạng ảnh xuất ra và mimetype tương ứng
        if output_format == 'png':
            mimetype = 'image/png'
            improved_img.save(buffered, format='PNG')
        else:
            mimetype = 'image/jpeg'
            improved_img.save(buffered, format='JPEG')

        buffered.seek(0)

        # Trả về ảnh trực tiếp
        return send_file(buffered, mimetype=mimetype)
    except Exception as e:
        print(e)  # Debug lỗi server nếu có
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
