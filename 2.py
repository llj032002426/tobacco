from PIL import Image

def emf_to_jpg(emf_file, jpg_file):
    try:
        img = Image.open(emf_file)
        img.save(jpg_file)
        print("转换成功!")
    except Exception as e:
        print("转换失败:", e)

# 调用转换函数
emf_to_jpg('./picture/图2.emf', './picture/图2.jpg')
