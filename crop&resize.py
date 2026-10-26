import os
from PIL import Image

# 定义源文件夹和目标文件夹
source_dir = 'F:\cjh\Low latitude\FZA0M'
target_dir = 'F:\cjh\Low latitude\FZA0M_re'

# 需要处理的年份，可以根据需要修改
years_to_process = [ '2009','2010', '2011', '2012','2013',
                  '2017', '2018', '2019',]  # 修改为你需要处理的年份列表

# 裁剪和调整大小的参数
crop_box = (176, 199, 809, 531)  # 裁剪区域 (左，上，右，下)
resize_size = (448, 448)  # 最终调整大小的目标尺寸

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 遍历每个年份文件夹
for year in years_to_process:
    source_year_dir = os.path.join(source_dir, year)
    target_year_dir = os.path.join(target_dir, year)

    # 确保目标年份文件夹存在
    os.makedirs(target_year_dir, exist_ok=True)

    # 遍历该年份文件夹中的所有图片
    for file_name in os.listdir(source_year_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(source_year_dir, file_name)

            # 打开图片
            img = Image.open(file_path)

            # 裁剪图片
            cropped_img = img.crop(crop_box)

            # 调整图片大小
            resized_img = cropped_img.resize(resize_size)

            # 保存处理后的图片到目标文件夹
            target_file_path = os.path.join(target_year_dir, file_name)
            resized_img.save(target_file_path)

            print(f"处理完成: {target_file_path}")

print("所有图片处理完成！")
