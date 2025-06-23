import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


data = np.load("results/frames_human_blood/Main_dictionnary.npz", allow_pickle=True)
main_dict = data["arr_0"].item()

print("available frames:", list(main_dict.keys())[:6])



selected_frame = "0000" 

img_path = main_dict[selected_frame ]['adress']
masks = main_dict[selected_frame ]['masks']

img = np.array(Image.open(img_path))
print(img)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("segmentation mask")
plt.imshow(img, cmap='gray')
plt.imshow(masks, alpha=0.4, cmap='jet')
plt.axis('off')

plt.tight_layout()
plt.show()



outlines = main_dict[selected_frame]['outlines']

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')

for outline in outlines:
    plt.plot(outline[:, 1], outline[:, 0], color='red')

plt.title("outlines")
plt.axis('off')
plt.show()
