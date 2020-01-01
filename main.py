import os
import time
import skimage.io
import sys
import torch
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


classes = np.asarray([
"bg",
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor",
])


def plot_side_by_side(image, segmentation_mask, label):
	fig = plt.figure()
	ax1 = fig.add_subplot(1,2,1)
	ax1.imshow(image)
	ax2 = fig.add_subplot(1,2,2)
	ax2.imshow(segmentation_mask)
	plt.title(label)
	plt.show()


if __name__ == '__main__':
	custom_image = False
	image_dir = "./images"

	model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
	model.eval()

	checkpoint_name = "deeplabv3_resnet101_pretrained_VOC_21_classes.pth"
	if not os.path.exists(checkpoint_name):
		torch.save(model.state_dict(), checkpoint_name) # 233Mb :c
	# print(model)

	if custom_image is False:
		# Download an example image from the pytorch website
		url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
		try: 
			urllib.URLopener().retrieve(url, filename)
		except: 
			urllib.request.urlretrieve(url, filename)
	else:
		filename = "cat.jpg"

	# sample execution (requires torchvision)
	input_image = Image.open(os.path.join(image_dir, filename))
	preprocess = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	input_tensor = preprocess(input_image)
	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

	# move the input and model to GPU for speed if available
	if torch.cuda.is_available():
	    input_batch = input_batch.to('cuda')
	    model.to('cuda')

	with torch.no_grad():
	    output = model(input_batch)['out'][0]
	output_predictions = output.argmax(0)

	# create a color pallette, selecting a color for each class
	palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
	colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
	colors = (colors % 255).numpy().astype("uint8")

	# plot the semantic segmentation predictions of 21 classes in each color
	r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
	r.putpalette(colors)

	unique_classes, classwise_pixel_amount = np.unique(r, return_counts=True)
	finded_labels = ", ".join(list(classes[unique_classes]))

	plot_side_by_side(input_image, r, finded_labels)