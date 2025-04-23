# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:19:28 2022

@author: wang
"""
import torchvision
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L2CarliniWagnerAttack
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import time






model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

# get data and test the model
# wrapping the tensors with ep.astensors is optional, but it allows
# us to work with EagerPy tensors in the following
images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=20))

labels = ep.astensor(labels.raw.long())
clean_acc = accuracy(fmodel, images, labels)
print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
attack = FGSM()
epsilons = [0.004]

start_time = time.time()
raw_advs, clipped_advs, success =  attack(fmodel, images, labels, epsilons=epsilons)



#Question 1
result = np.where(success.raw.cpu().numpy() == True)
print(f"Attack Success Rate:  {result[1].size/20 * 100:.1f} %")
print("--- Total Execution Time %s seconds ---" % (time.time() - start_time))




#Question 3
magnify = 1
image_num = 0

perturbation = (clipped_advs[0] - images[0]).raw[image_num].permute(1,2,0).cpu().detach().numpy()*255*magnify
perturb = Image.fromarray(perturbation.astype(np.uint8))
perturb.save("perturbation_only.jpg")

original_image = images.raw[image_num].permute(1,2,0).cpu().detach().numpy()*255*magnify
im = Image.fromarray(original_image.astype(np.uint8))
im.save("original_img.jpg")

adv = clipped_advs[0].raw[image_num].permute(1,2,0).cpu().detach().numpy()*255*magnify
adv_im = Image.fromarray(adv.astype(np.uint8))
adv_im.save("adv_img.jpg")



