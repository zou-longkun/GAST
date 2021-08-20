import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json

file = "F:\\SSL_PDA\\visualization\\convergence\\convergence_s2m.json"
finetune_file = "F:\\SSL_PDA\\visualization\\convergence\\finetune_convergence_s2m2(20).json"
unsu_file = "F:\\SSL_PDA\\visualization\\convergence\\unsu_convergence_s2m.json"
data = json.load(open(file, 'r'))
finetune_data = json.load(open(finetune_file, 'r'))
unsu_data = json.load(open(unsu_file, 'r'))

unsu_src_val_acc_list = np.array(unsu_data[0])[:150]
unsu_src_val_loss_list = np.array(unsu_data[1])[:150]
unsu_trgt_val_acc_list = np.array(unsu_data[2])[:150]
unsu_trgt_val_loss_list = np.array(unsu_data[3])[:150]

finetune_src_val_acc_list = np.array(finetune_data[0])[:20]
finetune_src_val_loss_list = np.array(finetune_data[1])[:20]
finetune_trgt_val_acc_list = np.array(finetune_data[2])[:20]
finetune_trgt_val_loss_list = np.array(finetune_data[3])[:20]

src_val_acc_list = np.append(np.array(data[0])[:130], finetune_src_val_acc_list)
src_val_loss_list = np.append(np.array(data[1])[:130], finetune_src_val_loss_list)
trgt_val_acc_list = np.append(np.array(data[2])[:130], finetune_trgt_val_acc_list)
trgt_val_loss_list = np.append(np.array(data[3])[:130], finetune_trgt_val_loss_list)

x1 = range(0, 150)
x2 = range(0, 150)
x3 = range(0, 150)
x4 = range(0, 150)
x5 = range(0, 150)
x6 = range(0, 150)
y1 = trgt_val_acc_list
y2 = trgt_val_loss_list
y3 = src_val_acc_list
y4 = src_val_loss_list
y5 = unsu_trgt_val_acc_list
y6 = unsu_trgt_val_loss_list

plt.subplot(2, 1, 1)
plt.plot(x1, y1, '.-', color='r', label='GAST_Trgt_Acc')
plt.plot(x3, y3, '.-', color='g', label='GAST_Src_Acc')
plt.plot(x5, y5, '.-', color='y', label='w/o Adapt_Trgt_Acc')
#plt.title('Test accuracy vs. epoches')
plt.legend(loc='lower right')
plt.ylabel('Test Accuracy')
x_major_locator=MultipleLocator(15)
# y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-', color='r', label='GAST_Trgt_Loss')
plt.plot(x4, y4, '.-', color='g', label='GAST_Src_Loss')
plt.plot(x6, y6, '.-', color='y', label='w/o Adapt_Trgt_Loss')
plt.legend(loc='upper right')
plt.xlabel('Number of Epochs')
plt.ylabel('Test Loss')

x_major_locator=MultipleLocator(15)
# y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

plt.savefig("accuracy_loss.jpg", dpi=300)
plt.show()
