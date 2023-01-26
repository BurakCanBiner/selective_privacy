from matplotlib import pyplot as plt
from os import listdir,path
from PIL import Image

loc = 'Resnet18/split6'

# create figure
fig = plt.figure(figsize=(9, 6))

# setting values to rows and column variables
rows = 3
columns = 7

# reading images
prune1 = path.join(loc,'prune1')
prune99 = path.join(loc,'prune99')
prune5 = path.join(loc,'prune5')

row_1 = [None for i in range(len(list(listdir(prune1))))]
row_99 = [None for i in range(len(list(listdir(prune99))))]
row_5 = [None for i in range(len(list(listdir(prune5))))]

# row_03 = [None for i in range(len(list(listdir(prune16))))]

loc_dic = {'25':1,'50':2,'75':3,'100':4,'125':5,'149':6,'150':6}


epoch_list = ['original','25', '50', '75', '100', '125', '149']
    
# str_list = []
row_1 = []
for ind, ep in enumerate(epoch_list):
    if ep == 'original':
        file = 'real_test_alt_1.png'
        row_1.append(Image.open(path.join(prune1, file)))
    else:
        file = 'recon_test_alt_' + ep + '.png'
        row_1.append( Image.open(path.join(prune1, file)))

row_5 = []
for ind, ep in enumerate(epoch_list):
    if ep == 'original':
        file = 'real_test_alt_1.png'
        row_5.append(Image.open(path.join(prune5, file)))
    else:
        file = 'recon_test_alt_' + ep + '.png'
        row_5.append( Image.open(path.join(prune5, file)))
        
row_99 = []
for ind, ep in enumerate(epoch_list):
    if ep == 'original':
        file = 'real_test_alt_1.png'
        row_99.append(Image.open(path.join(prune99, file)))
    else:
        file = 'recon_test_alt_' + ep + '.png'
        row_99.append( Image.open(path.join(prune99, file)))
        
all_images = [*row_1,*row_5,*row_99]

epocz = [25,50,75,100,125,150]

prunes = [0.1,0.5,0.99]

# Adds a subplot at the 1st position
for i,img in enumerate(all_images):
    fig.add_subplot(rows, columns, i+1)
    
    plt.imshow(img)
    
    if i % columns == 0:
        pruned = prunes[i //columns]
        plt.ylabel('Pruned {}'.format(pruned))
    if i <7:
        if i==0:
            plt.title("Original")
        else:
            plt.title('Epoch {}'.format(epocz[i-1]))
    plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    plt.tight_layout()
plt.show()