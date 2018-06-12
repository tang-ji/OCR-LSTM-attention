import random
import numpy as np
from PIL import Image
from skimage import exposure
import cv2
import os
from shutil import copyfile
from defaults import Config as defaults
import tensorflow as tf
from model import Model
tf.logging.set_verbosity(tf.logging.ERROR)

test_image_files = []
imagePath = 'images/'
rotatedPath = 'rotated/'
if not os.path.isdir(imagePath + rotatedPath):
    os.mkdir(imagePath + rotatedPath)
savedPath = 'saved/'
if not os.path.isdir(imagePath + savedPath):
    os.mkdir(imagePath + savedPath)
testF = []
for (dirpath, dirnames, filenames) in os.walk(imagePath):
    testF.extend(filenames)
    break
for testName in testF:
    if testName.split('.')[-1] == 'png' or testName.split('.')[-1] == 'jpg':
        test_image_files.append(imagePath + testName)



dataset_path = 'train_tickets.tfrecords'
num_epoch = 2
batch_size = 64
steps_per_checkpoint = 30
max_prediction_length = 13
initial_learning_rate = 1.0
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
model = Model(
    phase='predict',
    visualize=defaults.VISUALIZE,
    output_dir=defaults.OUTPUT_DIR,
    batch_size=batch_size,
    initial_learning_rate=initial_learning_rate,
    steps_per_checkpoint=steps_per_checkpoint,
    model_dir=defaults.MODEL_DIR,
    target_embedding_size=defaults.TARGET_EMBEDDING_SIZE,
    attn_num_hidden=128,
    attn_num_layers=8,
    clip_gradients=defaults.CLIP_GRADIENTS,
    max_gradient_norm=defaults.MAX_GRADIENT_NORM,
    session=sess,
    load_model=defaults.LOAD_MODEL,
    gpu_id=defaults.GPU_ID,
    use_gru=False,
    use_distance=defaults.USE_DISTANCE,
    max_image_width=defaults.MAX_WIDTH,
    max_image_height=defaults.MAX_HEIGHT,
    max_prediction_length=max_prediction_length,
    channels=1,
)


A = "confifence: A (0 fault: 67%, 1 fault: 25%, 2 faults: 5%, 3 faults: 2%, 3++ faults: 0%)"
B = "confifence: B (0 fault: 40%, 1 fault: 40%, 2 faults: 12%, 3 faults: 4%, 3++ faults: 2%)"
C = "confifence: C (0 fault: 25%, 1 fault: 30%, 2 faults: 30%, 3 faults: 1%, 3++ faults: 5%)"
D = "confifence: D (0 fault: 2%, 1 fault: 14%, 2 faults: 13%, 3 faults: 13%, 3++ faults: 60%)"

def level(str):
    s = float(str)
    if s > 0.98:
        return A
    if 0.9 < s and s <= 0.98:
        return B
    if 0.8 < s and s <= 0.9:
        return C
    else:
        return D


log = open(imagePath + 'result.txt', 'a')
log.close()
os.remove(imagePath + 'result.txt')
for testFile in test_image_files:
    max_file = testFile
    max_text, max_pro = model.predict(open(testFile, 'rb').read())

    if max_pro < 0.95:
        for i in range(10):
            img = cv2.imread(testFile)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = exposure.adjust_gamma(img, 0.6 + i * 0.1)
            img = cv2.equalizeHist(img)
            img = exposure.adjust_gamma(img, 0.6 + i * 0.1)
            img = cv2.resize(img,(160,60),interpolation=cv2.INTER_CUBIC)
            Image.fromarray(np.asarray(img)).save(imagePath + 'rotated/' + str(int(1000*random.random())) + '_' + testFile.split('_')[-1].split('/')[-1])


        rotated_image_files = []
        f = []
        for (dirpath, dirnames, filenames) in os.walk(imagePath + rotatedPath):
            f.extend(filenames)
            break
        for path in f:
            if path.split('.')[-1] == 'png' or path.split('.')[-1] == 'jpg':
                rotated_image_files.append(imagePath + rotatedPath + path)

        for filename in rotated_image_files:
            text, probability = model.predict(open(filename, 'rb').read())
            if probability > max_pro:
                max_pro = probability
                max_file = filename
                max_text = text
                copyfile(max_file, imagePath + 'saved/' + testFile.split('_')[-1].split('/')[-1])
            os.remove(filename)

    if max_pro < 0.95:
        for i in range(15):
            img = cv2.imread(testFile)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             img = cv2.GaussianBlur(img,(5,5),0)
            img = cv2.equalizeHist(img)
            img = exposure.adjust_gamma(img, 0.5 + i * 0.1)
            img = cv2.equalizeHist(img)
            img = exposure.adjust_gamma(img, 0.5 + i * 0.1)
            img = cv2.equalizeHist(img)
            img = exposure.adjust_gamma(img, 0.5 + i * 0.1)
            img = cv2.resize(img,(160,60),interpolation=cv2.INTER_CUBIC)
            Image.fromarray(np.asarray(img)).save(imagePath + 'rotated/' + str(int(1000*random.random())) + '_' + testFile.split('_')[-1].split('/')[-1])


        rotated_image_files = []
        f = []
        for (dirpath, dirnames, filenames) in os.walk(imagePath + rotatedPath):
            f.extend(filenames)
            break
        for path in f:
            if path.split('.')[-1] == 'png' or path.split('.')[-1] == 'jpg':
                rotated_image_files.append(imagePath + rotatedPath + path)

        for filename in rotated_image_files:
            text, probability = model.predict(open(filename, 'rb').read())
            if probability > max_pro:
                max_pro = probability
                max_file = filename
                max_text = text
                copyfile(max_file, imagePath + 'saved/' + testFile.split('_')[-1].split('/')[-1])
            os.remove(filename)

    if max_pro < 0.95:
        img = Image.open(testFile)
        [rows,cols] = img.size
        img = cv2.imread(imagePath + 'saved/' + testFile.split('_')[-1].split('/')[-1])
        for i in range(100):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),15 * (random.random()-0.5),1)
            try:
                img = cv2.warpAffine(img,M,(cols,rows))
                Image.fromarray(np.asarray(img)).save(imagePath + 'rotated/' + str(int(1000*random.random())) + '_' + testFile.split('_')[-1].split('/')[-1])
            except cv2.error:
                continue

        rotated_image_files = []
        f = []
        for (dirpath, dirnames, filenames) in os.walk(imagePath + rotatedPath):
            f.extend(filenames)
            break
        for path in f:
            if path.split('.')[-1] == 'png' or path.split('.')[-1] == 'jpg':
                rotated_image_files.append(imagePath + rotatedPath + path)

        for filename in rotated_image_files:
            text, probability = model.predict(open(filename, 'rb').read())
            if probability > max_pro:
                max_pro = probability
                max_file = filename
                max_text = text
                copyfile(max_file, imagePath + 'saved/' + testFile.split('_')[-1].split('/')[-1])
            os.remove(filename)

    log = open(imagePath + 'result.txt', 'a')
    log.write(testFile + ' '+ '\n' + str(max_text) + ' '+ '\n' + level(max_pro) + '\n-----------------------------------------------------------\n')
    log.close()