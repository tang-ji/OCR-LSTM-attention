# Attention-OCR

A multi-digit-letter recognition based on Attention OCR, run under Python 2.7 .

<img src="/picture/1.png" width="400" height="200" /> <img src="/picture/3.png" width="400" height="200" />
<img src="/picture/2.png" width="400" height="200" /> <img src="/picture/4.png" width="400" height="200" />

Pakages required: numpy, tensorflow==1.2.0, pillow, scikit-image, opencv2, six

Put every image in the folder "train" end with "_ABC.jpg", which has a label "ABC".

If we need a tiket enrichment, we could use the command for applying a random rotation between -10~10 degrees for all the images in the folder "images":

    python rotate.py

Also we have a ticket generator that could generate similar images as the original tickets:

    python generator.py

Then we could transform the images in the folder "images" into "tfrecords":

    python data.py
    
Then begin the trainning
    
    python train.py
    
Finally, put all the images in the folder "images" for prediction
    
    python predict.py

By the way, we could see the log in the file: log.txt
    
    
