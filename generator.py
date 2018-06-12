from PIL import Image
import os
import random
import string
import data
from PIL import Image,ImageDraw,ImageFont,ImageFilter

font_path = 'Arial.ttf'

number = 9
size = (165,55)

color = [[(196,168,68), (96,57,0), (96,57,0)],
         [(69,76,32), (131,130,84), (15,18,1)],
         [(140,157,127), (37,53,27), (97,111,78)],
         [(144,54,4), (162,93,15), (162,93,15)],
         [(97,126,104), (182,187,164), (182,187,164)],
         [(193,175,135), (153,137,78), (123,94,38)],
         [(79,52,9), (206,160,74), (216,153,51)],
         [(193,168,65), (177,147,51), (86,67,9)],
         [(186,166,131), (253,222,157), (253,222,157)],
         [(203,204,186), (176,179,160), (176,179,160)]]
Line_color = [(96,57,0), (15,18,1), (97,111,78), (162,93,15), (182,187,164), (123,94,38), (216,153,51), (86,67,9)]

draw_line = True
line_number = (0,3)

adress = os.path.abspath('.') + "/generate_image/"

def gene_text():
    source_l = list(string.ascii_uppercase)
    source_n = []
    source_blank = [' ', '']
    for index in range(0,10):
        source_n.append(str(index))

    st = '' + ''.join(random.sample(source_n,1))+''.join(random.choice(source_blank)) + ''.join(random.sample(source_l,3))+''.join(random.choice(source_blank)) + ''.join(random.sample(source_n,3))+''.join(random.choice(source_blank)) +''.join(random.sample(source_l,2))
    return st

def gene_randomtext():
    source_l = list(string.ascii_uppercase)
    source_n = []
    source_blank = [' ', '']
    for index in range(0,10):
        source_n.append(str(index))

    st = ''.join(random.sample(source_n + source_l + source_blank, random.randint(5, 30)))
    return st

def gene_line(draw,width,height,linecolor, number):
    line_color = linecolor
    while number > 0:
        begin = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([begin, end], fill = linecolor)
        line_color = random.choice(Line_color)
        number -= 1

def gene_code(n):
    if not os.path.exists(adress):
        os.makedirs(adress)
    f = open(adress + 'generate_text.txt', 'w')
    for i in range(n):
        font = ImageFont.truetype(font_path,25)
        subfont = ImageFont.truetype(font_path, 13 + int(7 * (random.random()-0.5)))
        text = gene_text()
        label = ''.join(text.split(' '))
        if random.random() < 0.1:
            text = '*' + text + '*'
        subtext = gene_randomtext()
        font_width, font_height = font.getsize(text)
        subfont_width, subfont_height = font.getsize(subtext)

        width = int(len(text) * font_width / 8 + 10)
        height = int(size[1]*(1+ 1.1*(random.random()-0.1)))
        bgcolor, fontcolor, linecolor = random.choice(color)
        image = Image.new('RGBA',(width,height),bgcolor)


        draw = ImageDraw.Draw(image)
        subheight = [25, 45]

        draw.text(((width - font_width) / number, (height - font_height) / number),text,\
                font= font,fill=fontcolor)
        draw.text(((width - subfont_width) / number, random.choice(subheight) + (height - subfont_height) / number),subtext,\
                font= subfont,fill=fontcolor)
        if draw_line:
            gene_line(draw,width,height,linecolor,random.choice(line_number))
        extra_data = (0.85 ,0.3*(random.random()-0.5),0.14*(random.random()-0.5),0.14*(random.random()-0.5),0.8+0.3*(random.random()-0.5),0.2)
        image = image.transform((width+ 3 + int(3*(random.random()-0.5)),height+int(5+10*(random.random()-0.5))), Image.AFFINE,extra_data ,Image.BILINEAR)

        # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

        image = image.convert('L')
        image = image.rotate(30 * (random.random()-0.5))
        image = image.resize([int(image.size[0] * (1 + 0.4 * (random.random()-0.5))), int(image.size[1]* (1 + 0.4 * (random.random()-0.5)))],Image.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        image = image.resize([160, 60], Image.LANCZOS)
        f.write(adress + str(i) +'.jpg ' + label +"\n")
        image.save(adress + str(i) +'.jpg', 'jpeg')
    f.close()

if __name__ == "__main__":
    gene_code(5000)
    data.generate(adress + 'generate_text.txt', 'train_g.tfrecords')
    print('Dataset saved as "train_g.tfrecords"')