# ImageNet

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '... (ImageNet)'
print now

#XMLBASE = '../Exploration/ImageNet/bbs'
XMLBASE = '../Data/Images/ImageNet/BBs/Annotation'


def xml_to_bb(xml_path):
    this_baseid = os.path.splitext(os.path.basename(xml_path))[0]
        
    if not os.path.exists(get_imagenet_filename(this_baseid)):
        #print this_baseid, 'does not exist. Skipping it.'
        return None
    
    img = plt.imread(get_imagenet_filename(this_baseid))
    if len(img.shape) != 3:
        return None
    real_x, real_y, _ = img.shape

    scale_x = 1
    scale_y = 1
    
    tree = ET.parse(xml_path)
    this_object = {}
    for element in tree.getroot():
        if element.tag == 'folder':
            this_object['folder'] = element.text
        if element.tag == 'filename':
            this_object['filename'] = element.text
        if element.tag == 'size':
            this_object['width'] = element.find('width').text
            this_object['height'] = element.find('height').text
            scale_x = img.shape[1] / int(this_object['width'])
            scale_y = img.shape[0] / int(this_object['height'])
        if element.tag == 'object':
            bb = element.find('bndbox')
            this_object['bb_x'] = int(bb.find('xmin').text) * scale_x
            this_object['bb_y'] = int(bb.find('ymin').text) * scale_y
            tmp_x = bb.find('xmax').text
            tmp_y = bb.find('ymax').text        
            this_object['bb_w'] = int(tmp_x) * scale_x - int(this_object['bb_x'])
            this_object['bb_h'] = int(tmp_y) * scale_y - int(this_object['bb_y'])
        # print element.tag, element.text
    return this_object




all_bb_xmls = glob(XMLBASE + '/*/*xml')

in_icorp = icorpus_code['image_net']

rows = []
for this_xml in all_bb_xmls:
    this_bb_dict = xml_to_bb(this_xml)
    if this_bb_dict is not None:
        this_image_id_n, this_region_id = this_bb_dict['filename'].split('_')
        this_image_id = this_image_id_n[1:] # remove the leading "n"
        this_bb = [this_bb_dict['bb_x'],
                   this_bb_dict['bb_y'],
                   this_bb_dict['bb_w'],
                   this_bb_dict['bb_h']]
        rows.append((in_icorp,
                     int(this_image_id),
                     int(this_region_id),
                     this_bb,
                     this_image_id_n))

bbdf_in = pd.DataFrame(rows,
                       columns=['i_corpus', 'image_id', 
                                'region_id', 'bb', 'cat'])

with gzip.open('PreProcOut/imagenet_bbdf.pklz', 'w') as f:
    pickle.dump(bbdf_in, f)
